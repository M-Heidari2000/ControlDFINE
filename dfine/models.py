import torch
import torch.nn as nn
from typing import Optional
import torch.nn.init as init
from torch.distributions import MultivariateNormal


class Encoder(nn.Module):
    """
        y_t -> a_t
    """

    def __init__(
        self,
        a_dim: int,
        y_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*y_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, a_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, y: torch.Tensor):
        return self.mlp_layers(y)
    

class Decoder(nn.Module):
    """
        a_t -> y_t
    """

    def __init__(
        self,
        a_dim: int,
        y_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*a_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(a_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, a: torch.Tensor):
        return self.mlp_layers(a)


class CostModel(nn.Module):
    """
        Learnable quadratic cost function in the latent space
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
    ):
        
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        
        self.A = nn.Parameter(torch.eye(x_dim, dtype=torch.float32))
        self.B = nn.Parameter(torch.eye(u_dim, dtype=torch.float32))
        self.q = nn.Parameter(torch.randn((x_dim, ), dtype=torch.float32))

    @property
    def Q(self):
        return self.A @ self.A.T
    
    @property
    def R(self):
        L = torch.tril(self.B)
        diagonals = nn.functional.softplus(L.diagonal()) + 1e-4
        X = 1 - torch.eye(self.u_dim, device=self.B.device, dtype=torch.float32)
        L = L * X + diagonals.diag()
        return L @ L.T
    
    def forward(self, x: torch.Tensor, u: torch.Tensor):
        # x: b x
        # u: b u
        xQx = torch.einsum('bi,ij,bj->b', x, self.Q, x)
        uRu = torch.einsum('bi,ij,bj->b', u, self.R, u)
        xq = torch.einsum('bi,i->b', x, self.q)
        cost = 0.5 * (xQx + uRu) + xq
        cost = cost.unsqueeze(1)
        return cost
        

class Dynamics(nn.Module):
    
    """
        KF that obtains belief over x_{t+1} using belief of x_t, u_t, and y_{t+1}
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        a_dim: int,
        hidden_dim: Optional[int]=128,
        min_var: float=1e-4,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self._min_var = min_var

        self.backbone = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.A_head = nn.Linear(hidden_dim, x_dim * x_dim)
        self.B_head = nn.Linear(hidden_dim, x_dim * u_dim)
        self.C_head = nn.Linear(hidden_dim, a_dim * x_dim)
        self.nx_head = nn.Linear(hidden_dim, x_dim)
        self.na_head = nn.Linear(hidden_dim, a_dim)
        self.alpha = nn.Parameter(torch.tensor([1e-2]))

        self._init_weights()

    def _init_weights(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def make_psd(self, P, eps=1e-6):
        b = P.shape[0]
        P = 0.5 * (P + P.transpose(-1, -2))
        P = P + eps * torch.eye(P.size(-1), device=P.device).expand([b, -1, -1])
        return P

    def get_dynamics(self, x):
        """
            get dynamics matrices depending on the state x
        """
        b = x.shape[0]
        hidden = self.backbone(x)
        I = torch.eye(self.x_dim, device=x.device).expand([b, -1, -1])
        A = I + self.alpha * self.A_head(hidden).reshape(b, self.x_dim, self.x_dim)
        B = self.B_head(hidden).reshape(b, self.x_dim, self.u_dim)
        C = self.C_head(hidden).reshape(b, self.a_dim, self.x_dim)
        Nx = torch.diag_embed(nn.functional.softplus(self.nx_head(hidden)) + self._min_var)
        Na = torch.diag_embed(nn.functional.softplus(self.na_head(hidden)) + self._min_var)

        return A, B, C, Nx, Na
    
    def get_a(self, x):
        """
        returns emissions (a) based on the input state (x)
        """

        _, _, C, _, _ = self.get_dynamics(x=x)
        return torch.einsum('bij,bj->bi', C, x)

    def dynamics_update(
        self,
        dist: MultivariateNormal,
        u: torch.Tensor,
    ):
        """
            Single step dynamics update

            mean: b x
            cov: b x x
            u: b u
        """

        mean = dist.loc
        cov = dist.covariance_matrix

        A, B, _, Nx, _ = self.get_dynamics(x=mean)

        next_mean = torch.einsum('bij,bj->bi', A, mean) + torch.einsum('bij,bj->bi', B, u)
        next_cov = torch.einsum('bij,bjk,bkl->bil', A, cov, A.transpose(1, 2)) + Nx
        next_cov = self.make_psd(next_cov)
        updated_dist = MultivariateNormal(loc=next_mean, covariance_matrix=next_cov)

        return updated_dist
    
    def measurement_update(
        self,
        dist: MultivariateNormal,
        a: torch.Tensor,
    ):
        """
            Single step measurement update
        
            mean: b x
            cov: b x x
            a: b a
        """

        mean = dist.loc
        cov = dist.covariance_matrix

        _, _, C, _, Na = self.get_dynamics(x=mean)

        S = torch.einsum('bij,bjk,bkl->bil', C, cov, C.transpose(1, 2)) + Na
        G = torch.einsum('bij,bjk,bkl->bil', cov, C.transpose(1, 2), torch.linalg.pinv(S))
        innovation = a - torch.einsum('bij,bj->bi', C, mean)
        next_mean = mean + torch.einsum('bij,bj->bi', G, innovation)
        next_cov = cov - torch.einsum('bij,bjk,bkl->bil', G, C, cov)
        next_cov = self.make_psd(next_cov)
        updated_dist = MultivariateNormal(loc=next_mean, covariance_matrix=next_cov)

        return updated_dist