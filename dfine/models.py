import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch.nn.init as init
from einops import rearrange
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
        hidden_dim: Optional[int]=64,
        input_penalty: Optional[float]=1.0,
    ):
        
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        
        self.backbone = nn.Sequential(
            nn.Linear(self.x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.L_head = nn.Linear(hidden_dim, x_dim * x_dim)
        self.q_head = nn.Linear(hidden_dim, x_dim)

        self.register_buffer("R", input_penalty * torch.eye(u_dim, dtype=torch.float32))

    def get_cost_matrices(self, x: torch.Tensor):
        h = self.backbone(x)
        L = self.L_head(h).reshape(-1, self.x_dim, self.x_dim)
        Q = L @ L.transpose(-2, -1)
        Q = 0.5 * (Q + Q.transpose(-2, -1))
        q = self.q_head(h)

        return Q, q
    

    def compute_window_cost(self, x: torch.Tensor, u: torch.Tensor, c: torch.Tensor, radius: int = 1):
        """
        x: (T, B, x_dim)
        u: (T, B, u_dim)
        c: (T, B) | (T, B, 1)
        """
        T, B, x_dim = x.shape
        W = 2 * radius + 1

        if c.ndim == 3:
            c = c.squeeze(-1)

        if T <= 2 * radius:
            return torch.zeros((), device=x.device)

        loss = torch.zeros((), device=x.device)
        count = 0

        for t in range(radius, T - radius):
            x_win = x[t-radius: t+radius+1]
            u_win = u[t-radius: t+radius+1]
            c_win = c[t-radius: t+radius+1]

            # compute (Q,q) from center x[t]
            Q, q = self.get_cost_matrices(x[t])

            # repeat across window
            Qw = Q.unsqueeze(0).expand(W, -1, -1, -1)
            qw = q.unsqueeze(0).expand(W, -1, -1)

            dx = x_win - qw
            x_term = torch.einsum("wbi,wbij,wbj->wb", dx, Qw, dx)
            u_term = torch.einsum("wbi,ij,wbj->wb", u_win, self.R, u_win)
            pred = 0.5 * (x_term + u_term)
            loss = loss + F.mse_loss(pred, c_win)
            count += 1

        return loss / max(count, 1)

    def forward(self, x: torch.Tensor, u: torch.Tensor, Q: Optional[torch.Tensor]=None, q: Optional[torch.Tensor]=None):
        # x: b x
        # u: b u
        if Q is None or q is None:
            Q, q = self.get_cost_matrices(x=x)
        dx = x - q
        xQx = torch.einsum('bi,bij,bj->b', dx, Q, dx)
        uRu = torch.einsum('bi,ij,bj->b', u, self.R, u)
        cost = 0.5 * (xQx + uRu)
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
        locally_linear: Optional[bool]=False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self._min_var = min_var
        self.locally_linear = locally_linear

        if self.locally_linear:
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
        else:
            self.A = nn.Parameter(torch.eye(x_dim))
            self.B = nn.Parameter(torch.randn(x_dim, u_dim))
            self.C = nn.Parameter(torch.randn(a_dim, x_dim))
            self.nx = nn.Parameter(torch.randn(x_dim))
            self.na = nn.Parameter(torch.randn(a_dim)) 

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

        if self.locally_linear:
            hidden = self.backbone(x)
            I = torch.eye(self.x_dim, device=x.device).expand([b, -1, -1])
            A = I + self.alpha * self.A_head(hidden).reshape(b, self.x_dim, self.x_dim)
            B = self.B_head(hidden).reshape(b, self.x_dim, self.u_dim)
            C = self.C_head(hidden).reshape(b, self.a_dim, self.x_dim)
            Nx = torch.diag_embed(nn.functional.softplus(self.nx_head(hidden)) + self._min_var)
            Na = torch.diag_embed(nn.functional.softplus(self.na_head(hidden)) + self._min_var)
        else:
            A = self.A.expand(b, -1, -1)
            B = self.B.expand(b, -1, -1)
            C = self.C.expand(b, -1, -1)
            Nx = torch.diag_embed(nn.functional.softplus(self.nx) + self._min_var).expand(b, -1, -1)
            Na = torch.diag_embed(nn.functional.softplus(self.na) + self._min_var).expand(b, -1, -1)

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