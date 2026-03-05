import torch
import torch.nn as nn
from typing import Optional
import torch.nn.init as init
from torch.distributions import MultivariateNormal


class Encoder(nn.Module):
    """
        y_t -> a_t
    """

    def __init__(self, a_dim: int, y_dim: int, hidden_dim: int):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, a_dim),
        )

    def forward(self, y: torch.Tensor):
        return self.mlp_layers(y)
    

class Decoder(nn.Module):
    """
        a_t -> y_t
    """

    def __init__(self, a_dim: int, y_dim: int, hidden_dim: int):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(a_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, a: torch.Tensor):
        return self.mlp_layers(a)


class CostModel(nn.Module):
    def __init__(self, x_dim: int, u_dim: int):
        """
            Learnable quadratic cost function
        """
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.A = nn.Parameter(torch.eye(self.x_dim, dtype=torch.float32))
        self.q = nn.Parameter(torch.randn((1, self.x_dim), dtype=torch.float32))
        self.register_buffer("R", 1e-6 * torch.eye(self.u_dim, dtype=torch.float32))

    @property
    def Q(self):
        return self.A @ self.A.T

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        res = x - self.q
        xQx = torch.einsum("bi,ij,bj->b", res, self.Q, res)
        uRu = torch.einsum("bi,ij,bj->b", u, self.R, u)
        cost = 0.5 * (xQx + uRu).reshape(-1, 1)
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
        min_var: float=1e-2,
        max_var: float=1.0,
        locally_linear: Optional[bool]=False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self._min_var = min_var
        self._max_var = max_var
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
        P = P + eps * torch.eye(P.size(-1), device=P.device).expand(b, -1, -1)
        return P

    def get_dynamics(self, x):
        """
            get dynamics matrices depending on the state x
        """
        b = x.shape[0]

        if self.locally_linear:
            hidden = self.backbone(x)
            I = torch.eye(self.x_dim, device=x.device).expand(b, -1, -1)
            A = I + self.alpha * self.A_head(hidden).reshape(b, self.x_dim, self.x_dim)
            B = self.B_head(hidden).reshape(b, self.x_dim, self.u_dim)
            C = self.C_head(hidden).reshape(b, self.a_dim, self.x_dim)
            Nx = torch.diag_embed(self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.nx_head(hidden)))
            Na = torch.diag_embed(self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.na_head(hidden)))
        else:
            A = self.A.expand(b, -1, -1)
            B = self.B.expand(b, -1, -1)
            C = self.C.expand(b, -1, -1)
            Nx = torch.diag_embed(self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.nx)).expand(b, -1, -1)
            Na = torch.diag_embed(self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.na)).expand(b, -1, -1)

        return A, B, C, Nx, Na
    
    def get_a(self, x):
        """
        returns emissions (a) based on the input state (x)
        """

        _, _, C, _, _ = self.get_dynamics(x=x)
        return torch.einsum('bij,bj->bi', C, x)

    def prior(
        self,
        dist: MultivariateNormal,
        u: torch.Tensor,
    ):
        """
            single step dynamics update

            dist: N(b x, b x x)
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
    
    def posterior(
        self,
        dist: MultivariateNormal,
        a: torch.Tensor,
    ):
        """
            single step measurement update
        
            dist: N(b x, b x x)
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
    
    def generate(self, dist: MultivariateNormal, u: torch.Tensor):
        """
            generates trajectory given the initial belief and list of actions
            uses mean decoding
        """

        with torch.no_grad():
            samples = []
            
            if u.dim() == 2:
                u = u.unsqueeze(0)
            d, _, _ = u.shape

            for l in range(d):
                dist = self.prior(dist=dist, u=u[l])
                samples.append(dist.loc)
            
            samples = torch.stack(samples, dim=0)

        return samples
    
    def forward(self, u: torch.Tensor, a: torch.Tensor):
        """
            multi step inference of priors and posteriors

            inputs:
                - a: a0:T-1
                - u: u0:T-1
            outputs:
                priors: one step priors over the states
                posteriors: posterior over the states
            
            Notes: u[T-1] is not used
        """
        
        T, B, _ = u.shape
        device = u.device
        prior = MultivariateNormal(
            loc=torch.zeros((B, self.x_dim), device=device),
            covariance_matrix=torch.eye(self.x_dim, device=device).expand(B, -1, -1),
        )
        posterior = self.posterior(dist=prior, a=a[0])

        priors = [prior]
        posteriors = [posterior]

        for t in range(T-1):
            prior = self.prior(dist=posterior, u=u[t])
            posterior = self.posterior(dist=prior, a=a[t+1])
            priors.append(prior)
            posteriors.append(posterior)

        return priors, posteriors