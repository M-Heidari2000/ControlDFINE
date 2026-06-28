import torch
import numpy as np
from typing import Union, List, Dict, Any
from torch.distributions import kl_divergence
from torch.distributions import MultivariateNormal


def bottle_mvn(dists: List[MultivariateNormal]):
    """
        concatenates a list of distributions along the batch dimension
    """
    mean = torch.cat([d.loc for d in dists], dim=0)
    cov = torch.cat([d.covariance_matrix for d in dists], dim=0)

    return MultivariateNormal(loc=mean, covariance_matrix=cov)


def pearson_corr(
    true: torch.Tensor,
    pred: torch.Tensor
):

    # mean and std along time dimension
    true_mean = true.mean(dim=0, keepdim=True)  # (1, B, D)
    pred_mean = pred.mean(dim=0, keepdim=True)
    true_std = true.std(dim=0, unbiased=False, keepdim=True)
    pred_std = pred.std(dim=0, unbiased=False, keepdim=True)

    # covariance across time
    cov = ((true - true_mean) * (pred - pred_mean)).mean(dim=0)  # (B, D)

    corr = cov / (true_std.squeeze(0) * pred_std.squeeze(0) + 1e-8)  # (B, D)
    return corr.mean()


def compute_consistency(
    prior: MultivariateNormal,
    posterior: MultivariateNormal,
    free_nats: float=3.0,
):
    prior_mean = prior.loc
    posterior_mean = posterior.loc
    mean_consistency = (
        2 * (prior_mean - posterior_mean).norm(dim=1, p=2) /
        (prior_mean.norm(dim=1, p=2) + posterior_mean.norm(dim=1, p=2)  + 1e-6)
    ).mean()
    kl_consistency = kl_divergence(posterior, prior).clamp(min=free_nats).mean()

    return mean_consistency, kl_consistency


def solve_discrete_lyapunov(A: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Solve W = A W Aᵀ + Q via (I − A⊗A) vec(W) = vec(Q).
    A must have spectral radius < 1.
    """
    n = A.shape[0]
    I = torch.eye(n * n, device=A.device, dtype=A.dtype)
    W = torch.linalg.solve(I - torch.kron(A, A), Q.reshape(-1)).reshape(n, n)
    return 0.5 * (W + W.mT)


def hankel_singular_values(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Infinite-horizon discrete-time Hankel singular values.
      W_c = A W_c Aᵀ + BBᵀ
      W_o = Aᵀ W_o A + CᵀC
      HSVs = sqrt(eig(W_c W_o))
    """
    W_c = solve_discrete_lyapunov(A, B @ B.mT)
    W_o = solve_discrete_lyapunov(A.mT, C.mT @ C)
    eigvals = torch.linalg.eigvals(W_c @ W_o).real
    return eigvals.clamp(min=0).sqrt()


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer — pushes p(a) → N(0, I)."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D)"""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return (err @ self.weights * proj.size(-2)).mean()


def make_grid(
    low: np.ndarray,
    high: np.ndarray,
    num_regions: Union[int, np.ndarray],
    num_points: int,
    rng: np.random.Generator | None = None,
    deterministic: bool = False,
) -> List[Dict[str, np.ndarray]]:

    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    assert low.shape == high.shape
    d = low.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if np.isscalar(num_regions):
        num_regions = np.full(d, int(num_regions), dtype=int)
    else:
        num_regions = np.asarray(num_regions, dtype=int)
        assert num_regions.shape == (d,)

    widths = (high - low) / num_regions

    regions: List[Dict[str, np.ndarray]] = []

    for cell_index in np.ndindex(*num_regions):
        cell_index = np.array(cell_index)

        cell_low = low + cell_index * widths
        cell_high = cell_low + widths

        if deterministic:
            center = (cell_low + cell_high) / 2.0
            samples = center.reshape(1, d).astype(np.float32)
        else:
            samples = rng.uniform(
                low=cell_low,
                high=cell_high,
                size=(num_points, d),
            ).astype(np.float32)

        regions.append(
            {
                "low": cell_low.astype(np.float32),
                "high": cell_high.astype(np.float32),
                "samples": samples,
            }
        )

    return regions


def jsonify(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = jsonify(v)
        elif isinstance(v, list):
            out[k] = [
                x.tolist() if isinstance(x, np.ndarray) else x for x in v
            ]
        else:
            out[k] = v
    return out