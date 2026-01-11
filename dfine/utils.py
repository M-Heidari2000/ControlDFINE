import torch
import numpy as np
from typing import Union, List, Dict, Any
from torch.distributions import kl_divergence
from torch.distributions import MultivariateNormal


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
):
    prior_mean = prior.loc
    posterior_mean = posterior.loc
    mean_consistency = (
        2 * (prior_mean - posterior_mean).norm(dim=1, p=2) /
        (prior_mean.norm(dim=1, p=2) + posterior_mean.norm(dim=1, p=2)  + 1e-6)
    ).mean()
    kl_consistency = kl_divergence(posterior, prior).mean()

    return mean_consistency, kl_consistency


def make_grid(
    low: np.ndarray,
    high: np.ndarray,
    num_regions: Union[int, np.ndarray],
    num_points: int,
    rng: np.random.Generator | None = None,
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
        assert num_regions.shape == (d, )

    widths = (high - low) / num_regions

    regions: List[Dict[str, np.ndarray]] = []

    for cell_index in np.ndindex(*num_regions):
        cell_index = np.array(cell_index)

        cell_low = low + cell_index * widths
        cell_high = cell_low + widths

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