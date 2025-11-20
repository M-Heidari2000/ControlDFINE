import torch
import numpy as np
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
    num_points: int=1,
):
    assert low.shape == high.shape

    axes = []
    for lo, hi in zip(low, high):
        step = (hi - lo) / num_points
        centers = lo + (np.arange(num_points) + 0.5) * step
        axes.append(centers)

    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack(mesh, axis=-1).reshape(-1, low.shape[0])
    return grid.astype(np.float32)