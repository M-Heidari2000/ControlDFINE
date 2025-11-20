import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple


def plot_costs(
    target_regions: List[Dict[str, Any]],
) -> Tuple[plt.Figure, plt.Figure]:

    if len(target_regions) == 0:
        raise ValueError("target_regions is empty")

    lows = np.array([r["low"] for r in target_regions])   # (N, 2)
    highs = np.array([r["high"] for r in target_regions]) # (N, 2)

    assert lows.shape[1] == 2, "Only 2D supported for heatmap."

    xs = np.unique(lows[:, 0])
    ys = np.unique(lows[:, 1])
    nx, ny = len(xs), len(ys)

    mean_grid = np.zeros((ny, nx), dtype=np.float32)
    std_grid = np.zeros((ny, nx), dtype=np.float32)

    dx = target_regions[0]["high"][0] - target_regions[0]["low"][0]
    dy = target_regions[0]["high"][1] - target_regions[0]["low"][1]

    for r in target_regions:
        low = r["low"]
        costs = np.asarray(r["costs"], dtype=float)
        mean_c = costs.mean()
        std_c = costs.std()

        ix = np.where(xs == low[0])[0][0]
        iy = np.where(ys == low[1])[0][0]

        mean_grid[iy, ix] = mean_c
        std_grid[iy, ix] = std_c

    x_min = xs.min()
    x_max = xs.max() + dx
    y_min = ys.min()
    y_max = ys.max() + dy
    extent = [x_min, x_max, y_min, y_max]

    fig_mean, ax_mean = plt.subplots()
    im_mean = ax_mean.imshow(
        mean_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm"
    )
    ax_mean.set_title("Mean Cost per Region")
    ax_mean.set_xlabel("x")
    ax_mean.set_ylabel("y")
    cbar_mean = fig_mean.colorbar(im_mean, ax=ax_mean)
    cbar_mean.set_label("Mean Cost")
    fig_mean.tight_layout()

    fig_std, ax_std = plt.subplots()
    im_std = ax_std.imshow(
        std_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm"
    )
    ax_std.set_title("Std of Cost per Region")
    ax_std.set_xlabel("x")
    ax_std.set_ylabel("y")
    cbar_std = fig_std.colorbar(im_std, ax=ax_std)
    cbar_std.set_label("Std of Cost")
    fig_std.tight_layout()

    return fig_mean, fig_std
