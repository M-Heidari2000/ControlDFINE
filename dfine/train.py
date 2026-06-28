import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from omegaconf.dictconfig import DictConfig
from .memory import ReplayBuffer
from .utils import compute_consistency, bottle_mvn, SIGReg, hankel_singular_values
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Dynamics,
    CostModel,
)


def train_backbone(
    config: DictConfig,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
    checkpoint_dir: Optional[Path] = None,
):

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=train_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    dynamics_model = Dynamics(
        x_dim=config.x_dim,
        u_dim=train_buffer.u_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        min_var=config.min_var,
        locally_linear=config.locally_linear,
    ).to(device)

    sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj).to(device)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pth", map_location=device))
        dynamics_model.load_state_dict(torch.load(checkpoint_dir / "dynamics_model.pth", map_location=device))
        print(f"loaded backbone weights from {checkpoint_dir}")

    wandb.watch([encoder, dynamics_model], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(dynamics_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    # train and test loop
    for update in tqdm(range(config.num_updates)):

        # train
        encoder.train()
        dynamics_model.train()

        y, u, _, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        priors, posteriors = dynamics_model(a=a, u=u)   # x0:T-1

        consistencies = compute_consistency(
            prior=bottle_mvn(priors),
            posterior=bottle_mvn(posteriors),
            free_nats=config.kl_free_nats
        )
        mean_consistency = consistencies[0]
        kl_consistency = consistencies[1]

        filter_a = dynamics_model.get_a(bottle_mvn(posteriors).loc)
        a_filter_loss = (
            (filter_a - einops.rearrange(a, "l b a -> (l b) a")) ** 2
        ).sum(dim=-1).mean()

        a_pred_loss = 0.0
        for k in range(1, config.prediction_k+1):
            pred_dist = bottle_mvn(posteriors[0:config.chunk_length-k])
            for t in range(k):
                pred_dist = dynamics_model.prior(
                    dist=pred_dist,
                    u=einops.rearrange(u[t:config.chunk_length-k+t], "l b u -> (l b) u"),
                )
            pred_a = dynamics_model.get_a(pred_dist.loc)
            true_a = einops.rearrange(a[k:config.chunk_length], "l b a -> (l b) a")
            a_pred_loss += (
                (pred_a - true_a) ** 2
            ).sum(dim=-1).mean() * (config.chunk_length - k) / config.chunk_length

        a_pred_loss /= config.prediction_k

        # sigreg loss
        sigreg_loss = sigreg(a)

        # balanced gramian loss (global linear only)
        gramian_loss = torch.tensor(0.0, device=device)
        if not dynamics_model.locally_linear:
            hsv = hankel_singular_values(dynamics_model.A, dynamics_model.B, dynamics_model.C)
            k = min(dynamics_model.u_dim, dynamics_model.a_dim)
            gramian_loss = 1.0 / hsv.topk(k).values.min().clamp(min=1e-8)

        total_loss = (
            a_pred_loss +
            config.filtering_weight * a_filter_loss +
            config.mean_consistency_weight * mean_consistency +
            config.kl_consistency_weight * kl_consistency +
            config.sigreg_weight * sigreg_loss +
            config.gramian_weight * gramian_loss
        )

        optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/a prediction loss": a_pred_loss.item(),
            "train/a filter loss": a_filter_loss.item(),
            "train/total loss": total_loss.item(),
            "train/mean consistency": mean_consistency.item(),
            "train/kl consistency": kl_consistency.item(),
            "train/sigreg loss": sigreg_loss.item(),
            "train/gramian loss": gramian_loss.item(),
            "global_step": update,
        })

        if update % config.test_interval == 0:
            # test
            with torch.no_grad():
                encoder.eval()
                dynamics_model.eval()

                y, u, _, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
                a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")

                priors, posteriors = dynamics_model(a=a, u=u)   # x0:T-1

                consistencies = compute_consistency(
                    prior=bottle_mvn(priors),
                    posterior=bottle_mvn(posteriors),
                    free_nats=config.kl_free_nats
                )
                mean_consistency = consistencies[0]
                kl_consistency = consistencies[1]

                filter_a = dynamics_model.get_a(bottle_mvn(posteriors).loc)
                a_filter_loss = (
                    (filter_a - einops.rearrange(a, "l b a -> (l b) a")) ** 2
                ).sum(dim=-1).mean()

                a_pred_loss = 0.0
                for k in range(1, config.prediction_k+1):
                    pred_dist = bottle_mvn(posteriors[0:config.chunk_length-k])
                    for t in range(k):
                        pred_dist = dynamics_model.prior(
                            dist=pred_dist,
                            u=einops.rearrange(u[t:config.chunk_length-k+t], "l b u -> (l b) u"),
                        )
                    pred_a = dynamics_model.get_a(pred_dist.loc)
                    true_a = einops.rearrange(a[k:config.chunk_length], "l b a -> (l b) a")
                    a_pred_loss += (
                        (pred_a - true_a) ** 2
                    ).sum(dim=-1).mean() * (config.chunk_length - k) / config.chunk_length

                a_pred_loss /= config.prediction_k

                # sigreg loss
                sigreg_loss = sigreg(a)

                # balanced gramian loss (global linear only)
                gramian_loss = torch.tensor(0.0, device=device)
                if not dynamics_model.locally_linear:
                    hsv = hankel_singular_values(dynamics_model.A, dynamics_model.B, dynamics_model.C)
                    k = min(dynamics_model.u_dim, dynamics_model.a_dim)
                    gramian_loss = 1.0 / hsv.topk(k).values.min().clamp(min=1e-8)

                total_loss = (
                    a_pred_loss +
                    config.filtering_weight * a_filter_loss +
                    config.mean_consistency_weight * mean_consistency +
                    config.kl_consistency_weight * kl_consistency +
                    config.sigreg_weight * sigreg_loss +
                    config.gramian_weight * gramian_loss
                )

                wandb.log({
                    "test/a prediction loss": a_pred_loss.item(),
                    "test/a filter loss": a_filter_loss.item(),
                    "test/total loss": total_loss.item(),
                    "test/mean consistency": mean_consistency.item(),
                    "test/kl consistency": kl_consistency.item(),
                    "test/sigreg loss": sigreg_loss.item(),
                    "test/gramian loss": gramian_loss.item(),
                    "global_step": update,
                })

    return encoder, dynamics_model


def train_cost(
    config: DictConfig,
    encoder: Encoder,
    dynamics_model: Dynamics,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    cost_model = CostModel(
        x_dim=dynamics_model.x_dim,
        u_dim=dynamics_model.u_dim,
    ).to(device)

    # save requires_grad state so we can restore it after cost training
    encoder_grad_state        = {p: p.requires_grad for p in encoder.parameters()}
    dynamics_model_grad_state = {p: p.requires_grad for p in dynamics_model.parameters()}

    # freeze backbone models
    for p in encoder.parameters():
        p.requires_grad = False
    for p in dynamics_model.parameters():
        p.requires_grad = False

    encoder.eval()
    dynamics_model.eval()

    wandb.watch([cost_model], log="all", log_freq=10)

    all_params = list(cost_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    # train and test loop
    for update in tqdm(range(config.num_updates)):
        # train
        cost_model.train()

        y, u, c, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")
        c = torch.as_tensor(c, device=device)
        c = einops.rearrange(c, "b l 1 -> l b 1")

        _, posteriors = dynamics_model(a=a, u=u)  # x0:T-1
        cost_loss = nn.MSELoss()(
            cost_model(x=bottle_mvn(posteriors).loc, u=einops.rearrange(u, "l b u -> (l b) u")),
            einops.rearrange(c, "l b 1 -> (l b) 1")
        )
        optimizer.zero_grad()
        cost_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/cost loss": cost_loss.item(),
            "global_step": update,
        })

        if update % config.test_interval == 0:
            # test
            with torch.no_grad():
                cost_model.eval()

                y, u, c, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
                a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")
                c = torch.as_tensor(c, device=device)
                c = einops.rearrange(c, "b l 1 -> l b 1")

                _, posteriors = dynamics_model(a=a, u=u)  # x0:T-1
                cost_loss = nn.MSELoss()(
                    cost_model(x=bottle_mvn(posteriors).loc, u=einops.rearrange(u, "l b u -> (l b) u")),
                    einops.rearrange(c, "l b 1 -> (l b) 1")
                )

                wandb.log({
                    "test/cost loss": cost_loss.item(),
                    "global_step": update,
                })

    # restore requires_grad so backbone can continue being used after cost training
    for p, state in encoder_grad_state.items():
        p.requires_grad = state
    for p, state in dynamics_model_grad_state.items():
        p.requires_grad = state

    return cost_model
