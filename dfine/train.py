import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
from .memory import ReplayBuffer
from .utils import compute_consistency, bottle_mvn
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    CostModel,
)


def train_backbone(
    config: DictConfig,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=train_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    decoder = Decoder(
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

    wandb.watch([encoder, dynamics_model, decoder], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
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
        decoder.train()
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
        y_filter_loss = nn.MSELoss()(
            decoder(filter_a),
            einops.rearrange(y, "l b y -> (l b) y")
        )

        y_pred_loss = 0.0
        for k in range(1, config.prediction_k+1):
            pred_dist = bottle_mvn(posteriors[0:config.chunk_length-k])
            for t in range(k):
                pred_dist = dynamics_model.prior(
                    dist=pred_dist,
                    u=einops.rearrange(u[t:config.chunk_length-k+t], "l b u -> (l b) u"),
                )
            pred_a = dynamics_model.get_a(pred_dist.loc)
            pred_y = decoder(pred_a)
            true_y = einops.rearrange(y[k:config.chunk_length], "l b y -> (l b) y")
            y_pred_loss += nn.MSELoss()(pred_y, true_y) * (config.chunk_length - k) / config.chunk_length

        # y prediction loss
        y_pred_loss /= config.prediction_k
        # autoencoder loss
        a_flatten = einops.rearrange(a, "l b a -> (l b) a")
        y_flatten = einops.rearrange(y, "l b y -> (l b) y")
        y_recon = decoder(a_flatten)
        ae_loss = nn.MSELoss()(y_recon, y_flatten)

        total_loss = (
            y_pred_loss +
            config.filtering_weight * y_filter_loss +
            config.mean_consistency_weight * mean_consistency +
            config.kl_consistency_weight * kl_consistency
        )

        optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/y prediction loss": y_pred_loss.item(),
            "train/y filter loss": y_filter_loss.item(),
            "train/ae loss": ae_loss.item(),
            "train/total loss": total_loss.item(),
            "train/mean consistency": mean_consistency.item(),
            "train/kl consistency": kl_consistency.item(),
            "global_step": update,
        })
            
        if update % config.test_interval == 0:
            # test
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
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
                y_filter_loss = nn.MSELoss()(
                    decoder(filter_a),
                    einops.rearrange(y, "l b y -> (l b) y")
                )

                y_pred_loss = 0.0
                for k in range(1, config.prediction_k+1):
                    pred_dist = bottle_mvn(posteriors[0:config.chunk_length-k])
                    for t in range(k):
                        pred_dist = dynamics_model.prior(
                            dist=pred_dist,
                            u=einops.rearrange(u[t:config.chunk_length-k+t], "l b u -> (l b) u"),
                        )
                    pred_a = dynamics_model.get_a(pred_dist.loc)
                    pred_y = decoder(pred_a)
                    true_y = einops.rearrange(y[k:config.chunk_length], "l b y -> (l b) y")
                    y_pred_loss += nn.MSELoss()(pred_y, true_y) * (config.chunk_length - k) / config.chunk_length

                # y prediction loss
                y_pred_loss /= config.prediction_k

                # autoencoder loss
                a_flatten = einops.rearrange(a, "l b a -> (l b) a")
                y_flatten = einops.rearrange(y, "l b y -> (l b) y")
                y_recon = decoder(a_flatten)
                ae_loss = nn.MSELoss()(y_recon, y_flatten)

                total_loss = (
                    y_pred_loss +
                    config.filtering_weight * y_filter_loss +
                    config.mean_consistency_weight * mean_consistency +
                    config.kl_consistency_weight * kl_consistency
                )
                
                wandb.log({
                    "test/y prediction loss": y_pred_loss.item(),
                    "test/y filter loss": y_filter_loss.item(),
                    "test/ae loss": ae_loss.item(),
                    "test/total loss": total_loss.item(),
                    "test/mean consistency": mean_consistency.item(),
                    "test/kl consistency": kl_consistency.item(),
                    "global_step": update,
                })
                
    return encoder, decoder, dynamics_model


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
        # compute cost loss
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
                # compute cost loss
                cost_loss = nn.MSELoss()(
                    cost_model(x=bottle_mvn(posteriors).loc, u=einops.rearrange(u, "l b u -> (l b) u")),
                    einops.rearrange(c, "l b 1 -> (l b) 1")
                )
                
                wandb.log({
                    "test/cost loss": cost_loss.item(),
                    "global_step": update,
                })

    return cost_model