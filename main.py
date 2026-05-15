import os
import envs
import wandb
import torch
import minari
import logging
import joblib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from minari import MinariDataset
from sklearn.preprocessing import StandardScaler
from dfine.memory import ReplayBuffer
from dfine.train import train_backbone


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFINE")
    parser.add_argument("--config", type=str, help="path to the config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to a previous run dir to load backbone weights from "
                             "(encoder.pth, decoder.pth, dynamics_model.pth). Optimizer state is NOT loaded.")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    config.run_id = datetime.now().strftime("%Y%m%d_%H%M")

    wandb.init(
        project="Manifolds control",
        name=config.run_name,
        notes=config.notes,
        config=OmegaConf.to_container(config, resolve=True)
    )

    # prepare logging
    save_dir = Path(config.log_dir) / config.run_id
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(config, save_dir / "config.yaml")
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    logger = logging.getLogger(__name__)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # create env and collect data
    env = envs.make(config=config.env)
    
    # create replay buffers
    dataset = MinariDataset(data=Path(config.data.data_dir) / "data")
    test_size = int(len(dataset) * config.data.test_ratio)
    train_size = len(dataset) - test_size
    train_data, test_data = minari.split_dataset(dataset=dataset, sizes=[train_size, test_size])
    train_buffer = ReplayBuffer.load_from_minari(dataset=train_data)
    test_buffer = ReplayBuffer.load_from_minari(dataset=test_data)

    # fit scaler on training observations, normalise both buffers in-place
    scaler = StandardScaler()
    n_train = len(train_buffer)
    scaler.fit(train_buffer.ys[:n_train])
    train_buffer.ys[:n_train] = scaler.transform(train_buffer.ys[:n_train]).astype(np.float32)
    n_test = len(test_buffer)
    test_buffer.ys[:n_test] = scaler.transform(test_buffer.ys[:n_test]).astype(np.float32)
    joblib.dump(scaler, save_dir / "scaler.joblib")

    # normalize costs by train std so cost model targets are O(1)
    cost_std = train_buffer.cs[:n_train].std() + 1e-8
    train_buffer.cs[:n_train] /= cost_std
    test_buffer.cs[:n_test]   /= cost_std

    # train and save the backbone
    logging.info("training backbone ...")
    encoder, dynamics_model = train_backbone(
        config=config.train,
        train_buffer=train_buffer,
        test_buffer=test_buffer,
        env=env,
        scaler=scaler,
        checkpoint_dir=Path(args.checkpoint) if args.checkpoint is not None else None,
    )
    torch.save(encoder.state_dict(), save_dir / "encoder.pth")
    torch.save(dynamics_model.state_dict(), save_dir / "dynamics_model.pth")
    
    wandb.finish()