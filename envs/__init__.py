from .pendulum import Pendulum
import numpy as np
from omegaconf.dictconfig import DictConfig


def make(config: DictConfig):
    
    match config.name:
        case "pendulum":
            env = Pendulum(
                render_mode="rgb_array",
                horizon=config.horizon,
                g=config.gravity,
            )
        case _:
            raise ValueError(f"env {config.name} not found!")
    return env