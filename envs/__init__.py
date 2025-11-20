from .torus import Torus
import numpy as np
from omegaconf.dictconfig import DictConfig


def make(config: DictConfig):
    
    match config.name:
        case "torus":
            env = Torus(
                A=np.array(config.A),
                B=np.array(config.B),
                horizon=config.horizon,
                render_mode="rgb_array",
            )
        case _:
            raise ValueError(f"env {config.name} not found!")
    return env