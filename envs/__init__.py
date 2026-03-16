from .pendulum import Pendulum
from omegaconf.dictconfig import DictConfig


def make(config: DictConfig):
    
    match config.name:
        case "pendulum":
            env = Pendulum(
                render_mode="rgb_array",
                horizon=config.horizon,
                g=config.gravity,
                action_repeat=config.action_repeat,
            )
        case _:
            raise ValueError(f"env {config.name} not found!")
    return env