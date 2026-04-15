from .pendulum import Pendulum
from .basal_ganglia import BasalGanglia
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
        case "basal_ganglia":
            env = BasalGanglia(
                horizon=config.horizon,
                action_low=list(config.action_low),
                action_high=list(config.action_high),
                **{k: v for k, v in config.items() if k not in ("name", "horizon", "action_low", "action_high")},
            )
        case _:
            raise ValueError(f"env {config.name} not found!")
    return env