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
            _explicit = {"name", "horizon_seconds", "action_low", "action_high", "reset_warmup_seconds", "reward_window"}
            env = BasalGanglia(
                horizon_seconds=config.horizon_seconds,
                action_low=list(config.action_low),
                action_high=list(config.action_high),
                reset_warmup_seconds=list(config.reset_warmup_seconds),
                reward_window=config.get("reward_window", 20),
                **{k: v for k, v in config.items() if k not in _explicit},
            )
        case _:
            raise ValueError(f"env {config.name} not found!")
    return env