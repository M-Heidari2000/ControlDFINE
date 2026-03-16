import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gymnasium import spaces


class Pendulum(gym.Env):

    x_dim = 2
    y_dim = 3
    u_dim = 2

    def __init__(
        self,
        render_mode: str | None = None,
        g: float = 10.0,
        horizon: int = 200,
        action_repeat: int = 2,
    ):
        self.env = PendulumEnv(render_mode=render_mode, g=g)  # fixed: removed trailing comma
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            shape=(1,),
            dtype=np.float32,
        )
        self.state_space = spaces.Box(
            low=np.array([-np.pi, 0.0]),
            high=np.array([np.pi, 8.0]),
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def state(self):
        theta, thetadot = self.env.state
        theta = angle_normalize(theta)
        return np.array([theta, thetadot])

    def step(self, action):
        scaled_action = action * 2.0   # map [-1, 1] → [-2, 2]
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(scaled_action)
            total_reward += reward
            self._step += 1
            truncated = truncated or bool(self._step >= self.horizon)
            if terminated or truncated:
                break
        info = info | {"state": self.state}
        return obs, total_reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._step = 0
        info = info | {"state": self.state}
        return obs, info

    def render(self):
        return self.env.render()

    def manifold(self, s: np.ndarray):
        assert s.shape[1] == self.x_dim
        x = np.cos(s[:, 0])
        y = np.sin(s[:, 0])
        z = s[:, 1]
        return np.stack([x, y, z], axis=1)