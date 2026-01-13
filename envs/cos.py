import gymnasium as gym
from pathlib import Path
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class Cos(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    x_dim = 1
    y_dim = 2
    u_dim = 1

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Ns: Optional[np.ndarray]=None,
        No: Optional[np.ndarray]=None,
        render_mode: str=None,
        horizon: int= 1000,
        periodic: Optional[bool]=True,
    ):
        
        super().__init__()

        self.A = A.astype(np.float32)
        self.B = B.astype(np.float32)
        self.Ns = Ns.astype(np.float32) if Ns is not None else None
        self.No = No.astype(np.float32) if No is not None else None

        self._verify_parameters()
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.horizon = horizon
        self.periodic = periodic

        self.state_space = spaces.Box(
            low=np.array([-np.pi]),
            high=np.array([np.pi]),
            shape=(1, ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2, ),
            dtype=np.float32,
        )

    def _verify_parameters(self):
        assert self.A.shape == (self.x_dim, self.x_dim)
        assert self.B.shape == (self.x_dim, self.u_dim)
        if self.Ns is not None:
            assert self.Ns.shape == (self.x_dim, self.x_dim)
        if self.No is not None:
            assert self.No.shape == (self.y_dim, self.y_dim)

    def manifold(self, s: np.ndarray):
        assert s.shape[1] == self.x_dim
        # Embed 2D latent state into 3D torus manifold
        e = np.hstack([np.cos(s), np.sin(s)])
        return e

    def _get_obs(self):
        obs = self.manifold(self._state)
        if self.No is not None:
            no = self.np_random.multivariate_normal(
                mean=np.zeros(self.observation_space.shape),
                cov=self.No,
            ).astype(np.float32).reshape(1, -1)
            obs = obs + no
        return obs

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        
        super().reset(seed=seed)
        options = options or {}
        initial_state = options.get("initial_state")
        target_state = options.get("target_state")
        
        if initial_state is not None:
            assert initial_state.shape == self.state_space.shape
            self._state = initial_state.astype(np.float32).reshape(1, -1)
        else:
            self._state = self.state_space.sample().reshape(1, -1)

        if target_state is not None:
            assert target_state.shape == self.state_space.shape
            self._target = target_state.astype(np.float32).reshape(1, -1)
        else:
            self._target = self.state_space.sample().reshape(1, -1)

        self._step = 0
        observation = self._get_obs().flatten()
        info = {
            "state": self._state.copy().flatten(),
            "target": self._target.copy().flatten(),
        }

        return observation, info
    

    def step(
        self,
        action: np.ndarray,
    ):
        assert action.shape == self.action_space.shape
        action = action.astype(np.float32).reshape(1, -1)
        action = np.clip(
            action, 
            a_min=self.action_space.low,
            a_max=self.action_space.high,
        )

        self._state = self._state @ self.A.T + action @ self.B.T
        if self.Ns is not None:
            ns = self.np_random.multivariate_normal(
                mean=np.zeros(self.state_space.shape),
                cov=self.Ns,
            ).astype(np.float32).reshape(1, -1)
            self._state = self._state + ns

        self._step += 1
        truncated = bool(self._step >= self.horizon)
        terminated = False
        reward = 0.0
        obs = self._get_obs().flatten()
        
        if self.periodic:
            rng = self.state_space.high - self.state_space.low
            self._state = ((self._state - self.state_space.low) % rng) + self.state_space.low

        else:
            # Check if the state is valid
            is_valid = (
                np.all(self.state_space.low < self._state.flatten()) and np.all(self._state.flatten() < self.state_space.high)
            )
            if not is_valid:
                terminated = True

        info = {
            "state": self._state.copy().flatten(),
            "target": self._target.copy().flatten(),
        }

        return obs, reward, terminated, truncated, info 

            
    def render(self):
        if self.render_mode != "rgb_array":
            return None

        # Current observation (what the agent actually sees; includes No noise if present)
        obs_cur = self._get_obs().reshape(-1)          # shape (2,)
        # Target observation (noise-free, since target obs isn't actually emitted by env)
        obs_tgt = self.manifold(self._target).reshape(-1)  # shape (2,)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

        # Plot the observation manifold: unit circle
        theta = np.linspace(-np.pi, np.pi, 600)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        ax.plot(circle[:, 0], circle[:, 1], linewidth=2, label="manifold (cos x, sin x)")

        # Current + target in observation space
        ax.scatter([obs_cur[0]], [obs_cur[1]], s=120, marker="o", label="current obs")
        ax.scatter([obs_tgt[0]], [obs_tgt[1]], s=140, marker="X", label="target (noise-free)")

        # Cosmetics / axes
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("y[0] = cos(x)")
        ax.set_ylabel("y[1] = sin(x)")
        ax.set_title("Cos env render (observation space)")
        ax.legend(loc="upper right")

        # Convert to RGB array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return img
