import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class Circle(gym.Env):
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

        # Noise-free embeddings (pure manifold of latent state)
        obs_true = self.manifold(self._state).reshape(-1)      # (2,)
        obs_tgt  = self.manifold(self._target).reshape(-1)     # (2,)

        # What the agent actually observes (may include observation noise)
        obs_seen = self._get_obs().reshape(-1)                 # (2,)

        # Scalars for annotations / arc
        x  = float(self._state.reshape(-1)[0])
        xt = float(self._target.reshape(-1)[0])

        # Wrapped angular difference in [-pi, pi)
        d = ((xt - x + np.pi) % (2.0 * np.pi)) - np.pi

        fig, ax = plt.subplots(figsize=(5.8, 5.8), dpi=140)

        # --- Draw manifold (unit circle) using manifold() ---
        lo = float(self.state_space.low.reshape(-1)[0])
        hi = float(self.state_space.high.reshape(-1)[0])
        theta = np.linspace(lo, hi, 900, dtype=np.float32).reshape(-1, 1)
        circle = self.manifold(theta)
        ax.plot(circle[:, 0], circle[:, 1], linewidth=2.5, alpha=0.55, label="manifold")

        # --- Shortest arc from current -> target ---
        theta_arc = np.linspace(x, x + d, 160, dtype=np.float32).reshape(-1, 1)
        arc = self.manifold(theta_arc)
        ax.plot(arc[:, 0], arc[:, 1], linewidth=6, alpha=0.18, label="shortest arc")

        # Origin marker
        ax.scatter([0.0], [0.0], s=40, marker="+", alpha=0.6)

        # Radial lines (helpful visually)
        ax.plot([0.0, obs_true[0]], [0.0, obs_true[1]], linewidth=1.6, alpha=0.25)
        ax.plot([0.0, obs_tgt[0]],  [0.0, obs_tgt[1]],  linewidth=1.6, alpha=0.35, linestyle="--")

        # --- Points ---
        ax.scatter([obs_tgt[0]],  [obs_tgt[1]],  s=190, marker="X", label="target (noise-free)")
        ax.scatter([obs_true[0]], [obs_true[1]], s=130, marker="o", alpha=0.55, label="current (noise-free)")

        # If observation noise exists, show the noisy observation separately
        if self.No is not None:
            ax.scatter(
                [obs_seen[0]], [obs_seen[1]],
                s=130, marker="o",
                edgecolors="k", linewidths=1.2,
                label="current (observed)"
            )
            # Link noise-free -> observed
            ax.plot([obs_true[0], obs_seen[0]], [obs_true[1], obs_seen[1]], linewidth=1.2, alpha=0.5)

        # --- Info box ---
        info_txt = (
            f"x   = {x:.3f} rad\n"
            f"x*  = {xt:.3f} rad\n"
            f"Δwrap = {d:.3f} rad\n"
            f"step = {self._step}"
        )
        ax.text(
            0.02, 0.98, info_txt,
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.85)
        )

        # Cosmetics
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("y[0]")
        ax.set_ylabel("y[1]")
        ax.set_title("Circle env render (observation space)")
        ax.legend(loc="lower right", framealpha=0.92)

        # Convert to RGB array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return img
