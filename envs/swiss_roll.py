import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class SwissRoll(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    x_dim = 2
    y_dim = 3
    u_dim = 2

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
            low=np.array([-np.pi, -np.pi]),
            high=np.array([np.pi, np.pi]),
            shape=(2, ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, ),
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
        x = s[:, 0] * np.cos(s[:, 0]) / 2
        y = s[:, 1]
        z = s[:, 0] * np.sin(s[:, 0]) / 2
        e = np.stack([x, y, z], axis=1)
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

        # --- Figure / axis (bright, readable) ---
        fig = plt.figure(figsize=(7.2, 6.2), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --- Sample the latent grid and map via manifold() ---
        lo = self.state_space.low.astype(np.float32)
        hi = self.state_space.high.astype(np.float32)

        s0 = np.linspace(lo[0], hi[0], 240, dtype=np.float32)
        s1 = np.linspace(lo[1], hi[1], 80, dtype=np.float32)
        S0, S1 = np.meshgrid(s0, s1)

        s_samples = np.column_stack([S0.ravel(), S1.ravel()]).astype(np.float32)
        M = self.manifold(s_samples)

        X = M[:, 0].reshape(S0.shape)
        Y = M[:, 1].reshape(S0.shape)
        Z = M[:, 2].reshape(S0.shape)

        # Color the surface by the roll parameter (S0) to reveal the spiral clearly
        c = (S0 - S0.min()) / (S0.max() - S0.min() + 1e-8)

        ax.plot_surface(
            X, Y, Z,
            facecolors=plt.cm.viridis(c),  # clear gradient along the roll
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.95,
        )

        # Wireframe overlay = depth/shape readability
        ax.plot_wireframe(
            X, Y, Z,
            rstride=10, cstride=20,
            linewidth=0.5,
            alpha=0.25,
        )

        # --- Centerline curve (s1 = 0) so you can "see the curve" instantly ---
        s1_center = np.zeros_like(s0, dtype=np.float32)
        center_states = np.stack([s0, s1_center], axis=1)
        center_curve = self.manifold(center_states)

        ax.plot(
            center_curve[:, 0], center_curve[:, 1], center_curve[:, 2],
            linewidth=3.0, alpha=0.9,
            label="centerline (s1=0)",
        )

        # --- Current/target points (noise-free), and observed (if No) ---
        obs_true = self.manifold(self._state).reshape(-1)   # noise-free current
        obs_tgt  = self.manifold(self._target).reshape(-1)  # noise-free target

        ax.scatter(obs_true[0], obs_true[1], obs_true[2], s=120, marker="o",
                label="current (noise-free)", depthshade=False)
        ax.scatter(obs_tgt[0], obs_tgt[1], obs_tgt[2], s=160, marker="X",
                label="target (noise-free)", depthshade=False)

        if self.No is not None:
            obs_seen = self._get_obs().reshape(-1)  # what agent actually sees
            ax.scatter(obs_seen[0], obs_seen[1], obs_seen[2], s=120, marker="o",
                    label="current (observed)", depthshade=False)
            ax.plot([obs_true[0], obs_seen[0]],
                    [obs_true[1], obs_seen[1]],
                    [obs_true[2], obs_seen[2]],
                    linewidth=1.5, alpha=0.6)

        # --- Nice limits / aspect ---
        pad = 0.08
        xmin, xmax = float(X.min()), float(X.max())
        ymin, ymax = float(Y.min()), float(Y.max())
        zmin, zmax = float(Z.min()), float(Z.max())
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin

        ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
        ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
        ax.set_zlim(zmin - pad * dz, zmax + pad * dz)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True, alpha=0.25)

        # View angle that usually shows the "roll" clearly
        ax.view_init(elev=22, azim=35)

        ax.set_title("Swiss Roll (observation manifold)")
        ax.legend(loc="upper left")

        fig.tight_layout(pad=0.3)

        # --- Convert to RGB array ---
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return img
