import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class Torus(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    x_dim = 2
    y_dim = 3
    u_dim = 2
    radius1 = 1  # Minor radius (tube radius)
    radius2 = 2  # Major radius (distance from center to tube center)

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Ns: Optional[np.ndarray]=None,
        No: Optional[np.ndarray]=None,
        render_mode: str=None,
        horizon: int= 1000,
    ):

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Ns = Ns
        self.No = No
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.horizon = horizon

        self.state_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([2*np.pi, 2*np.pi]),
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
        assert self.Q.shape == (self.x_dim, self.x_dim)
        assert self.R.shape == (self.u_dim, self.u_dim)
        if self.Ns is not None:
            assert self.Ns.shape == (self.x_dim, self.x_dim)
        if self.No is not None:
            assert self.No.shape == (self.y_dim, self.y_dim)

    def manifold(self, s: np.ndarray):
        assert s.shape[1] == self.x_dim
        # Embed 2D latent state into 3D torus manifold
        x = (self.radius2 + self.radius1 * np.sin(s[:, 0:1])) * np.cos(s[:, 1:2])
        y = (self.radius2 + self.radius1 * np.sin(s[:, 0:1])) * np.sin(s[:, 1:2])
        z = self.radius1 * np.cos(s[:, 0:1])
        e = np.hstack([x, y, z])
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

        self._step = 1
        observation = self._get_obs().flatten()
        info = {
            "state": self._state.copy().flatten(),
            "target": self._target.copy().flatten(),
        }

        return observation, info
            
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        
        # Create figure and 3D axis with clean background
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        
        # Generate torus surface with higher resolution
        theta = np.linspace(0, 2*np.pi, 150)
        phi = np.linspace(0, 2*np.pi, 150)
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Compute 3D coordinates for the surface
        X = (self.radius2 + self.radius1 * np.sin(THETA)) * np.cos(PHI)
        Y = (self.radius2 + self.radius1 * np.sin(THETA)) * np.sin(PHI)
        Z = self.radius1 * np.cos(THETA)
        
        # Plot the torus surface with improved styling
        surf = ax.plot_surface(
            X, 
            Y, 
            Z, 
            alpha=0.35,
            color='#94a3b8',
            edgecolor='none',
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        
        # Get current state and target observations in 3D
        current_obs = self.manifold(self._state).flatten()
        target_obs = self.manifold(self._target).flatten()
        
        # Plot current state as red dot with glow effect
        ax.scatter(
            current_obs[0], 
            current_obs[1], 
            current_obs[2], 
            c='#ef4444', 
            s=400, 
            marker='o', 
            label='Current State', 
            edgecolors='#f87171', 
            linewidths=3,
            alpha=1.0,
            depthshade=False,
        )
        ax.scatter(
            current_obs[0], 
            current_obs[1], 
            current_obs[2], 
            c='#ef4444', 
            s=800, 
            marker='o', 
            edgecolors='none',
            linewidths=0,
            alpha=0.3,
            depthshade=False,
        )
        
        # Plot target state as yellow cross with glow
        ax.scatter(
            target_obs[0], 
            target_obs[1], 
            target_obs[2], 
            c='#eab308', 
            s=400, 
            marker='X', 
            label='Target State', 
            linewidths=0,
            edgecolors='#facc15',
            alpha=1.0,
            depthshade=False,
        )
        ax.scatter(
            target_obs[0], 
            target_obs[1], 
            target_obs[2], 
            c='#eab308', 
            s=800, 
            marker='X', 
            linewidths=0,
            alpha=0.3,
            depthshade=False,
        )
        
        # Set labels and title with better styling
        ax.set_xlabel('X', fontsize=14, color='#e2e8f0', labelpad=10)
        ax.set_ylabel('Y', fontsize=14, color='#e2e8f0', labelpad=10)
        ax.set_zlabel('Z', fontsize=14, color='#e2e8f0', labelpad=10)
        ax.set_title(
            'Torus Manifold Environment',
            fontsize=18,
            fontweight='bold',
            color='#f1f5f9',
            pad=20,
        )
        
        # Style the legend
        legend = ax.legend(
            loc='upper left',
            fontsize=12,
            framealpha=0.9,
            facecolor='#1e293b',
            edgecolor='#475569',
        )
        for text in legend.get_texts():
            text.set_color('#e2e8f0')
        
        # Set equal aspect ratio with proper limits
        max_range = self.radius1 + self.radius2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_box_aspect([1, 1, 1])
        
        # Style the axes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#334155')
        ax.yaxis.pane.set_edgecolor('#334155')
        ax.zaxis.pane.set_edgecolor('#334155')
        ax.grid(True, alpha=0.2, color='#475569')
        ax.tick_params(colors='#cbd5e1', labelsize=10)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Convert plot to RGB array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        
        return img
    
    def save(self, filepath: str):
        """
        Save environment parameters (dynamics and noise matrices) to a file.
        
        Args:
            filepath: Path to save the environment parameters
        """
        params = {
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'Q': self.Q.tolist(),
            'R': self.R.tolist(),
            'Ns': self.Ns.tolist() if self.Ns is not None else None,
            'No': self.No.tolist() if self.No is not None else None,
            'horizon': self.horizon,
            'radius1': self.radius1,
            'radius2': self.radius2,
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, render_mode: str = None):
        """
        Load environment parameters from a file and create a new environment.
        
        Args:
            filepath: Path to load the environment parameters from
            render_mode: Render mode for the environment
            
        Returns:
            Torus environment initialized with loaded parameters
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        A = np.array(params['A'])
        B = np.array(params['B'])
        Q = np.array(params['Q'])
        R = np.array(params['R'])
        Ns = np.array(params['Ns']) if params['Ns'] is not None else None
        No = np.array(params['No']) if params['No'] is not None else None
        horizon = params['horizon']
        
        env = cls(
            A=A,
            B=B,
            Q=Q,
            R=R,
            Ns=Ns,
            No=No,
            render_mode=render_mode,
            horizon=horizon,
        )
        
        # Load radius parameters if they exist (for backward compatibility)
        if 'radius1' in params:
            env.radius1 = params['radius1']
        if 'radius2' in params:
            env.radius2 = params['radius2']
        
        return env