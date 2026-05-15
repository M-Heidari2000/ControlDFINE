import torch
import numpy as np
from mpc import mpc
from mpc.mpc import QuadCost, LinDx
from typing import Optional
from torch.distributions import MultivariateNormal
from sklearn.preprocessing import StandardScaler
from .models import Encoder, Dynamics, CostModel


class MPCAgent:
    """
        action planning by the MPC method
        c = 0.5 * (x-q).T @ Q @ (x-q) + 0.5 * (u-r).T @ R @ (u-r)
    """
    def __init__(
        self,
        encoder: Encoder,
        dynamics_model: Dynamics,
        cost_model: CostModel,
        planning_horizon: int,
        scaler: Optional[StandardScaler] = None,
        action_noise: float = 0.3,
    ):
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.cost_model = cost_model
        self.planning_horizon = planning_horizon
        self.scaler = scaler
        self.action_noise = action_noise

        self.device = next(encoder.parameters()).device

        # MPC matrices
        C = torch.block_diag(self.cost_model.Q, self.cost_model.R).expand(self.planning_horizon, 1, -1, -1)
        c = torch.cat([
            -self.cost_model.q @ self.cost_model.Q,
            -self.cost_model.r @ self.cost_model.R,
        ], dim=1).expand(self.planning_horizon, -1, -1)
        F = torch.cat((self.dynamics_model.A, self.dynamics_model.B), dim=1).expand(self.planning_horizon, 1, -1, -1)
        f = torch.zeros((1, self.cost_model.x_dim), device=self.device).expand(self.planning_horizon, -1, -1)

        self.quadcost = QuadCost(C, c)
        self.lindx = LinDx(F, f)

        self.planner = mpc.MPC(
            n_batch=1,
            n_state=self.cost_model.x_dim,
            n_ctrl=self.cost_model.u_dim,
            T=self.planning_horizon,
            u_lower=-1.0,
            u_upper=1.0,
            lqr_iter=50,
            backprop=False,
            exit_unconverged=False,
        )

        # Initialize belief with zeros
        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device),
        )

    def __call__(self, y: torch.Tensor, u: Optional[torch.Tensor], explore: bool=False):
        """
        inputs: y_t, u_{t-1}
            outputs: planned u_t
            explore: add random values to planned actions for exploration purpose

        notes: if u_{t-1} is None then that's the first observation
        """

        with torch.no_grad():
            if self.scaler is not None:
                y = self.scaler.transform(y.reshape(1, -1)).flatten()
            y = torch.as_tensor(y, device=self.device).unsqueeze(0)
            a = self.encoder(y)
            if u is not None:
                u = torch.as_tensor(u, device=self.device).unsqueeze(0)
                self.dist = self.dynamics_model.prior(dist=self.dist, u=u)
            self.dist = self.dynamics_model.posterior(dist=self.dist, a=a)
            planned_u = self._plan()

            if explore:
                planned_u += self.action_noise * torch.randn_like(planned_u)

        return np.clip(planned_u.cpu().numpy(), a_min=-1.0, a_max=1.0)

    def _plan(self):
        _, planned_u, _ = self.planner(self.dist.loc, self.quadcost, self.lindx)

        return planned_u.squeeze(1)


    def reset(self):
        # Initialize belief with zeros
        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device),
        )


class IMPCAgent:
    """
    Iterative MPC for locally-linear dynamics using the mpc package.
    At each iteration: roll out nominal actions → get local A_t, B_t at each
    state → pass as time-varying LinDx to the MPC solver → repeat.
    """

    def __init__(
        self,
        encoder: Encoder,
        dynamics_model: Dynamics,
        cost_model: CostModel,
        planning_horizon: int,
        num_iterations: int=10,
        scaler: Optional[StandardScaler] = None,
        action_noise: float=0.3,
    ):
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.cost_model = cost_model
        self.planning_horizon = planning_horizon
        self.num_iterations = num_iterations
        self.scaler = scaler
        self.action_noise = action_noise
        self.device = next(encoder.parameters()).device

        # cost matrices — fixed across iterations
        C = torch.block_diag(self.cost_model.Q, self.cost_model.R).expand(planning_horizon, 1, -1, -1)
        c = torch.cat([
            -self.cost_model.q @ self.cost_model.Q,
            -self.cost_model.r @ self.cost_model.R,
        ], dim=1).expand(planning_horizon, -1, -1)
        self.quadcost = QuadCost(C, c)

        self.planner = mpc.MPC(
            n_batch=1,
            n_state=self.cost_model.x_dim,
            n_ctrl=self.cost_model.u_dim,
            T=planning_horizon,
            u_lower=-1.0,
            u_upper=1.0,
            lqr_iter=50,
            backprop=False,
            exit_unconverged=False,
        )

        self.reset()

    def reset(self):
        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device),
        )
        self.u_nom = torch.zeros(
            self.planning_horizon, 1, self.dynamics_model.u_dim, device=self.device
        )

    def __call__(self, y: torch.Tensor, u: Optional[torch.Tensor], explore: bool=False):
        """
        inputs: y_t, u_{t-1}
            outputs: all planned actions (T, u_dim)
            explore: add random values to planned actions for exploration purpose

        notes: if u_{t-1} is None then that's the first observation
        """
        with torch.no_grad():
            if self.scaler is not None:
                y = self.scaler.transform(y.reshape(1, -1)).flatten()
            y = torch.as_tensor(y, device=self.device).unsqueeze(0)
            a = self.encoder(y)
            if u is not None:
                u = torch.as_tensor(u, device=self.device).unsqueeze(0)
                self.dist = self.dynamics_model.prior(dist=self.dist, u=u)
            self.dist = self.dynamics_model.posterior(dist=self.dist, a=a)
            planned_u = self._plan()
            if explore:
                planned_u += self.action_noise * torch.randn_like(planned_u)
        return np.clip(planned_u.cpu().numpy(), a_min=-1.0, a_max=1.0)

    def _build_lindx(self, x0: torch.Tensor) -> LinDx:
        """
        Roll out u_nom from x0 and collect local linearizations.
        Returns a time-varying LinDx with F of shape (T, 1, x_dim, x_dim+u_dim).
        """
        x = x0
        F_list = []
        for t in range(self.planning_horizon):
            A, B, _, _, _ = self.dynamics_model.get_dynamics(x)   # (1, x_dim, x_dim/u_dim)
            F_list.append(torch.cat([A, B], dim=2))                # (1, x_dim, x_dim+u_dim)
            x = (torch.einsum('bij,bj->bi', A, x) +
                 torch.einsum('bij,bj->bi', B, self.u_nom[t]))
        F = torch.stack(F_list, dim=0)                             # (T, 1, x_dim, x_dim+u_dim)
        f = torch.zeros(self.planning_horizon, 1, self.dynamics_model.x_dim, device=self.device)
        return LinDx(F, f)

    def _plan(self) -> torch.Tensor:
        x0 = self.dist.loc   # (1, x_dim)

        for _ in range(self.num_iterations):
            lindx = self._build_lindx(x0)
            _, self.u_nom, _ = self.planner(x0, self.quadcost, lindx)

        # warm-start: shift left, pad with zeros
        planned = self.u_nom.clone()
        self.u_nom = torch.cat([
            self.u_nom[1:],
            torch.zeros(1, 1, self.dynamics_model.u_dim, device=self.device),
        ], dim=0)

        return planned.squeeze(1)   # (T, u_dim)

class OracleMPC:
    """
        action planning by MPC method using the actual states
        c = 0.5 * (x-q).T @ Q @ (x-q) + 0.5 * u.T @ R @ u
    """

    def __init__(
        self,
        Q: torch.Tensor,
        R: torch.Tensor,
        q: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        planning_horizon: int=10
    ):
        
        x_dim = Q.shape[0]
        u_dim = R.shape[0]
        self.device = A.device

        C = torch.block_diag(Q, R).expand(planning_horizon, 1, -1, -1)
        c = torch.cat([
            -q @ Q,
            torch.zeros((1, u_dim), device=self.device)
        ], dim=1).expand(planning_horizon, -1, -1)
        
        F = torch.cat((A, B), dim=1).expand(planning_horizon, 1, -1, -1)
        f = torch.zeros((1, x_dim), device=self.device).expand(planning_horizon, -1, -1)
        
        self.quadcost = QuadCost(C, c)
        self.lindx = LinDx(F, f)
        
        self.planner = mpc.MPC(
            n_batch=1,
            n_state=x_dim,
            n_ctrl=u_dim,
            T=planning_horizon,
            u_lower=-1.0,
            u_upper=1.0,
            lqr_iter=50,
            backprop=False,
            exit_unconverged=False,
        )

    def __call__(self, x: torch.Tensor):
        _, planned_u, _ = self.planner(
            x,
            self.quadcost,
            self.lindx
        )        
        return np.clip(planned_u.squeeze(1).cpu().numpy(), a_min=-1.0, a_max=1.0)