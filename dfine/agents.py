import torch
import numpy as np
from mpc import mpc
from mpc.mpc import QuadCost, LinDx
from torch.distributions import MultivariateNormal


class MPCAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        dynamics_model,
        cost_model,
        planning_horizon: int,
        action_noise: float=0.3,
    ):
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.cost_model = cost_model
        self.planning_horizon = planning_horizon
        self.action_noise = action_noise

        self.device = next(encoder.parameters()).device

        x_dim, u_dim = self.dynamics_model.B.shape

        C = torch.block_diag(self.cost_model.Q, self.cost_model.R).repeat(
            self.planning_horizon, 1, 1, 1,
        )

        c = torch.cat([
            self.cost_model.q.reshape(1, -1),
            torch.zeros((1, u_dim), device=self.device)
        ], dim=1).repeat(self.planning_horizon, 1, 1)

        F = torch.cat((self.dynamics_model.A, self.dynamics_model.B), dim=1).repeat(
            self.planning_horizon, 1, 1, 1
        )
        f = torch.zeros((1, x_dim), device=self.device).repeat(
            self.planning_horizon, 1, 1
        )

        self.quadcost = QuadCost(C, c)
        self.lindx = LinDx(F, f)

        self.planner = mpc.MPC(
            n_batch=1,
            n_state=x_dim,
            n_ctrl=u_dim,
            T=self.planning_horizon,
            u_lower=-1.0,
            u_upper=1.0,
            lqr_iter=50,
            backprop=False,
            exit_unconverged=False,
        )

        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)
        )

    def __call__(self, y, u, explore: bool=False):

        """
            inputs: y_t, u_{t-1}
            outputs: planned u_t
        """

        # convert y_t to a torch tensor and add a batch dimension
        y = torch.as_tensor(y, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.dynamics_model.eval()
        
            a = self.encoder(y)

            # update belief using u_{t-1}
            self.dist = self.dynamics_model.dynamics_update(
                dist=self.dist,
                u=torch.as_tensor(u, device=self.device).unsqueeze(0),
            )
            # update belief using a_t
            self.dist = self.dynamics_model.measurement_update(dist=self.dist, a=a)

            planned_x, planned_u, _ = self.planner(
                self.dist.loc,
                self.quadcost,
                self.lindx
            )

            if explore:
                planned_actions += self.action_noise * torch.randn_like(planned_actions)

        
        return np.clip(planned_u.squeeze(1).cpu().numpy(), a_min=-1.0, a_max=1.0)
    
    def reset(self):
        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)
        )