import torch
import einops
from typing import Optional
import numpy as np
from mpc import mpc
from mpc.mpc import QuadCost, LinDx
from torch.distributions import MultivariateNormal
from .models import Encoder, Dynamics, CostModel


class CEMAgent:
    """
        action planning by the Cross Entropy Method (CEM)
    """
    def __init__(
        self,
        encoder: Encoder,
        dynamics_model: Dynamics,
        cost_model: CostModel,
        planning_horizon: int,
        num_iterations: int = 10,
        num_candidates: int = 100,
        num_elites: int = 10,
        action_noise: float = 0.3,
    ):
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.cost_model = cost_model
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.planning_horizon = planning_horizon
        self.action_noise = action_noise

        self.device = next(encoder.parameters()).device

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
        action_dist = MultivariateNormal(
            loc=torch.zeros((self.planning_horizon, self.dynamics_model.u_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.u_dim, device=self.device).expand([self.planning_horizon, -1, -1])
        )
        dist = MultivariateNormal(
            loc=self.dist.loc.expand(self.num_candidates, -1),
            covariance_matrix=self.dist.covariance_matrix.expand(self.num_candidates, -1, -1),
        )

        for _ in range(self.num_iterations):
            action_candidates = action_dist.sample([self.num_candidates])
            action_candidates = einops.rearrange(action_candidates, "n h u -> h n u")
            action_candidates = action_candidates.clamp(min=-1.0, max=1.0)
            prior_samples = self.dynamics_model.generate(dist=dist, u=action_candidates)
            total_predicted_cost = torch.zeros(self.num_candidates, device=self.device)
            for t in range(self.planning_horizon):
                total_predicted_cost += self.cost_model(x=prior_samples[t]).squeeze()
            # find the elite sequences
            elite_indexes = total_predicted_cost.argsort(descending=False)[:self.num_elites]
            elites = action_candidates[:, elite_indexes, :]

            # fit a new distribution on the elites
            mean = elites.mean(dim=1)
            cov = torch.diag_embed(elites.var(dim=1, unbiased=False) + 1e-4)
            action_dist = MultivariateNormal(loc=mean, covariance_matrix=cov)

        return action_dist.loc


    def reset(self):
        # Initialize belief with zeros
        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device),
        )
        

class OracleMPC:
    """
        action planning by MPC method using the actual states
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

        C = torch.block_diag(Q, R).repeat(planning_horizon, 1, 1, 1)
        c = torch.cat([
            q,
            torch.zeros((1, u_dim), device=self.device)
        ], dim=1).repeat(planning_horizon, 1, 1)
        
        F = torch.cat((A, B), dim=1).repeat(planning_horizon, 1, 1, 1)
        f = torch.zeros((1, x_dim), device=self.device).repeat(planning_horizon, 1, 1)
        
        self.quadcost = QuadCost(C, c)
        self.lindx = LinDx(F, f)
        
        self.planner = mpc.MPC(
            n_batch=1,
            n_state=x_dim,
            n_ctrl=u_dim,
            T=planning_horizon,
            u_lower=-1.0,
            u_upper=1.0,
            lqr_iter=10,
            backprop=False,
            exit_unconverged=False,
        )

    def __call__(self, x: torch.Tensor):
        planned_x, planned_u, _ = self.planner(
            x,
            self.quadcost,
            self.lindx
        )        
        return np.clip(planned_u.squeeze(1).cpu().numpy(), a_min=-1.0, a_max=1.0)