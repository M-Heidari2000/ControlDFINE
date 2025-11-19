import torch
import numpy as np
from mpc import mpc
from mpc.mpc import QuadCost, LinDx


class ILQRAgent:
    """
        action planning by the IQLR method
    """
    def __init__(
        self,
        encoder,
        dynamics_model,
        cost_model,
        planning_horizon: int,
        num_iterations: int = 10,
        action_noise: float = 0.3
    ):
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.cost_model = cost_model
        self.num_iterations = num_iterations
        self.planning_horizon = planning_horizon
        self.action_noise = action_noise

        self.device = next(encoder.parameters()).device

        self.mean = torch.zeros((1, self.dynamics_model.x_dim), device=self.device)
        self.cov = torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)

    def __call__(self, y, u, explore: bool=False):
        """
        inputs: y_t, u_{t-1}
            outputs: planned u_t
            explore: add random values to planned actions for exploration purpose
        """

        # convert y_t to a torch tensor and add a batch dimension
        y = torch.as_tensor(y, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.dynamics_model.eval()
        
            a = self.encoder(y)

            # update belief using u_{t-1}
            self.mean, self.cov = self.dynamics_model.dynamics_update(
                mean=self.mean,
                cov=self.cov,
                u=torch.as_tensor(u, device=self.device).unsqueeze(0)
            )

            # update belief using y_t
            self.mean, self.cov = self.dynamics_model.measurement_update(
                mean=self.mean,
                cov=self.cov,
                a=a,
            )

            # initial policy (zero actions)
            Ks = [
                torch.zeros(
                    (self.dynamics_model.u_dim, self.dynamics_model.x_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
            ] * self.planning_horizon

            ks = [
                torch.zeros(
                    (self.dynamics_model.u_dim, 1),
                    device=self.device,
                    dtype=torch.float32,
                )
            ] * self.planning_horizon

            for _ in range(self.num_iterations + 1):
                state = self.mean
                As = []
                Bs = []
                actions = torch.zeros(
                    (self.planning_horizon, self.dynamics_model.u_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                # rollout a trajectory with current policy
                for t in range(self.planning_horizon):
                    action = (state @ Ks[t].T + ks[t].T).clamp(min=-1.0, max=1.0)
                    actions[t] = action
                    A, B, _, _, _ = self.dynamics_model.get_dynamics(state)
                    A, B = A.squeeze(0), B.squeeze(0)
                    state = state @ A.T + action @ B.T
                    As.append(A)
                    Bs.append(B)
                # compute a new policy
                Ks, ks = self._compute_policy(
                    As=As,
                    Bs=Bs,
                )

            if explore:
                actions += self.action_noise * torch.randn_like(actions)

        return np.clip(actions.cpu().numpy(), a_min=-1.0, a_max=1.0)
    
    def _compute_policy(self, As, Bs):
        state_dim, action_dim = Bs[0].shape

        Ks = []
        ks = []

        V = torch.zeros((state_dim, state_dim), device=self.device)
        v = torch.zeros((state_dim, 1), device=self.device)

        C = torch.block_diag(self.cost_model.Q, self.cost_model.R)
        c = torch.cat([
            self.cost_model.q,
            torch.zeros((action_dim, 1), device=self.device)
        ])

        for t in range(self.planning_horizon-1, -1, -1):
            F = torch.concatenate((As[t], Bs[t]), dim=1)
            Q = C + F.T @ V @ F
            q = c + F.T @ v
            Qxx = Q[:state_dim, :state_dim]
            Qxu = Q[:state_dim, state_dim:]
            Qux = Q[state_dim:, :state_dim]
            Quu = Q[state_dim:, state_dim:]
            qx = q[:state_dim, :]
            qu = q[state_dim:, :]

            K = - torch.linalg.pinv(Quu) @ Qux
            k = - torch.linalg.pinv(Quu) @ qu
            V = Qxx + Qxu @ K + K.T @ Qux + K.T @ Quu @ K
            v = qx + Qxu @ k + K.T @ qu + K.T @ Quu @ k

            Ks.append(K)
            ks.append(k)
        
        return Ks[::-1], ks[::-1]

    def reset(self):
        self.mean = torch.zeros((1, self.dynamics_model.x_dim), device=self.device)
        self.cov = torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)


class MPCAgent:
    """
        action planning by the MPC method
    """
    def __init__(
        self,
        encoder,
        dynamics_model,
        cost_model,
        planning_horizon: int,
        num_iterations: int = 10,
        action_noise: float = 0.3

    ):
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.cost_model = cost_model
        self.num_iterations = num_iterations
        self.planning_horizon = planning_horizon
        self.action_noise = action_noise

        self.device = next(encoder.parameters()).device

        self.mean = torch.zeros((1, self.dynamics_model.x_dim), device=self.device)
        self.cov = torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)

        self.reference_actions = torch.zeros(
            (self.planning_horizon, 1, self.dynamics_model.u_dim),
            device=self.device,
            dtype=torch.float32,
        )

    def __call__(self, y, u, explore: bool=False):
        """
        inputs: y_t, u_{t-1}
            outputs: planned u_t
            explore: add random values to planned actions for exploration purpose
        """

        # convert y_t to a torch tensor and add a batch dimension
        y = torch.as_tensor(y, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.dynamics_model.eval()
        
            a = self.encoder(y)

            # update belief using u_{t-1}
            self.mean, self.cov = self.dynamics_model.dynamics_update(
                mean=self.mean,
                cov=self.cov,
                u=torch.as_tensor(u, device=self.device).unsqueeze(0)
            )

            # update belief using y_t
            self.mean, self.cov = self.dynamics_model.measurement_update(
                mean=self.mean,
                cov=self.cov,
                a=a,
            )

            planned_actions = self.reference_actions.clone()

            for _ in range(self.num_iterations + 1):
                state = self.mean
                As = []
                Bs = []
                # rollout a trajectory with current policy
                for t in range(self.planning_horizon):
                    A, B, _, _, _ = self.dynamics_model.get_dynamics(state)
                    A, B = A.squeeze(0), B.squeeze(0)
                    state = state @ A.T + planned_actions[t] @ B.T
                    As.append(A)
                    Bs.append(B)
                # compute a new policy
                planned_actions, planned_states = self._plan(
                    As=As,
                    Bs=Bs,
                )

            if explore:
                planned_actions += self.action_noise * torch.randn_like(planned_actions)

            self.reference_actions[:-1] = planned_actions[1:].detach()
            self.reference_actions[-1].zero_()

        return np.clip(planned_actions.cpu().numpy(), a_min=-1.0, a_max=1.0)
    
    def _plan(self, As, Bs):
        x_dim, u_dim = Bs[0].shape
        C = torch.block_diag(self.cost_model.Q, self.cost_model.R).repeat(
            self.planning_horizon, 1, 1, 1,
        )
        c = torch.cat([
            self.cost_model.q.reshape(1, -1),
            torch.zeros((1, u_dim), device=self.device)
        ], dim=1).repeat(self.planning_horizon, 1, 1)
        F_list = []
        for A, B in zip(As, Bs):
            Ft = torch.cat((A, B), dim=1)
            Ft = Ft.unsqueeze(0)
            F_list.append(Ft)
        F = torch.stack(F_list, dim=0)
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
            lqr_iter=10,
            backprop=False,
            exit_unconverged=False,
        )

        planned_x, planned_u, _ = self.planner(
            self.mean,
            self.quadcost,
            self.lindx
        )
        
        return planned_u, planned_x

    def reset(self):
        self.mean = torch.zeros((1, self.dynamics_model.x_dim), device=self.device)
        self.cov = torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)

        self.reference_actions = torch.zeros(
            (self.planning_horizon, 1, self.dynamics_model.u_dim),
            device=self.device,
            dtype=torch.float32,
        )