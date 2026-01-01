import torch
import numpy as np
import gymnasium as gym
from .agents import IMPCAgent, OracleMPC
from omegaconf.dictconfig import DictConfig
from .models import Dynamics, Encoder
from .utils import make_grid
from .memory import ReplayBuffer
from .train import train_cost


def trial(
    env: gym.Env,
    agent: IMPCAgent,
    oracle: OracleMPC,
    target: np.ndarray,
):
    # initialize the environment in the middle of the state space
    initial_state = (env.state_space.low + env.state_space.high) / 2
    obs_target = env.manifold(target.reshape(1, -1)).flatten()
    options={
        "initial_state": initial_state,
        "target_state": target,
    }

    # control with oracle
    obs, info = env.reset(options=options)
    done = False
    oracle_cost = 0.0
    while not done:
        x = torch.as_tensor(info["state"], device=oracle.Q.device).unsqueeze(0)
        planned_actions = oracle(x=x)
        planned_u = agent(x=x)
        obs, _, terminated, truncated, info = env.step(planned_u[0].flatten())
        oracle_cost += np.linalg.norm(obs - obs_target) ** 2
        done = terminated or truncated

    # control with the learned model
    obs, _ = env.reset(options=options)
    agent.reset()
    action = env.action_space.sample()
    done = False
    total_cost = 0.0
    while not done:
        planned_actions = agent(y=obs, u=action, explore=False)
        action = planned_actions[0].flatten()
        obs, _, terminated, truncated, _ = env.step(action=action)
        total_cost += np.linalg.norm(obs - obs_target) ** 2
        done = terminated or truncated

    return total_cost.item() / oracle_cost.item()


def evaluate(
    eval_config: DictConfig,
    cost_train_config: DictConfig,
    env: gym.Env,
    dynamics_model: Dynamics,
    encoder: Encoder,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    target_regions = make_grid(
        low=env.state_space.low,
        high=env.state_space.high,
        num_regions=eval_config.num_regions,
        num_points=eval_config.num_points,
    )

    for region in target_regions:
        costs = []
        for sample in region["samples"]:
            # train a cost function for this target
            obs_target = env.manifold(sample.reshape(1, -1)).flatten()
            train_buffer = train_buffer.map_costs(obs_target=obs_target)
            test_buffer = test_buffer.map_costs(obs_target=obs_target)
            cost_model = train_cost(
                config=cost_train_config,
                encoder=encoder,
                dynamics_model=dynamics_model,
                train_buffer=train_buffer,
                test_buffer=test_buffer,
            )
            # create agent
            agent = IMPCAgent(
                encoder=encoder,
                dynamics_model=dynamics_model,
                cost_model=cost_model,
                planning_horizon=eval_config.planning_horizon,
            )

            # create oracle
            device = next(cost_model.parameters()).device
            Q = torch.eye(cost_model.x_dim, device=device)
            R = cost_model.R
            q = -torch.as_tensor(sample, device=device).reshape(1, -1) @ Q
            A=torch.as_tensor(env.A, device=device)
            B=torch.as_tensor(env.B, device=device)
            oracle = OracleMPC(Q=Q, R=R, q=q, A=A, B=B)

            # get a trial
            trial_cost = trial(env=env, agent=agent, oracle=oracle, target=sample)
            costs.append(trial_cost)
        
        region["costs"] = np.array(costs)

    return target_regions