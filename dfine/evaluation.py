import torch
import numpy as np
import gymnasium as gym
from .agents import IMPCAgent, OracleMPC
from omegaconf.dictconfig import DictConfig
from .models import Dynamics, Encoder
from .utils import make_grid


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
        x = torch.as_tensor(info["state"], device=oracle.device).unsqueeze(0)
        planned_actions = oracle(x=x)
        action = planned_actions[0].flatten()
        obs, _, terminated, truncated, info = env.step(action=action)
        if terminated:
            oracle_cost += np.inf
        else:
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
        if terminated:
            total_cost += np.inf
        else:
            total_cost += np.linalg.norm(obs - obs_target) ** 2
        done = terminated or truncated

    return total_cost.item() / oracle_cost.item()


def evaluate(
    eval_config: DictConfig,
    env: gym.Env,
    dynamics_model: Dynamics,
    encoder: Encoder,
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
            # create agent
            agent = IMPCAgent(
                encoder=encoder,
                dynamics_model=dynamics_model,
                obs_target=obs_target,
                planning_horizon=eval_config.planning_horizon,
            )

            # create oracle
            device = next(dynamics_model.parameters()).device
            Q = torch.eye(env.state_space.shape[0], device=device)
            R = torch.eye(dynamics_model.u_dim, dtype=torch.float32, device=device) * 1e-6
            q = -torch.as_tensor(sample, device=device).reshape(1, -1) @ Q
            A=torch.as_tensor(env.A, device=device)
            B=torch.as_tensor(env.B, device=device)
            oracle = OracleMPC(Q=Q, R=R, q=q, A=A, B=B)

            # get a trial
            trial_cost = trial(env=env, agent=agent, oracle=oracle, target=sample)
            costs.append(trial_cost)
        
        region["costs"] = np.array(costs)

    return target_regions