import wandb
import numpy as np
import gymnasium as gym
from .agents import IMPCAgent
from omegaconf.dictconfig import DictConfig
from .models import Dynamics, Encoder
from .utils import make_grid
from .memory import ReplayBuffer
from .train import train_cost


def trial(
    env: gym.Env,
    agent: IMPCAgent,
    obs_target: np.ndarray,
):
    # initialize the environment in the middle of the state space
    mid = (env.state_space.low + env.state_space.high) / 2
    obs, _ = env.reset(options={"initial_state": mid})
    agent.reset()
    action = env.action_space.sample()
    done = False
    total_cost = 0.0
    while not done:
        planned_actions = agent(y=obs, u=action, explore=False)
        action = planned_actions[0].flatten()
        next_obs, _, terminated, truncated, _ = env.step(action=action)
        total_cost += np.linalg.norm(obs - obs_target) ** 2
        done = terminated or truncated
        obs = next_obs
    return total_cost.item()


def evaluate(
    eval_config: DictConfig,
    train_config: DictConfig,
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
                config=train_config,
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
            trial_cost = trial(env=env, agent=agent, obs_target=obs_target)
            costs.append(trial_cost)
        region["costs"] = np.array(costs)

    return target_regions