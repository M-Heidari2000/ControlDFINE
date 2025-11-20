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
    config: DictConfig,
    train_config: DictConfig,
    env: gym.Env,
    dynamics_model: Dynamics,
    encoder: Encoder,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    state_targets = make_grid(
        low=env.state_space.low,
        high=env.state_space.high,
        num_points=config.num_points,
    )
    obs_targets = env.manifold(state_targets)
    costs = np.zeros((obs_targets.shape[0], config.num_trials))

    for i, target in enumerate(obs_targets):
        # train a cost function for each target
        train_buffer = train_buffer.map_costs(obs_target=target)
        test_buffer = test_buffer.map_costs(obs_target=target)
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
            planning_horizon=config.planning_horizon,
        )
        for j in range(config.num_trials):
            costs[i, j] = trial(env=env, agent=agent, obs_target=target)
