import torch
import numpy as np
import gymnasium as gym
from .agents import MPCAgent


def trial(
    env: gym.Env,
    agent: MPCAgent,
):
    obs, info = agent.reset()
    action = None
    done = False
    total_cost = np.array(0.0)
    while not done:
        planned_actions = agent(y=obs, u=action, explore=False)
        action = planned_actions[0].flatten()
        obs, reward, terminated, truncated, _ = env.step(action=action)
        if terminated:
            total_cost += np.inf
        else:
            total_cost += -reward
        done = terminated or truncated

    return total_cost.item()