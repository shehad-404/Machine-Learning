# Author: Andrea Pierré
# License: GPLv3+

from typing import NamedTuple
from enum import Enum
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

sns.set_theme()

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    seed: int  # Define a seed so that we get reproducible results
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    epsilon: float  # Exploration probability  

params = Params(
    total_episodes=50,
    learning_rate=0.3,
    gamma=0.95,
    seed=42,
    n_runs=100,
    action_size=None,
    state_size=None,
    epsilon=0.1,
)

rng = np.random.default_rng(params.seed)

class Actions(Enum):
    Left = 0
    Right = 1

# Environment Definition
class RandomWalk1D:
    def __init__(self):
        self.observation_space = np.arange(0, 7)
        self.action_space = [item.value for item in list(Actions)]
        self.right_boundary = 6
        self.left_boundary = 0
        self.reset()

    def reset(self):
        self.current_state = 3
        return self.current_state

    def step(self, action):
        if action == Actions.Left.value:
            new_state = max(self.left_boundary, self.current_state - 1)
        elif action == Actions.Right.value:
            new_state = min(self.right_boundary, self.current_state + 1)
        else:
            raise ValueError("Invalid action type")
        self.current_state = new_state
        reward = self.reward(self.current_state)
        is_terminated = self.is_terminated(self.current_state)
        return new_state, reward, is_terminated

    def reward(self, observation):
        if observation == self.right_boundary:
            return 1
        elif observation == self.left_boundary:
            return -1
        return 0

    def is_terminated(self, observation):
        return observation == self.right_boundary or observation == self.left_boundary

env = RandomWalk1D()
params = params._replace(action_size=len(env.action_space), state_size=len(env.observation_space))

# Q-learning Class
class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        delta = (
            reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action]
        )
        self.qtable[state, action] += self.learning_rate * delta

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

# Epsilon-Greedy Policy
class EpsilonGreedy:
    def __init__(self, epsilon, rng=None):
        self.epsilon = epsilon
        self.rng = rng or default_rng()

    def choose_action(self, action_space, state, qtable):
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.choice(action_space)  # Exploration
        return np.argmax(qtable[state, :])  # Exploitation

learner = Qlearning(
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    state_size=params.state_size,
    action_size=params.action_size,
)

explorer = EpsilonGreedy(epsilon=params.epsilon, rng=rng)

rewards = np.zeros((params.total_episodes, params.n_runs))
steps = np.zeros((params.total_episodes, params.n_runs))
episodes = np.arange(params.total_episodes)
qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
all_states = []
all_actions = []

for run in range(params.n_runs):
    learner.reset_qtable()
    for episode in tqdm(episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False):
        state = env.reset()
        done = False
        total_rewards = 0

        while not done:
            action = explorer.choose_action(
                action_space=env.action_space, state=state, qtable=learner.qtable
            )
            all_states.append(state)
            all_actions.append(action)

            new_state, reward, done = env.step(action)
            learner.update(state, action, reward, new_state)

            total_rewards += reward
            state = new_state

        rewards[episode, run] = total_rewards

# Add Plotting and Results as Required
