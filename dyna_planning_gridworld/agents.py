import numpy as np
from collections import defaultdict
import random

class DynaQAgent():
    """
    Dyna-Q (and Dyna-Q+) tabular agent.
 
    Args:
        plus          : enable Dyna-Q+ exploration bonus
        env           : gymnasium environment instance
        k             : Dyna-Q+ bonus coefficient
        alpha         : Q-learning step size
        gamma         : discount factor
        epsilon_start : initial ε for ε-greedy policy
        epsilon_end   : minimum ε after decay
        epsilon_decay : exponential decay rate
        planning_steps: simulated updates per real step
    """

    def __init__(
        self,
        env,
        plus: bool = False,
        k: float = 1e-4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.005,
        planning_steps: int = 10,
    ):

        self.env = env
        self.actions = self.env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.planning_steps = planning_steps
        self.plus = plus
        self.q_table = defaultdict(lambda: np.zeros(self.actions))
        self.model: dict[tuple[int, int], tuple[float, int]] = {}

        if self.plus:
            self.k = k
            self.timestep: int = 0
            self.last_visit = defaultdict(lambda: np.zeros(self.actions))

    # functions
    def e_greedy(self, state, epsilon):
        """epsilon-greedy action selection"""
        if np.random.rand() > epsilon:
            return int(np.argmax(self.q_table[state]))
        return self.env.action_space.sample()

    def q_update(self, state, action, reward, next_state):
        """one-step Q-learning update"""
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )

    def update_model(self, state, action, next_state, reward):
        self.model[(state, action)] = (reward, next_state)

        if self.plus:
            self.timestep += 1
            self.last_visit[state, action] = self.timestep

    def epsilon_exp_decay(self, eps):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(- self.epsilon_decay * eps)

    def planning(self):
        for _ in range(self.planning_steps):
            # randomly sample a state and action from the model
            state, action = random.choice(list(self.model))
            reward, next_state = self.model[(state, action)]
            # adding dyna q+ option
            if self.plus:
                tau = self.timestep - self.last_visit[state][action]
                reward += self.k * np.sqrt(tau)
            # update q_table
            self.q_update(state, action, reward, next_state)