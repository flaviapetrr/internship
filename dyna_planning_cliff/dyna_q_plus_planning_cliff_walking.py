# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt

# parameters
n_training_eps = 500
steps_eps = 200

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
        self.states = self.env.observation_space.n
        self.actions = self.env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.planning_steps = planning_steps
        self.plus = plus
        self.q_table = np.zeros((self.states, self.actions))
        self.model: dict[tuple[int, int], tuple[float, int]] = {}

        if self.plus:
            self.k = k
            self.timestep: int = 0
            self.last_visit = np.zeros((self.states, self.actions))

    # functions
    def e_greedy(self, state, epsilon):
        """epsilon-greedy action selection"""
        if np.random.rand() > epsilon:
            return int(np.argmax(self.q_table[state]))
        return self.env.action_space.sample()

    def q_update(self, state, action, reward, next_state):
        """one-step Q-learning update"""
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action]
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
            
def train(agent, n_training_eps, steps_eps):
    episode_rewards = []
    episode_lengths = []
    epsilons = []

    for eps in range (n_training_eps):
        # exponential epsilon decay
        epsilon = agent.epsilon_exp_decay(eps)
        epsilons.append(epsilon)

        state, _ = agent.env.reset()
        total_reward = 0
        truncated = False
        terminated = False

        for step in range (steps_eps):
            action = agent.e_greedy(state, epsilon)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            #update q_table
            agent.q_update(state, action, reward, next_state)
            # update model -> if i'm in state s and do action a what will happen?
            agent.update_model(state, action, next_state, reward)

            # planning
            if agent.model: # planning only once a model exists
                agent.planning()

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        
    return agent.q_table, episode_rewards, episode_lengths, epsilons


def plot_results(agent, episode_rewards, episode_lengths, epsilons, q_table, window=20):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if agent.plus:
        fig.suptitle("Dyna-Q+ on CliffWalking", fontsize=14, fontweight="bold")
    else:
        fig.suptitle("Dyna-Q on CliffWalking", fontsize=14, fontweight="bold")

    # 1. Reward per episode + smoothed
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, color="steelblue", label="Raw")
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
    ax.plot(range(window - 1, len(episode_rewards)), smoothed, color="steelblue", label=f"Smoothed (w={window})")
    ax.axhline(-13, color="green", linestyle="--", label="Optimal (-13)")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()

    # 2. Episode length
    ax = axes[0, 1]
    smoothed_len = np.convolve(episode_lengths, np.ones(window)/window, mode="valid")
    ax.plot(episode_lengths, alpha=0.3, color="coral")
    ax.plot(range(window - 1, len(episode_lengths)), smoothed_len, color="coral")
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")

    # 3. Epsilon decay
    ax = axes[1, 0]
    ax.plot(epsilons, color="purple")
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")

    # 4. Learned policy on the grid (4 rows x 12 cols)
    ax = axes[1, 1]
    action_symbols = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    grid = np.full((4, 12), "", dtype=object)
    best_actions = np.argmax(q_table, axis=1)

    for state in range(48):
        row, col = divmod(state, 12)
        if row == 3 and 1 <= col <= 10:  # cliff
            grid[row, col] = "X"
        elif state == 47:                 # goal
            grid[row, col] = "G"
        elif state == 36:                 # start
            grid[row, col] = "S"
        else:
            grid[row, col] = action_symbols[best_actions[state]]

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Learned Policy")

    for row in range(4):
        for col in range(12):
            color = "lightcoral" if grid[row, col] == "X" else \
                    "lightgreen"  if grid[row, col] == "G" else "white"
            ax.add_patch(plt.Rectangle((col, 3 - row), 1, 1,
                         linewidth=0.5, edgecolor="gray", facecolor=color))
            ax.text(col + 0.5, 3 - row + 0.5, grid[row, col],
                    ha="center", va="center", fontsize=11)

    plt.tight_layout()
    if agent.plus:
        plt.savefig("dyna_q_plus_results.png", dpi=150, bbox_inches="tight")
    else:
        plt.savefig("dyna_q_results.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    #define environment
    env_id = "CliffWalking-v1"
    # define agent
    agent = DynaQAgent(plus = False, env = gym.make(env_id, render_mode="rgb_array"))
    # train agent
    q_table, rewards, lengths, epsilons = train(agent, n_training_eps, steps_eps)
    plot_results(agent, rewards, lengths, epsilons, q_table)
    