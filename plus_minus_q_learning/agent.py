import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class QLearningAgent():
    """
    Q-Learning tabular agent.
    
    Args:
        env:                gymnasium environment
        training_episodes:  number of episodes performed
        alpha:              learning rate
        gamma:              discount factor
        episode_steps:      max nr. of steps performed for each episode
        epsilon_start :     initial ε for ε-greedy policy
        epsilon_end   :     minimum ε after decay
        epsilon_decay :     exponential decay rate
        epsilon:            current ε value
    """

    def __init__(
            self,
            env,
            training_episodes: int = 400,
            alpha: float = 0.1,
            gamma: float = 0.99,
            episode_steps: int = 100,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay: float = 0.005, # λ == decay rate
            epsilon: float = 1,
    ):
        
        self.env = env
        self.training_episodes = training_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.episode_steps = episode_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.q_table = np.zeros ((self.state_space, self.action_space))

        self.episode_rewards = []
        self.episode_lengths = []

    def reset(self):
        state, _ = self.env.reset()
        step = 0
        truncated = terminated = False

        return state, step, truncated, terminated

    def action_selection(self, state, episode):
        """epsilon-greedy action selection"""
        if np.random.rand() > self.epsilon:
            return int(np.argmax(self.q_table[state]))
        return self.env.action_space.sample()

    def q_table_update_std(self, state, action, reward, next_state):
        """one-step bellman equation update"""
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )
    
    def q_table_update_opposite(self, state, action, reward, next_state):
        """one-step bellman equation update"""
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )

    def q_table_update_relative(self, state, action, reward, next_state):
        """one-step bellman equation update"""
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )

    def epsilon_exponential_decay(self, episode):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(- self.epsilon_decay * episode)
    
    def train(self, mode):
        print("--- TRAINING ---")

        for eps in range (self.training_episodes):
            self.epsilon = self.epsilon_exponential_decay(eps)

            state, step, truncated, terminated = self.reset()
            total_reward = 0

            for step in range(self.episode_steps):
                if terminated or truncated:
                    break
 
                action = self.action_selection(state, eps)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if mode == "std":
                    self.q_table_update_std(state, action, reward, next_state)
                elif mode == "opposite":
                    self.q_table_update_opposite(state, action, reward, next_state)
                elif mode == "relative":
                    self.q_table_update_relative(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            if (eps + 1) % 50 == 0:
                print("Episode: ", eps + 1)

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step + 1)

        
        print("--- COMPLETED ---")

    def plot_training(self, grid_size=4):
        """
        Subplot con:
          1. Reward per episodio + media mobile
          2. Lunghezza degli episodi + media mobile
          3. Reward cumulativo nel tempo
          4. Policy appresa (griglia con frecce)
        """
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        window = max(1, len(rewards) // 20)   # finestra ~5% degli episodi
 
        def moving_avg(x, w):
            return np.convolve(x, np.ones(w) / w, mode='valid')
 
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Q-Learning — Training Summary", fontsize=14, fontweight='bold')
 
        eps_x = np.arange(len(rewards))
        ma_x  = np.arange(window - 1, len(rewards))
 
        # ── 1. Reward per episodio ──────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(eps_x, rewards, alpha=0.35, color='steelblue', linewidth=0.8, label='Reward')
        ax.plot(ma_x, moving_avg(rewards, window), color='steelblue', linewidth=2, label=f'Media mobile ({window})')
        ax.set_title("Reward per Episodio")
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
        # ── 2. Lunghezza episodi ────────────────────────────────────────
        ax = axes[0, 1]
        ax.plot(eps_x, lengths, alpha=0.35, color='darkorange', linewidth=0.8, label='Lunghezza')
        ax.plot(ma_x, moving_avg(lengths, window), color='darkorange', linewidth=2, label=f'Media mobile ({window})')
        ax.set_title("Lunghezza degli Episodi")
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Steps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
        # ── 3. Reward cumulativo ────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(eps_x, np.cumsum(rewards), color='seagreen', linewidth=1.8)
        ax.set_title("Reward Cumulativo nel Tempo")
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Reward cumulativo")
        ax.grid(True, alpha=0.3)
 
        # ── 4. Policy (griglia con frecce) ─────────────────────────────
        ax = axes[1, 1]
        policy = np.argmax(self.q_table, axis=1)
 
        # 0=LEFT 1=DOWN 2=RIGHT 3=UP  (FrozenLake convention)
        arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        colors = {0: '#4a90d9', 1: '#e67e22', 2: '#27ae60', 3: '#8e44ad'}
 
        grid = policy.reshape(grid_size, grid_size)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, color='gray', alpha=0.4)
        ax.set_aspect('equal')
        ax.set_title("Policy Appresa (azione greedy)")
        ax.invert_yaxis()
 
        for r in range(grid_size):
            for c in range(grid_size):
                a = grid[r, c]
                ax.text(c, r, arrows[a], ha='center', va='center',
                        fontsize=22, color=colors[a], fontweight='bold')
 
        legend_patches = [
            mpatches.Patch(color=colors[a], label=f"{arrows[a]} {name}")
            for a, name in zip([2, 1, 0, 3], ['RIGHT', 'DOWN', 'LEFT', 'UP'])
        ]
        ax.legend(handles=legend_patches, loc='upper right',
                  fontsize=7, ncol=2, framealpha=0.8)
 
        plt.tight_layout()
        plt.savefig("./training_summary.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("Plot salvato in training_summary.png")
 
