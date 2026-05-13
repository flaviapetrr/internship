import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
import imageio
 

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
            mode: str = "std", # "std", "std_punish", "opposite", "relative" 
            training_episodes: int = 400,
            alpha: float = 0.1,
            alpha_v: float = 0.1,
            gamma: float = 0.99,
            episode_steps: int = 100,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay: float = 0.005, # λ == decay rate
            epsilon: float = 1,
    ):
        
        self.env = env
        valid_modes = ["std", "std_punish", "opposite", "relative", "relative_punish"]
        if mode not in valid_modes:
            raise ValueError(f"Erroe: '{mode}' not valid.\nValid options: {valid_modes}")
            
        self.mode = mode
        self.training_episodes = training_episodes
        self.alpha = alpha
        self.alpha_v = alpha_v
        self.gamma = gamma
        self.episode_steps = episode_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.q_table = np.zeros ((self.state_space, self.action_space))
        self.v_table = np.zeros(self.state_space)

        self.episode_rewards = []
        self.episode_lengths = []

        self.valid_actions = {}
        for s in range(self.state_space):
            self.valid_actions[s] = [
                a for a in range(self.action_space)
                if any(ns != s for prob, ns, r, done in self.env.unwrapped.P[s][a])
            ]

    def reset(self):
        state, _ = self.env.reset()
        step = 0
        truncated = terminated = False

        return state, step, truncated, terminated

    def action_selection(self, state, episode):
        """epsilon-greedy action selection"""
        if np.random.rand() > self.epsilon:
            if self.mode in ["opposite", "relative_punish"]:
                return int(np.argmin(self.q_table[state]))
            else:
                return int(np.argmax(self.q_table[state]))
        return self.env.action_space.sample()

    def q_table_update(self, state, action, reward, next_state):
        """
        update equation, depends on mode:
            1. std:             classical one-step bellman equation
            2. opposite:        punishment based one-step bellman equation
            3. relative:        contextual update equation
            4. relative_punish: contextual update equation for punishment values
        """
        if self.mode in ["std", "std_punish"]:
            self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )
        elif self.mode == "opposite":
            self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.min(self.q_table[next_state]) - self.q_table[state][action]
        )
        elif self.mode == "relative":
            # r_v = (r_chosen + sum(stored value of all unchosen states)) / nr.space
            # first getting only the possible actions, no hitting  wall considered
            valid = self.valid_actions[state]
            unchosen_valid = [a for a in valid if a != action]
            if unchosen_valid:
                sum_unchosen_q = sum(self.q_table[state][a] for a in unchosen_valid)
                r_v = (reward + sum_unchosen_q) / (len(unchosen_valid) + 1)
            else:
                r_v = reward
                    
            # update V first
            # V(s) = V(s) + alpha * (r_v - V(s))
            self.v_table[state] += self.alpha_v * (r_v - self.v_table[state])
            
            self.q_table[state][action] += self.alpha * (
                reward - self.v_table[state] + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
            )
        elif self.mode == "relative_punish":
            # r_v = (r_chosen + sum(stored value of all unchosen states)) / nr.space
            # first getting only the possible actions, no hitting  wall considered
            valid = self.valid_actions[state]
            unchosen_valid = [a for a in valid if a != action]
            if unchosen_valid:
                sum_unchosen_q = sum(self.q_table[state][a] for a in unchosen_valid)
                r_v = (reward + sum_unchosen_q) / (len(unchosen_valid) + 1)
            else:
                r_v = reward
            # update V first
            # V(s) = V(s) + alpha * (r_v - V(s))
            self.v_table[state] += self.alpha_v * (r_v - self.v_table[state])
            
            self.q_table[state][action] += self.alpha * (
                reward - self.v_table[state] + self.gamma * np.min(self.q_table[next_state]) - self.q_table[state][action]
            )

    def epsilon_exponential_decay(self, episode):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(- self.epsilon_decay * episode)
    
    def train(self):
        """train agent, update equation depends on mode"""
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

                self.q_table_update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            if (eps + 1) % 50 == 0:
                print("Episode: ", eps + 1)

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step + 1)

        
        print("--- COMPLETED ---\n")

    def plot_training(self, path="training_summary.png", grid_size=4):
        """
        subplots:
            1. reward per episode + moving average
            2. episode length + moving average
            3. cumulative reward
            4. learned policy grid
        """
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        window  = max(1, len(rewards) // 20)
 
        def moving_avg(x, w):
            return np.convolve(x, np.ones(w) / w, mode='valid')
 
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        fig.suptitle(f"{self.mode} Q-Learning - Training Summary", fontsize=14, fontweight='bold')
 
        eps_x = np.arange(len(rewards))
        ma_x  = np.arange(window - 1, len(rewards))
 
        # subplot 1
        ax = axes[0, 0]
        ax.set_ylim(-1.5, 1.5)
        ax.plot(eps_x, rewards, alpha=0.35, color='steelblue', linewidth=0.8, label='Reward')
        ax.plot(ma_x, moving_avg(rewards, window), color='steelblue', linewidth=2,
                label=f'Moving avg ({window})')
        ax.set_title("Reward per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
        # subplot 2
        ax = axes[0, 1]
        ax.set_ylim(0, 25)
        ax.plot(eps_x, lengths, alpha=0.35, color='darkorange', linewidth=0.8, label='Length')
        ax.plot(ma_x, moving_avg(lengths, window), color='darkorange', linewidth=2,
                label=f'Moving avg ({window})')
        ax.set_title("Episode Length")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
        # subplot 3
        ax = axes[1, 0]
        ax.plot(eps_x, np.cumsum(rewards), color='seagreen', linewidth=1.8)
        ax.set_title("Cumulative Reward over Time")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
        ax.grid(True, alpha=0.3)
 
        # subplot 4
        ax = axes[1, 1]
        # greedy policy for the plot
        if self.mode in ["opposite", "relative_punish"]:
            policy = np.argmin(self.q_table, axis=1)
        else:
            policy = np.argmax(self.q_table, axis=1)
 
        # moves: 0=left 1=down 2=right 3=up
        arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        arrow_colors = {0: '#4a90d9', 1: '#e67e22', 2: '#27ae60', 3: '#8e44ad'}
 
        grid_policy = policy.reshape(grid_size, grid_size)
 
        # read the actual map from the env (works for any custom map too)
        desc = self.env.unwrapped.desc.astype(str)   # shape (grid_size, grid_size)
 
        bg = {'S': '#d4edda',   # start
              'F': '#f8f9fa',   # frozen
              'H': '#adb5bd',   # hole
              'G': '#fff3cd'}   # goal
 
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_aspect('equal')
        ax.set_title("Learned Policy  (S=start H=hole G=goal)")
        ax.invert_yaxis()
 
        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc[r, c]
 
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     color=bg.get(cell, '#f8f9fa'), zorder=0)
                ax.add_patch(rect)
 
                if cell == 'H':
                    # hole: only label, no arrow
                    ax.text(c, r, 'H', ha='center', va='center',
                            fontsize=18, color='#495057', fontweight='bold')
                elif cell == 'G':
                    ax.text(c, r, 'G', ha='center', va='center',
                            fontsize=18, color='#856404', fontweight='bold')
                else:
                    # start or frozen: greedy arrow
                    a = grid_policy[r, c]
                    ax.text(c, r, arrows[a], ha='center', va='center',
                            fontsize=22, color=arrow_colors[a], fontweight='bold')
                    if cell == 'S':
                        ax.text(c + 0.35, r - 0.35, 'S', ha='center', va='center',
                                fontsize=7, color='#155724', fontweight='bold')
 
        ax.grid(True, color='gray', alpha=0.4, zorder=1)
 
        """legend_patches = [
            mpatches.Patch(color=arrow_colors[a], label=f"{arrows[a]} {name}")
            for a, name in zip([2, 1, 0, 3], ['RIGHT', 'DOWN', 'LEFT', 'UP'])
        ] + [
            mpatches.Patch(color=bg['H'], label='H hole',  edgecolor='gray'),
            mpatches.Patch(color=bg['G'], label='G goal',  edgecolor='gray'),
            mpatches.Patch(color=bg['S'], label='S start', edgecolor='gray'),
        ]
        ax.legend(handles=legend_patches, loc='upper right',
                  fontsize=7, ncol=2, framealpha=0.9) """
 
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to {path}")

    def record_gif(self, env_id, env_kwargs, path="agent_episode.gif", fps=4):
        """run one greedy episode"""
        render_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        state, _ = render_env.reset()
        frames = [render_env.render()]
 
        terminated = truncated = False
        for _ in range(self.episode_steps):
            if terminated or truncated:
                break
            # greedy policy for the recording
            if self.mode in ["opposite", "relative_punish"]:
                action = int(np.argmin(self.q_table[state]))
            else:
                action = int(np.argmax(self.q_table[state]))
            state, _, terminated, truncated, _ = render_env.step(action)
            frames.append(render_env.render())
 
        render_env.close()
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print(f"GIF saved to {path}  ({len(frames)} frames)")