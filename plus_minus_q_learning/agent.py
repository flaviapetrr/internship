import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
import imageio
import heapq
import io

class QLearningAgent():
    """
    Q-Learning tabular agent
    
    Args:
        env:                gymnasium environment
        training_episodes:  number of episodes performed
        episode_steps:      max nr. of steps performed for each episode
        q_init:             initial values of q_table
        alpha:              learning rate
        gamma:              discount factor
        epsilon_start :     initial ε for ε-greedy policy
        epsilon_end   :     minimum ε after decay
        epsilon_decay :     exponential decay rate
        epsilon:            current ε value
        replay_steps:       number of replay performed for each step
        theta:              prioritized sweeping surprise threshold
    """

    def __init__(
            self,
            env,
            mode: str = "std", # "std", "std_punish", "opposite", "relative", "relative_punish"
            replay_mode: str = "none",
            training_episodes: int = 400,
            episode_steps: int = 100,
            q_init: float = 0.0,
            alpha: float = 0.1,
            alpha_v: float = 0.1,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay: float = 0.005, # λ == decay rate
            epsilon: float = 1,
            replay_steps: int = 10,
            theta: float = 0.0001
    ):
        
        self.env = env
        valid_modes = ["std", "std_punish", "opposite", "relative", "relative_punish"]
        if mode not in valid_modes:
            raise ValueError(f"Error: '{mode}' not valid.\nValid options: {valid_modes}")
            
        self.mode = mode
        self.replay_mode = replay_mode
        self.training_episodes = training_episodes
        self.episode_steps = episode_steps
        self.q_init = q_init
        self.alpha = alpha
        self.alpha_v = alpha_v
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.replay_steps = replay_steps
        self.theta = theta
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.q_table = np.full((self.state_space, self.action_space), self.q_init)
        self.v_table = np.zeros(self.state_space)

        self.valid_actions = {}
        for s in range(self.state_space):
            self.valid_actions[s] = list(range(self.action_space))
            # uncomment if not considering hitting walls as valid actions
            # self.valid_actions[s] = [
            #    a for a in range(self.action_space)
            #    if any(ns != s for prob, ns, r, done in self.env.unwrapped.P[s][a])
            # ]
        
        self.episode_memory = []
        
        self.model = {}
        self.predecessors = {}
        self.priority_queue = []

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_td_errors = [] 
        self.episode_success = [] 
        self.replay_counts = []        
        self.model_sizes = []
        self.sampled_paths = []   
        self.state_visits = np.zeros(self.state_space)

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

    def q_table_update(self, state, action, reward, next_state):
        """
        update equation, depends on mode:
            1. std && std_punish:   classical one-step bellman equation
            2. opposite:            punishment based one-step bellman equation
            3. relative:            contextual update equation
            4. relative_punish:     contextual update equation for punishment values
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
            total_td_error = 0
            episode_replay_count = 0 
            self.episode_memory = []
            self.priority_queue = []

            current_path = [state]
            self.state_visits[state] += 1

            for step in range(self.episode_steps):
                if terminated or truncated:
                    break
 
                action = self.action_selection(state, eps)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                current_path.append(next_state)
                self.state_visits[next_state] += 1

                # saving experience
                self.episode_memory.append((state, action, reward, next_state))

                # TD error before update
                if self.mode in ["opposite", "relative_punish"]:
                    best_next = np.min(self.q_table[next_state])
                else:
                    best_next = np.max(self.q_table[next_state])

                td_error = abs(reward + self.gamma * best_next - self.q_table[state][action])
                total_td_error += td_error
 
                # updating q-table
                if self.replay_mode == "none":
                    self.q_table_update(state, action, reward, next_state)
 
                # updating model
                if state not in self.model:
                    self.model[state] = {}
                self.model[state][action] = (reward, next_state) 
 
                if next_state not in self.predecessors:
                    self.predecessors[next_state] = set()
                self.predecessors[next_state].add((state, action)) 
 
                # seed priority queue for prioritized sweeping
                if self.replay_mode == "forward" and td_error > self.theta:
                    heapq.heappush(self.priority_queue, (-td_error, (state, action)))
 
                if self.replay_mode == "forward":
                    self.prioritized_sweeping(self.replay_steps)
                    episode_replay_count += self.replay_steps
                    
                state = next_state
                total_reward += reward

                if self.replay_mode == "backward":
                    self.backward_replay(self.replay_steps)
                    episode_replay_count += min(self.replay_steps, len(self.episode_memory))
 
            if self.replay_mode == "dyna":
                self.dyna_replay(self.replay_steps)
                episode_replay_count += self.replay_steps

            if (eps + 1) % 100 == 0:
                print("Episode: ", eps + 1)
                
            if (eps + 1) % 50 == 0:
                self.sampled_paths.append((eps + 1, current_path))

            n_steps_done = step + 1
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(n_steps_done)
            self.episode_td_errors.append(total_td_error / max(n_steps_done, 1))
            if self.mode in ["std", "relative"]:
                self.episode_success.append(1 if total_reward > 0 else 0)
            else:
                self.episode_success.append(1 if total_reward < 0 else 0)
            self.replay_counts.append(episode_replay_count)
            self.model_sizes.append(sum(len(v) for v in self.model.values()))
        
        print("--- COMPLETED ---\n")

    def backward_replay(self, n_steps=10):
        """
        MF-RL
        associated with backward and unordered replays
        -> rapid learning
        """
        replay_batch = self.episode_memory[-n_steps:][::-1]

        for state, action, reward, next_state in replay_batch:
            self.q_table_update(state, action, reward, next_state)

    def prioritized_sweeping(self, n_steps=10):
        """
        MB-RL: prioritized sweeping
        associated with forward and imaginary replays
        -> planning
        """
        steps = 0
       
        while self.priority_queue and steps < n_steps:
            # popping experience with max priority
            _, (state, action) = heapq.heappop(self.priority_queue)
            
            # recovering experience from model
            reward, next_state = self.model[state][action]
            
            # updating q-table based on model
            self.q_table_update(state, action, reward, next_state)
            
            # proparating
            if state in self.predecessors:
                for pre_state, pre_action in self.predecessors[state]:
                    pre_reward, _ = self.model[pre_state][pre_action]
                    
                    # computing "surprise" and checking against threshold theta
                    if self.mode in ["opposite", "relative_punish"]:
                        best_next_q = np.min(self.q_table[state])
                    else:
                        best_next_q = np.max(self.q_table[state])
                    current_q = self.q_table[pre_state][pre_action]
                    td_error = abs(pre_reward + self.gamma * best_next_q - current_q)

                    if td_error > self.theta:
                        heapq.heappush(self.priority_queue, (-td_error, (pre_state, pre_action)))
                        
            steps += 1

    def dyna_replay(self, n_steps=10):
        """
        Dyna algorithm
        associated with a mix of both forward and backward replays
        -> efficient
        """
        if not self.model:
            return
        import random
        states_with_model = list(self.model.keys())

        for _ in range(n_steps):
            s = random.choice(states_with_model)
            a = random.choice(list(self.model[s].keys()))
            reward, next_state = self.model[s][a]
            self.q_table_update(s, a, reward, next_state)

# --------------- VISUALIZING RESULTS ---------------

    def plot_training(self, path="training_summary.png", grid_size=10):
        """
        subplots:
            1. reward per episode + moving average
            2. episode length + moving average
            3. Q-value heatmap (max Q per state)
            4. learned policy grid
        """
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        window  = max(1, len(rewards) // 20)
 
        def moving_avg(x, w):
            return np.convolve(x, np.ones(w) / w, mode='valid')
 
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
 
        fig.suptitle(
            f"Training Summary  |  mode={self.mode}  α={self.alpha}  episodes={self.training_episodes}  q_table initial values={self.q_init}",
            fontsize=13, fontweight='bold'
        )
 
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
        ax.set_ylim(0, self.episode_steps * 1.05)
        ax.set_xlim(0, self.training_episodes * 1.05)
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
        q_vals = np.max(self.q_table, axis=1)
        grid_q = q_vals.reshape(grid_size, grid_size)
        desc_q = self.env.unwrapped.desc.astype(str)
        vmin, vmax = float(q_vals.min()), float(q_vals.max())
        im = ax.imshow(grid_q,  cmap="plasma", aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        mid = (vmin + vmax) / 2
        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc_q[r, c]
                val  = grid_q[r, c]

                text_color = 'white' if val < mid else 'black' 
                ax.text(c, r, f"{val:.2f}", ha='center', va='center',
                            fontsize=5.5, color = text_color)


        ax.set_title("Q-value Heatmap  (max Q per state)")
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.tick_params(labelsize=7)
 
        # subplot 4
        ax = axes[1, 1]
        policy = np.argmax(self.q_table, axis=1)
 
        # moves: 0=left 1=down 2=right 3=up
        arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
 
        grid_policy = policy.reshape(grid_size, grid_size)
 
        # reading the actual map from the env
        desc = self.env.unwrapped.desc.astype(str)
 
        bg = {'S': '#d4edda', # start
              'F': '#f8f9fa', # frozen
              'H': '#adb5bd', # hole
              'G': '#fff3cd'} # goal
 
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_aspect('equal')
        ax.set_title("Learned Policy (S=start H=hole G=goal)")
        ax.invert_yaxis()
 
        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc[r, c]
 
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     color=bg.get(cell, '#f8f9fa'), zorder=0)
                ax.add_patch(rect)
 
                if cell == 'H':
                    ax.text(c, r, 'H', ha='center', va='center',
                            fontsize=18, color='#495057', fontweight='bold')
                elif cell == 'G':
                    ax.text(c, r, 'G', ha='center', va='center',
                            fontsize=18, color='#856404', fontweight='bold')
                else:
                    # start or frozen: greedy arrow
                    a = grid_policy[r, c]
                    ax.text(c, r, arrows[a], ha='center', va='center',
                            fontsize=22, color="#8F8F8F", fontweight='bold')
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
        #plt.show()
        print(f"Plot saved to {path}")

    def plot_replay_analysis(self, path="replay_analysis.png", grid_size=10, window=50):
        """
        subplots:
            1. rolling success rate
            2. mean TD error per episode
            3. model coverage 
        """
        n = len(self.episode_rewards)
        w = max(1, min(window, n // 5))
        eps_x = np.arange(n)
 
        def rolling(x, w):
            return np.convolve(x, np.ones(w) / w, mode='valid')
 
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(
            f"Replay analysis  |  mode={self.mode}  α={self.alpha}  episodes={self.training_episodes}   q_table initial values={self.q_init}",
            fontsize=13, fontweight='bold'
        )    

        # subplot 1
        ax = axes[0]
        success = np.array(self.episode_success, dtype=float)
        roll_x  = np.arange(w - 1, n)
        ax.plot(roll_x, rolling(success, w) * 100,
                color='steelblue', linewidth=2)
        ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_title(f"Rolling Success Rate  (window={w})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success %")
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
 
        # subplot 2
        ax = axes[1]
        td = np.array(self.episode_td_errors)
        ax.plot(eps_x, td, alpha=0.3, color='tomato', linewidth=0.7)
        ax.plot(np.arange(w - 1, n), rolling(td, w),
                color='tomato', linewidth=2, label=f'Moving avg ({w})')
        ax.set_title("Mean TD Error per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("TD error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
        # subplot 3
        ax = axes[2]
        max_pairs = self.state_space * self.action_space
        model_arr = np.array(self.model_sizes)
        ax.plot(eps_x, model_arr, color='mediumpurple', linewidth=1.8,
                label='(s,a) pairs in model')
        ax.axhline(max_pairs, color='gray', linestyle='--', linewidth=0.8,
                   label=f'Max possible ({max_pairs})')
        ax.set_title("Model Coverage over Episodes")
        ax.set_xlabel("Episode")
        ax.set_ylabel("# (s, a) pairs known")
        ax.set_ylim(0, max_pairs * 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        #plt.show()
        print(f"Replay plot saved to {path}")

    def record_gif(self, env_id, max_steps, env_kwargs, path="agent_episode.gif", fps=4):
        """run one greedy episode"""
        render_env = gym.make(env_id, max_episode_steps=max_steps, render_mode="rgb_array", **env_kwargs)
        state, _ = render_env.reset()
        frames = [render_env.render()]
 
        terminated = truncated = False
        for _ in range(self.episode_steps):
            if terminated or truncated:
                break
            # greedy policy for the recording
            action = int(np.argmax(self.q_table[state]))
            state, _, terminated, truncated, _ = render_env.step(action)
            frames.append(render_env.render())
 
        render_env.close()
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print(f"GIF saved to {path} ({len(frames)} frames)")

    def plot_trajectories(self, env_id, max_steps, env_kwargs, num_episodes=50, epsilon_test=0.1, path="all_trajectories.png", grid_size=10):
            """performing N episodes
               after training
               epsilon-greedy police
               poltting the trajectories + visual jitter"""
            test_env = gym.make(env_id, max_episode_steps=max_steps, **env_kwargs)
            desc = test_env.unwrapped.desc.astype(str)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f"Trajectories | episodes={num_episodes} epsilon={epsilon_test}", fontsize=14, fontweight='bold')
            ax.set_xlim(-0.5, grid_size - 0.5)
            ax.set_ylim(-0.5, grid_size - 0.5)
            ax.set_xticks(range(grid_size))
            ax.set_yticks(range(grid_size))
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # background
            bg = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd'}
            for r in range(grid_size):
                for c in range(grid_size):
                    cell = desc[r, c]
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg.get(cell, '#f8f9fa'), zorder=0)
                    ax.add_patch(rect)
                    if cell in ['H', 'G', 'S']:
                        ax.text(c, r, cell, ha='center', va='center', fontsize=16, color='black', alpha=0.5)

            ax.grid(True, color='gray', alpha=0.3, zorder=1)

            # getting and plotting trajectories
            for _ in range(num_episodes):
                state, _ = test_env.reset()
                terminated = truncated = False
                
                # conversing state in row,col
                path_x = [state % grid_size]
                path_y = [state // grid_size]
                
                for _ in range(self.episode_steps):
                    if terminated or truncated:
                        break
                    
                    # epsilon-greedy policy
                    if np.random.rand() < epsilon_test:
                        action = test_env.action_space.sample()
                    else:
                        action = int(np.argmax(self.q_table[state]))
                        
                    state, _, terminated, truncated, _ = test_env.step(action)
                    path_x.append(state % grid_size)
                    path_y.append(state // grid_size)
                    
                # adding visual jitter
                jitter_x = np.array(path_x) + np.random.uniform(-0.15, 0.15, size=len(path_x))
                jitter_y = np.array(path_y) + np.random.uniform(-0.15, 0.15, size=len(path_y))
                
                # plotting trajectory
                ax.plot(jitter_x, jitter_y, color='royalblue', alpha=0.15, linewidth=2, zorder=2)
                
                # final point
                end_color = 'green' if desc[path_y[-1], path_x[-1]] == 'G' else ('red' if desc[path_y[-1], path_x[-1]] == 'H' else 'orange')
                ax.scatter(jitter_x[-1], jitter_y[-1], color=end_color, s=20, zorder=3, alpha=0.5)

            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches='tight')
            #plt.show()
            print(f"Trajectories plot saved to {path}")
            test_env.close()

    def swarm_gif(self, env_id, max_steps, env_kwargs, num_agents=20, epsilon_test=0.1, path="swarm.gif", grid_size=10, fps=5):
        """GIF with N agents running simultaneously
           after training
           epsilon-greedy policy"""
        
        test_env = gym.make(env_id, max_episode_steps=max_steps, **env_kwargs)
        desc = test_env.unwrapped.desc.astype(str)
        
        all_paths = []
        max_steps = 0
        
        # simulating all agents
        for _ in range(num_agents):
            state, _ = test_env.reset()
            terminated = truncated = False
            path_coords = [(state % grid_size, state // grid_size)]
            
            for _ in range(self.episode_steps):
                if terminated or truncated:
                    break
                if np.random.rand() < epsilon_test:
                    action = test_env.action_space.sample()
                else:
                    action = int(np.argmax(self.q_table[state]))
                    
                state, _, terminated, truncated, _ = test_env.step(action)
                path_coords.append((state % grid_size, state // grid_size))
                
            all_paths.append(path_coords)
            if len(path_coords) > max_steps:
                max_steps = len(path_coords)

        # frames
        frames = []
        bg = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd'}
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        for step in range(max_steps):
            ax.clear()
            ax.set_title(f"Swarm | step {step} | epsilon={epsilon_test}", fontsize=14, fontweight='bold')
            ax.set_xlim(-0.5, grid_size - 0.5)
            ax.set_ylim(-0.5, grid_size - 0.5)
            ax.set_xticks(range(grid_size))
            ax.set_yticks(range(grid_size))
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, color='gray', alpha=0.3, zorder=1)

            # background
            for r in range(grid_size):
                for c in range(grid_size):
                    cell = desc[r, c]
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg.get(cell, '#f8f9fa'), zorder=0)
                    ax.add_patch(rect)
                    if cell in ['H', 'G', 'S']:
                        ax.text(c, r, cell, ha='center', va='center', fontsize=14, color='black', alpha=0.5)
            
            # getting position of each agent for the current step
            x_coords = []
            y_coords = []
            
            for p in all_paths:
                # if step > episode lenght keep last position
                if step < len(p):
                    coord_corrente = p[step]
                else:
                    coord_corrente = p[-1]
                    
                # visual jitter
                x_coords.append(coord_corrente[0] + np.random.uniform(-0.2, 0.2))
                y_coords.append(coord_corrente[1] + np.random.uniform(-0.2, 0.2))
                
            # Ora disegna gli agenti come facevi prima
            ax.scatter(x_coords, y_coords, color='purple', edgecolors='white', s=50, zorder=3, alpha=0.8)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            
        plt.close(fig)
        
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print(f"Swarm GIF saved to {path} ({len(frames)} frames)")
        test_env.close()

    def plot_training_evolution(self, path="training_evolution.png", grid_size=10):
        """
        subplots:
           1.   trajectory gradient
           2.   visits heatmap
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"Training Evolution | mode={self.mode} | episodes={self.training_episodes}", fontsize=14, fontweight='bold')
        
        desc = self.env.unwrapped.desc.astype(str)
        bg = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd'}
        
        # subplot 1
        ax1 = axes[0]
        ax1.set_title("Sampled Trajectories (50 eps)", fontsize=12)
        ax1.set_xlim(-0.5, grid_size - 0.5)
        ax1.set_ylim(-0.5, grid_size - 0.5)
        ax1.set_xticks(range(grid_size)); ax1.set_yticks(range(grid_size))
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        
        # background
        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc[r, c]
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg.get(cell, '#f8f9fa'), zorder=0)
                ax1.add_patch(rect)
                if cell in ['H', 'G', 'S']:
                    ax1.text(c, r, cell, ha='center', va='center', fontsize=12, color='black', alpha=0.5)
        ax1.grid(True, color='gray', alpha=0.3, zorder=1)

        # plotting sampled trajectories (every 50 eps)
        cmap = plt.colormaps['coolwarm']
        for eps_num, path_states in self.sampled_paths:
            # normalization between 0 and 1
            color_intensity = eps_num / self.training_episodes
            color = cmap(color_intensity)
            
            path_x = [s % grid_size for s in path_states]
            path_y = [s // grid_size for s in path_states]
            
            # visual jitter
            jitter_x = np.array(path_x) + np.random.uniform(-0.15, 0.15, size=len(path_x))
            jitter_y = np.array(path_y) + np.random.uniform(-0.15, 0.15, size=len(path_y))
            
            # changing line trasparency as the episode increases
            alpha_val = 0.2 + 0.5 * color_intensity
            ax1.plot(jitter_x, jitter_y, color=color, alpha=alpha_val, linewidth=1.5, zorder=2)
            
        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=self.training_episodes))
        cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Training Episode', rotation=270, labelpad=15)


        # subplot 2
        ax2 = axes[1]
        ax2.set_title("State Visitation Frequency (%)", fontsize=12)
        
        # computing %
        total_visits = max(1, np.sum(self.state_visits))
        visit_pct = (self.state_visits / total_visits) * 100
        grid_visits = visit_pct.reshape(grid_size, grid_size)
        
        im = ax2.imshow(grid_visits, cmap='plasma', aspect='equal')
        cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cb.set_label('% of total steps', rotation=270, labelpad=15)
        
        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc[r, c]
                val = grid_visits[r, c]
                text_color = "white" if val < np.max(grid_visits)/2 else "black"
                ax2.text(c, r, f"{val:.1f}%", ha='center', va='center', fontsize=6, color=text_color)
                
                if cell in ['H', 'G', 'S']:
                    ax2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red' if cell=='H' else 'orange', linewidth=1.5))

        ax2.set_xticks(range(grid_size)); ax2.set_yticks(range(grid_size))
        ax2.tick_params(labelsize=8)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        #plt.show()
        print(f"Training evolution plot saved to {path}")

    def plot_sampled_trajectories_gif(self, path="sampled_evolution.gif", grid_size=10, fps=2):
        """GIF showing every 50 episodes chronologically"""
        import matplotlib.pyplot as plt
        import io
        import imageio
        
        if not self.sampled_paths:
            print("Error: sampled_paths = Null")
            return

        desc = self.env.unwrapped.desc.astype(str)
        bg = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd'}
        
        frames = []
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.colormaps['coolwarm']
        
        for eps_num, path_states in self.sampled_paths:
            ax.clear()
            ax.set_title(f"Trajectory evolution | episode {eps_num}", fontsize=13, fontweight='bold')
            ax.set_xlim(-0.5, grid_size - 0.5)
            ax.set_ylim(-0.5, grid_size - 0.5)
            ax.set_xticks(range(grid_size))
            ax.set_yticks(range(grid_size))
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, color='gray', alpha=0.3, zorder=1)
            
            # background
            for r in range(grid_size):
                for c in range(grid_size):
                    cell = desc[r, c]
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg.get(cell, '#f8f9fa'), zorder=0)
                    ax.add_patch(rect)
                    if cell in ['H', 'G', 'S']:
                        ax.text(c, r, cell, ha='center', va='center', fontsize=14, color='black', alpha=0.4)
            
            # getting coords for the current episode
            path_x = [s % grid_size for s in path_states]
            path_y = [s // grid_size for s in path_states]
            
            # color changing as episode nr progresses
            color_intensity = eps_num / self.training_episodes
            color = cmap(color_intensity)
            
            # plotting single trajectory
            ax.plot(path_x, path_y, color=color, linewidth=2.5, zorder=2, alpha=0.85)
            
            # highlighting starting and finishing points
            ax.scatter(path_x[0], path_y[0], color='lime', s=50, zorder=3, edgecolors='black')
            
            end_cell = desc[path_y[-1], path_x[-1]]
            end_color = 'gold' if end_cell == 'G' else ('red' if end_cell == 'H' else 'orange')
            ax.scatter(path_x[-1], path_y[-1], color=end_color, s=80, marker='X', zorder=3, edgecolors='black')
            
            # saving frame in a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            
        plt.close(fig)
        
        # final GIF
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print(f"Trajectory chronological GIF saved to {path}")