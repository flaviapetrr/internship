import numpy as np
import heapq

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
        reward_shift_ep:    episode at which to swap the goal position (None = disabled)
        shift_env_fn:       callable () -> new_env, called at reward_shift_ep

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
            theta: float = 0.0001,
            reward_shift_ep: int = None,
            shift_env_fn=None,

    ):
        
        self.env = env
        self.initial_desc = env.unwrapped.desc.astype(str).copy()
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
        
        self.reward_shift_ep = reward_shift_ep
        self.shift_env_fn = shift_env_fn
        self.shift_happened_ep = None # actual episode when env was swapped
        self.first_success_ep = None # first episode reaching goal
        self.five_success_ep = None # fifth time reaching goal ep         
        self.first_success_after_shift_ep = None # first success on the new goal
        self.five_success_after_shift_ep = None # fifth time reaching goal ep 
 
        self.success_count = 0
        self.success_after_shift_count = 0
        # Q-value snapshots: key -> (episode_nr, q_table_copy, desc_copy)
        # Keys: "first_goal", "first_goal_p10", "5_times_goal",
        #       "at_shift", "first_new_goal", "first_new_goal_p10", "5_times_new_goal",
        #       "final"
        self.q_snapshots = {}

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
        self.replay_paths = []
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
    
    def _save_snapshot(self, key, episode):
        """Save a copy of the current q_table + env desc under key"""
        self.q_snapshots[key] = (
            episode,
            self.q_table.copy(),
            self.env.unwrapped.desc.astype(str).copy(),
        )

    def train(self):
        """train agent, update equation depends on mode"""
        print("--- TRAINING ---")

        for eps in range (self.training_episodes):
            self.epsilon = self.epsilon_exponential_decay(eps)

            if (self.reward_shift_ep is not None
                and eps == self.reward_shift_ep
                and self.shift_env_fn is not None
                and self.shift_happened_ep is None):
            # Save snapshot BEFORE switching (shows q-table at moment of change)
                self._save_snapshot("at_shift", eps)
                old_env = self.env
                self.env = self.shift_env_fn()
                old_env.close()
                self.shift_happened_ep = eps
                print(f"  [shift] goal moved at episode {eps}")

            state, step, truncated, terminated = self.reset()
            total_reward = 0
            total_td_error = 0
            episode_replay_count = 0 
            self.episode_memory = []
            self.priority_queue = []
            episode_replay_transitions = []

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
                if self.replay_mode == "f_ps" and td_error > self.theta:
                    heapq.heappush(self.priority_queue, (-td_error, (state, action)))
 
                if self.replay_mode == "f_ps":
                    trans = self.prioritized_sweeping(self.replay_steps)
                    episode_replay_transitions.extend(trans)
                    episode_replay_count += self.replay_steps
   
                state = next_state
                total_reward += reward

                if self.replay_mode == "backward":
                    trans = self.backward_replay(self.replay_steps)
                    episode_replay_transitions.extend(trans)
                    episode_replay_count += min(self.replay_steps, len(self.episode_memory))

            if self.replay_mode == "dyna":
                trans = self.dyna_replay(self.replay_steps)
                episode_replay_transitions.extend(trans)
                episode_replay_count += self.replay_steps                

            if (eps + 1) % 100 == 0:
                print("Episode: ", eps + 1)

            # for trajectory plotting 
            self.sampled_paths.append((eps + 1, current_path))

            n_steps_done = step + 1
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(n_steps_done)
            self.episode_td_errors.append(total_td_error / max(n_steps_done, 1))
            self.replay_counts.append(episode_replay_count)
            self.model_sizes.append(sum(len(v) for v in self.model.values()))

            is_success = (total_reward > 0) if self.mode in ["std", "relative"] else (total_reward < 0)
            self.episode_success.append(1 if is_success else 0)
            
            if is_success:
                self.success_count += 1
                if self.shift_happened_ep is not None and eps > self.shift_happened_ep:
                    self.success_after_shift_count += 1

            # shapshots
            # first ever goal
            if self.first_success_ep is None and is_success:
                self.first_success_ep = eps
                self._save_snapshot("first_goal", eps)
 
            # +10 episodes after first goal
            if (self.first_success_ep is not None
                    and eps == self.first_success_ep + 10):
                self._save_snapshot("first_goal_p10", eps)
 
            # fifth time reaching  goal
            if self.five_success_ep is None and self.success_count == 5:
                self.five_success_ep = eps
                self._save_snapshot("5_times_goal", eps)

            # First success on the new goal (after shift)
            if (self.shift_happened_ep is not None
                    and self.first_success_after_shift_ep is None
                    and eps > self.shift_happened_ep
                    and is_success):
                self.first_success_after_shift_ep = eps
                self._save_snapshot("first_new_goal", eps)
 
            # +10 episodes after first new goal
            if (self.first_success_after_shift_ep is not None
                    and eps == self.first_success_after_shift_ep + 10):
                self._save_snapshot("first_new_goal_p10", eps)
 
            # fifth time reaching new goal
            if (self.five_success_after_shift_ep is None
                    and self.success_after_shift_count == 5):
                self.five_success_after_shift_ep = eps
                self._save_snapshot("5_times_new_goal", eps)

        # final snapshot
        self._save_snapshot("final", self.training_episodes - 1)

        print("--- COMPLETED ---\n")

    def backward_replay(self, n_steps=10):
        """
        MF-RL
        associated with backward and unordered replays
        -> rapid learning
        """
        replay_batch = self.episode_memory[-n_steps:][::-1]
        transitions = []

        for state, action, reward, next_state in replay_batch:
            self.q_table_update(state, action, reward, next_state)
            transitions.append((state, next_state))

        return transitions


    def prioritized_sweeping(self, n_steps=10):
        """
        MB-RL: prioritized sweeping
        associated with forward and imaginary replays
        -> planning
        """
        steps = 0
        transitions = []

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
            
            transitions.append((state, next_state))

            steps += 1
        return transitions

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
        transitions = []

        for _ in range(n_steps):
            s = random.choice(states_with_model)
            a = random.choice(list(self.model[s].keys()))
            reward, next_state = self.model[s][a]
            self.q_table_update(s, a, reward, next_state)
            transitions.append((s, next_state))
        
        return transitions
