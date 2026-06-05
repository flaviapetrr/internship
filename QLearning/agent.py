# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import gymnasium as gym
import numpy as np

UPDATE_MODES = ["std", "std_punish", "opposite", "relative", "relative_punish"]    
REPLAY_MODES = ["none", "prioritized_sweeping", "value_iteration", "backward", "dyna"]

class QLearningAgent():

    def __init__(
            self,
            env,
            update_mode: str            = "std",
            replay_mode: str            = "none",
            action_selection: str       = "epsilon_greedy",
            training_eps: int           = 400,
            max_episode_steps: int      = 100,
            epsilon_start: float        = 1.0,
            epsilon_min: float          = 0.01,
            decay_rate: float           = 0.99,
            tau: float                  = 5,
            q_init: float               = 0.0,
            gamma: float                = 0.99,
            alpha: float                = 0.1,
            alpha_v: float              = 0.1,
            backward_steps: int         = 10,
            dyna_steps: int             = 10,
            ps_steps: int               = 15,
            vi_steps: int               = 15,
            theta: float                = 0.0001,
            shift_goal_ep: int          = None,
            shift_goal_pos: list        = (2, 8),
            add_obs_ep: int             = None
    ):  
        if update_mode not in UPDATE_MODES:
            raise ValueError(f"Error: '{update_mode}' not valid.\nValid options: {UPDATE_MODES}")
        
        if replay_mode not in REPLAY_MODES:
            raise ValueError(f"Error: '{replay_mode}' not valid.\nValid options: {REPLAY_MODES}")
        
        self.env                    = env
        self.update_mode            = update_mode
        self.mode                   = update_mode # alias for plot_utils
        self.replay_mode            = replay_mode
        self.action_selection       = action_selection
        self.training_eps           = training_eps
        self.training_episodes      = training_eps # alias for plot_utils
        self.max_episode_steps      = max_episode_steps
        self.epsilon                = epsilon_start
        self.epsilon_start          = epsilon_start
        self.epsilon_min            = epsilon_min
        self.decay_rate             = decay_rate
        self.tau                    = tau
        self.q_init                 = q_init
        self.gamma                  = gamma
        self.alpha                  = alpha
        self.alpha_v                = alpha_v
        self.backward_steps         = backward_steps
        self.dyna_steps             = dyna_steps
        self.ps_steps               = ps_steps
        self.vi_steps               = vi_steps
        self.theta                  = theta
        self.shift_goal_ep          = shift_goal_ep
        self.shift_goal_pos         = shift_goal_pos
        self.add_obs_ep             = add_obs_ep
        self.state_space            = env.observation_space.n
        self.action_space           = env.action_space.n
        self.q_table                = np.full((self.state_space, self.action_space), self.q_init)
        self.v_table                = np.zeros(self.state_space)

        self.shift_happened_ep      = None # actual episode in which goal shifted in env
        self.obs_added_ep           = None # actual episode in which the obstacles are added
        self.first_goal_ep          = None # first time reaching goal ep
        self.five_goal_ep           = None # fifth time reaching goal ep         
        self.first_new_goal_ep      = None # first time reaching new goal ep -> after shift
        self.five_new_goal_ep       = None # fifth time reaching new goal ep -> after shift
        self.q_snapshots            = {} # snapshots dictionary
        self.eq_snapshots           = []
        self.ep_reach_goal          = [] # list to track reaching goals
        self.goal_count             = 0 # goal reaching counter
        self.new_goal_count         = 0 # new goal reaching counter -> after shift
  
        self.initial_desc           = None # set in Trainer.training()
        self.replay_paths           = [] # [(ep, replay_batches, q_table_snap), ...]
        self.sampled_paths          = [] # [(ep, agent_path), ...]

        self.episode_rewards        = [] # to store accumulated reward
        self.episode_times          = [] # to store accumulated decision making time

        # variables to track exploration
        self.ep_physical_normal     = []
        self.ep_physical_terminal   = []
        self.ep_replay_normal       = []
        self.ep_replay_terminal     = []

    def q_table_update(self, state, action, next_state, reward, terminated):
        """
        Update equation, depends on mode:
            1. std && std_punish:   classical one-step bellman equation
            2. opposite:            punishment based one-step bellman equation
            3. relative:            contextual update equation
            4. relative_punish:     contextual update equation for punishment values
        """
        if terminated:
            optimal_next = 0.0
        elif self.update_mode in ["opposite", "relative_punish"]:
            optimal_next = np.min(self.q_table[next_state])
        else:
            optimal_next = np.max(self.q_table[next_state])

        if self.update_mode in ["std", "std_punish", "opposite"]:
            self.q_table[state][action] += self.alpha * (
            reward + self.gamma * optimal_next - self.q_table[state][action])
        # "relative", "relative_punish"
        else:
            # r_v = (r_chosen + sum(stored value of all unchosen states)) / nr.space
            # first getting possible actions
            sum_unchosen_q = sum(self.q_table[state][a] for a in range(self.action_space) if a != action)
            r_v = (reward + sum_unchosen_q) / (self.action_space)
                    
            # update V first
            # V(s) = V(s) + alpha * (r_v - V(s))
            self.v_table[state] += self.alpha_v * (r_v - self.v_table[state])
            
            self.q_table[state][action] += self.alpha * (
                reward - self.v_table[state] + self.gamma * optimal_next - self.q_table[state][action]
            )

    def epsilon_greedy(self, epsilon, state):
        """epsilon-greedy action selection policy"""
        if np.random.random() > epsilon:
            q_vals = self.q_table[state]
            return np.random.choice(np.flatnonzero(q_vals == q_vals.max()))
        else:
            return np.random.randint(len(self.q_table[state]))
        
    def epsilon_decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
        return self.epsilon
        
    def softmax(self, tau, state):
        """softmax action selection policy with Gibbs distribution"""
        q_vals = self.q_table[state]

        # taking max q-va lto then subtracting
        # trick to ensure numeric stability, avoiding huge nrs
        max_q = np.max(q_vals)

        # comuting softmax
        exp_q = np.exp((q_vals - max_q) / tau)
        prob = exp_q / np.sum(exp_q)

        # returns selected action

        return np.random.choice(len(q_vals), p=prob)
    
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, update_mode):
        super().__init__(env)
        self.update_mode = update_mode

    def reward(self, reward):
    
        if self.update_mode in ["std_punish", "opposite", "relative_punish"]:
            if reward == 1.0:
                return -1.0
            
        return reward