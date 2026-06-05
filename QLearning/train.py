import time
import numpy as np
from agent import QLearningAgent
from replay import BackwardReplay, DynaReplay, PrioritizedSweepingReplay, ValueIterationReplay

class Trainer(QLearningAgent):

    def __init__(self, *args, env_factory=None, n_samples=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory
        self.n_samples = n_samples
        self.count=np.zeros(self.state_space)

    def training(self):
        """
        Train agent
        update equation depends on mode
        """
        print("="*100)
        print(" "*40, "TRAINING\n"
              f"ep: {self.training_eps}  |  update: {self.update_mode}  |  replay: {self.replay_mode}  |    action: {self.action_selection}")
        print("="*100)

        # storing the initial grid description for the plot
        self.initial_desc = self.env.unwrapped.desc.astype(str)
 
        # selecting equally distributed episodes for plot
        snapshot_ep = set(
            int(round(i * (self.training_eps - 1) / (self.n_samples - 1)))
            for i in range(self.n_samples)
        )

        # initializing buffer
        if self.replay_mode == "backward":
            self.buffer = BackwardReplay()
        elif self.replay_mode == "dyna":
            self.buffer = DynaReplay()
        elif self.replay_mode == "prioritized_sweeping":
            self.buffer = PrioritizedSweepingReplay(theta=self.theta)
        elif self.replay_mode == "value_iteration":
            self.buffer = ValueIterationReplay()

        # checking if obstacles are active
        obstacles_active = (self.add_obs_ep == 0)

        for ep in range(self.training_eps):
            # checking if need to shift goal
            if (self.shift_goal_ep is not None
                and ep == self.shift_goal_ep
                and self.shift_happened_ep is None):
                # saving snapshot
                self._save_snapshot("at_shift", ep, episode_replay_batches, agent_path)
                old_env = self.env
                old_env.close()

                # generating and assigning new env
                if self.env_factory is not None:
                    row, col = self.shift_goal_pos
                    self.env = self.env_factory(row, col, obstacles_active)
                else:
                    raise RuntimeError("agent if configured for goal shift but env_factory does not exist")
                
                self.shift_happened_ep = ep
                print(f"\n[env]     goal moved at episode {ep}\n")

            # checking if need to add obstacles
            if (self.add_obs_ep is not None
                and self.add_obs_ep > 0
                and ep == self.add_obs_ep 
                and self.obs_added_ep is None):
                
                self._save_snapshot("at_obs_add", ep, episode_replay_batches, agent_path)
                old_env = self.env
                old_env.close()
                # if wanting to reinitialize epsilon when obs spawn
                # self.epsilon = self.epsilon_start

                obstacles_active = True
                
                if self.env_factory is not None:
                    # keeping goal position if not shifted
                    row, col = self.shift_goal_pos if self.shift_happened_ep is not None else (9, 9)
                    self.env = self.env_factory(row, col, obstacles_active)
                
                self.obs_added_ep = ep
                print(f"\n[env]     obstacles added at episode {ep}\n")
            # reset
            current_state, _ = self.env.reset()
            step = 0
            truncated = terminated = False
            total_reward = 0
            ep_decision_time = 0.0
            ep_phys_norm = 0
            ep_phys_term = 0
            ep_rep_norm = 0
            ep_rep_term = 0

            # computing epsilon / tau for the current episode
            if self.action_selection == "epsilon_greedy":
                self.epsilon = self.epsilon_decay()
                action_select_param = (f"epsilon: {self.epsilon}")
            elif self.action_selection == "softmax":
                action_select_param = (f"tau: {self.tau}")

            if self.replay_mode == "backward":
                self.buffer.clear()
            
            agent_path = [current_state]
            episode_replay_batches = [] # list of [(s, ns)] per replay event
            
            if ep % 100 == 0:
                self.count = np.zeros(self.state_space)
            for step in range(self.max_episode_steps):
                if terminated or truncated:
                    break
                
                # sarting counter
                step_start_time = time.perf_counter()

                # selecting action
                if self.action_selection == "epsilon_greedy":
                    action = self.epsilon_greedy(self.epsilon, current_state)
                elif self.action_selection == "softmax":
                    action = self.softmax(self.tau, current_state)

                # performing selected action on the env
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if terminated:
                    ep_phys_term += 1
                else:
                    ep_phys_norm += 1
                    
                #computing TD error aka surprise for prioritized sweeping
                if self.update_mode in ["opposite", "relative_punish"]:
                    optimal_next = np.min(self.q_table[next_state])
                else:
                    optimal_next = np.max(self.q_table[next_state])

                if terminated:
                    optimal_next = 0.0

                target = reward + self.gamma * optimal_next
                td_error = abs(target - self.q_table[current_state][action])

                # updating q_table
                self.q_table_update(current_state, action, next_state, reward, terminated)
                
                self.count[next_state] += 1 
                # per-step replay
                if self.replay_mode == "prioritized_sweeping":
                        # updating model and predecessors
                        self.buffer.store_step(current_state, action, next_state, reward, terminated)
                        # pushing them according to priority
                        self.buffer.push(td_error, current_state, action)
                        
                        batch = []

                        for _ in range(self.ps_steps):
                            if self.buffer.is_empty():
                                break
                                
                            # popping highest priority state
                            p_state, p_action = self.buffer.pop()
                            
                            # getting what happened from the model
                            p_next_state, p_reward, p_terminated = self.buffer.model[(p_state, p_action)]
                            
                            # updating q_table
                            self.q_table_update(p_state, p_action, p_next_state, p_reward, p_terminated)
                            
                            if p_terminated: ep_rep_term += 1
                            else: ep_rep_norm += 1

                            batch.append((p_state, p_next_state))
 

                            # backpropagating to the predecessors
                            for pred_state, pred_action, pred_reward, pred_terminated in self.buffer.predecessors[p_state]:
                                # computing TD error
                                if pred_terminated:
                                    pred_opt_next = 0.0
                                elif self.update_mode in ["opposite", "relative_punish"]:
                                    pred_opt_next = np.min(self.q_table[p_state]) # p_state is the next_state of the predecessor
                                else:
                                    pred_opt_next = np.max(self.q_table[p_state])
                                    
                                pred_target = pred_reward + self.gamma * pred_opt_next
                                pred_error = abs(pred_target - self.q_table[pred_state][pred_action])
                                
                                # pushing in line if the error is high enough
                                self.buffer.push(pred_error, pred_state, pred_action)
                            
                            if batch:
                                episode_replay_batches.append(batch)

                elif self.replay_mode == "dyna":
                
                    self.buffer.store_step(current_state, action, next_state, reward, terminated)
                    
                    batch = []
                    for _ in range(self.dyna_steps):
                        # randomly selects a memory
                        d_state, d_action, d_next_state, d_reward, d_terminated = self.buffer.random_sample()
                        
                        # updating q_table using mental replay
                        self.q_table_update(d_state, d_action, d_next_state, d_reward, d_terminated)

                        if d_terminated: ep_rep_term += 1
                        else: ep_rep_norm += 1

                        batch.append((d_state, d_next_state))

                    episode_replay_batches.append(batch)
                    
                elif self.replay_mode in ["backward", "value_iteration"]:
                
                    self.buffer.store_step(current_state, action, next_state, reward, terminated)

                # adding step time to accumulator
                step_end_time = time.perf_counter()
                ep_decision_time += (step_end_time - step_start_time)

                agent_path.append(next_state)
                current_state = next_state
                total_reward += reward

            end_replay_start = time.perf_counter()

            # end of episode replay
            if self.replay_mode == "backward":
                backward_trajectory = self.buffer.backward_traj(self.backward_steps)

                batch = []

                for b_state, b_action, b_next_state, b_reward, b_terminated in backward_trajectory:
                    self.q_table_update(b_state, b_action, b_next_state, b_reward, b_terminated)
                    if b_terminated: ep_rep_term += 1
                    else: ep_rep_norm += 1
                    batch.append((b_state, b_next_state))
                if batch:
                    episode_replay_batches.append(batch)

            elif self.replay_mode == "value_iteration":
                delta = float('inf')
                sweeps = 0
                max_sweeps = 100 # security limit if no convergence is reached


                while delta > self.theta and sweeps < max_sweeps:
                    delta = 0.0
                    batch = []

                # taking n_samples random transitions from the model
                transitions = self.buffer.sample_n(self.vi_steps)
                    
                for v_state, v_action, v_next_state, v_reward, v_terminated in transitions:
                    
                    # saving old q-val before update
                    old_q = self.q_table[v_state][v_action]
                    
                    # updating q_table
                    self.q_table_update(v_state, v_action, v_next_state, v_reward, v_terminated)
                    
                    if v_terminated: ep_rep_term += 1
                    else: ep_rep_norm += 1

                    # computing q-vals abs difference
                    new_q = self.q_table[v_state][v_action]
                    delta = max(delta, abs(old_q - new_q))
                    
                    batch.append((v_state, v_next_state))

                if batch:
                    episode_replay_batches.append(batch)
                    
                sweeps += 1
                
            # adding end of episode replay time to the counter
            ep_decision_time += (time.perf_counter() - end_replay_start)

            if (ep + 1) % 100 == 0:
                    print(f"Episode: {ep + 1} - {action_select_param}")
                    #print(self.count[:10])
                    #print(self.count[11:20])
                    #print(self.count[21:30])
                    #print(self.count[31:40])
                    #print(self.count[41:50])
                    #print(self.count[51:60])
                    #print(self.count[61:70])
                    #print(self.count[71:80])
                    #print(self.count[81:90])
                    #print(self.count[91:100])


            reach_goal = (total_reward > 0) if self.mode in ["std", "relative"] else (total_reward < 0)
            self.ep_reach_goal.append(1 if reach_goal else 0)

            # saving data
            self.episode_times.append(ep_decision_time)
            self.episode_rewards.append(total_reward)

            self.ep_physical_normal.append(ep_phys_norm)
            self.ep_physical_terminal.append(ep_phys_term)
            self.ep_replay_normal.append(ep_rep_norm)
            self.ep_replay_terminal.append(ep_rep_term)

            if reach_goal:
                self.goal_count += 1
                if self.shift_happened_ep is not None and ep > self.shift_happened_ep:
                    self.new_goal_count += 1

            #checking snapshots
            self._check_and_save_snapshots(ep, reach_goal, episode_replay_batches, agent_path)

            # storing equally distributed replay trajs and relative q-val heatmap
            if ep in snapshot_ep:
                self.eq_snapshots.append((
                    ep,
                    self.q_table.copy(),
                    self.env.unwrapped.desc.astype(str).copy(),
                    list(episode_replay_batches),
                    list(agent_path)
                ))

        # final snapshot
        self._save_snapshot("final", self.training_eps - 1, episode_replay_batches, agent_path)

        print("="*100)
        print(" "*40, "TRAINING COMPLETE")
        print("="*100)

    def _check_and_save_snapshots(self, ep, reach_goal, replay_batches, agent_path):
        """Check conditions and save snapshot if met"""
        
        # first time reaching goal
        if self.first_goal_ep is None and reach_goal:
            self.first_goal_ep = ep
            self._save_snapshot("first_goal", ep, replay_batches, agent_path)
        
        # +10 episodes after first goal
        if self.first_goal_ep is not None and ep == self.first_goal_ep + 10:
            self._save_snapshot("first_goal_p10", ep, replay_batches, agent_path)

        # fifth time reaching goal (Corretto typo: success -> goal)
        if self.five_goal_ep is None and self.goal_count == 5:
            self.five_goal_ep = ep
            self._save_snapshot("5_times_goal", ep, replay_batches, agent_path)

        # first time reaching new goal (after shift)
        if (self.shift_happened_ep is not None
                and self.first_new_goal_ep is None
                and ep > self.shift_happened_ep
                and reach_goal):
            self.first_new_goal_ep = ep
            self._save_snapshot("first_new_goal", ep, replay_batches, agent_path)

        # +10 episodes after first new goal
        if self.first_new_goal_ep is not None and ep == self.first_new_goal_ep + 10:
            self._save_snapshot("first_new_goal_p10", ep, replay_batches, agent_path)

        # fifth time reaching new goal
        if self.five_new_goal_ep is None and self.new_goal_count == 5:
            self.five_new_goal_ep = ep
            self._save_snapshot("5_times_new_goal", ep, replay_batches, agent_path)

    def _save_snapshot(self, key, episode, replay_batches=[], agent_path=[]):
        """Save a copy of the current q_table + env desc under key"""
        self.q_snapshots[key] = (
            episode,
            self.q_table.copy(),
            self.env.unwrapped.desc.astype(str).copy(),
            list(replay_batches),
            list(agent_path)
        )