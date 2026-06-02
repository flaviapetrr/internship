import numpy as np
from agent import QLearningAgent
from replay import BackwardReplay, DynaReplay, PrioritizedSweepingReplay

# how many episode snapshots to save for the replay-trajectory plot
N_PLOT_SNAPSHOTS = 8

class Trainer(QLearningAgent):

    def __init__(self, *args, env_factory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory

    def training(self):
        """
        Train agent
        update equation depends on mode
        """
        print("="*70)
        print(f"TRAINING  |  ep: {self.training_eps}  |  update: {self.update_mode}  |  replay: {self.replay_mode}")
        print("="*70)

        # storing the initial grid description for the plot
        self.initial_desc = self.env.unwrapped.desc.astype(str)
 
        # uncomment if equally distributed plots are wanted for visual plot
        # selecting equally distributed episodes for plot
        #snapshot_ep = set(
        #    int(round(i * (self.training_eps - 1) / (N_PLOT_SNAPSHOTS - 1)))
        #    for i in range(N_PLOT_SNAPSHOTS)
        #)

        # initializing buffer
        if self.replay_mode == "backward":
            self.buffer = BackwardReplay()
        elif self.replay_mode == "dyna":
            self.buffer = DynaReplay()
        elif self.replay_mode == "prioritized_sweeping":
            self.buffer = PrioritizedSweepingReplay(theta=self.theta)

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
                    self.env = self.env_factory(row, col)
                else:
                    raise RuntimeError("agent if configured for goal shift but env_factory does not exist")
                
                self.shift_happened_ep = ep
                print(f"[shift]     goal moved at episode {ep}")

            # reset
            current_state, _ = self.env.reset()
            step = 0
            truncated = terminated = False
            total_reward = 0
            # computing epsilon for the current episode
            self.epsilon = self.epsilon_decay()

            if self.replay_mode == "backward":
                self.buffer.clear()
            
            agent_path = [current_state]
            episode_replay_batches = [] # list of [(s, ns)] per replay event
            
            for step in range(self.max_episode_steps):
                if terminated or truncated:
                    break
                # selecting action
                action = self.action_selection(self.epsilon, current_state)
                # performing selected action on the env
                next_state, reward, terminated, truncated, _ = self.env.step(action)

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

                        batch.append((d_state, d_next_state))

                    episode_replay_batches.append(batch)
                    if reward is not 0: print(self.model)
                elif self.replay_mode == "backward":
                
                    self.buffer.store_step(current_state, action, next_state, reward, terminated)

                agent_path.append(next_state)
                current_state = next_state
                total_reward += reward

            # end of episode replay
            if self.replay_mode == "backward":
                backward_trajectory = self.buffer.backward_traj(self.backward_steps)

                batch = []

                for b_state, b_action, b_next_state, b_reward, b_terminated in backward_trajectory:
                    self.q_table_update(b_state, b_action, b_next_state, b_reward, b_terminated)
                    batch.append((b_state, b_next_state))
                if batch:
                    episode_replay_batches.append(batch)

            if (ep + 1) % 100 == 0:
                    print(f"Episode: {ep + 1} - epsilon: {self.epsilon}")

            reach_goal = (total_reward > 0) if self.mode in ["std", "relative"] else (total_reward < 0)
            self.ep_reach_goal.append(1 if reach_goal else 0)

            if reach_goal:
                self.goal_count += 1
                if self.shift_happened_ep is not None and ep > self.shift_happened_ep:
                    self.new_goal_count += 1

            #checking snapshots
            self._check_and_save_snapshots(ep, reach_goal, episode_replay_batches, agent_path)

            # uncomment if equally distributed plots are wanted for visual plot
            #if ep in snapshot_ep:
            #    self.replay_paths.append((ep, episode_replay_batches, self.q_table.copy()))
            #    self.sampled_paths.append((ep, agent_path))

        # final snapshot
        self._save_snapshot("final", self.training_eps - 1, episode_replay_batches, agent_path)

        print("="*70)
        print(" "*25, "TRAINING COMPLETE")
        print("="*70)

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