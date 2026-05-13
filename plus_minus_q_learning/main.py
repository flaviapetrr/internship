
# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import gymnasium as gym
# from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import agent

if __name__ == "__main__":

    mode = "relative" # "std", "std_punish", "opposite", "relative", "relative_punish"

    if mode in ["std", "relative"]:
        reward_schedule = (1, 0, 0) #goal, hole, step/frozen
    else:
        reward_schedule = (-1, 0, 0)

    env_id = 'FrozenLake-v1'
    env_kwargs = {"map_name": "4x4",
                  "is_slippery": False, # if True "success_rate": 1.0/3.0,
                  "reward_schedule": reward_schedule
                 }
    env = gym.make(env_id,
                   **env_kwargs
                )
    
    agent = agent.QLearningAgent(env,
                                 mode = mode, 
                                 training_episodes=2500,
                                 episode_steps=100,
                                 alpha=0.8,
                                 alpha_v = 0.8,           
                                 gamma=0.99,
                                 epsilon_start=1.0,
                                 epsilon_end=0.01,
                                 epsilon_decay=0.001,
                                 epsilon=1.0
                                )

 
    agent.train()
    
    agent.plot_training(path=f"./plus_minus_q_learning/training_summary_{agent.mode}.png", grid_size=4)
    agent.record_gif(env_id, env_kwargs, path=f"./plus_minus_q_learning/agent_episode_{agent.mode}.gif", fps=3)

env = gym.make(env_id, render_mode="rgb_array")