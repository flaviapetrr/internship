
import gymnasium as gym
import agent
 
env_id = "FrozenLake-v1"
 
if __name__ == "__main__":
 
    env = gym.make(env_id,
                   map_name="4x4",
                   is_slippery=False,
                   render_mode="rgb_array")
 
    agent_std = agent.QLearningAgent(env,
                                     training_episodes=2500,
                                     episode_steps=100,
                                     alpha=0.8,            
                                     gamma=0.99,
                                     epsilon_start=1.0,
                                     epsilon_end=0.01,
                                     epsilon_decay=0.001,
                                     epsilon=1.0)

 
    agent_std.train(mode="std")
    agent_std.plot_training(grid_size=4)
