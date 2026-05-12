# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
import numpy as np
import gymnasium as gym
import random
import imageio
from pyvirtualdisplay import Display
from IPython.display import Image

# defining hyperparameters
# training parameters
n_training_episodes = 10000
alpha = 0.7 # step_size parameter / learning rate   

# evaluation parameters
n_eval_episodes = 100      

# environment parameters
env_id = "FrozenLake-v1"   
max_steps = 99             
gamma = 0.95 # discount rate               
eval_seed = []             

# exploration parameters
max_epsilon = 1.0           
min_epsilon = 0.05           
decay_rate = 0.0005    

# to not visualize the training
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

env = gym.make(env_id, map_name="4x4",is_slippery=False, render_mode="rgb_array")

# prints to get to know the environment / what we are working with
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample())

print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")

# initializing Q-table
def init_Q_table (state_space, action_space):

    Q_table = np.zeros ((state_space, action_space))

    return Q_table

# REMINDER: in Q-learning acting policy != updating policy

# defining epsilon-greedy policy -> acting policy
def epsilon_greedy_policy (Q_table, epsilon, state):

    if np.random.rand() < epsilon: # could also use random.uniform depends of what i wanna do
        action =  env.action_space.sample()
    else:
        action = np.argmax(Q_table[state])
    
    return action

# defining greedy policy -> updating policy
def greedy_policy (Q_table, state):

    action = np.argmax(Q_table[state])

    return action

# defining training loop
def training (n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Q_table):

    print("\n--- STARTING TRAINING ---")
    for episode in range (n_training_episodes):
        # exponential decay of epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

        # resetting environment
        state, info = env.reset()
        step = 0
        truncated = False
        terminated = False

        for step in range (max_steps):
            # choosing action with epsilon-greedy
            action = epsilon_greedy_policy (Q_table, epsilon, state)
            # taking action in env and getting reward and next state
            new_state, reward, terminated, truncated, info = env.step(action)
            # updating Q_function -> using update equation
            Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state, action])

            if terminated or truncated:
                break

            #changing current state into the new state
            state = new_state
        
        # printing every &k episodes
        if (episode + 1) % 1000 == 0:
            print(f"Episode: {episode + 1} | Epsilon: {epsilon:.4f} | Max Q: {np.max(Q_table):.4f}")
    
    print("--- COMPLETED ---\n")
    return Q_table

# evaluating agent
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):

  print(f"\n-> Evaluating... \n(considering {n_eval_episodes} episodes)")
  episode_rewards = []
  for episode in range(n_eval_episodes):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    step = 0
    truncated = False
    terminated = False
    total_rewards_ep = 0
   
    for step in range(max_steps):
      action = np.argmax(Q[state][:])
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward
       
      if terminated or truncated:
        break
      
      state = new_state

    episode_rewards.append(total_rewards_ep)

  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

# recording visuals
def record_video(env, Qtable, out_directory, fps=1):

    print("\n-> Recording...")
    images = [] 
    truncated = False
    terminated = False 
    state, info = env.reset(seed=random.randint(0,500))
    img = env.render()
    images.append(img)

    while not (terminated or truncated):
        action = np.argmax(Qtable[state][:])
        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for img in images], fps=fps)
    print("GIF saved")

Q_table_lake = init_Q_table(state_space, action_space)
Q_table_lake = training(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Q_table_lake)

print(" Final Q-Table:\n", Q_table_lake)

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Q_table_lake, eval_seed)
print(f"Results: Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

video_path="replay_lake.gif"
video_fps=1
record_video(env, Q_table_lake, video_path, video_fps)

Image(filename='./replay_lake.gif')