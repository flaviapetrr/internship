# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
import numpy as np
import gymnasium as gym
import random
import imageio
from pyvirtualdisplay import Display
from IPython.display import Image
import matplotlib.pyplot as plt

# defining hyperparameters
# training parameters
#  testing parameters
planning_comparison = [0, 5, 10, 25, 50]
episodes_comparison = [250, 500, 750] 
# dictionary to spore comparison results
results = {}
eval_results = {}
alpha = 0.1 # step_size parameter / learning rate   

# evaluation parameters
n_eval_episodes = 100 

# environment parameters
env_id = "CliffWalking-v1"
max_steps = 199             
gamma = 0.99 # discount rate               
eval_seed = []             

# exploration parameters
max_epsilon = 1.0           
min_epsilon = 0.01
decay_rate = 0.001

# to not visualize the training
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

env = gym.make(env_id, is_slippery=False, render_mode="rgb_array")

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

def update_Q_table(state, action, next_state, reward, Q_table, terminated):

    if terminated:
        target = reward
    else:
        target = reward + gamma * np.max(Q_table[next_state])
    # one-step Q-learning
    Q_table[state, action] += alpha * (target - Q_table[state, action])
    return Q_table


# defining epsilon-greedy policy
def epsilon_greedy_policy (Q_table, epsilon, state, env):

  if np.random.rand() < epsilon: # could also use random.uniform depends of what i wanna do
      action =  env.action_space.sample()
  else:
      action = np.argmax(Q_table[state])
  
  return action

# defining greedy policy
def greedy_policy (Q_table, state):

  return np.argmax(Q_table[state])

     

# updating environment model -> if i'm in state s and do action a what will happen?
def update_model(state, action, next_state, reward, model):

  # if state = Null, creating a dictionary for its actions
  if state not in model:
        model[state] = {}
    
  # saving action outcome
  model[state][action] = (reward, next_state)

  return model

# defining planning phase
def planning_phase(model, Q_table, n_steps, terminated):

  # loop for #steps defined for planning
  for step in range (n_steps):
    # randomly picking a state in our model
    s_sim = random.choice(list(model.keys()))
    # randomly performing and action associated to model state
    a_sim = random.choice(list(model[s_sim].keys()))
    # taking next state and reward from simulated action
    r_sim, next_s_sim = model[s_sim][a_sim]
    # updating Q_tablle based on model planning
    Q_table = update_Q_table(s_sim, a_sim, next_s_sim, r_sim, Q_table, terminated)

  return Q_table

# defining training loop
def training (n_training_episodes, n_planning_steps, min_epsilon, max_epsilon, decay_rate, env, max_steps, Q_table, model):
  reward_history = []

  print("\n--- STARTING TRAINING ---")
  for episode in range (n_training_episodes):
      # exponential decay of epsilon
      # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp (- decay_rate * episode)

      # resetting environment
      state, info = env.reset()
      truncated = False
      terminated = False
      total_reward = 0

      for step in range (max_steps):
        # choosing action
        action = greedy_policy (Q_table, state)
        # action = epsilon_greedy_policy(Q_table, epsilon, state, env)
        # taking action in env and getting reward and next state
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # updating Q_function -> using update equation
        Q_table = update_Q_table(state, action, next_state, reward, Q_table, terminated)
        # updating model based on next_state
        model = update_model (state, action, next_state, reward, model)
        # planning
        if len(model) > 0:
          # executing planning if we have at least one memory
          Q_table = planning_phase(model, Q_table, n_planning_steps, terminated)

        if terminated or truncated:
            break

        #changing current state into the new state
        state = next_state

      reward_history.append(total_reward)

      # printing every 100 episodes
      if (episode + 1) % 100 == 0:
          # print(f"Episode: {episode + 1} | Epsilon: {epsilon:.4f} | Max Q: {np.max(Q_table):.4f}")
          print(f"Episode: {episode + 1} | Max Q: {np.max(Q_table):.4f}")
  
  print("--- COMPLETED ---\n")

  return Q_table, reward_history

# recording visuals
def record_video(env, Qtable, out_directory, fps=1, max_frames = 100):

  print("\n-> Recording...")
  images = []
  frame_count = 0
  truncated = False
  terminated = False 
  state, info = env.reset(seed=random.randint(0,500))
  img = env.render()
  images.append(img)

  while not (terminated or truncated) and frame_count < max_frames:
      action = np.argmax(Qtable[state][:])
      state, reward, terminated, truncated, info = env.step(action)
      img = env.render()
      images.append(img)
  imageio.mimsave(out_directory, [np.array(img) for img in images], fps=fps)
  print("GIF saved")

# plotting
def plot_comparison(results, episodes_list, planning_list, window=20):

  print("\n-> Plotting...")
  fig, axes = plt.subplots(1, len(episodes_list), figsize=(20, 6), sharey=True)
  
  # if tested just with one # of eps, axes must be transformed in a list
  if len(episodes_list) == 1:
      axes = [axes]

  for i, eps in enumerate(episodes_list):
      ax = axes[i]
      print("# eps plotting: ", eps)
      for p_steps in planning_list:
          # recovering past
          history = results.get((eps, p_steps))
          
          if history is not None:
              # using media mobile to avg rewards
              if len(history) > window:
                  smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
                  ax.plot(smoothed, label=f'Planning: {p_steps}')
              else:
                  ax.plot(history, label=f'Planning: {p_steps}')
      
      ax.set_title(f"Testing over {eps} eps")
      ax.set_xlabel("eps")
      if i == 0:
          ax.set_ylabel("mean avg reward")

      ax.axhline(y=-13, color='red', linestyle='--', alpha=0.6, label='Optimal target')
      
      # ax.set_ylim([-150, 5]) 
      ax.legend(loc='lower right')
      ax.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('comparison_cliff.png', dpi=300, bbox_inches='tight')
  # plt.savefig('comparison_cliff.pdf', bbox_inches='tight') # in vettoriale
  print("PNG saved")
  # print("PDF saved")
  plt.close(fig)

for eps in episodes_comparison:
  for p_steps in planning_comparison:
    print(f"\n-> Running experiment with n_planning_steps = {p_steps} and episodes = {eps}")
    # for each experiment

    # initializing Q table
    Q_table_cliff = init_Q_table(state_space, action_space)
    # initializing model (dictionary)
    model_cliff = {}
    
    # training
    Q_table_cliff, history = training(eps, p_steps, min_epsilon, max_epsilon, decay_rate, env, max_steps, Q_table_cliff, model_cliff)

    results[(eps, p_steps)] = history

video_path="replay_cliff.gif"
video_fps=1
record_video(env, Q_table_cliff, video_path, video_fps)
Image(filename='./replay_cliff.gif')

plot_comparison(results, episodes_comparison, planning_comparison)