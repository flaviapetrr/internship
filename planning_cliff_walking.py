# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
import numpy as np
import gymnasium as gym
import random

# parameters
n_training_eps = 500
steps_eps = 100
epsilon_start  = 1
epsilon_end = 0.05
env_id = "CliffWalking-v1"
env = gym.make(env_id, is_slippery=False, render_mode="rgb_array")

def e_greedy(env, state, e, q_table):
    if np.random.rand() > e:
        action = np.argmax(q_table[state])
    else:
        action = env.action_space.sample()

    return action

def greedy(state, q_table):
    return np.argmax(q_table[state])

def train(env, n_training_eps, steps_eps, epsilon_end, epsilon_start):

    for eps in range (n_training_eps):
        # exponential epsilon decay
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-  / epsilon_decay)
    
        state, info = env.reset()

        for step in range (steps_eps):
            action = e_greedy(env, state, epsilon, q_table)
            next_state, reward, terminated, truncated, info = env.step(action)

if __name__ == "__main__":

    # get spaces from env
    state_space = env.observation_space.n
    action_space = env.action_space_space.n
    # init q_table
    q_table = np.zeros((state_space, action_space))