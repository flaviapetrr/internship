# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import gymnasium as gym
import numpy as np
import math
from trainer import Trainer
from agent import CustomRewardWrapper
from plot_utils import plot_replay_trajectories, plot_qvalue_snapshots

UPDATE_MODES = ["std", "std_punish", "opposite", "relative", "relative_punish"]    
REPLAY_MODES = ["none", "prioritized_sweeping", "value_iteration", "backward", "dyna"]

# --------------- CONFIG ---------------
UPDATE_MODE     = UPDATE_MODES[2]
REPLAY_MODE     = REPLAY_MODES[4]

TRAINING_EPS    = 700
MAX_EPS_STEPS   = 70

REPLAY_STEPS    = 15

SHIFT_GOAL_EP   = None # TRAINING_EPS // 2
SHIFT_GOAL_POS  = (2, 8)

GRID_SIZE       = 10    

# --------------- PARAMS ---------------
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
DECAY_RATE      = math.exp(math.log(EPSILON_MIN / EPSILON_START) / (TRAINING_EPS * 0.8)) # epsilon decay factor
Q_INIT          = 0.0
GAMMA           = 0.99
ALPHA           = 0.5
ALPHA_V         = 0.5

# --------------- REPLAY PARAMS ---------------
BACKWARD_STEPS  = 10
DYNA_STEPS      = 15
PS_STEPS        = 15
THETA           = 0.0001

OUTDIR           = "./QLearning/visuals"
SHIFTDIR         = (f"_shift" if SHIFT_GOAL_EP is not None else "")
    
def make_env(update_mode, goal_row=9, goal_col=9):
    
    # creating custom grid based on grid_size, goal_row, goal_col
    rows = []
    for r in range(GRID_SIZE):
        row = ""
        for c in range(GRID_SIZE):
            if r == 0 and c == 0:
                row += "S"
            elif r == goal_row and c == goal_col:
                row += "G"
            else:
                row += "F"
        rows.append(row)

    env_kwargs = {
        #map_name": "4x4",
        "is_slippery": False,
        "desc": rows
    }

    env = gym.make("FrozenLake-v1", max_episode_steps=MAX_EPS_STEPS, **env_kwargs)
    env = CustomRewardWrapper(env, update_mode)

    return env, env_kwargs

def make_agent(env, update_mode, replay_mode):
    # Creiamo una funzione factory che genera un ambiente con il goal nelle coordinate riga/colonna fornite
    # Prendiamo solo il primo elemento restituito da make_env (l'oggetto env effettivo)
    env_factory = lambda row, col: make_env(update_mode, goal_row=row, goal_col=col)[0]

    return Trainer(
        env,
        update_mode=update_mode,
        replay_mode=replay_mode,
        training_eps=TRAINING_EPS,
        max_episode_steps=MAX_EPS_STEPS,
        epsilon_start= EPSILON_START,
        epsilon_min= EPSILON_MIN,
        decay_rate=DECAY_RATE,
        q_init=Q_INIT,
        gamma=GAMMA,
        alpha=ALPHA,
        alpha_v=ALPHA_V,
        backward_steps=BACKWARD_STEPS,
        dyna_steps=DYNA_STEPS,
        ps_steps=PS_STEPS,
        theta=THETA,
        shift_goal_ep=SHIFT_GOAL_EP,
        shift_goal_pos=SHIFT_GOAL_POS,
        env_factory=env_factory
    )

if __name__ == "__main__":

    #for u in UPDATE_MODES:
    #    for r in REPLAY_MODES:
            env, env_kwargs = make_env(UPDATE_MODE)
            agent = make_agent(env, UPDATE_MODE, REPLAY_MODE)
            
            agent.training()

            plot_replay_trajectories(
                agent,
                path=f"{OUTDIR}/replay_trajs/{UPDATE_MODE}/{UPDATE_MODE}_{REPLAY_MODE}{SHIFTDIR}.png",
                grid_size=GRID_SIZE,
                n_sample=8,
                only_longest_replay=True,
            )
        
            plot_qvalue_snapshots(
                agent,
                path=f"{OUTDIR}/heatmaps/{UPDATE_MODE}/{UPDATE_MODE}_{REPLAY_MODE}{SHIFTDIR}.png",
                grid_size=GRID_SIZE,
            )
