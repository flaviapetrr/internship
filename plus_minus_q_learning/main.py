# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import agent

# --------------- CHANGE CONFIG HERE ---------------

MODE             = "std" # "std", "std_punish", "opposite", "relative", "relative_punish"
TRAINING_EPS     = 600 
EPISODE_STEPS    = 100
Q_INIT           = 0.0
ALPHA            = 0.3
ALPHA_V          = 0.3
GAMMA            = 0.99
EPS_START        = 1.0
EPS_END          = 0.01
EPS_DECAY        = 0.005 
REPLAY_STEPS     = 10
THETA            = 0.0001
OUTDIR           = "./plus_minus_q_learning/visuals"

REPLAY_MODES = ["none", "forward", "backward", "dyna"]

COLORS = {
    "none":     "#888888",
    "forward":  "#2196F3",
    "backward": "#E91E63",
    "dyna":     "#4CAF50",
}

def make_env(mode):
    if mode in ["std", "relative"]:
        reward_schedule = (1, 0, 0)
    else:
        reward_schedule = (-1, 0, 0)

    env_kwargs = {
        "map_name": "4x4",
        "is_slippery": False,
        "reward_schedule": reward_schedule,
    }

    return gym.make("FrozenLake-v1", **env_kwargs), env_kwargs

def make_agent(env, mode, replay_mode):
    return agent.QLearningAgent(
        env,
        mode=mode,
        replay_mode=replay_mode,
        training_episodes=TRAINING_EPS,
        episode_steps=EPISODE_STEPS,
        q_init=Q_INIT,
        alpha=ALPHA,
        alpha_v=ALPHA_V,
        gamma=GAMMA,
        epsilon_start=EPS_START,
        epsilon_end=EPS_END,
        epsilon_decay=EPS_DECAY,
        epsilon=1.0,
        replay_steps=REPLAY_STEPS,
        theta=THETA,
    )


# moving average mean
def rolling(x, w):
    """
    Moving Avg Mean
    
    Args:
    x:  data
    w: window
    """
    return np.convolve(np.array(x, dtype=float), np.ones(w) / w, mode='valid')

def episodes_to_criterion(success_list, window=50, threshold=80.0):
    """first episode where rolling success rate crosses threshold%"""
    r = rolling(success_list, window) * 100
    hits = np.where(r >= threshold)[0]
    return int(hits[0] + window - 1) if len(hits) > 0 else None

# --------------- SINGLE RUN ---------------

def run_single(replay_mode):
    print(f"\n{'='*50}")
    print(f"  mode={MODE}  replay={replay_mode}")
    print(f"{'='*50}")
    env, env_kwargs = make_env(MODE)
    agent = make_agent(env, MODE, replay_mode)
    agent.train()
    agent.plot_training(
        path=f"{OUTDIR}/plots/training_summary/{MODE}_{replay_mode}.png", grid_size=4
    )
    if replay_mode != "none":
        agent.plot_replay_analysis(
            path=f"{OUTDIR}/plots/replay_analysis/{replay_mode}_{MODE}.png", grid_size=4
        )

    agent.record_gif(
        "FrozenLake-v1", env_kwargs,
        path=f"{OUTDIR}/gifs/{MODE}.gif", fps=3
    )

# --------------- COMPARISON RUN ---------------

def run_comparison():
    """
    Running #seed agents for each replay method
    subplots:
        1. mean rolling success rate + error
        2. mean TD error
        3. mean episodes to reach 80% + error
    """
    print("\n" + "="*55)
    print("  COMPARISON RUN — all replay modes")
    print("="*55)
 
    N_SEEDS = 10
    window  = max(10, TRAINING_EPS // 20)
 
    # collecting data
    all_success  = {rm: [] for rm in REPLAY_MODES}
    all_td_error = {rm: [] for rm in REPLAY_MODES}
 
    for seed in range(N_SEEDS):
        print(f"\n  seed {seed+1}/{N_SEEDS}")
        np.random.seed(seed)
        for rm in REPLAY_MODES:
            env, _ = make_env(MODE)
            agent = make_agent(env, MODE, rm)
            agent.train()
            all_success[rm].append(agent.episode_success)
            all_td_error[rm].append(agent.episode_td_errors)
 
    # computing mean +- std dev
    mean_suc = {rm: np.mean(all_success[rm],  axis=0) for rm in REPLAY_MODES}
    std_suc  = {rm: np.std(all_success[rm],   axis=0) for rm in REPLAY_MODES}
    mean_td  = {rm: np.mean(all_td_error[rm], axis=0) for rm in REPLAY_MODES}
 
    x = np.arange(window - 1, TRAINING_EPS)
 
    # plotting
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Replay comparison  |  mode={MODE}  α={ALPHA}  episodes={TRAINING_EPS}"
        f"  (mean ± std over {N_SEEDS} seeds)   q_table initial values={Q_INIT}",
        fontsize=13, fontweight='bold'
    )
 
    # subplot 1
    ax = axes[0]
    for rm in REPLAY_MODES:
        mean_r = rolling(mean_suc[rm], window) * 100
        std_r  = rolling(std_suc[rm],  window) * 100
        ax.plot(x, mean_r, color=COLORS[rm], linewidth=2, label=rm)
        ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                        color=COLORS[rm], alpha=0.15)
        # mark mean 80% crossing
        crit = episodes_to_criterion(mean_suc[rm], window=window, threshold=80)
        if crit is not None:
            ax.axvline(crit, color=COLORS[rm], linestyle=':', linewidth=1, alpha=0.7)
            ax.text(crit + 3, 3, str(crit), color=COLORS[rm], fontsize=7, va='bottom')
 
    ax.axhline(80, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, label='80% target')
    ax.set_title(f"Rolling Success Rate  (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success %")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
    # subplot 2
    ax = axes[1]
    for rm in REPLAY_MODES:
        ax.plot(x, rolling(mean_td[rm], window),
                color=COLORS[rm], linewidth=2, label=rm)
    ax.set_title(f"Mean TD Error  (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("TD error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
    # subplot 3
    ax = axes[2]
    labels, values, bar_colors, err_vals = [], [], [], []
    for rm in REPLAY_MODES:
        crits = [
            episodes_to_criterion(s, window=window, threshold=80) or TRAINING_EPS
            for s in all_success[rm]
        ]
        labels.append(rm)
        values.append(np.mean(crits))
        err_vals.append(np.std(crits))
        bar_colors.append(COLORS[rm])
 
    bars = ax.bar(labels, values, yerr=err_vals, color=bar_colors,
                  edgecolor='white', width=0.5, capsize=5, error_kw={"linewidth": 1.5})
    for bar, v in zip(bars, values):
        label = f"{v:.0f}" if v < TRAINING_EPS else "never"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(err_vals) + 10,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(f"Mean episodes to 80%  (±std, {N_SEEDS} seeds)")
    ax.set_ylabel("Episode nr.")
    ax.set_ylim(0, TRAINING_EPS * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
 
    plt.tight_layout()
    path = f"{OUTDIR}/plots/comparisons/{MODE}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nComparison plot saved to {path}")
 
    # printing summary table
    print(f"\n{'─'*45}")
    print(f"  {'replay':<12} {'mean eps to 80%':>15}  {'±std':>8}")
    print(f"{'─'*45}")
    for rm, v, e in zip(labels, values, err_vals):
        tag = f"{v:.0f}" if v < TRAINING_EPS else "never"
        print(f"  {rm:<12} {tag:>15}  {e:>8.1f}")
    print(f"{'─'*45}")

if __name__ == "__main__":

    # --------------- option A: single replay mode ---------------
    run_single(replay_mode="backward")

    # --------------- option B: compare all replay modes ---------------
    # run_comparison()