# conda activate /home/etu-admin/Desktop/Flavia/internship/cyu_tut
# Gym docs: https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import agent
import plot
import gif

# --------------- CHANGE CONFIG HERE ---------------

RUN              = "single" # "single", "mode_comparison", "replay_comparison"
FIXED_REPLAY     = "prioritized_sweeping" # "none", "prioritized_sweeping", "backward", "dyna"
MODE             = "opposite" # "std", "std_punish", "opposite", "relative", "relative_punish"

TRAINING_EPS     = 900
EPISODE_STEPS    = 90

REPLAY_STEPS     = 20

REWARD_SHIFT_EP  = TRAINING_EPS // 2
SHIFT_GOAL_POS   = (2, 8) 

Q_INIT           = 0.0
ALPHA            = 0.3
ALPHA_V          = 0.3
GAMMA            = 0.99
EPS_START        = 1.0
EPS_END          = 0.01
EPS_DECAY        = 0.0005
THETA            = 0.0001
OUTDIR           = "./plus_minus_q_learning/visuals"
SHIFTDIR         = (f"_shift" if REWARD_SHIFT_EP is not None else "")

REPLAY_MODES = ["none", "prioritized_sweeping", "backward", "dyna"]
MODES = ["std", "std_punish", "opposite", "relative", "relative_punish"]    

COLORS = {
    "none":     "#888888",
    "prioritized_sweeping":  "#2196F3",
    "backward": "#E91E63",
    "dyna":     "#4CAF50",
}

GRID_SIZE       = 10    

def make_env(mode,  goal_row=9, goal_col=9):
    # defining reward/punishment based on mode
    if mode in ["std", "relative"]:
        reward_schedule = (1, 0, 0)
    else:
        reward_schedule = (-1, 0, 0)

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
        "reward_schedule": reward_schedule,
        "desc": rows
    }

    return gym.make("FrozenLake-v1", max_episode_steps=EPISODE_STEPS, **env_kwargs), env_kwargs

def make_agent(env, mode, replay_mode,reward_shift_ep=REWARD_SHIFT_EP, shift_env_fn=None):
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
        replay_steps=REPLAY_STEPS,
        theta=THETA,
        reward_shift_ep=reward_shift_ep,
        shift_env_fn=shift_env_fn,

    )

def episodes_to_criterion(success_list, window=50, threshold=80.0):
    """first episode where moving avg success rate crosses threshold%"""
    r = plot._moving_avg(success_list, window) * 100
    hits = np.where(r >= threshold)[0]
    return int(hits[0] + window - 1) if len(hits) > 0 else None

# --------------- SINGLE RUN ---------------

def run_single(mode, replay_mode):

    print(f"\n{'='*50}")
    print(f" mode={mode} - replay={replay_mode}")
    if REWARD_SHIFT_EP is not None:
        print(f" goal shift at ep {REWARD_SHIFT_EP} to {SHIFT_GOAL_POS}")
    print(f"{'='*50}")

    env, env_kwargs = make_env(mode)
    shift_fn = None

    if REWARD_SHIFT_EP is not None:
        shift_fn = lambda: make_env(mode, SHIFT_GOAL_POS[0], SHIFT_GOAL_POS[1])[0]
 
    agent = make_agent(
        env, mode, replay_mode,
        reward_shift_ep=REWARD_SHIFT_EP,
        shift_env_fn=shift_fn,
    )
    agent.train()

    plot.plot_training(agent,
        path=f"{OUTDIR}/plots/training_summary/{mode}/{mode}_{replay_mode}{SHIFTDIR}.png",
        grid_size=GRID_SIZE,
    )
 
    if replay_mode != "none":
        plot.plot_replay_analysis(agent,
            path=f"{OUTDIR}/plots/replay_analysis/{mode}/{mode}_{replay_mode}{SHIFTDIR}.png",
        )
        plot.plot_replay_trajectories(agent,
            path=f"{OUTDIR}/plots/replay_trajectories/{mode}/{mode}_{replay_mode}{SHIFTDIR}.png",
            grid_size=GRID_SIZE,
        )
 
    plot.plot_training_evolution(agent,
        path=f"{OUTDIR}/plots/training_evolution/{mode}/{mode}_{replay_mode}{SHIFTDIR}.png",
        grid_size=GRID_SIZE,
    )
 
    plot.plot_qvalue_snapshots(agent,
        path=f"{OUTDIR}/plots/qvalue_snapshots/{mode}/{mode}_{replay_mode}{SHIFTDIR}.png",
        grid_size=GRID_SIZE,
    )
#
#    plot.plot_trajectories(agent,
#        "FrozenLake-v1", EPISODE_STEPS, env_kwargs, num_episodes=50,
#        path=f"{OUTDIR}/plots/egreedy_trajectories/{mode}/{mode}_{replay_mode}.png", grid_size=GRID_SIZE
#    )
#
#    gif.record_gif(agent,
#        "FrozenLake-v1", EPISODE_STEPS, env_kwargs,
#        path=f"{OUTDIR}/gifs/{MODE}.gif", fps=3
#    )
#
#    gif.plot_sampled_trajectories_gif(agent,
#        path=f"{OUTDIR}/gifs/training_evolution/{mode}.gif",
#        grid_size=GRID_SIZE,
#        fps=3
#    )
#    
#    gif.swarm_gif(
#       path=f"{OUTDIR}/gifs/egreedy_trajectories/{mode}.gif",
#       "FrozenLake-v1", EPISODE_STEPS, env_kwargs, num_agents=30, grid_size=GRID_SIZE
#    )
    
# --------------- MODE COMPARISON RUN ---------------

def run_mode_comparison(replay_mode="none"):
    """
    Comparison of all Q-learning update modes
 
    2 groups:
        reward-based:     std, relative
        punishment-based: std_punish, opposite, relative_punish

    subplots:
        1. moving avg mean success rate + error
        2. mean TD error
        3. mean episodes to reach 80% + error
    """
 
    REWARD_MODES = ["std", "relative"]
    PUNISH_MODES = ["std_punish", "opposite", "relative_punish"]
    ALL_MODES    = REWARD_MODES + PUNISH_MODES
 
    MODE_COLORS = {
        "std":              "#2196F3",
        "std_punish":       "#FF9800",
        "relative":         "#9C27B0",
        "opposite":         "#E91E63",
        "relative_punish":  "#4CAF50",
    }
 
    print("\n" + "="*50)
    print(f" MODE COMPARISON - replay={replay_mode}")
    print("="*50)
 
    N_SEEDS = 10
    window  = max(10, TRAINING_EPS // 20)
 
    all_success  = {m: [] for m in ALL_MODES}
    all_td_error = {m: [] for m in ALL_MODES}
 
    for seed in range(N_SEEDS):
        print(f"\n  seed {seed+1}/{N_SEEDS}")
        np.random.seed(seed)
        for m in ALL_MODES:
            env, _ = make_env(m)
            ag = make_agent(env, mode=m, replay_mode=replay_mode)
            ag.train()
            all_success[m].append(ag.episode_success)
            all_td_error[m].append(ag.episode_td_errors)
 
    x = np.arange(window - 1, TRAINING_EPS)
 
    def _plot_group(modes, group_label):
 
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
 
        fig.suptitle(
            f"Mode comparison  ({group_label})  |  replay={replay_mode}"
            f"  \u03b1={ALPHA}  episodes={TRAINING_EPS}  Q_init={Q_INIT}",
            fontsize=13, fontweight='bold',
        )
 
        ax_suc, ax_td, ax_bar = axes
 
        mean_suc = {m: np.mean(all_success[m],  axis=0) for m in modes}
        std_suc  = {m: np.std(all_success[m],   axis=0) for m in modes}
        mean_td  = {m: np.mean(all_td_error[m], axis=0) for m in modes}
 
        # subplot 1
        for m in modes:
            mr = plot._moving_avg(mean_suc[m], window) * 100
            sr = plot._moving_avg(std_suc[m],  window) * 100
            ax_suc.plot(x, mr, color=MODE_COLORS[m], linewidth=2, label=m)
            ax_suc.fill_between(x, mr - sr, mr + sr, color=MODE_COLORS[m], alpha=0.15)
            crit = episodes_to_criterion(mean_suc[m], window=window, threshold=80)
            if crit is not None:
                ax_suc.axvline(crit, color=MODE_COLORS[m], linestyle=':', linewidth=1, alpha=0.7)
                ax_suc.text(crit + 3, 2, str(crit), color=MODE_COLORS[m], fontsize=7)
        ax_suc.axhline(80, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='80%')
        ax_suc.set_title(f"Moving Avg Success Rate  (window={window})")
        ax_suc.set_xlabel("Episode"); ax_suc.set_ylabel("Success %")
        ax_suc.set_ylim(0, 115); ax_suc.legend(fontsize=8); ax_suc.grid(True, alpha=0.3)
 
        # subplot 2
        for m in modes:
            ax_td.plot(x, plot._moving_avg(mean_td[m], window),
                       color=MODE_COLORS[m], linewidth=2, label=m)
        ax_td.set_title(f"Mean TD Error  (window={window})")
        ax_td.set_xlabel("Episode"); ax_td.set_ylabel("TD error")
        ax_td.legend(fontsize=8); ax_td.grid(True, alpha=0.3)
 
        # subplot 3
        labels_b, values_b, colors_b, errs_b = [], [], [], []
        for m in modes:
            crits = [
                episodes_to_criterion(s, window=window, threshold=80) or TRAINING_EPS
                for s in all_success[m]
            ]
            labels_b.append(m); values_b.append(np.mean(crits))
            errs_b.append(np.std(crits)); colors_b.append(MODE_COLORS[m])
        bars = ax_bar.bar(labels_b, values_b, yerr=errs_b, color=colors_b,
                          edgecolor='white', width=0.5, capsize=5,
                          error_kw={"linewidth": 1.5})
        for bar, v in zip(bars, values_b):
            lbl = f"{v:.0f}" if v < TRAINING_EPS else "never"
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(errs_b) if errs_b else 10) + 10,
                        lbl, ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax_bar.set_title(f"Mean episodes to 80%  (\u00b1std, {N_SEEDS} seeds)")
        ax_bar.set_ylabel("Episode nr.")
        ax_bar.set_ylim(0, TRAINING_EPS * 1.2)
        ax_bar.grid(True, alpha=0.3, axis='y')
 
        slug = group_label.lower().replace(" ", "_").replace("-", "_")
        path = f"{OUTDIR}/plots/comparisons/mode_comparisons/{slug}_replay_{replay_mode}{SHIFTDIR}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nMode comparison plot saved to {path}")
 
        print(f"\n{'─'*50}")
        print(f"  {'mode':<18} {'mean eps to 80%':>15}  {'±std':>8}")
        print(f"{'─'*50}")
        for m, v, e in zip(labels_b, values_b, errs_b):
            tag = f"{v:.0f}" if v < TRAINING_EPS else "never"
            print(f"  {m:<18} {tag:>15}  {e:>8.1f}")
        print(f"{'─'*50}")
 
    _plot_group(REWARD_MODES, "reward_based")
    _plot_group(PUNISH_MODES, "punishment_based")

# --------------- REPLAY COMPARISON RUN ---------------

def run_replay_comparison():
    """
    Running N seed agents for each replay method

    subplots:
        1. moving avg mean success rate + error
        2. mean TD error
        3. mean episodes to reach 80% + error
    """
    print("\n" + "="*50)
    print(f" REPLAY COMPARISON - mode={MODE}")
    print("="*50)
 
    N_SEEDS = 10
    window  = max(10, TRAINING_EPS // 20)
 
    # collecting data
    all_success  = {rm: [] for rm in REPLAY_MODES}
    all_td_error = {rm: [] for rm in REPLAY_MODES}
 
    for seed in range(N_SEEDS):
        print(f"    seed {seed+1}/{N_SEEDS}")
        np.random.seed(seed)
        for rm in REPLAY_MODES:
            env, _ = make_env(MODE)
            agent = make_agent(env, MODE, rm)
            agent.train()
            all_success[rm].append(agent.episode_success)
            all_td_error[rm].append(agent.episode_td_errors)
 
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
        mean_r = plot._moving_avg(mean_suc[rm], window) * 100
        std_r  = plot._moving_avg(std_suc[rm],  window) * 100
        ax.plot(x, mean_r, color=COLORS[rm], linewidth=2, label=rm)
        ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                        color=COLORS[rm], alpha=0.15)
        # mark mean 80% crossing
        crit = episodes_to_criterion(mean_suc[rm], window=window, threshold=80)
        if crit is not None:
            ax.axvline(crit, color=COLORS[rm], linestyle=':', linewidth=1, alpha=0.7)
            ax.text(crit + 3, 3, str(crit), color=COLORS[rm], fontsize=7, va='bottom')
 
    ax.axhline(80, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, label='80% target')
    ax.set_title(f"Moving Avg Successes Rate  (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success %")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
    # subplot 2
    ax = axes[1]
    for rm in REPLAY_MODES:
        ax.plot(x, plot._moving_avg(mean_td[rm], window),
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
    path = f"{OUTDIR}/plots/comparisons/replay_comparisons/{MODE}{SHIFTDIR}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nReplay comparison plot saved to {path}")
 
    # printing summary table
    print(f"\n{'─'*45}")
    print(f"  {'replay':<12} {'mean eps to 80%':>15}  {'±std':>8}")
    print(f"{'─'*45}")
    for rm, v, e in zip(labels, values, err_vals):
        tag = f"{v:.0f}" if v < TRAINING_EPS else "never"
        print(f"  {rm:<12} {tag:>15}  {e:>8.1f}")
    print(f"{'─'*45}")

if __name__ == "__main__":

    if RUN == "single":
        # single replay mode
#        for m in MODES:
#            for r in REPLAY_MODES:
#                run_single(mode=m, replay_mode=r)
#
        run_single(mode=MODE, replay_mode=FIXED_REPLAY)

    elif RUN == "mode_comparison":
        # compare all Q-learning modes
        run_mode_comparison(replay_mode=FIXED_REPLAY)

    elif RUN == "replay_comparison":
        # compare all replay modes
        run_replay_comparison()