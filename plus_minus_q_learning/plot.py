import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym

BG_COLORS = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd', 'G2': '#ffd5cd'}

def _draw_grid_bg(ax, grid_size, desc, initial_desc, bg, shifted):
    """Draw cell backgrounds + S/G/H labels"""
    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')
            bg_color = "#cdebff" if is_old_goal else bg.get(cell, '#f8f9fa')
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg_color, zorder=0))
            if cell in ['H', 'S']:
                ax.text(c, r, cell, ha='center', va='center',
                        fontsize=12, color='black', alpha=0.5)
            elif cell == 'G':
                lbl = 'G2' if shifted else 'G'
                ax.text(c, r, lbl, ha='center', va='center',
                        fontsize=12, color='black', fontweight='bold', alpha=0.8)
            elif is_old_goal:
                ax.text(c, r, 'G1', ha='center', va='center',
                        fontsize=12, color='gray', fontweight='bold', alpha=0.8)
    ax.grid(True, color='gray', alpha=0.3, zorder=1)

def _heatmap(agent, grid_size, desc, initial_desc, plot, q_table=None, shifted=None, snap=False, min=0, max=0):
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
 
    if q_table is None:
        q_table = agent.q_table

    if agent.mode in ["std", "relative"]:
        q_vals = np.max(q_table, axis=1)
        nr ="max"
    else:
        q_vals = np.min(q_table, axis=1)
        nr = "min"
     
    grid = q_vals.reshape(grid_size, grid_size)

    vmin, vmax = (min, max) if snap else (float(q_vals.min()), float(q_vals.max())) 
    mid = (vmin + vmax) / 2
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    policy = np.argmax(q_table, axis=1)
    grid_policy = policy.reshape(grid_size, grid_size)

    im = plot.imshow(grid, cmap="plasma", norm=norm, aspect='equal')
    cb = plt.colorbar(im, ax=plot, fraction=0.046, pad=0.04)
    cb.set_label("Q value", rotation=270, labelpad=15)
    cb.ax.tick_params(labelsize=7)

    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            val  = grid[r, c]
            text_color = 'white' if val < mid else 'black'
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')

            plot.text(c, r - 0.18, arrows[grid_policy[r, c]],
                    ha='center', va='center', fontsize=9, color=text_color)

            plot.text(c, r + 0.22, f"{val:.3f}",
                    ha='center', va='center', fontsize=7.5, color=text_color)

            if cell == 'G':
                plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, zorder=3))
            elif is_old_goal:
                plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='steelblue', linestyle='--', linewidth=2.5, zorder=3))
            elif cell == 'S':
                plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2.5, zorder=3))

    plot.set_xticks(range(grid_size))
    plot.set_yticks(range(grid_size))
    plot.tick_params(labelsize=8)

    return nr

def _draw_traj_panel(fig, ax, grid_size, desc, initial_desc, bg, shifted, sampled_paths, ep_min, ep_max, ):
    """
    Draw one trajectory panel
    sampled_paths: subset of agent.sampled_paths already filtered
    ep_min/ep_max: define the colour range for the coolwarm cmap
    """

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_aspect('equal')
    ax.invert_yaxis()
 
    _draw_grid_bg(ax, grid_size, desc, initial_desc, bg, shifted)
 
    cmap  = plt.colormaps['plasma']
    ep_range = max(ep_max - ep_min, 1)
 
    for eps_num, path_states in sampled_paths:
        t = (eps_num - ep_min) / ep_range   
        color = cmap(t)
        alpha_val = 0.2 + 0.5 * t
        path_x = [s % grid_size for s in path_states]
        path_y = [s // grid_size for s in path_states]
        jx = np.array(path_x) + np.random.uniform(-0.15, 0.15, size=len(path_x))
        jy = np.array(path_y) + np.random.uniform(-0.15, 0.15, size=len(path_y))
        ax.plot(jx, jy, color=color, alpha=alpha_val, linewidth=1.5, zorder=2)
 
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=ep_min, vmax=ep_max))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Episode', rotation=270, labelpad=15)
    return cbar
 
 
def _draw_visits_panel(ax, grid_size, desc, initial_desc, shifted, visits, vmax=None):
    """Draw state-visit heatmap"""
    grid_visits = visits.reshape(grid_size, grid_size)

    max_val = np.max(grid_visits) if vmax is None else vmax
    if max_val == 0: max_val = 1

    im = ax.imshow(grid_visits, cmap='plasma', aspect='equal', vmin=0, vmax=max_val)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('visit count', rotation=270, labelpad=15)
    cb.ax.tick_params(labelsize=7)
 
    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            val = int(grid_visits[r, c])
            text_color = "black" if val < val < (max_val / 2) else "white"
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')
            ax.text(c, r, str(val), ha='center', va='center', fontsize=9, color=text_color)
            if cell == 'G':
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, zorder=3))
            elif is_old_goal:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor='steelblue', linestyle='--',
                                           linewidth=2.5, zorder=3))
            elif cell == 'S':
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor='green', linewidth=2.5, zorder=3))
    ax.set_xticks(range(grid_size)); ax.set_yticks(range(grid_size))
    ax.tick_params(labelsize=8)

def plot_training(agent, path="training_summary.png", grid_size=10):
    """
    subplots:
        1. reward per episode + moving average
        2. episode length + moving average

        3. two cases:
            if shifted:
                3. Q-value heatmap right before shifting
                4. final Q-value heatmap 
            else:
                3. final Q-value heatmap 
    """
    rewards = np.array(agent.episode_rewards)
    lengths = np.array(agent.episode_lengths)
    window  = max(1, len(rewards) // 20)

    def moving_avg(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    desc = agent.env.unwrapped.desc.astype(str)
    initial_desc = getattr(agent, 'initial_desc', desc)
    shifted = getattr(agent, 'shift_happened_ep', None) is not None

    ncols = 2 if shifted else 3
    nrows = 2 if shifted else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()

    shift_note = (f"  |  goal shift at ep {agent.shift_happened_ep}"
                    if shifted else "")
    fig.suptitle(
        f"Training Summary  |  mode={agent.mode}  alpha={agent.alpha}  episodes={agent.training_episodes}  q_table initial values={agent.q_init}{shift_note}",
        fontsize=13, fontweight='bold'
    )

    eps_x = np.arange(len(rewards))
    ma_x  = np.arange(window - 1, len(rewards))
    
    # subplot 1
    ax = axes[0]
    ylim = (-0.2, 1.2) if agent.mode in ["std", "relative"] else (-1.2, 0.2)
    ax.set_ylim(ylim)
    ax.set_xlim(0, agent.training_episodes * 1.05)
    ax.plot(eps_x, rewards, alpha=0.35, color='steelblue', linewidth=0.8, label='Reward')
    ax.plot(ma_x, moving_avg(rewards, window), color='steelblue', linewidth=2,
            label=f'avg ({window})')
    if shifted:
        ax.axvline(agent.shift_happened_ep, color='gray', linestyle='--', linewidth=0.8,
            label=f'Goal shift: ep {agent.shift_happened_ep}')
    ax.set_title("Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # subplot 2
    ax = axes[2] if shifted else axes[1]
    ax.set_ylim(0, agent.episode_steps * 1.05)
    ax.set_xlim(0, agent.training_episodes * 1.05)
    ax.plot(eps_x, lengths, alpha=0.35, color='darkorange', linewidth=0.8, label='Length')
    ax.plot(ma_x, moving_avg(lengths, window), color='darkorange', linewidth=2,
            label=f'avg ({window})')
    if shifted:
        ax.axvline(agent.shift_happened_ep, color='gray', linestyle='--', linewidth=0.8,
            label=f'Goal shift: ep {agent.shift_happened_ep}')
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if shifted:
        ax_snap = axes[1]
        # subplot 3
        if "at_shift" in agent.q_snapshots:
            ep, q_snap, desc_snap = agent.q_snapshots["at_shift"]
            nr = _heatmap(agent, grid_size, desc_snap, initial_desc, ax_snap, q_table=q_snap, shifted=shifted)
            ax_snap.set_title(f"Q-value ({nr} Q) - At Shift (ep {ep})")

        
        else:
            nr = _heatmap(agent, grid_size, desc, initial_desc, ax_snap, q_table=agent.q_table, shifted=shifted)
            ax_snap.set_title(f"Q-value ({nr} Q) - Snapshot")
        
        # subplot 4 - if shifted
        ax_final = axes[3]
        nr_final = _heatmap(agent, grid_size, desc, initial_desc, ax_final, q_table=agent.q_table, shifted=shifted)
        ax_final.set_title(f"Q-value Heatmap    |   {nr_final} Q + policy arrows")

    # subplot 3 
    else:
        ax_final = axes[2]
        nr = _heatmap(agent, grid_size, desc, initial_desc, ax_final, q_table=agent.q_table, shifted=shifted)
        ax_final.set_title(f"Q-value Heatmap    |   {nr} Q + policy arrows")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    print(f"Training Summary plot saved to {path}")

def plot_replay_analysis(agent, path="replay_analysis.png", window=50):
    """
    subplots:
        1. rolling success rate
        2. mean TD error per episode
        3. model coverage 
    """
    n = len(agent.episode_rewards)
    w = max(1, min(window, n // 5))
    eps_x = np.arange(n)
    shifted = getattr(agent, 'shift_happened_ep', None) is not None

    def rolling(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Replay analysis  |  mode={agent.mode}  alpha={agent.alpha}  episodes={agent.training_episodes}   q_table initial values={agent.q_init}",
        fontsize=13, fontweight='bold'
    )    

    # subplot 1
    ax = axes[0]
    success = np.array(agent.episode_success, dtype=float)
    roll_x  = np.arange(w - 1, n)
    ax.set_ylim(0, 110)
    ax.set_xlim(0, agent.training_episodes * 1.05)
    ax.plot(roll_x, rolling(success, w) * 100,
            color='steelblue', linewidth=2, label="Success rate")
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
            label=f'100% success rate')
    if shifted:
        ax.axvline(agent.shift_happened_ep, color='gray', linestyle='--', linewidth=0.8,
            label=f'Goal shift: ep {agent.shift_happened_ep}')
    ax.set_title(f"Rolling Success Rate  (window={w})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # subplot 2
    ax = axes[1]
    td = np.array(agent.episode_td_errors)
    ax.plot(eps_x, td, alpha=0.3, color='darkorange', linewidth=0.7, label=f'TD error')
    ax.plot(np.arange(w - 1, n), rolling(td, w),
            color='tomato', linewidth=2, label=f'avg ({w})')
    if shifted:
        ax.axvline(agent.shift_happened_ep, color='gray', linestyle='--', linewidth=0.8,
            label=f'Goal shift: ep {agent.shift_happened_ep}')
    ax.set_title("Mean TD Error per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("TD error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # subplot 3
    ax = axes[2]
    max_pairs = agent.state_space * agent.action_space
    model_arr = np.array(agent.model_sizes)
    ax.plot(eps_x, model_arr, color='green', linewidth=1.8,
            label='(s,a) pairs in model')
    ax.axhline(max_pairs, color='gray', linestyle='--', linewidth=0.8,
                label=f'Max possible ({max_pairs})')
    if shifted:
        ax.axvline(agent.shift_happened_ep, color='gray', linestyle='--', linewidth=0.8,
            label=f'Goal shift: ep {agent.shift_happened_ep}')
    ax.set_title("Model Coverage over Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("# (s, a) pairs known")
    ax.set_ylim(0, max_pairs * 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    print(f"Replay Analysis plot saved to {path}")

def plot_trajectories(agent, env_id, max_steps, env_kwargs, num_episodes=50, epsilon_test=0.1, path="all_trajectories.png", grid_size=10):
        """performing N episodes
            after training
            epsilon-greedy police
            poltting the trajectories + visual jitter"""
        test_env = gym.make(env_id, max_episode_steps=max_steps, **env_kwargs)
        desc = test_env.unwrapped.desc.astype(str)
        
        _, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Trajectories | episodes={num_episodes} epsilon={epsilon_test}", fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        _draw_grid_bg(ax, grid_size, desc, desc, BG_COLORS, shifted=False)

        ax.grid(True, color='gray', alpha=0.3, zorder=1)

        # getting and plotting trajectories
        for _ in range(num_episodes):
            state, _ = test_env.reset()
            terminated = truncated = False
            
            # conversing state in row,col
            path_x = [state % grid_size]
            path_y = [state // grid_size]
            
            for _ in range(agent.episode_steps):
                if terminated or truncated:
                    break
                
                # epsilon-greedy policy
                if np.random.rand() < epsilon_test:
                    action = test_env.action_space.sample()
                else:
                    action = int(np.argmax(agent.q_table[state]))
                    
                state, _, terminated, truncated, _ = test_env.step(action)
                path_x.append(state % grid_size)
                path_y.append(state // grid_size)
                
            # adding visual jitter
            jitter_x = np.array(path_x) + np.random.uniform(-0.15, 0.15, size=len(path_x))
            jitter_y = np.array(path_y) + np.random.uniform(-0.15, 0.15, size=len(path_y))
            
            # plotting trajectory
            ax.plot(jitter_x, jitter_y, color='royalblue', alpha=0.15, linewidth=2, zorder=2)
            
            # final point
            end_color = 'green' if desc[path_y[-1], path_x[-1]] == 'G' else ('red' if desc[path_y[-1], path_x[-1]] == 'H' else 'orange')
            ax.scatter(jitter_x[-1], jitter_y[-1], color=end_color, s=20, zorder=3, alpha=0.5)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        #plt.show()
        print(f"Test Trajectories plot saved to {path}")
        test_env.close()

def plot_qvalue_snapshots(agent, path="qvalue_snapshots.png", grid_size=10):
    """
    Heatmap snapshots of the Q-table at key moments during training

    Without goal shift  (4 panels):
        first_goal - first_goal +10 ep - 5 times_goal - final

    With goal shift  (up to 8 panels):
        first_goal - first_goal +10 ep - at_shift - 5 times_goal
        first_new_goal - first_new_goal +10 ep - final

    Each panel:
        shows max-Q per state with policy arrows overlaid
        highlights the goal cell with a border.
    """
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    CMAP = 'plasma'

    LABEL_MAP = {
        "first_goal":        "First Goal Reached",
        "first_goal_p10":    "+10 ep After First Goal",
        "5_times_goal":      "Reach First Goal 5 Times",
        "at_shift":          "At Reward / Goal Shift",
        "first_new_goal":    "First New Goal Reached",
        "first_new_goal_p10": "+10 ep After New Goal",
        "5_times_new_goal":      "Reach New Goal 5 Times",
        "final":             "Final (end of training)",
    }

    shifted = agent.shift_happened_ep is not None

    desc = agent.env.unwrapped.desc.astype(str)
    initial_desc = getattr(agent, 'initial_desc', desc)

    if shifted:
        key_order = ["first_goal", "first_goal_p10",
                        "5_times_goal", "at_shift",
                        "first_new_goal", "first_new_goal_p10",
                        "5_times_new_goal", "final"]
    else:
        key_order = ["first_goal", "first_goal_p10", "5_times_goal", "final"]

    snapshots = [(k, agent.q_snapshots[k])
                    for k in key_order if k in agent.q_snapshots]

    if not snapshots:
        print("[plot_qvalue_snapshots] No snapshots to plot - skipping")
        return

    n = len(snapshots)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                                figsize=(ncols * 6, nrows * 6),
                                squeeze=False)
    axes_flat = axes.flatten()

    if agent.mode in ["std", "relative"]:
        q_vals = np.concatenate(
                [np.max(qtable, axis=1) for _, (_, qtable, _) in snapshots]
            )
    else:
        q_vals = np.concatenate(
                [np.min(qtable, axis=1) for _, (_, qtable, _) in snapshots]
            )

    vmin, vmax = float(q_vals.min()), float(q_vals.max())

    # drawing each snapshot
    for i, (key, (ep, qtable, desc)) in enumerate(snapshots):
        ax = axes_flat[i]

        nr = _heatmap(agent, grid_size, desc, initial_desc, ax, q_table=qtable, shifted=shifted, snap=True, min=vmin, max=vmax)

        label_text = LABEL_MAP.get(key, key)
        ax.set_title(f"{label_text}   |   ep {ep}", fontsize=11, fontweight='bold')

    shift_note = (f"  |  goal shift at ep {agent.shift_happened_ep}"
                    if shifted else "")
    fig.suptitle(
        f"{nr} Q-value Snapshots  |  mode={agent.mode}  alpha={agent.alpha}"
        f"  episodes={agent.training_episodes}{shift_note}",
        fontsize=13, fontweight='bold',
    )

    # hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Q-value Snapshots plot saved to {path}")

def plot_training_evolution(agent, path="training_evolution.png", grid_size=10):
    """
    subplots:
        1.  trajectory gradient
        2.  Q-value heatmap
        3.  state visits heatmap
    note: doubled if shifted
    """
   
    shifted = agent.shift_happened_ep is not None
    desc = agent.env.unwrapped.desc.astype(str)
    initial_desc = getattr(agent, 'initial_desc', desc)

    nrows = 2 if shifted else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 6 * nrows))
    axes = np.array(axes).reshape(nrows, 3) 

    if shifted:
        shift_ep = agent.shift_happened_ep
        shift_note = f" | goal shift at ep {shift_ep}"

        paths_before = [(ep, p) for ep, p in agent.sampled_paths if ep <= shift_ep]
        paths_after  = [(ep, p) for ep, p in agent.sampled_paths if ep > shift_ep]
        
        visits_before = np.zeros(agent.state_space)
        visits_after  = np.zeros(agent.state_space)
        for _, p in paths_before:
            for s in p: visits_before[s] += 1
        for _, p in paths_after:
            for s in p: visits_after[s] += 1
            
        vmax_visits = max(visits_before.max(), visits_after.max())
        
        q_shift_table = agent.q_snapshots["at_shift"][1] if "at_shift" in agent.q_snapshots else agent.q_table
        if agent.mode in ["std", "relative"]:
            q_max = max(np.max(agent.q_table, axis=1).max(), np.max(q_shift_table, axis=1).max())
            q_min = min(np.max(agent.q_table, axis=1).min(), np.max(q_shift_table, axis=1).min())
        else:
            q_max = max(np.min(agent.q_table, axis=1).max(), np.min(q_shift_table, axis=1).max())
            q_min = min(np.min(agent.q_table, axis=1).min(), np.min(q_shift_table, axis=1).min())

    title = (f"Training Evolution   |   mode={agent.mode} replay={agent.replay_mode}    |   episodes={agent.training_episodes}{shift_note}")
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if shifted:
        ax = axes[0, 0]
        _draw_traj_panel(fig, ax, grid_size, initial_desc, initial_desc, BG_COLORS, False, sampled_paths=paths_before, ep_min=0, ep_max=agent.training_episodes)
        ax.set_title(f"Trajectories before shift", fontsize=12)

        ax = axes[0, 1]
        if "at_shift" in agent.q_snapshots:
            ep_s, q_s, desc_s = agent.q_snapshots["at_shift"]
            nr = _heatmap(agent, grid_size, desc_s, initial_desc, ax, q_table=q_s, shifted=False, snap=True, min=q_min, max=q_max)
            ax.set_title(f"At shift Q-value Heatmap    |   {nr} Q + policy arrows", fontsize=12)
        else:
            ax.set_visible(False)

        ax = axes[0, 2]
        _draw_visits_panel(ax, grid_size, initial_desc, initial_desc, False, visits_before, vmax=vmax_visits)
        ax.set_title("State Visits before shift", fontsize=12)

        ax = axes[1, 0]
        _draw_traj_panel(fig, ax, grid_size, desc, initial_desc, BG_COLORS, True, sampled_paths=paths_after, ep_min=0, ep_max=agent.training_episodes)
        ax.set_title(f"Trajectories after shift", fontsize=12)

        ax = axes[1, 1]
        nr = _heatmap(agent, grid_size, desc, initial_desc, ax, q_table=agent.q_table, shifted=True, snap=True, min=q_min, max=q_max)
        ax.set_title(f"Final Q-value Heatmap    |   {nr} Q + policy arrows", fontsize=12)

        ax = axes[1, 2]
        _draw_visits_panel(ax, grid_size, desc, initial_desc, True, visits_after, vmax=vmax_visits)
        ax.set_title("State Visits after shift", fontsize=12)

    else:
        ax = axes[0, 0]
        _draw_traj_panel(fig, ax, grid_size, desc, initial_desc, BG_COLORS, False, sampled_paths=agent.sampled_paths, ep_min=0, ep_max=agent.training_episodes)
        ax.set_title("Trajectories", fontsize=12)

        ax = axes[0, 1]
        nr = _heatmap(agent, grid_size, desc, initial_desc, ax, q_table=agent.q_table, shifted=False)
        ax.set_title(f"Final Q-value Heatmap    |   {nr} Q + policy arrows", fontsize=12)

        ax = axes[0, 2]
        _draw_visits_panel(ax, grid_size, desc, initial_desc, False, agent.state_visits)
        ax.set_title("State Visits", fontsize=12)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training Evolution plot saved to {path}")
    
