import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm, Normalize
import agent

LABEL_MAP = {
        "first_goal":        "First Goal Reached",
        "first_goal_p10":    "+10 ep After First Goal",
        "5_times_goal":      "Reach First Goal 5 Times",
        "at_shift":          "At Reward / Goal Shift",
        "at_obs_add":        "Obstacles Added",
        "first_new_goal":    "First New Goal Reached",
        "first_new_goal_p10": "+10 ep After New Goal",
        "5_times_new_goal":      "Reach New Goal 5 Times",
        "final":             "Final (end of training)",
    }

def _draw_cell_borders(plot, c, r, cell, is_old_goal, zorder):
    
    if cell == 'G':
        plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red', 
                                     linewidth=2.5, zorder=zorder))
    elif is_old_goal:
        plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='yellow', 
                                     linestyle='--', linewidth=2.5, zorder=zorder))
    elif cell == 'S':
        plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='green', 
                                     linewidth=2.5, zorder=zorder))
    elif cell == 'H':
        plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color='#2c2c2c', alpha=0.3, zorder=1))
        plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.5, zorder=zorder))
    
def _write_cell_text(plot, c, r, cell, is_old_goal, shifted, text_color='black', 
                     pos=False, val=None, arrow=None, zorder=None):

    text_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 12}

    if zorder is not None:
        text_kwargs['zorder'] = zorder

    if pos:
        if cell == 'H':
            plot.text(c, r, 'X', color='black', fontweight='bold', alpha=1.0, **text_kwargs)
        elif cell == 'F':  
            plot.text(c, r, cell, color='black', fontweight='bold', alpha=0.5, **text_kwargs)
        elif cell == 'G':
            label = 'G2' if shifted else 'G'
            plot.text(c, r, label, color='black', fontweight='bold', alpha=0.8, **text_kwargs)
        elif is_old_goal:
            plot.text(c, r, 'G1', color='gray', fontweight='bold', alpha=0.8, **text_kwargs)

    if cell != 'H':
        if arrow is not None:
            plot.text(c, r - 0.18, arrow, ha='center', va='center', fontsize=9, fontweight='bold', color=text_color)
        if val is not None:
            plot.text(c, r + 0.22, f"{val:.3f}", ha='center', va='center', fontsize=7.5, fontweight='bold', color=text_color)

def _heatmap(agent, grid_size, desc, initial_desc, plot, shifted, q_table=None, vmin=None, vmax=None, 
             cmap="Blues", alpha=1.0, show_text=True, show_cbar=True, zorder=1):
    """
    Draw heatmap
    
    Args:
        agent:          QLearningAgent instance
        grid_size:      grid dimension, std = 10x10
        desc:           str matrix that describes current env state ('S', 'F', 'G')
        initial_desc:   str matrix that describes env state before a possible goal shift ('S', 'F', 'G1', 'G2')
        plot:           obj in which we are drawing
        shifted:        bool to tell if there is a goal shift

        the others are optional (kwargs), for visualization purposes
    """
    # mapping gym action indices into arrows
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    # if no q_table is given takes the current one
    if q_table is None:
        q_table = agent.q_table

    # q_table = (grid_size*grid_size)xactions
    # transforming it in a 1D list of size (grid_size*grid_size)
    # taking max or min action based on mode
    if agent.mode in ["std", "relative"]:
        q_vals = np.max(q_table, axis=1)
        nr ="max"
    else:
        q_vals = np.min(q_table, axis=1)
        nr = "min"
    
    # reshaping q_vals from 1D (grid_size*grid_size) to 2D grid_sizexgrid_size
    grid = q_vals.reshape(grid_size, grid_size)

    # if vmin and vmax are given using them as global thresholds -> useful for coerence in comparisons
    # else computing current min and max from q_vals
    min_val = float(q_vals.min()) if vmin is None else vmin
    max_val = float(q_vals.max()) if vmax is None else vmax

    # if all values are identical add a little epsilon to avoid /0 in plt.Normalize
    if min_val == max_val: max_val += 1e-5

    # computing mean for text color purposes   
    mid = (min_val + max_val) / 2

    policy = np.argmax(q_table, axis=1)
    # reshaping policy from 1D (grid_size*grid_size) to 2D grid_sizexgrid_size
    grid_policy = policy.reshape(grid_size, grid_size)

    norm = SymLogNorm(linthresh=0.01, vmin=min_val, vmax=max_val, base=10)

    im = plot.imshow(grid, cmap=cmap, vmin=min_val, vmax=max_val, aspect='equal', alpha=alpha, zorder=zorder)    
    
    if show_cbar:
        cb = plt.colorbar(im, ax=plot, fraction=0.046, pad=0.04)
        cb.set_label("Q value", rotation=270, labelpad=15)
        cb.ax.tick_params(labelsize=7)

    # iterating for every row and every col of the env
    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            val = grid[r, c]
            text_color = 'white' if val > mid else 'black'
            # understanding if the current cell is the old goal
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')

            if show_text:
                arrow = arrows[grid_policy[r, c]]
                # text and arrows
                _write_cell_text(plot, c, r, cell, is_old_goal, shifted, text_color=text_color, val=val, arrow=arrow)

            rect_z = max(zorder + 2, 3)
            # borders
            _draw_cell_borders(plot, c, r, cell, is_old_goal, zorder=rect_z)

    # uncomment if want numbers on the side of the grid
    #plot.set_xticks(range(grid_size))
    #plot.set_yticks(range(grid_size))
    #plot.tick_params(labelsize=7)

    plot.set_xticks([])
    plot.set_yticks([])

    # returns max or min to change plot titles according to mode
    return nr

def plot_qvalue_snapshots(agent, path="qvalue_snapshots.svg", grid_size=10):
    """
    Heatmap snapshots of the Q-table at key moments during training

    Without goal shift  (4 panels):
        first_goal - first_goal +10 ep - 5 times_goal - final

    With goal shift  (up to 8 panels):
        first_goal - first_goal +10 ep - at_shift - 5 times_goal
        first_new_goal - first_new_goal +10 ep - final

    Each panel:
        shows Q-val per state with policy arrows overlaid
        highlights the special cells with a border
    """
    heatmap_cmap = plt.colormaps['Blues']

    shifted = agent.shift_happened_ep is not None

    desc = agent.env.unwrapped.desc.astype(str)
    initial_desc = getattr(agent, 'initial_desc', desc)

    snapshots = [(k, agent.q_snapshots[k]) for k in LABEL_MAP.keys() if k in agent.q_snapshots]
    
    if not snapshots:
        print("[plot_qvalue_snapshots] No snapshots to plot - skipping")
        return
    
    # sorting snapshots per episode
    snapshots.sort(key=lambda x: x[1][0])

    n = len(snapshots)
    ncols = min(n, 3) if agent.add_obs_ep is not None else min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False,
                             constrained_layout=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    axes_flat = axes.flatten()

    # uncomment if wanting to use different range
    #if agent.mode in ["std", "relative"]:
    #    q_vals = np.concatenate(
    #            [np.max(qtable, axis=1) for _, (_, qtable, _, _, _) in snapshots]
    #        )
    #else:
    #    q_vals = np.concatenate(
    #            [np.min(qtable, axis=1) for _, (_, qtable, _, _, _) in snapshots]
    #        )
    #
    #vmin, vmax = float(q_vals.min()), float(q_vals.max())

    # imposing range
    vmin, vmax = (0.0, 1.0) if agent.mode in ["std", "relative"] else (-1.0, 0.0)
        
    # drawing snapshots
    for i, (key, (ep, qtable, desc, _, _)) in enumerate(snapshots):
        ax = axes_flat[i]

        nr = _heatmap(agent, grid_size, desc, initial_desc, ax, q_table=qtable, shifted=shifted, vmin=vmin, vmax=vmax, 
                          cmap=heatmap_cmap, show_cbar=False)

        # single subtitles
        label_text = LABEL_MAP.get(key, key)
        ax.set_title(f"{label_text}   |   ep {ep}", fontsize=12, fontweight='bold')

    # main title
    shift_note = (f"  |  goal shift at ep {agent.shift_happened_ep}"
                    if shifted else "")
    fig.suptitle(
        f"{nr} Q-value Snapshots  |  episodes={agent.training_episodes}{shift_note}\n"
        f"mode={agent.mode}      replay_mode={agent.replay_mode}     action_select={agent.action_selection}",
        fontsize=16, fontweight='bold', y=1.10
    )

    # hiding unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # single colorbar
    sm = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=SymLogNorm(linthresh=0.01, vmin=vmin, vmax=vmax, base=10))
    cb = fig.colorbar(sm, ax=axes, location='right', shrink=0.8, aspect=30, pad=0.02)
    cb.set_label("Q value", rotation=270, labelpad=20, fontsize=10, fontweight='bold')
    cb.ax.tick_params(labelsize=7)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot_qvalue_snapshots]         saved to -> {path}")

def plot_replay_trajectories(agent, path="replay_trajectories.svg", grid_size=10, key_moments=True, n_samples=8, only_one=True):
    """
    Plots replay trajectories over the Q-value heatmap
    
    Args:
        key_moments:            bool, if true plots same snapshots as the q-val heatmap, else samples n_samples equally spaced
        n_samples:              nr of sampled episodes
        only_one:               option to plot only one traj out of all the replays per step for visual ease
    """
    heatmap_cmap = plt.colormaps['Greys']
    replay_cmap = plt.colormaps['Blues']

    JITTER_VAL = 0.2

    shifted = agent.shift_happened_ep is not None

    desc = agent.env.unwrapped.desc.astype(str)
    initial_desc = getattr(agent, 'initial_desc', desc)
    
    if key_moments:
        snapshots = [(k, agent.q_snapshots[k]) for k in LABEL_MAP.keys() if k in agent.q_snapshots]
    else:
        snapshots = [("", snap) for snap in agent.eq_snapshots]

    if not snapshots:
        print("[plot_qvalue_snapshots] No snapshots to plot - skipping")
        return
    
    # sorting snapshots per episode
    snapshots.sort(key=lambda x: x[1][0])

    n = len(snapshots)
    ncols = min(n, 3) if agent.add_obs_ep is not None else min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False,
                             constrained_layout=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    axes_flat = axes.flatten()

    # uncomment if wanting to use different range
    #if agent.mode in ["std", "relative"]:
    #    q_vals = np.concatenate(
    #            [np.max(qtable, axis=1) for _, (_, qtable, _, _, _) in snapshots]
    #        )
    #else:
    #    q_vals = np.concatenate(
    #            [np.min(qtable, axis=1) for _, (_, qtable, _, _, _) in snapshots]
    #        )
    #
    #vmin, vmax = float(q_vals.min()), float(q_vals.max())

    # imposing range
    vmin, vmax = (0.0, 1.0) if agent.mode in ["std", "relative"] else (-1.0, 0.0)

    for i, (key, (ep, qtable, desc, batches, agent_path)) in enumerate(snapshots):
        ax = axes_flat[i]

        # bg heatmap
        nr = _heatmap(
                agent, grid_size, desc, initial_desc, ax, q_table=qtable, 
                shifted=shifted, vmin=vmin, vmax=vmax, cmap=heatmap_cmap, alpha=0.8, 
                show_text=False, show_cbar=False, zorder=1
            )
        # adding grid
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color="#A1A1A1", lw=1.0, alpha=0.6, zorder=2)
            ax.axvline(i - 0.5, color='#A1A1A1', lw=1.0, alpha=0.6, zorder=2)

        # drawing real trajectory
        if len(agent_path) > 1:
            px = [s % grid_size for s in agent_path]
            py = [s // grid_size for s in agent_path]
            jx = np.array(px, float) + np.random.uniform(-0.1, 0.1, len(px))
            jy = np.array(py, float) + np.random.uniform(-0.1, 0.1, len(py))
            ax.plot(jx, jy, color="#B848DA", alpha=0.7, linewidth=1.5, linestyle='--', zorder=3)

        # if replay_mode == backward and only_longest=True
        # selects only the longest replay for the episode
        if batches and agent.replay_mode == "backward" and only_one:
            batches = [max(batches, key=len)]
        # else selects the last one
        elif batches and agent.replay_mode in ["dyna", "prioritized_sweeping", "value_iteration"] and only_one:
                batches = [batches[-1]]
        
        # computing (x, y) coords and adding visual jitter
        node_x = {st: (st % grid_size) + np.random.uniform(-JITTER_VAL, JITTER_VAL) for st in range(agent.state_space)}
        node_y = {st: (st // grid_size) + np.random.uniform(-JITTER_VAL, JITTER_VAL) for st in range(agent.state_space)}
        
        for batch in batches:
            b_len = len(batch)
            if b_len == 0: continue

            if agent.replay_mode in ["dyna", "value_iteration"]:
                for i, (s, ns) in enumerate(batch):
                    t = i / max(b_len - 1, 1) 
                    dyna_color = replay_cmap(t)
                    sx, sy = node_x[s], node_y[s]
                    nsx, nsy = node_x[ns], node_y[ns]
                    
                    if s == ns: 
                        ax.scatter(sx, sy, s=30, color=dyna_color, alpha=0.8, zorder=5)
                    else:
                        ax.annotate('', xy=(nsx, nsy), xytext=(sx, sy),
                                    arrowprops=dict(arrowstyle="-|>", color=dyna_color, lw=1.8, mutation_scale=9, shrinkA=0, shrinkB=0),
                                    zorder=4)
            else:
                first_s = batch[0][0]
                ax.scatter(node_x[first_s], node_y[first_s], marker='x', s=60, color=replay_cmap(0.0), alpha=1.0, zorder=7, linewidths=2.5)
                lines, line_colors = [], []
                static_x, static_y, static_c = [], [], []
                end_x, end_y, end_c = [], [], []
                
                for i, (s, ns) in enumerate(batch):
                    t = i / max(b_len - 1, 1) 
                    color = replay_cmap(t)
                    sx, sy = node_x[s], node_y[s]
                    nsx, nsy = node_x[ns], node_y[ns]

                    if s == ns: 
                        static_x.append(sx)
                        static_y.append(sy)
                        static_c.append(color)
                    else:
                        lines.append([(sx, sy), (nsx, nsy)])
                        line_colors.append(color)
                        end_x.append(nsx)
                        end_y.append(nsy)
                        end_c.append(color)
                
                if static_x: ax.scatter(static_x, static_y, s=30, c=static_c, alpha=0.9, zorder=5)
                if lines:
                    lc = LineCollection(lines, colors=line_colors, alpha=0.8, linewidths=3.0, capstyle='round', zorder=4)
                    ax.add_collection(lc)
                if end_x: ax.scatter(end_x, end_y, s=15, c=end_c, alpha=1.0, zorder=5)
    
        # single subtitles
        if key_moments:
            label_text = LABEL_MAP.get(key, key)
            ax.set_title(f"{label_text}   |   ep {ep}", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"Ep {ep}", fontsize=12, fontweight='bold')

    # main title
    shift_note = (f"  |  goal shift at ep {agent.shift_happened_ep}"
                    if shifted else "")
    fig.suptitle(
        f"{nr} Q-value Snapshots  |  episodes={agent.training_episodes}{shift_note}\n"
        f"mode={agent.mode}      replay_mode={agent.replay_mode}     action_select={agent.action_selection}",
        fontsize=16, fontweight='bold', y=1.10
    )

    # hiding unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # colorbars
    sm_q = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=SymLogNorm(linthresh=0.01, vmin=vmin, vmax=vmax, base=10))
    cb_q = fig.colorbar(sm_q, ax=axes, location='left', shrink=0.8, aspect=30, pad=0.02)
    cb_q.set_label("Background: Q-Value", rotation=90, labelpad=20, fontsize=10, fontweight='bold')
    cb_q.ax.tick_params(labelsize=10)

    sm_r = plt.cm.ScalarMappable(cmap=replay_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cb_r = fig.colorbar(sm_r, ax=axes, location='right', shrink=0.8, aspect=30, pad=0.02)
    cb_r.set_label("Replay Sequence (timesteps)", rotation=270, labelpad=20, fontsize=10, fontweight='bold')
    cb_r.set_ticks([0, 1])
    cb_r.set_ticklabels(['Start', 'End'])
    cb_r.ax.tick_params(labelsize=7)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot_replay_trajectories]      saved to -> {path}")

def plot_metrics(results_dict, shift_ep=None, path="training_metrics.svg"):
    from collections import defaultdict
    
    # Grouping data by label: "update_mode + replay_mode"
    groups = defaultdict(dict)
    for label, data in results_dict.items():
        if " + " in label:
            u_mode, r_mode = label.split(" + ")
            groups[r_mode][u_mode] = data
        else:
            groups["all"][label] = data 

    num_groups = len(groups)
    cols = 2
    rows = num_groups

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 5))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # creating std color palette
    all_u_modes = sorted({u for g in groups.values() for u in g.keys()})
    colors = plt.cm.tab10.colors[:len(all_u_modes)]
    color_map = dict(zip(all_u_modes, colors))

    # drawing plots by line
    for i, (r_mode, group_data) in enumerate(groups.items()):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        
        for u_mode, data in group_data.items():
            color = color_map[u_mode]
            
            # subplot 1
            cum_rewards = np.cumsum(data["rewards"], axis=1)
            mean_cum_rew = np.mean(cum_rewards, axis=0)
            std_cum_rew = np.std(cum_rewards, axis=0)
            episodes = np.arange(len(mean_cum_rew))
            
            ax1.plot(episodes, mean_cum_rew, label=u_mode, color=color, linewidth=2)
            ax1.fill_between(episodes, mean_cum_rew - std_cum_rew, mean_cum_rew + std_cum_rew, color=color, alpha=0.2)
            
            # subplot 2
            times = data["times"]
            mean_times = np.mean(times, axis=0)
            std_times = np.std(times, axis=0)
            
            ax2.plot(episodes, mean_times, label=u_mode, color=color, linewidth=2)
            ax2.fill_between(episodes, mean_times - std_times, mean_times + std_times, color=color, alpha=0.2)

        # subplot reward
        ax1.set_title(f"Cumulative Reward  |  Replay: {r_mode}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Episodes", fontsize=12)
        ax1.set_ylabel("Cumulative Reward", fontsize=12)
        ax1.grid(True, alpha=0.3)

        # subplot time
        ax2.set_title(f"Elapsed Time  |  Replay: {r_mode}", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Episodes", fontsize=12)
        ax2.set_ylabel("Seconds", fontsize=12)
        ax2.grid(True, alpha=0.3)

        if shift_ep is not None:
            ax1.axvline(shift_ep, color='black', linestyle='--', linewidth=1.5, zorder=10)
            ax2.axvline(shift_ep, color='black', linestyle='--', linewidth=1.5, zorder=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    #fig.subplots_adjust(bottom=0.05, hspace=0.35, wspace=0.15)
    
    # global labels
    fig.legend(handles, labels, loc='lower center', ncol=len(all_u_modes), bbox_to_anchor=(0.5, 0.01), fontsize=12)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot_metrics]                  saved to -> {path}")


def plot_exploration_stats(results_dict, shift_ep=None, path="exploration_stats.svg"):

    num_configs = len(results_dict)
    cols = 5
    rows = (num_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(40, rows * 5))
    axes = np.atleast_1d(axes).flatten()

    for i, (label, data) in enumerate(results_dict.items()):
        ax = axes[i]
        
        # comuting means
        m_phys_norm = np.mean(data["phys_norm"], axis=0)
        m_phys_term = np.mean(data["phys_term"], axis=0)
        m_rep_norm  = np.mean(data["rep_norm"], axis=0)
        m_rep_term  = np.mean(data["rep_term"], axis=0)
        
        episodes = np.arange(len(m_phys_norm))
        
        # stackplot
        ax.stackplot(episodes, m_phys_norm, m_phys_term, m_rep_norm, m_rep_term,
                     labels=['Physical: Std Steps', 'Physical: Goal/Punish', 'Replay: Std Steps', 'Replay: Goal/Punish'],
                     colors=['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e'], alpha=0.85)
        
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_ylabel("Steps", fontsize=10)
        ax.grid(True, alpha=0.3)
        if shift_ep is not None:
            ax.axvline(shift_ep, color='gray', linestyle='--', linewidth=1.5, zorder=10)
        
    # hiding unused spots
    for j in range(num_configs, len(axes)):
        axes[j].set_visible(False)
        
    # global lables
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.2)
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=12)
    
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot_exploration_stats]        saved to -> {path}")

def plot_update_stats(results_dict, shift_ep=None, path="update_stats.svg"):

    num_configs = len(results_dict)
    cols = 5
    rows = (num_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(40, rows * 5))
    axes = np.atleast_1d(axes).flatten()

    for i, (label, data) in enumerate(results_dict.items()):
        ax = axes[i]
        
        # computing means
        m_upd_phys = np.mean(data["upd_phys"], axis=0)
        m_upd_rep  = np.mean(data["upd_rep"], axis=0)
        
        episodes = np.arange(len(m_upd_phys))
        
        # stackplot
        ax.stackplot(episodes, m_upd_phys, m_upd_rep,
                     labels=['Physical Updates', 'Replay Updates'],
                     colors=['#2ca02c', '#1f77b4'], alpha=0.85)
        
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_ylabel("Updated States", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if shift_ep is not None:
            ax.axvline(shift_ep, color='gray', linestyle='--', linewidth=1.5, zorder=10)

    # hiding unused lines
    for j in range(num_configs, len(axes)):
        axes[j].set_visible(False)
        
    # global labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.06, hspace=0.35, wspace=0.15)
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01), fontsize=12)
    
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot_update_stats]             saved to -> {path}")