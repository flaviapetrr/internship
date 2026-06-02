import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm, Normalize

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
    
def _write_cell_text(plot, c, r, cell, is_old_goal, shifted, text_color='black', 
                     pos=None, val=None, arrow=None, zorder=None):

    text_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 12}

    if zorder is not None:
        text_kwargs['zorder'] = zorder

    if pos is not None:
        if cell in ['H', 'S']:
            plot.text(c, r, cell, color='black', fontweight='bold', alpha=0.5, **text_kwargs)
        elif cell == 'G':
            label = 'G2' if shifted else 'G'
            plot.text(c, r, label, color='black', fontweight='bold', alpha=0.8, **text_kwargs)
        elif is_old_goal:
            plot.text(c, r, 'G1', color='gray', fontweight='bold', alpha=0.8, **text_kwargs)
            
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

def plot_qvalue_snapshots(agent, path="qvalue_snapshots.png", grid_size=10):
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
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False,
                             constrained_layout=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    axes_flat = axes.flatten()

    # uncomment if wanting to use different range
    #if agent.update_mode in ["std", "relative"]:
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
    vmin, vmax = (0.0, 1.0) if agent.update_mode in ["std", "relative"] else (-1.0, 0.0)
        
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
        f"{nr} Q-value Snapshots  |  mode={agent.mode}  alpha={agent.alpha}"
        f"  episodes={agent.training_episodes}{shift_note}",
        fontsize=16, fontweight='bold', y=1.05
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

def plot_replay_trajectories(agent, path="replay_trajectories.png", grid_size=10, n_samples=8, only_one=True):
    """
    Plots replay trajectories over the Q-value heatmap
    
    Args:
        n_samples:              nr of sampled episodes
        only_one:               option to plot only one traj out of all the replays per step for visual ease
    """
    heatmap_cmap = plt.colormaps['Greys']
    replay_cmap = plt.colormaps['Blues']

    JITTER_VAL = 0.2

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
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False,
                             constrained_layout=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    axes_flat = axes.flatten()

     # uncomment if wanting to use different range
    #if agent.update_mode in ["std", "relative"]:
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
    vmin, vmax = (0.0, 1.0) if agent.update_mode in ["std", "relative"] else (-1.0, 0.0)

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
        elif agent.replay_mode in ["dyna", "prioritized_sweeping"] and only_one:
                batches = [batches[-1]]
        
        # computing (x, y) coords and adding visual jitter
        node_x = {st: (st % grid_size) + np.random.uniform(-JITTER_VAL, JITTER_VAL) for st in range(agent.state_space)}
        node_y = {st: (st // grid_size) + np.random.uniform(-JITTER_VAL, JITTER_VAL) for st in range(agent.state_space)}
        
        for batch in batches:
            b_len = len(batch)
            if b_len == 0: continue

            if agent.replay_mode == "dyna":
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
        label_text = LABEL_MAP.get(key, key)
        ax.set_title(f"{label_text}   |   ep {ep}", fontsize=12, fontweight='bold')

    # main title
    shift_note = (f"  |  goal shift at ep {agent.shift_happened_ep}"
                    if shifted else "")
    fig.suptitle(
        f"{nr} Q-value Snapshots  |  mode={agent.mode}  alpha={agent.alpha}"
        f"  episodes={agent.training_episodes}{shift_note}",
        fontsize=16, fontweight='bold', y=1.05
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