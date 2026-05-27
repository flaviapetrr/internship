import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym

BG_COLORS = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd', 'G1': '#cdebff', 'G2': '#ffd5cd'}

def _moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')

def _draw_cell_borders(plot, c, r, cell, is_old_goal, zorder):
    
    if cell == 'G':
        plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, zorder=zorder))
    elif is_old_goal:
        plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='steelblue', linestyle='--', linewidth=2.5, zorder=zorder))
    elif cell == 'S':
        plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2.5, zorder=zorder))
    
def _write_cell_text(plot, c, r, cell, is_old_goal, shifted, text_color='black', val=None, arrow=None):

    text_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 12}

    if cell in ['H', 'S']:
        plot.text(c, r, cell, color='black', alpha=0.5, **text_kwargs)
    elif cell == 'G':
        label = 'G2' if shifted else 'G'
        plot.text(c, r, label, color='black', fontweight='bold', alpha=0.8, **text_kwargs)
    elif is_old_goal:
        plot.text(c, r, 'G1', color='gray', fontweight='bold', alpha=0.8, **text_kwargs)
        
    if arrow is not None:
        plot.text(c, r - 0.18, arrow, ha='center', va='center', fontsize=9, color=text_color)
    if val is not None:
        plot.text(c, r + 0.22, f"{val:.3f}", ha='center', va='center', fontsize=7.5, color=text_color)
        
def _draw_grid_bg(plot, grid_size, desc, initial_desc, bg, shifted):
    """
    Draw cell backgrounds + S/G/H labels
    
    Args:
        plot:           obj in which we are drawing           
        grid_size:      grid dimension, std = 10x10
        desc:           str matrix that describes current env state ('S', 'F', 'G')
        initial_desc:   str matrix that describes env state before a possible goal shift ('S', 'F', 'G1', 'G2')
        bg:             colors dictionary
        shifted:        bool to tell if there is a goal shift
    """

    # iterating for every row and every col of the env
    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            # understanding if the current cell is the old goal
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')
            bg_color = bg.get('G1') if is_old_goal else bg.get(cell, 'F')
            # adding backgroung color on the cell based on type
            plot.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg_color, zorder=0))
            # text
            _write_cell_text(plot, c, r, cell, is_old_goal, shifted)
            
    # drawing drid
    plot.grid(True, color='gray', alpha=0.3, zorder=1)

def _heatmap(agent, grid_size, desc, initial_desc, plot, shifted, q_table=None, vmin=None, vmax=None, cmap="plasma", alpha=1.0, show_text=True, show_cbar=True, zorder=1):
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
    mid = (vmin + vmax) / 2

    policy = np.argmax(q_table, axis=1)
    # reshaping policy from 1D (grid_size*grid_size) to 2D grid_sizexgrid_size
    grid_policy = policy.reshape(grid_size, grid_size)

    im = plot.imshow(grid, cmap=cmap, vmin=min_val, vmax=max_val, aspect='equal', alpha=alpha, zorder=zorder)    
    
    if show_cbar:
        cb = plt.colorbar(im, ax=plot, fraction=0.046, pad=0.04)
        cb.set_label("Q value", rotation=270, labelpad=15)
        cb.ax.tick_params(labelsize=7)

    # iterating for every row and every col of the env
    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            val  = grid[r, c]
            text_color = 'white' if val < mid else 'black'
            # understanding if the current cell is the old goal
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')

            if show_text:
                arrow = arrows[grid_policy[r, c]]
                # text and arrows
                _write_cell_text(plot, c, r, cell, is_old_goal, shifted, text_color=text_color, val=val, arrow=arrow)

            rect_z = max(zorder + 2, 3)
            # borders
            _draw_cell_borders(plot, c, r, cell, is_old_goal, zorder=rect_z)

    # putting numbers on the side of the grid
    plot.set_xticks(range(grid_size))
    plot.set_yticks(range(grid_size))
    plot.tick_params(labelsize=8)

    # returns max or min to change plot titles according to mode
    return nr

def _draw_traj_panel(fig, plot, grid_size, desc, initial_desc, bg, shifted, sampled_paths, ep_min, ep_max, cmap='plasma', linestyle='-', show_cbar=True):
    """
    Draw one trajectory panel
    Args:
        plot:           obj in which we are drawing
        grid_size:      grid dimension, std = 10x10
        desc:           str matrix that describes current env state ('S', 'F', 'G')
        initial_desc:   str matrix that describes env state before a possible goal shift ('S', 'F', 'G1', 'G2')
        shifted:        bool to tell if there is a goal shift
        sampled_paths:  subset of agent.sampled_paths, tuple containing (ep_number, visited_states_list)
        ep_min/ep_max:  temporal interval for colour range

        the others are optional (kwargs), for visualization purposes   
    """

    # seting up
    plot.set_xlim(-0.5, grid_size - 0.5)
    plot.set_ylim(-0.5, grid_size - 0.5)
    plot.set_xticks(range(grid_size))
    plot.set_yticks(range(grid_size))
    plot.set_aspect('equal')
    # to make y start from the top, like the env grid
    plot.invert_yaxis()
 
    # drawing bg and grid
    _draw_grid_bg(plot, grid_size, desc, initial_desc, bg, shifted)

    # setting up the colormap
    # accepts bot gradients and solid colors
    try:
        cmap_obj = plt.colormaps[cmap]
        use_cmap = True
        norm = plt.Normalize(vmin=ep_min, vmax=ep_max)
    except KeyError:
        use_cmap = False
        color = cmap
 
    # iterating for every trajectory
    for eps_num, path_states in sampled_paths:

        if use_cmap:
            t = norm(eps_num) 
            color = cmap_obj(t)
        
        # getting 2D coords (x, y) from 1D std gym position
        path_x = [s % grid_size for s in path_states]
        path_y = [s // grid_size for s in path_states]
        # adding jitter for visual purposes
        jx = np.array(path_x) + np.random.uniform(-0.15, 0.15, size=len(path_x))
        jy = np.array(path_y) + np.random.uniform(-0.15, 0.15, size=len(path_y))
        # drawing line
        plot.plot(jx, jy, color=color, alpha=0.7, linewidth=1.5, linestyle=linestyle, zorder=2)
 
    if show_cbar and use_cmap:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        cbar = fig.colorbar(sm, ax=plot, fraction=0.046, pad=0.04)
        cbar.set_label('Episode', rotation=270, labelpad=15)
        return cbar
 
def _draw_visits_panel(plot, grid_size, desc, initial_desc, shifted, visits, vmax=None):
    """
    Draw state-visit heatmap
    Args:
        plot:           obj in which we are drawing
        grid_size:      grid dimension, std = 10x10
        desc:           str matrix that describes current env state ('S', 'F', 'G')
        initial_desc:   str matrix that describes env state before a possible goal shift ('S', 'F', 'G1', 'G2')
        shifted:        bool to tell if there is a goal shift
        visits:         1D visits list

        the others are optional (kwargs), for visualization purposes 
    """
    # reshaping visits from 1D (grid_size*grid_size) to 2D grid_sizexgrid_size
    grid_visits = visits.reshape(grid_size, grid_size)

    # using global threshold if given
    max_val = np.max(grid_visits) if vmax is None else vmax
    if max_val == 0: max_val = 1

    im = plot.imshow(grid_visits, cmap='plasma', aspect='equal', vmin=0, vmax=max_val)
    cb = plt.colorbar(im, ax=plot, fraction=0.046, pad=0.04)
    cb.set_label('visit count', rotation=270, labelpad=15)
    cb.ax.tick_params(labelsize=7)
 
    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            val = int(grid_visits[r, c])
            text_color = "white" if val < (max_val / 2) else "black"
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')
            plot.text(c, r, str(val), ha='center', va='center', fontsize=9, color=text_color)

            _draw_cell_borders(plot, c, r, cell, is_old_goal, zorder=3)
    plot.set_xticks(range(grid_size));
    plot.set_yticks(range(grid_size))
    plot.tick_params(labelsize=8)

def _get_replay_batches(transitions):
    """Encapsule untied transitions in countinous ones"""
    batches, current_batch = [], []
    for s, ns in transitions:
        if not current_batch:
            current_batch.append((s, ns))
        else:
            prev_s, prev_ns = current_batch[-1]
            if s == prev_ns or ns == prev_s:
                current_batch.append((s, ns))
            else:
                batches.append(current_batch)
                current_batch = [(s, ns)]
    if current_batch:
        batches.append(current_batch)
    return batches

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
    # rewards and lengths populated in agent.train()
    # now translated in np arrays
    rewards = np.array(agent.episode_rewards)
    lengths = np.array(agent.episode_lengths)
    # computing 5% window wrt tot episodes
    window = max(1, len(rewards) // 20)

    desc = agent.env.unwrapped.desc.astype(str)
    # getting initial_desc attribute from obj agent
    # if it does not exists use desc
    initial_desc = getattr(agent, 'initial_desc', desc)
    shifted = getattr(agent, 'shift_happened_ep', None) is not None

    ncols = 2 if shifted else 3
    nrows = 2 if shifted else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()

    if shifted:
        ax_rew, ax_snap, ax_len, ax_final = axes
    else:
        ax_rew, ax_len, ax_final = axes

    shift_note = (f"  |  goal shift at ep {agent.shift_happened_ep}" if shifted else "")
    fig.suptitle(
        f"Training Summary  |  mode={agent.mode}  alpha={agent.alpha}  episodes={agent.training_episodes}  q_table initial values={agent.q_init}{shift_note}",
        fontsize=13, fontweight='bold'
    )

    # getting length of arrays for axis length
    eps_x = np.arange(len(rewards))
    # ma_x starts late since it does not have enough info to compute info before
    ma_x  = np.arange(window - 1, len(rewards))
    
    # subplot 1
    ylim = (-0.2, 1.2) if agent.mode in ["std", "relative"] else (-1.2, 0.2)
    ax_rew.set_ylim(ylim)
    ax_rew.set_xlim(0, agent.training_episodes * 1.05)
    ax_rew.plot(eps_x, rewards, alpha=0.35, color='steelblue', linewidth=0.8, label='Reward')
    ax_rew.plot(ma_x, _moving_avg(rewards, window), color='steelblue', linewidth=2, label=f'avg ({window})')
    # drawing line where the goal shift occurred
    if shifted:
        ax_rew.axvline(agent.shift_happened_ep, color='gray', linestyle='--', linewidth=0.8, label=f'Goal shift: ep {agent.shift_happened_ep}')
    ax_rew.set_title("Reward per Episode")
    ax_rew.set_xlabel("Episode")
    ax_rew.set_ylabel("Reward")
    ax_rew.legend(fontsize=8)
    ax_rew.grid(True, alpha=0.3)

    # subplot 2
    ax_len.set_ylim(0, agent.episode_steps * 1.05)
    ax_len.set_xlim(0, agent.training_episodes * 1.05)
    ax_len.plot(eps_x, lengths, alpha=0.35, color='darkorange', linewidth=0.8, label='Length')
    ax_len.plot(ma_x, _moving_avg(lengths, window), color='darkorange', linewidth=2, label=f'avg ({window})')
    if shifted:
        ax_len.axvline(agent.shift_happened_ep, color='gray', linestyle='--', linewidth=0.8, label=f'Goal shift: ep {agent.shift_happened_ep}')
    ax_len.set_title("Episode Length")
    ax_len.set_xlabel("Episode")
    ax_len.set_ylabel("Steps")
    ax_len.legend(fontsize=8)
    ax_len.grid(True, alpha=0.3)

    # subplot 3 (and 4)
    if shifted:
        if "at_shift" in agent.q_snapshots:
            ep, q_snap, desc_snap = agent.q_snapshots["at_shift"]
            nr = _heatmap(agent, grid_size, desc_snap, initial_desc, ax_snap, q_table=q_snap, shifted=shifted)
            ax_snap.set_title(f"Q-value ({nr} Q)    |   At Shift (ep {ep})")

        else:
            nr = _heatmap(agent, grid_size, desc, initial_desc, ax_snap, q_table=agent.q_table, shifted=shifted)
            ax_snap.set_title(f"First Goal Q-value Heatmap  |   {nr} Q = policy arrows")
        
    nr_final = _heatmap(agent, grid_size, desc, initial_desc, ax_final, q_table=agent.q_table, shifted=shifted)
    ax_final.set_title(f"Final Q-value Heatmap    |   {nr_final} Q + policy arrows")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    print(f"Training Summary plot saved to {path}")

def plot_replay_analysis(agent, path="replay_analysis.png", window=50):
    """
    subplots:
        1. moving avg mean success rate
        2. mean TD error per episode
        3. model coverage 
    """
    n = len(agent.episode_rewards)
    w = max(1, min(window, n // 5))
    eps_x = np.arange(n)
    shifted = getattr(agent, 'shift_happened_ep', None) is not None

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
    ax.plot(roll_x, _moving_avg(success, w) * 100,
            color='steelblue', linewidth=2, label="Success rate")
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
            label=f'100% success rate')
    if shifted:
        ax.axvline(agent.shift_happened_ep, color='gray', linestyle='--', linewidth=0.8,
            label=f'Goal shift: ep {agent.shift_happened_ep}')
    ax.set_title(f"Moving Avg Success Rate  (window={w})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # subplot 2
    ax = axes[1]
    td = np.array(agent.episode_td_errors)
    ax.plot(eps_x, td, alpha=0.3, color='darkorange', linewidth=0.7, label=f'TD error')
    ax.plot(np.arange(w - 1, n), _moving_avg(td, w),
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
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Trajectories | episodes={num_episodes} epsilon={epsilon_test}", fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_aspect('equal')
        ax.invert_yaxis()

        ax.grid(True, color='#888888', alpha=0.3, zorder=1)

        # getting and plotting trajectories
        sampled_paths = []
        for ep in range(num_episodes):
            state, _ = test_env.reset()
            terminated = truncated = False
            
            path_states = [state]
            
            for _ in range(agent.episode_steps):
                if terminated or truncated:
                    break
                
                if np.random.rand() < epsilon_test:
                    action = test_env.action_space.sample()
                else:
                    action = int(np.argmax(agent.q_table[state]))
                    
                state, _, terminated, truncated, _ = test_env.step(action)
                path_states.append(state)
                
            sampled_paths.append((ep, path_states))

        # 
        _draw_traj_panel(
            fig, ax, grid_size, desc, desc, BG_COLORS, 
            shifted=False, 
            sampled_paths=sampled_paths, 
            ep_min=0, ep_max=num_episodes
        )
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

        nr = _heatmap(agent, grid_size, desc, initial_desc, ax, q_table=qtable, shifted=shifted, vmin=vmin, vmax=vmax)

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

    shift_note = f" | goal shift at ep {shift_ep}"
    title = (f"Training Evolution   |   mode={agent.mode} replay={agent.replay_mode}    |   episodes={agent.training_episodes}{shift_note}")
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if shifted:
        ax = axes[0, 0]
        _draw_traj_panel(fig, ax, grid_size, initial_desc, initial_desc, BG_COLORS, False, sampled_paths=paths_before, ep_min=0, ep_max=agent.training_episodes)
        ax.set_title(f"Trajectories before shift", fontsize=12)

        ax = axes[0, 1]
        if "at_shift" in agent.q_snapshots:
            ep_s, q_s, desc_s = agent.q_snapshots["at_shift"]
            nr = _heatmap(agent, grid_size, desc_s, initial_desc, ax, q_table=q_s, shifted=False, vmin=q_min, vmax=q_max)
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
        nr = _heatmap(agent, grid_size, desc, initial_desc, ax, q_table=agent.q_table, shifted=True, vmin=q_min, vmax=q_max)
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
    
def plot_replay_trajectories(agent, path="replay_trajectories.png", grid_size=10, n_sample=8, only_longest_replay=True):
    """
    Plots replay trajectories over the Q-value heatmap
    
    Args:
        only_longest_replay:    option to plot only one traj out of all the replays per step for visual ease
    """

    # checking if replays exist
    if getattr(agent, 'replay_paths', None) is None or not agent.replay_paths:
        print("[plot_replay_trajectories] No replay paths stored - skipping")
        return

    # setup
    shifted = agent.shift_happened_ep is not None
    desc = agent.env.unwrapped.desc.astype(str)
    initial_desc = getattr(agent, 'initial_desc', desc)

    # getting all episodes stored
    all_eps = [data[0] for data in agent.replay_paths]
    # selecting n_sample episodes equally distributed
    indices = np.round(np.linspace(0, len(all_eps) - 1, min(n_sample, len(all_eps)))).astype(int)
    sampled = [agent.replay_paths[i] for i in indices]
    # creating dictionary to find the real agent trajectory in the selected episodes
    agent_path_dict = {ep: path for ep, path in agent.sampled_paths}

    n = len(sampled)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False, constrained_layout=True)

    heatmap_cmap = plt.colormaps['Greys']
    replay_cmap = plt.colormaps['plasma']
    
    shift_note = f" | goal shift at ep {agent.shift_happened_ep}" if shifted else ""
    title = (f"Replay Trajectories   |   mode={agent.mode} replay={agent.replay_mode}    |   episodes={agent.training_episodes}{shift_note}")
    fig.suptitle(title, fontsize=14, fontweight='bold')

    JITTER_VAL = 0.2

    # computing global range for heatmap
    sampled_qtables = [data[2] for data in sampled if len(data) > 2]
    
    if agent.mode in ["std", "relative"]:
        all_q_vals = np.concatenate([np.max(qt, axis=1) for qt in sampled_qtables])
    else:
        all_q_vals = np.concatenate([np.min(qt, axis=1) for qt in sampled_qtables])
        
    q_vmin, q_vmax = float(all_q_vals.min()), float(all_q_vals.max())
    if q_vmin == q_vmax: q_vmax += 1e-5

    for idx, data in enumerate(sampled):
        eps_num = data[0]
        replay_trans = data[1]
        q_table_snap = data[2] if len(data) > 2 else agent.q_table

        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        
        panel_desc = initial_desc if (shifted and eps_num <= agent.shift_happened_ep) else desc
        is_panel_shifted = (shifted and eps_num > agent.shift_happened_ep)

        # heatmap
        _heatmap(
            agent, grid_size, panel_desc, initial_desc, ax, q_table=q_table_snap, 
            shifted=is_panel_shifted, min=q_vmin, max=q_vmax, cmap=heatmap_cmap, alpha=0.8, 
            show_text=False, show_cbar=False, zorder=1
        )

        # drawing grid
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color='white', lw=1.0, alpha=0.6, zorder=2)
            ax.axvline(i - 0.5, color='white', lw=1.0, alpha=0.6, zorder=2)

        # real trajectory
        agent_path = agent_path_dict.get(eps_num, [])
        if len(agent_path) > 1:
            px = [s % grid_size for s in agent_path]
            py = [s // grid_size for s in agent_path]
            jx = np.array(px, float) + np.random.uniform(-0.1, 0.1, len(px))
            jy = np.array(py, float) + np.random.uniform(-0.1, 0.1, len(py))
            ax.plot(jx, jy, color='#888888', alpha=0.9, linewidth=1.5, linestyle='--', zorder=3)

        # replay trajectory
        batches = _get_replay_batches(replay_trans)

        # checking if drawing all replays or only the longest one
        if batches:
            batches = [max(batches, key=len)]
            
        # genetrating coords (x, y) + jitter
        node_x = {st: (st % grid_size) + np.random.uniform(-JITTER_VAL, JITTER_VAL) for st in range(agent.state_space)}
        node_y = {st: (st // grid_size) + np.random.uniform(-JITTER_VAL, JITTER_VAL) for st in range(agent.state_space)}
        
        for batch in batches:
            b_len = len(batch)
            if b_len == 0: continue

            first_s = batch[0][0]
            # X di partenza
            ax.scatter(node_x[first_s], node_y[first_s], marker='x', s=45, color=replay_cmap(0.0), alpha=1.0, zorder=6, linewidths=2.5)
            
            for i, (s, ns) in enumerate(batch):
                t = i / max(b_len - 1, 1) 
                color = replay_cmap(t)
                sx, sy = node_x[s], node_y[s]
                nsx, nsy = node_x[ns], node_y[ns]

                if s == ns: 
                    ax.scatter(sx, sy, s=20, color=color, alpha=0.9, zorder=4)
                else:
                    ax.plot([sx, nsx], [sy, nsy], color=color, alpha=0.8, linewidth=2.5, solid_capstyle='round', zorder=5)

        ax.set_title(f"Episode {eps_num}", fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    # colorbars

    sm_q = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=plt.Normalize(vmin=q_vmin, vmax=q_vmax))
    cb_q = fig.colorbar(sm_q, ax=axes, location='left', fraction=0.03, pad=0.04)
    cb_q.set_label("Background: Q-Value", rotation=90, labelpad=20, fontsize=12, fontweight='bold')

    sm_r = plt.cm.ScalarMappable(cmap=replay_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cb_r = fig.colorbar(sm_r, ax=axes, location='right', fraction=0.03, pad=0.04)
    cb_r.set_label("Replay Sequence (timesteps)", rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    cb_r.set_ticks([0, 1])
    cb_r.set_ticklabels(['Start', 'End'])

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Replay trajectories plot saved to {path}")