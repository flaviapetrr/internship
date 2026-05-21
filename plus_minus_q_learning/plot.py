import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
import imageio
import io

def plot_training(agent, path="training_summary.png", grid_size=10):
    """
    subplots:
        1. reward per episode + moving average
        2. episode length + moving average
        3. Q-value heatmap (max Q per state + policy arrows)
    """
    rewards = np.array(agent.episode_rewards)
    lengths = np.array(agent.episode_lengths)
    window  = max(1, len(rewards) // 20)

    def moving_avg(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    desc = agent.env.unwrapped.desc.astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    fig.suptitle(
        f"Training Summary  |  mode={agent.mode}  alpha={agent.alpha}  episodes={agent.training_episodes}  q_table initial values={agent.q_init}",
        fontsize=13, fontweight='bold'
    )

    eps_x = np.arange(len(rewards))
    ma_x  = np.arange(window - 1, len(rewards))

    initial_desc = getattr(agent, 'initial_desc', desc)
    shifted = getattr(agent, 'shift_happened_ep', None) is not None

    # subplot 1
    ax = axes[0]
    ylim = (-0.2, 1.2) if agent.mode in ["std", "relative"] else (-1.2, 0.2)
    ax.set_ylim(ylim)
    ax.set_xlim(0, agent.training_episodes * 1.05)
    ax.plot(eps_x, rewards, alpha=0.35, color='steelblue', linewidth=0.8, label='Reward')
    ax.plot(ma_x, moving_avg(rewards, window), color='steelblue', linewidth=2,
            label=f'AVG ({window})')
    ax.set_title("Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # subplot 2
    ax = axes[1]
    ax.set_ylim(0, agent.episode_steps * 1.05)
    ax.set_xlim(0, agent.training_episodes * 1.05)
    ax.plot(eps_x, lengths, alpha=0.35, color='darkorange', linewidth=0.8, label='Length')
    ax.plot(ma_x, moving_avg(lengths, window), color='darkorange', linewidth=2,
            label=f'AVG ({window})')
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # subplot 3
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    policy = np.argmax(agent.q_table, axis=1)
    grid_policy = policy.reshape(grid_size, grid_size)

    ax = axes[2]
    q_vals = np.max(agent.q_table, axis=1)
    grid_q = q_vals.reshape(grid_size, grid_size)
    desc_q = agent.env.unwrapped.desc.astype(str)
    vmin, vmax = float(q_vals.min()), float(q_vals.max())
    mid = (vmin + vmax) / 2

    im = ax.imshow(grid_q, cmap="plasma", vmin=vmin, vmax=vmax, aspect='equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc_q[r, c]
            val  = grid_q[r, c]
            text_color = 'white' if val < mid else 'black'
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')

            ax.text(c, r - 0.18, arrows[grid_policy[r, c]],
                    ha='center', va='center', fontsize=9, color=text_color)

            ax.text(c, r + 0.22, f"{val:.3f}",
                    ha='center', va='center', fontsize=7.5, color=text_color)

            if cell == 'G':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, zorder=3))
            elif is_old_goal:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='steelblue', linestyle='--', linewidth=2.5, zorder=3))
            elif cell == 'S':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2.5, zorder=3))

    ax.set_title("Final Q-value Heatmap  (max Q per state + policy arrows)")
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    print(f"Plot saved to {path}")

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

    def rolling(x, w):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
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
            color='steelblue', linewidth=2)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(f"Rolling Success Rate  (window={w})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success %")
    ax.grid(True, alpha=0.3)

    # subplot 2
    ax = axes[1]
    td = np.array(agent.episode_td_errors)
    ax.plot(eps_x, td, alpha=0.3, color='darkorange', linewidth=0.7)
    ax.plot(np.arange(w - 1, n), rolling(td, w),
            color='tomato', linewidth=2, label=f'Moving avg ({w})')
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
    ax.set_title("Model Coverage over Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("# (s, a) pairs known")
    ax.set_ylim(0, max_pairs * 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    print(f"Replay plot saved to {path}")

def record_gif(agent, env_id, max_steps, env_kwargs, path="agent_episode.gif", fps=4):
    """run one greedy episode"""
    render_env = gym.make(env_id, max_episode_steps=max_steps, render_mode="rgb_array", **env_kwargs)
    state, _ = render_env.reset()
    frames = [render_env.render()]

    terminated = truncated = False
    for _ in range(agent.episode_steps):
        if terminated or truncated:
            break
        # greedy policy for the recording
        action = int(np.argmax(agent.q_table[state]))
        state, _, terminated, truncated, _ = render_env.step(action)
        frames.append(render_env.render())

    render_env.close()
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"GIF saved to {path} ({len(frames)} frames)")

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
        
        # background
        bg = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd'}
        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc[r, c]
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg.get(cell, '#f8f9fa'), zorder=0)
                ax.add_patch(rect)
                if cell in ['H', 'G', 'S']:
                    ax.text(c, r, cell, ha='center', va='center', fontsize=16, color='black', alpha=0.5)

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
        print(f"Trajectories plot saved to {path}")
        test_env.close()

def plot_qvalue_snapshots(agent, path="qvalue_snapshots.png", grid_size=10):
    """
    Heatmap snapshots of the Q-table at key moments during training

    Without goal shift  (3 panels):
        first_goal - first_goal +10 ep - final

    With goal shift  (up to 6 panels):
        first_goal - first_goal +10 ep - at_shift
        first_new_goal - first_new_goal +10 ep - final

    Each panel:
        shows max-Q per state with policy arrows overlaid
        highlights the goal cell with a border.
    """
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    CMAP = 'coolwarm'
    vmin, vmax = -1.0, 1.0

    LABEL_MAP = {
        "first_goal":        "First Goal Reached",
        "first_goal_p10":    "+10 ep After First Goal",
        "at_shift":          "At Reward / Goal Shift",
        "first_new_goal":    "First New Goal Reached",
        "first_new_goal_p10": "+10 ep After New Goal",
        "final":             "Final (end of training)",
    }

    shifted = agent.shift_happened_ep is not None

    desc = agent.env.unwrapped.desc.astype(str)
    initial_desc = getattr(agent, 'initial_desc', desc)

    if shifted:
        key_order = ["first_goal", "first_goal_p10",
                        "at_shift",
                        "first_new_goal", "first_new_goal_p10",
                        "final"]
    else:
        key_order = ["first_goal", "first_goal_p10", "final"]

    snapshots = [(k, agent.q_snapshots[k])
                    for k in key_order if k in agent.q_snapshots]

    if not snapshots:
        print("[plot_qvalue_snapshots] No snapshots to plot - skipping.")
        return

    n = len(snapshots)
    # Layout: up to 3 columns
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                                figsize=(ncols * 5.2, nrows * 5.8),
                                squeeze=False)
    axes_flat = axes.flatten()

    # Shared colour scale across all snapshots
    all_q_max = np.concatenate(
        [np.max(qtable, axis=1) for _, (_, qtable, _) in snapshots]
    )

    # Draw each snapshot
    for i, (key, (ep, qtable, desc)) in enumerate(snapshots):
        ax = axes_flat[i]
        q_best = np.max(qtable, axis=1)
        policy = np.argmax(qtable, axis=1).reshape(grid_size, grid_size)
        grid_q = q_best.reshape(grid_size, grid_size)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(grid_q, cmap=CMAP, norm=norm, aspect='equal')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc[r, c]
                val = grid_q[r, c]
                text_color = 'white' if val < - 0.4 or val > + 0.4 else 'black'
                is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')

                # policy arrow (top half of cell)
                ax.text(c, r - 0.18, arrows[policy[r, c]],
                        ha='center', va='center', fontsize=9, color=text_color)
                # Q value (bottom half of cell)
                ax.text(c, r + 0.22, f"{val:.3f}",
                        ha='center', va='center', fontsize=7.5, color=text_color)

                if cell == 'G':
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, zorder=3))
                elif is_old_goal:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='steelblue', linestyle='--', linewidth=2.5, zorder=3))
                elif cell == 'S':
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2.5, zorder=3))

        label_text = LABEL_MAP.get(key, key)
        ax.set_title(f"{label_text} |   ep {ep}", fontsize=11, fontweight='bold')
        ax.set_xticks(range(grid_size)); ax.set_yticks(range(grid_size))

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    shift_note = (f"  |  goal shift at ep {agent.shift_happened_ep}"
                    if shifted else "")
    fig.suptitle(
        f"Q-value Snapshots  |  mode={agent.mode}  alpha={agent.alpha}"
        f"  episodes={agent.training_episodes}{shift_note}",
        fontsize=13, fontweight='bold',
    )

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Q-value snapshots saved to {path}")

def swarm_gif(agent, env_id, max_steps, env_kwargs, num_agents=20, epsilon_test=0.1, path="swarm.gif", grid_size=10, fps=5):
    """GIF with N agents running simultaneously
        after training
        epsilon-greedy policy"""
    
    test_env = gym.make(env_id, max_episode_steps=max_steps, **env_kwargs)
    desc = test_env.unwrapped.desc.astype(str)
    
    all_paths = []
    max_steps = 0
    
    # simulating all agents
    for _ in range(num_agents):
        state, _ = test_env.reset()
        terminated = truncated = False
        path_coords = [(state % grid_size, state // grid_size)]
        
        for _ in range(agent.episode_steps):
            if terminated or truncated:
                break
            if np.random.rand() < epsilon_test:
                action = test_env.action_space.sample()
            else:
                action = int(np.argmax(agent.q_table[state]))
                
            state, _, terminated, truncated, _ = test_env.step(action)
            path_coords.append((state % grid_size, state // grid_size))
            
        all_paths.append(path_coords)
        if len(path_coords) > max_steps:
            max_steps = len(path_coords)

    # frames
    frames = []
    bg = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd'}
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for step in range(max_steps):
        ax.clear()
        ax.set_title(f"Swarm | step {step} | epsilon={epsilon_test}", fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, color='gray', alpha=0.3, zorder=1)

        # background
        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc[r, c]
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg.get(cell, '#f8f9fa'), zorder=0)
                ax.add_patch(rect)
                if cell in ['H', 'G', 'S']:
                    ax.text(c, r, cell, ha='center', va='center', fontsize=14, color='black', alpha=0.5)
        
        # getting position of each agent for the current step
        x_coords = []
        y_coords = []
        
        for p in all_paths:
            # if step > episode lenght keep last position
            if step < len(p):
                coord_corrente = p[step]
            else:
                coord_corrente = p[-1]
                
            # visual jitter
            x_coords.append(coord_corrente[0] + np.random.uniform(-0.2, 0.2))
            y_coords.append(coord_corrente[1] + np.random.uniform(-0.2, 0.2))
            
        # Ora disegna gli agenti come facevi prima
        ax.scatter(x_coords, y_coords, color='purple', edgecolors='white', s=50, zorder=3, alpha=0.8)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        
    plt.close(fig)
    
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"Swarm GIF saved to {path} ({len(frames)} frames)")
    test_env.close()

def plot_training_evolution(agent, path="training_evolution.png", grid_size=10):
    """
    subplots:
        1.  trajectory gradient
        2.  Q-value heatmap with policy arrows (final q_table)
        3.  state visits heatmap

    If a goal shift occurred, a dashed vertical line is added to the
    colorbar of subplot 1 to mark the shift episode.
    """
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))

    title = (f"Training Evolution | mode={agent.mode} replay={agent.replay_mode} | episodes={agent.training_episodes}")
    
    if agent.shift_happened_ep is not None:
        title += f"  |  goal shift at ep {agent.shift_happened_ep}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    shifted = agent.shift_happened_ep is not None
    if shifted:
        title += f"  |  goal shift at ep {agent.shift_happened_ep}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    desc = agent.env.unwrapped.desc.astype(str)
    initial_desc = getattr(agent, 'initial_desc', desc)

    bg = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': "#ffd5cd"}
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    # subplot 1
    ax1 = axes[0]
    ax1.set_title("Trajectories", fontsize=12)
    ax1.set_xlim(-0.5, grid_size - 0.5); ax1.set_ylim(-0.5, grid_size - 0.5)
    ax1.set_xticks(range(grid_size)); ax1.set_yticks(range(grid_size))
    ax1.set_aspect('equal'); ax1.invert_yaxis()
    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')
            
            bg_color = "#cdebff" if is_old_goal else bg.get(cell, '#f8f9fa')
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg_color, zorder=0)
            ax1.add_patch(rect)
            
            if cell in ['H', 'S']:
                ax1.text(c, r, cell, ha='center', va='center', fontsize=12, color='black', alpha=0.5)
            elif cell == 'G':
                lbl = 'G2' if shifted else 'G'
                ax1.text(c, r, lbl, ha='center', va='center', fontsize=12, color='black', fontweight='bold', alpha=0.8)
            elif is_old_goal:
                ax1.text(c, r, 'G1', ha='center', va='center', fontsize=12, color='gray', fontweight='bold', alpha=0.8)
    
    ax1.grid(True, color='gray', alpha=0.3, zorder=1)

    cmap = plt.colormaps['coolwarm']
    for eps_num, path_states in agent.sampled_paths:
        color_intensity = eps_num / agent.training_episodes
        color = cmap(color_intensity)
        path_x = [s % grid_size for s in path_states]
        path_y = [s // grid_size for s in path_states]
        jitter_x = np.array(path_x) + np.random.uniform(-0.15, 0.15, size=len(path_x))
        jitter_y = np.array(path_y) + np.random.uniform(-0.15, 0.15, size=len(path_y))
        alpha_val = 0.2 + 0.5 * color_intensity
        ax1.plot(jitter_x, jitter_y, color=color, alpha=alpha_val, linewidth=1.5, zorder=2)

    sm   = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(vmin=0, vmax=agent.training_episodes))
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Training Episode', rotation=270, labelpad=15)

    # Mark shift on colorbar
    if agent.shift_happened_ep is not None:
        cbar.ax.axhline(agent.shift_happened_ep, color='black', linewidth=2, linestyle='--')
        cbar.ax.text(1.05, agent.shift_happened_ep, f'shift\nep {agent.shift_happened_ep}',
                        transform=cbar.ax.get_yaxis_transform(),
                        va='center', fontsize=7, color='black')

    # subplot 2
    ax2 = axes[1]
    ax2.set_title("Q-value Heatmap  (max Q + policy arrows)", fontsize=12)
    q_vals = np.max(agent.q_table, axis=1)
    grid_q = q_vals.reshape(grid_size, grid_size)
    desc = agent.env.unwrapped.desc.astype(str)
    vmin, vmax = float(q_vals.min()), float(q_vals.max())
    mid = (vmin + vmax) / 2

    policy  = np.argmax(agent.q_table, axis=1).reshape(grid_size, grid_size)
    grid_q  = q_vals.reshape(grid_size, grid_size)

    im2 = ax2.imshow(grid_q, cmap='plasma', aspect='equal',
                        vmin=vmin, vmax=vmax)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label('max Q', rotation=270, labelpad=15)
    cb2.ax.tick_params(labelsize=7)

    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            val = grid_q[r, c]
            text_color = 'white' if val < mid else 'black'
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')

            ax2.text(c, r - 0.18, arrows[policy[r, c]],
                        ha='center', va='center', fontsize=9, color=text_color)
            ax2.text(c, r + 0.22, f"{val:.3f}",
                    ha='center', va='center', fontsize=7.5, color=text_color)
            if cell == 'G':
                ax2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, zorder=3))
            elif is_old_goal:
                ax2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='steelblue', linestyle='--', linewidth=2.5, zorder=3))
            elif cell == 'S':
                ax2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2.5, zorder=3))

    ax2.set_xticks(range(grid_size)); ax2.set_yticks(range(grid_size))
    ax2.tick_params(labelsize=8)

    if shifted:
        cbar.ax.axhline(agent.shift_happened_ep, color='black', linewidth=2, linestyle='--')
        cbar.ax.text(1.05, agent.shift_happened_ep, f'shift\nep {agent.shift_happened_ep}',
                        transform=cbar.ax.get_yaxis_transform(), va='center', fontsize=7, color='black')
    
    # subplot 3
    ax3 = axes[2]
    ax3.set_title("State Visits  (count)", fontsize=12)
    grid_visits = agent.state_visits.reshape(grid_size, grid_size)
    im3 = ax3.imshow(grid_visits, cmap='plasma', aspect='equal')
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label('visit count', rotation=270, labelpad=15)
    cb3.ax.tick_params(labelsize=7)

    for r in range(grid_size):
        for c in range(grid_size):
            cell = desc[r, c]
            val  = int(grid_visits[r, c])
            text_color = "white" if val < np.max(grid_visits) / 2 else "black"
            is_old_goal = shifted and (initial_desc[r, c] == 'G') and (cell != 'G')

            ax3.text(c, r, str(val), ha='center', va='center',
                        fontsize=9, color=text_color)
            if cell == 'G':
                ax3.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, zorder=3))
            elif is_old_goal:
                ax3.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='steelblue', linestyle='--', linewidth=2.5, zorder=3))
            elif cell == 'S':
                ax3.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2.5, zorder=3))

    ax3.set_xticks(range(grid_size)); ax3.set_yticks(range(grid_size))
    ax3.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Training evolution plot saved to {path}")


def plot_sampled_trajectories_gif(agent, path="sampled_evolution.gif", grid_size=10, fps=2):
    """GIF showing every 50 episodes chronologically"""
    import matplotlib.pyplot as plt
    import io
    import imageio
    
    if not agent.sampled_paths:
        print("Error: sampled_paths = Null")
        return

    desc = agent.env.unwrapped.desc.astype(str)
    bg = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd'}
    
    frames = []
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.colormaps['coolwarm']
    
    for eps_num, path_states in agent.sampled_paths:
        ax.clear()
        ax.set_title(f"Trajectory evolution | episode {eps_num}", fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, color='gray', alpha=0.3, zorder=1)
        
        # background
        for r in range(grid_size):
            for c in range(grid_size):
                cell = desc[r, c]
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=bg.get(cell, '#f8f9fa'), zorder=0)
                ax.add_patch(rect)
                if cell in ['H', 'G', 'S']:
                    ax.text(c, r, cell, ha='center', va='center', fontsize=14, color='black', alpha=0.4)
        
        # getting coords for the current episode
        path_x = [s % grid_size for s in path_states]
        path_y = [s // grid_size for s in path_states]
        
        # color changing as episode nr progresses
        color_intensity = eps_num / agent.training_episodes
        color = cmap(color_intensity)
        
        # plotting single trajectory
        ax.plot(path_x, path_y, color=color, linewidth=2.5, zorder=2, alpha=0.85)
        
        # highlighting starting and finishing points
        ax.scatter(path_x[0], path_y[0], color='lime', s=50, zorder=3, edgecolors='black')
        
        end_cell = desc[path_y[-1], path_x[-1]]
        end_color = 'gold' if end_cell == 'G' else ('red' if end_cell == 'H' else 'orange')
        ax.scatter(path_x[-1], path_y[-1], color=end_color, s=80, marker='X', zorder=3, edgecolors='black')
        
        # saving frame in a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        
    plt.close(fig)
    
    # final GIF
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"Trajectory chronological GIF saved to {path}")