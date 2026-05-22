import plot
import gymnasium as gym
import imageio
import io
import numpy as np
import matplotlib.pyplot as plt

BG_COLORS = {'S': '#d4edda', 'F': '#f8f9fa', 'H': '#adb5bd', 'G': '#fff3cd', 'G2': '#ffd5cd'}

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

        plot._draw_grid_bg(ax, grid_size, desc, desc, BG_COLORS, shifted=False)

        # getting position of each agent for the current step
        x_coords = []
        y_coords = []
        
        for p in all_paths:
            # if step > episode lenght keep last position
            if step < len(p):
                current_coord = p[step]
            else:
                current_coord = p[-1]
                
            # visual jitter
            x_coords.append(current_coord[0] + np.random.uniform(-0.2, 0.2))
            y_coords.append(current_coord[1] + np.random.uniform(-0.2, 0.2))
            
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
    print(f"Sampled Trajectories GIF saved to {path}")