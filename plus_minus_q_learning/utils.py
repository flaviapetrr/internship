import numpy as np
import matplotlib.pyplot as plt

def plot_results(agent, episode_rewards, episode_lengths, epsilons, q_table, window=20):

    env = agent.env.unwrapped
    grid_size = env.size
    tx, ty = env.target_pos
    sx, sy = env.start_pos

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = "Dyna-Q+ " if agent.plus else "Dyna-Q "
    fig.suptitle(f"{title} on GridWorld {grid_size}x{grid_size}", fontsize=16, fontweight="bold")

    safe_window = min(window, len(episode_rewards))

    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, color="steelblue", label="Raw")
    if safe_window > 0:
        smoothed = np.convolve(episode_rewards, np.ones(safe_window)/safe_window, mode="valid")
        ax.plot(range(safe_window - 1, len(episode_rewards)), smoothed, color="steelblue", label=f"Smoothed (w={safe_window})")
    ax.set_title("Reward x Eps")
    ax.set_xlabel("Eps")
    ax.set_ylabel("Total Reward")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(episode_lengths, alpha=0.3, color="coral")
    if safe_window > 0:
        smoothed_len = np.convolve(episode_lengths, np.ones(safe_window)/safe_window, mode="valid")
        ax.plot(range(safe_window - 1, len(episode_lengths)), smoothed_len, color="coral")
    ax.set_title("Episode length")
    ax.set_xlabel("Eps")
    ax.set_ylabel("Steps")

    ax = axes[1, 0]
    ax.plot(epsilons, color="purple", linewidth=2)
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Eps")
    ax.set_ylabel("Epsilon")

    ax = axes[1, 1]
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Learned policy")

    for y in range(grid_size):
        for x in range(grid_size):
            pos = np.array([x, y])
            state = (x, y, tx, ty)
            
            text = ""
            color = "white"
            
            if np.array_equal(pos, [tx, ty]):
                text = "T"
                color = "lightgreen"
            elif env._is_obstacle(pos):
                text = "X"
                color = "gray"
            elif np.array_equal(pos, [sx, sy]):
                color = "lightblue"
                if state in q_table:
                    best_action = np.argmax(q_table[state])
                    text = f"S\n{action_symbols[best_action]}"
                else:
                    text = "S"
            else:
                if state in q_table:
                    best_action = np.argmax(q_table[state])
                    text = action_symbols[best_action]
                else:
                    text = "." # never explored

            ax.add_patch(plt.Rectangle((x, y), 1, 1, linewidth=1, edgecolor="black", facecolor=color))
            ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if agent.plus:
        plt.savefig("dyna_q_plus_gridworld.png", dpi=150, bbox_inches="tight")
    else:
        plt.savefig("dyna_q_gridworld.png", dpi=150, bbox_inches="tight")
    
    plt.show()