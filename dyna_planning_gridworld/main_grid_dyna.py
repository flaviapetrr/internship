import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from envs import GridWorldEnv
from agents import DynaQAgent


def get_state_tuple(obs):
    """Converts observation dictionary in an hashabile tuple x Q-Table"""
    return tuple(obs["agent"]) + tuple(obs["target"])

PHASE1_OBSTACLES = [[1, 1], [2, 1], [3, 1], [4,1]]
PHASE2_OBSTACLES = [[1, 1], [2,1], [3, 1]]
CHANGE_EP = 150                       

def train(agent, n_episodes=300, max_steps=100, render_last=True,
          dynamic=True, verbose=True):
    """
    Train `agent` for `n_episodes`.
 
    If `dynamic=True` the obstacles change at episode CHANGE_EP (phase 2).
 
    Returns
    -------
    rewards_history   : total reward per episode
    lengths_history   : steps per episode
    epsilons_history  : epsilon per episode
    cumulative_rewards: running sum of rewards (useful for comparison plots)
    """

    label = "Dyna-Q+" if agent.plus else "Dyna-Q "
    if verbose:
        print(f"\n{'='*50}")
        print(f"  TRAINING {label}  ({n_episodes} eps, dynamic={dynamic})")
        print(f"{'='*50}")
 
    rewards_history    = []
    lengths_history    = []
    epsilons_history   = []
    cumulative_rewards = []
    cumulative         = 0.0

    start_state_q_vals  = []   # max Q at start state over time
    model_sizes         = []   # |model| over time


    for eps in range(n_episodes):
        if dynamic and eps == CHANGE_EP:
            env = agent.env.unwrapped
            env.change_obstacles(PHASE2_OBSTACLES)
            if verbose:
                print(f"\n  *** OBSTACLE CHANGE at episode {eps} ***")
                print(f"      Shortcut opened at [2,1]\n")

        obs, _ = agent.env.reset()
        
        state = get_state_tuple(obs)
        epsilon = agent.epsilon_exp_decay(eps)
        total_reward = 0

        render_this_episode = render_last and (eps == n_episodes - 1)

        if render_this_episode:
            print(f"\n--- RENDERING EPISODE ({eps + 1}) ---")
            agent.env.render()
            
        for step in range(max_steps):
            action = agent.e_greedy(state, epsilon)
            next_obs, reward, terminated, truncated, _ = agent.env.step(action)

            next_state = get_state_tuple(next_obs)

            if render_this_episode:
                time.sleep(0.4) 
                print(f"Step: {step + 1} | Action: {action} | Reward: {reward}")
                agent.env.render()

            # update q-learning and dyna model
            agent.q_update(state, action, reward, next_state)
            agent.update_model(state, action, next_state, reward)

            if agent.model:
                agent.planning()

            total_reward += reward
            state = next_state

            if terminated or truncated:
                if render_this_episode:
                    print(f"Episode finished. Total reward: {total_reward}\n")
                break
                
        rewards_history.append(total_reward)
        lengths_history.append(step + 1)
        epsilons_history.append(epsilon)
        cumulative += total_reward
        cumulative_rewards.append(cumulative)

        
        # print every 50 eps
        if (eps + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Ep: {eps + 1}/{n_episodes} - Mean reward: {avg_reward:.2f} - Epsilon: {epsilon:.3f}\n")

        model_sizes.append(len(agent.model))
        env_inner = agent.env.unwrapped
        tx, ty    = env_inner.target_pos
        sx, sy    = env_inner.start_pos
        start_key = (sx, sy, tx, ty)
        q_at_start = float(np.max(agent.q_table[start_key])) if start_key in agent.q_table else 0.0
        start_state_q_vals.append(q_at_start)
 
        if verbose and (eps + 1) % 50 == 0:
            avg_r = np.mean(rewards_history[-50:])
            print(f"  Ep {eps+1:>4}/{n_episodes} | mean_r={avg_r:+.3f} "
                  f"| |model|={len(agent.model):>4} "
                  f"| Q(start)={q_at_start:+.4f} "
                  f"| eps={epsilon:.3f}")
 
    if verbose:
        print(f"\n  Done. Final model size: {len(agent.model)}")
 
    _sanity_checks(agent, model_sizes, start_state_q_vals, verbose)

    print("--- COMPLETED ---\n")
    return rewards_history, lengths_history, epsilons_history, cumulative_rewards

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

def _sanity_checks(agent, model_sizes, q_vals, verbose):
    label = "Dyna-Q+" if agent.plus else "Dyna-Q "
    if not verbose:
        return
 
    print(f"\n--- Sanity checks for {label} ---")
 
    # 1. Model grows (or at least doesn't stay empty)
    grew = model_sizes[-1] > model_sizes[0]
    print(f"  [{'✓' if grew else '✗'}] Model grew: {model_sizes[0]} → {model_sizes[-1]} entries")
 
    # 2. Q-value at start state improved
    improved = q_vals[-1] > q_vals[0]
    print(f"  [{'✓' if improved else '✗'}] Q(start) improved: {q_vals[0]:+.4f} → {q_vals[-1]:+.4f}")
 
    # 3. Dyna-Q+: last_visit timestamps should span a range
    if agent.plus:
        all_timestamps = np.concatenate([np.asarray(v).ravel()
                                         for v in agent.last_visit.values()]) \
                         if agent.last_visit else np.array([0])
        non_zero = all_timestamps[all_timestamps > 0]
        span = int(non_zero.max()) - int(non_zero.min()) if len(non_zero) > 1 else 0
        ok   = span > 0
        print(f"  [{'✓' if ok else '✗'}] Dyna-Q+ last_visit timestamps span {span} steps")

 
    # 4. Planning actually touched states
    touched = len(agent.q_table)
    print(f"  [✓] Q-table contains {touched} distinct states")
    print()

def plot_comparison(agent_dq, agent_dqp,
                    r_dq,  r_dqp,
                    cum_dq, cum_dqp,
                    len_dq, len_dqp,
                    window=20):
    """
    4-panel figure:
      [0,0] Smoothed per-episode reward  both agents + change marker
      [0,1] Cumulative reward            the definitive Dyna-Q vs Dyna-Q+ test
      [1,0] Episode length               shorter = found shortcut
      [1,1] Learned policy grid          side-by-side mini-grids
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dyna-Q vs Dyna-Q+  ·  Dynamic GridWorld 5x5",
                 fontsize=15, fontweight="bold")
 
    n      = len(r_dq)
    sw     = min(window, n)
    eps_ax = np.arange(n)
 
    def smooth(arr):
        return np.convolve(arr, np.ones(sw) / sw, mode="valid")
 
    ax = axes[0, 0]
    ax.plot(eps_ax, r_dq,  alpha=0.15, color="steelblue")
    ax.plot(eps_ax, r_dqp, alpha=0.15, color="tomato")
    ax.plot(range(sw - 1, n), smooth(r_dq),  color="steelblue", lw=2, label="Dyna-Q")
    ax.plot(range(sw - 1, n), smooth(r_dqp), color="tomato",    lw=2, label="Dyna-Q+")
    ax.axvline(CHANGE_EP, color="black", ls="--", lw=1.5, label=f"Env change (ep {CHANGE_EP})")
    ax.set_title("Reward per episode (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend(fontsize=9)
 
    # ── [0,1] Cumulative reward  (KEY diagnostic) ─────────────────
    ax = axes[0, 1]
    ax.plot(eps_ax, cum_dq,  color="steelblue", lw=2, label="Dyna-Q")
    ax.plot(eps_ax, cum_dqp, color="tomato",    lw=2, label="Dyna-Q+")
    ax.axvline(CHANGE_EP, color="black", ls="--", lw=1.5, label=f"Env change (ep {CHANGE_EP})")
    ax.set_title("Cumulative reward  ← main comparison metric")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.legend(fontsize=9)
    # Annotate which agent is ahead at the end
    gap = cum_dqp[-1] - cum_dq[-1]
    winner = "Dyna-Q+" if gap >= 0 else "Dyna-Q"
    ax.annotate(f"{winner} leads by {abs(gap):.1f}",
                xy=(n - 1, max(cum_dq[-1], cum_dqp[-1])),
                xytext=(-60, -20), textcoords="offset points",
                fontsize=9, arrowprops=dict(arrowstyle="->"))
 
    # ── [1,0] Episode length ──────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(eps_ax, len_dq,  alpha=0.15, color="steelblue")
    ax.plot(eps_ax, len_dqp, alpha=0.15, color="tomato")
    ax.plot(range(sw - 1, n), smooth(len_dq),  color="steelblue", lw=2, label="Dyna-Q")
    ax.plot(range(sw - 1, n), smooth(len_dqp), color="tomato",    lw=2, label="Dyna-Q+")
    ax.axvline(CHANGE_EP, color="black", ls="--", lw=1.5)
    ax.set_title("Episode length (fewer steps = better path)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=9)
 
    # ── [1,1] Policy grids ────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Learned policies after training\n(left=Dyna-Q, right=Dyna-Q+)")
    ax.axis("off")
 
    env      = agent_dq.env.unwrapped
    g        = env.size
    tx, ty   = env.target_pos
    sx, sy   = env.start_pos
    symbols  = {0: "↑", 1: "↓", 2: "←", 3: "→"}
 
    for col_offset, agent, col_label in [
        (0,     agent_dq,  "Dyna-Q"),
        (g + 1, agent_dqp, "Dyna-Q+"),
    ]:
        ax.text(col_offset + g / 2, g + 0.4, col_label,
                ha="center", va="bottom", fontsize=11, fontweight="bold")
 
        for y in range(g):
            for x in range(g):
                cx = col_offset + x
                pos   = np.array([x, y])
                state = (x, y, tx, ty)
 
                if np.array_equal(pos, [tx, ty]):
                    fc, txt = "lightgreen", "T"
                elif env._is_obstacle(pos):
                    fc, txt = "dimgray", "X"
                elif np.array_equal(pos, [sx, sy]):
                    fc = "lightblue"
                    txt = "S\n" + symbols[np.argmax(agent.q_table[state])] \
                          if state in agent.q_table else "S"
                elif state in agent.q_table:
                    fc  = "white"
                    txt = symbols[np.argmax(agent.q_table[state])]
                else:
                    fc, txt = "lightyellow", "."
 
                ax.add_patch(plt.Rectangle(
                    (cx, y), 1, 1, lw=1, edgecolor="black", facecolor=fc
                ))
                ax.text(cx + 0.5, y + 0.5, txt,
                        ha="center", va="center", fontsize=11, fontweight="bold")
 
    ax.set_xlim(0, 2 * g + 1)
    ax.set_ylim(0, g + 1)
 
    plt.tight_layout()
    plt.savefig("./dyna_planning_gridworld/dyna_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved")
    plt.show()

def print_policy(agent, label=""):
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    env = agent.env.unwrapped

    tx, ty = env.target_pos
    
    print(f"\n--- POLICY MAP {label} ---")
    
    for y in range(env.size - 1, -1, -1):
        row_str = ""
        for x in range(env.size):
            pos = np.array([x, y])
            state = (x, y, tx, ty)
            
            if np.array_equal(pos, env.target_pos):
                row_str += " T "
            elif env._is_obstacle(pos):
                row_str += " # "
            else:
                if state in agent.q_table:
                    best_action = np.argmax(agent.q_table[state])
                    row_str += f" {action_symbols[best_action]} "
                else:
                    row_str += " . "
        print(row_str)

def make_env():
    return gym.make(
        "GridWorld-v0",
        size=5,
        start_pos=[0, 0],
        target_pos=[4, 4],
        obstacles=PHASE1_OBSTACLES,
    )

if __name__ == "__main__":

    N_EPISODES    = 500
    MAX_STEPS     = 100
    PLANNING_STEPS = 20

    # DYNA Q
    env_dq  = make_env()
    agent_dq = DynaQAgent(plus=False, env=env_dq, planning_steps=PLANNING_STEPS)
    r_dq, len_dq, eps_dq, cum_dq = train(
        agent_dq, n_episodes=N_EPISODES, max_steps=MAX_STEPS, dynamic=True
    )

    # DYNA Q+
    env_dqp   = make_env()
    agent_dqp = DynaQAgent(plus=True,  env=env_dqp, planning_steps=PLANNING_STEPS, k=1e-3)
    r_dqp, len_dqp, eps_dqp, cum_dqp = train(
        agent_dqp, n_episodes=N_EPISODES, max_steps=MAX_STEPS, dynamic=True
    )

    # plotting results
    print_policy(agent_dq,  label="Dyna-Q  (phase-2 obstacles)")
    print_policy(agent_dqp, label="Dyna-Q+ (phase-2 obstacles)")

    plot_comparison(
        agent_dq, agent_dqp,
        r_dq,    r_dqp,
        cum_dq,  cum_dqp,
        len_dq,  len_dqp,
    )