# QLearning

This directory contains the implementation of Q-Learning, a model-free, value-based reinforcement learning algorithm. It is designed to find the optimal action-selection policy for a given finite Markov Decision Process (MDP).

It explores multiple Q-Learning variants, including both reward-based and punishment-based update rules, allowing the agent to seamlessly handle environments with stochastic transitions and various feedback structures without requiring structural adaptations.

Additionally, it implements the possibility to enhance the learning phase with various experience replay and planning mechanisms, inspired by the biological hippocampal replays that occur during learning.

## Contents

- **`agent.py`**: Defines the Q-Learning agent allowing for different update Q-table modes
- **`main.py`**: The entry point for running the Q-Learning algorithm with the possibility to customize training parameters
- **`plot.py`**: Contains utilities for visualizing results and metrics
- **`replay.py`**: Implements various replay algorithms
- **`train.py`**: Handles the training loop for the Q-Learning agent
- **`visuals/`**: Directory in which are stored the generated plots and heatmaps for analysis
- **`README.md`**: This file, which provides an overview of the project

## Prerequisites

- Python 3.x
- Libraries: `numpy`, `matplotlib`, `gym` (or other dependencies as required)

## Features

- Implementation of Q-Learning algorithms
- Possibility to add replays during training
- Customizable training parameters
- Visualization of learning progress and results
- Modular design for easy extension and experimentation

## Update Rules

This project implements different variations of the Bellman equation to update the Q-values, exploring both standard reward-based and punishment-based behaviours. 

Below are the mathematical formulations used in the `q_table_update` method inside the `QLearningAgent`, based on the selected `update_mode`.

### 1. Standard Q-Learning (`std` & `std_punish`)
The classical one-step Bellman equation, which aims to maximize the expected cumulative reward. 

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right)$$

`std_punish` differs for having punishment feedback: `r < 0`.

### 2. Punishment-Based Standard Q-Learning (`opposite`)
A variation designed for punishment-based environments, where the goal is to minimize negative outcomes. It evaluates the optimal next state using the minimum Q-value instead of the maximum.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \min_{a'} Q(s',a') - Q(s,a) \right)$$

### 3. Contextual / Relative Q-Learning (`relative`)
This approach updates the state-value tracking variable $V(s)$ based on the average value of unchosen actions, adjusting the Q-value update relative to this baseline.

First, we calculate a relative reward baseline $r_v$ based on the total number of actions $|A|$:
$$r_v = \frac{r + \sum_{a' \neq a} Q(s,a')}{|A|}$$

Then, we update the state-value table $V$:
$$V(s) \leftarrow V(s) + \alpha_v (r_v - V(s))$$

Finally, we update the Q-table using the standard maximization approach, adjusted by $V(s)$:
$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r - V(s) + \gamma \max_{a'} Q(s',a') - Q(s,a) \right)$$

### 4. Contextual Punishment Q-Learning (`relative_punish`)
This applies the contextual baseline logic from the `relative` mode, but adapts the final update to minimize punishment by selecting the minimum Q-value for the next state.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r - V(s) + \gamma \min_{a'} Q(s',a') - Q(s,a) \right)$$

## Replay Methods

To enhance learning efficiency and enable environment planning, this repository implements multiple **Experience Replay** architectures. These range from Model-Free (MF-RL) rapid learning buffers to Model-Based (MB-RL) planning algorithms.

### 1. Backward Replay (Model-Free)
Designed for rapid learning, this mechanism tracks the agent's trajectory during an episode. When an episode ends, it replays the last $n$ steps in reverse order. This allows the reward signal (or punishment) to propagate backwards through the exact trajectory immediately, highly accelerating convergence in environments with sparse rewards.

### 2. Dyna Replay (Integrated Architecture)
An implementation inspired by Sutton's Dyna architecture. The agent builds a deterministic internal model of the environment by storing observed transitions: $(s, a) \rightarrow (s', r, terminated)$. During the planning phase, it randomly samples previously visited state-action pairs from this model to perform simulated Q-value updates, efficiently blending real experience with simulated forward and backward planning.

### 3. Prioritized Sweeping (Model-Based)
A sophisticated planning approach that focuses computational resources where they are most needed. It maintains a priority queue of transitions based on the magnitude of the expected Q-value update. 
- Updates are only queued if their magnitude exceeds a specific threshold ($\theta$).
- It dynamically tracks **predecessors** (states that lead to the current state) to intelligently propagate significant value changes backwards through the state space, prioritizing the most impactful updates.

### 4. Value Iteration Replay (Background Planning)
A Model-Based mechanism designed for continuous background planning. It stores a comprehensive dictionary of all uniquely observed transitions. During training, it samples a batch of $n$ known transitions to perform broad Value Iteration, ensuring the Q-table is continuously refined based on the agent's cumulative historical knowledge.

## Usage

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Navigate to the `QLearning` directory:
    ```bash
    cd QLearning
    ```
3. Run the main script:
    ```bash
    python main.py
    ```

## References

- [Q-Learning Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
- - **Environment:**
    * Farama Foundation. (2023). *Gymnasium: A standard interface for reinforcement learning environments (FrozenLake)*. URL: [https://gymnasium.farama.org/environments/toy_text/frozen_lake/](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- * **Q-Learning Foundations:**
    * Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
    * Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
* **Contextual Modulation (`relative` modes):**
    * Palminteri, S., Khamassi, M., Joffily, M., & Coricelli, G. (2015). Contextual modulation of value signals in reward and punishment learning. *Nature Communications*, 6(1), 8096.
* **Hippocampal Replay & Prioritized Planning:**
    * Mattar, M. G., & Daw, N. D. (2018). Prioritized memory access explains planning and hippocampal replay. *Nature Neuroscience*, 21(11), 1609-1617.
    * Cazé, R., Khamassi, M., Lise, A., & Girard, B. (2018). Hippocampal replays under the scrutiny of reinforcement learning models. *Journal of Neurophysiology*, 120(6), 2877-2896.
