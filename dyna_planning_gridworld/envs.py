from typing import Optional
import numpy as np
import gymnasium as gym

class GridWorldEnv (gym.Env):
    """
    A configurable N×N GridWorld environment compatible with Gymnasium.
 
    Observation (Dict):
        - "agent": (row, col) of the agent
        - "goal":  (row, col) of the goal
 
    Actions:
        0 = up, 1 = down, 2 = left, 3 = right
 
    Args:
        size        : side length of the square grid (default 5)
        start_pos   : fixed agent start (row, col), random if None
        goal_pos    : fixed goal position (row, col), random if None
        obstacles   : list of (row, col) cells that are impassable
        reward_goal : reward for reaching the goal (default +1.0)
        reward_step : reward per step to encourage efficiency (default -0.01)
        reward_wall : reward for bumping into a wall/obstacle (default -0.1)
        max_steps   : episode length limit (default size * size * 2)
    """

    def __init__(
        self,
        size: int = 5,
        start_pos: Optional[list] = None,
        target_pos: Optional[list] = None,
        obstacles: Optional[list] = None,
        reward_goal: float = 1.0,
        reward_step: float = -0.01,
        reward_wall: float = -0.1,
    ):
        super().__init__()

        assert size >= 2, "Grid size must be at least 2."
        self.size = size
        self.start_pos = np.array(start_pos, dtype=np.int32) if start_pos else None
        self.target_pos = np.array(target_pos, dtype=np.int32) if target_pos else None
        self._obstacles = [np.array(o, dtype=np.int32) for o in (obstacles or [])]
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_wall = reward_wall
        
        self._agent_pos  = np.array([-1, -1], dtype=np.int32)
        self._target_pos = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),  # [x, y] coordinates
            }
        )
 
        # 4 possible actions
        self.action_space = gym.spaces.Discrete(4)

        # mapping actions to actual grid movements
        self._action_ = {
            0: np.array([0, 1]),  # up
            1: np.array([0, -1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),  # right
        }

    def _get_obs(self):
        return {
            "agent":  self._agent_pos,
            "target": self._target_pos,
        }
    
    def _get_info(self):
        return {
            # Manhattan distance x debugging
            "distance": np.linalg.norm(
                self._agent_pos - self._target_pos, ord=1
            )   # nessun time-limit interno; usa max_episode_steps in gym.register
        }
 
    def _is_obstacle(self, pos: np.ndarray) -> bool:
        return any(np.array_equal(pos, o) for o in self._obstacles)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # init random generator
        super().reset(seed=seed)
 
        if self.start_pos is not None:
            self._agent_pos = self.start_pos.copy()
        else:
            self._agent_pos = self._sample_free_pos()
 
        # Se c'è un target_pos fisso lo usiamo, altrimenti random
        if self.target_pos is not None:
            self._target_pos = self.target_pos.copy()
        else:
            self._target_pos = self._agent_pos.copy()
            while np.array_equal(self._target_pos, self._agent_pos):
                self._target_pos = self._sample_free_pos()

        return self._get_obs(), self._get_info()
 
    def _sample_free_pos(self) -> np.ndarray:
        """Sampling position != obstacles"""
        while True:
            pos = self.np_random.integers(0, self.size, size=2, dtype=int)
            if not self._is_obstacle(pos):
                return pos

    def change_obstacles(self, new_obstacles: list) -> None:
        self._obstacles = [np.array(o, dtype=np.int32) for o in new_obstacles]
        
    def step(self, action: int):
        direction = self._action_[action]
        new_pos= self._agent_pos + direction
 
        # check if new_pos is inside grid and != ostacles
        in_bounds = np.all((new_pos >= 0) & (new_pos < self.size))
        not_obstacle = not self._is_obstacle(new_pos)
 
        if in_bounds and not_obstacle:
            self._agent_pos = new_pos
            reward = self.reward_step # small penalty for every step
        else:
            # wall or obstacle -> agent stops and gets penalized
            reward = self.reward_wall
 
        terminated = np.array_equal(self._agent_pos, self._target_pos)
        truncated  = False
 
        if terminated:
            reward = self.reward_goal
 
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        for y in range(self.size - 1, -1, -1):
            row = ""
            for x in range(self.size):
                pos = np.array([x, y])
                if np.array_equal(pos, self._agent_pos):
                    row += "A "
                elif np.array_equal(pos, self._target_pos):
                    row += "T "
                elif self._is_obstacle(pos):
                    row += "# "
                else:
                    row += ". "
            print(row)
        print()

gym.register(
    id="GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,
)
