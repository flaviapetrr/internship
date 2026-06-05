import random
import heapq
from collections import defaultdict

class BackwardReplay():
    """
    MF-RL
    associated with backward and unordered replays
    -> rapid learning
    """
    def __init__(self):
        # defining list that tracks the steps made in an episode
        self.episode_history = []

    def store_step(self, current_state, action, next_state, reward, terminated):
        """save step just performed"""
        self.episode_history.append((current_state, action, next_state, reward, terminated))

    def backward_traj(self, n_steps):
        """take n_steps last performed and returns them in opposite sense"""
        trajectory = self.episode_history[-n_steps:]

        return reversed(trajectory)

    def clear(self):
        """empty buffer"""
        self.episode_history = []

class DynaReplay():
    """
    Dyna algorithm
    associated with a mix of both forward and backward replays
    -> efficient
    """
    def __init__(self):
        self.model = {} # {(current_state, action): (reward, next_state, terminated)}
        self.visited_states = set()
        self.state_actions = defaultdict(set)
    
    def store_step(self, current_state, action, next_state, reward, terminated):
        """save step just performed in the algorithm model"""
        self.model[(current_state, action)] = (next_state, reward, terminated)
        self.visited_states.add(current_state)
        self.state_actions[current_state].add(action)

        # stupid debug to check model storage
        #if next_state == 99:
        #if reward != 0:
        #    print(f"Update: State {current_state} | Action {action} -> Stored vals: {self.model[(current_state, action)]}")

    def random_sample(self):
        """samples a situation already experienced in the past"""
        # randomly chooses key
        state = random.choice(list(self.visited_states))
        action = random.choice(list(self.state_actions[state]))

        # getting what actually happened from its model
        next_state, reward, terminated = self.model[(state, action)]
        
        return state, action, next_state, reward, terminated

    def size(self):
        """how many situations are known to him"""
        return len(self.model)
    
class PrioritizedSweepingReplay():
    """
    MB-RL: prioritized sweeping
    associated with forward and imaginary replays
    -> planning
    """
    def __init__(self, theta=1e-4):
        self.model = {} # {(current_state, action): (reward, next_state, terminated)}
        self.predecessors = defaultdict(set)  # {(next_state): (current_state, action, reward, terminated)}
        self.priority_queue = []
        self.theta = theta

    def store_step(self, current_state, action, next_state, reward, terminated):
        """save step just performed in the algorithm model and the predecessors"""
        self.model[(current_state, action)] = (next_state, reward, terminated)
        self.predecessors[next_state].add((current_state, action, reward, terminated))

        # stupid debug to check model storage
        #if next_state == 99: 
        #if reward != 0:
        #    print(f"Update: State {current_state} | Action {action} -> Stored vals: {self.model[(current_state, action)]}")

    def push(self, priority, state, action):
        if priority > self.theta:
            # using -priority as heapq pushes smaller nrs first
            heapq.heappush(self.priority_queue, (-priority, state, action))

    def pop(self):
        if not self.is_empty():
            _neg_priority, state, action = heapq.heappop(self.priority_queue)
            return state, action
        return None, None

    def is_empty(self):
        return len(self.priority_queue) == 0
    
class ValueIterationReplay():
    """
    MB-RL: value iteration (background planning)
    """
    def __init__(self):
        self.model = {} # {(current_state, action): (next_state, reward, terminated)}
        self.known_states = set()

    def store_step(self, current_state, action, next_state, reward, terminated):
        """saves or updates transition in the agent model"""
        self.model[(current_state, action)] = (next_state, reward, terminated)
        self.known_states.add(current_state)

        # stupid debug to check model storage
        #if next_state == 99:
        #if reward != 0:
        #    print(f"Update: State {current_state} | Action {action} -> Stored vals: {self.model[(current_state, action)]}")

    def sample_n(self, n_samples):
        """returns n_samples known trasitions"""
        known_transitions = [
            (state, action, next_state, reward, terminated) 
            for (state, action), (next_state, reward, terminated) in self.model.items()
        ]
        sample_size = min(n_samples, len(known_transitions))
        return random.sample(known_transitions, sample_size)
    
    def size(self):
        """how many transitions are known to him"""
        return len(self.model)
    
    def clear(self):
        """empty buffer"""
        self.model = {}
        self.known_states = set()