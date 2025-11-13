"""
4x4 Maze Environment
===================

A custom 4x4 maze with obstacles at (1,1) and (2,2), goal at (3,3).
"""

import numpy as np
from typing import Tuple, List


class MazeEnv:
    """4x4 Maze Environment with obstacles and goal."""
    
    def __init__(self, size=4):
        self.size = size
        self.obstacles = [(1, 1), (2, 2)]  # Obstacles at (1,1) and (2,2)
        self.goal = (3, 3)  # Goal at (3,3)
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        self.action_names = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        self.action_size = 4
        
        # Get all valid start positions (excluding obstacles and goal)
        self.valid_starts = self._get_valid_starts()
        
    def _get_valid_starts(self) -> List[Tuple[int, int]]:
        """Get all valid starting positions (excluding obstacles and goal)."""
        valid = []
        for y in range(self.size):
            for x in range(self.size):
                pos = (y, x)
                if pos not in self.obstacles and pos != self.goal:
                    valid.append(pos)
        return valid
    
    def _is_valid_position(self, y: int, x: int) -> bool:
        """Check if position is within bounds and not an obstacle."""
        if not (0 <= y < self.size and 0 <= x < self.size):
            return False
        return (y, x) not in self.obstacles
    
    def reset(self, start_pos: Tuple[int, int] = None) -> Tuple[int, int]:
        """Reset environment to a random or specified start position."""
        if start_pos is None:
            start_pos = tuple(self.valid_starts[np.random.randint(len(self.valid_starts))])
        return start_pos
    
    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a step in the environment.
        Returns: (next_state, reward, done)
        """
        y, x = state
        dy, dx = self.actions[action]
        new_y, new_x = y + dy, x + dx
        
        # Check if move is valid
        if not self._is_valid_position(new_y, new_x):
            # Hit obstacle or out of bounds - stay in place, get penalty
            reward = -10.0
            next_state = state
        else:
            next_state = (new_y, new_x)
            reward = -1.0  # Step penalty
        
        # Check if goal reached
        if next_state == self.goal:
            reward = 100.0
            done = True
        else:
            done = False
        
        return next_state, reward, done
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (y, x) state to linear index."""
        y, x = state
        return y * self.size + x
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert linear index to (y, x) state."""
        y = index // self.size
        x = index % self.size
        return (y, x)
    
    def state_to_features(self, state: Tuple[int, int]) -> np.ndarray:
        """Convert state to feature vector (x, y coordinates)."""
        y, x = state
        return np.array([x, y], dtype=np.float32)

