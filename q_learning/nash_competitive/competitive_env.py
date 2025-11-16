"""
Competitive Pursuer-Evader Environment
======================================

A 4x4 grid environment for competitive pursuer-evader game.
"""

import numpy as np
from typing import Tuple, List, Optional


class CompetitiveEnv:
    """Competitive environment with pursuer and evader."""
    
    def __init__(self, size=4):
        self.size = size
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        self.action_names = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        self.action_size = 4
        self.joint_action_size = 16  # 4 × 4
        
        # Get all valid positions
        self.valid_positions = [(y, x) for y in range(size) for x in range(size)]
        
    def _is_valid_position(self, y: int, x: int) -> bool:
        """Check if position is within bounds."""
        return 0 <= y < self.size and 0 <= x < self.size
    
    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Apply action to position, staying in place if invalid."""
        y, x = pos
        dy, dx = self.actions[action]
        new_y, new_x = y + dy, x + dx
        
        if self._is_valid_position(new_y, new_x):
            return (new_y, new_x)
        else:
            return pos  # Stay in place if out of bounds
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def reset(self, pursuer_pos: Optional[Tuple[int, int]] = None,
              evader_pos: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Reset environment to random or specified positions.
        Ensures pursuer and evader are in distinct cells.
        """
        if pursuer_pos is None or evader_pos is None:
            # Sample random distinct positions
            available = self.valid_positions.copy()
            pursuer_pos = tuple(available.pop(np.random.randint(len(available))))
            evader_pos = tuple(available[np.random.randint(len(available))])
        
        if pursuer_pos == evader_pos:
            raise ValueError("Pursuer and evader must start in distinct cells")
        
        return pursuer_pos, evader_pos
    
    def step(self, pursuer_pos: Tuple[int, int], pursuer_action: int,
             evader_pos: Tuple[int, int], evader_action: int) -> Tuple[Tuple[int, int], Tuple[int, int], 
                                                                       float, float, bool]:
        """
        Take a step in the environment.
        Returns: (new_pursuer_pos, new_evader_pos, pursuer_reward, evader_reward, done)
        """
        # Apply actions (simultaneous moves)
        new_pursuer_pos = self._apply_action(pursuer_pos, pursuer_action)
        new_evader_pos = self._apply_action(evader_pos, evader_action)
        
        # Check if pursuer caught evader
        caught = (new_pursuer_pos == new_evader_pos)
        
        # Compute distance-based rewards
        old_distance = self._manhattan_distance(pursuer_pos, evader_pos)
        new_distance = self._manhattan_distance(new_pursuer_pos, new_evader_pos)
        
        # Pursuer reward: positive for getting closer, +100 for catching
        if caught:
            pursuer_reward = 100.0
            evader_reward = -100.0  # Evader gets penalized for being caught
        else:
            # Pursuer: reward for reducing distance
            pursuer_reward = old_distance - new_distance  # Positive if distance decreased
            # Evader: reward for increasing distance
            evader_reward = new_distance - old_distance  # Positive if distance increased
        
        done = caught
        
        return new_pursuer_pos, new_evader_pos, pursuer_reward, evader_reward, done
    
    def state_to_features(self, pursuer_pos: Tuple[int, int], 
                         evader_pos: Tuple[int, int]) -> np.ndarray:
        """Convert state to feature vector: [pursuer_x, pursuer_y, evader_x, evader_y]."""
        pursuer_y, pursuer_x = pursuer_pos
        evader_y, evader_x = evader_pos
        return np.array([pursuer_x, pursuer_y, evader_x, evader_y], dtype=np.float32)
    
    def joint_action_to_index(self, pursuer_action: int, evader_action: int) -> int:
        """Convert joint action (pursuer_action, evader_action) to linear index (0-15)."""
        return pursuer_action * 4 + evader_action
    
    def index_to_joint_action(self, index: int) -> Tuple[int, int]:
        """Convert linear index to joint action (pursuer_action, evader_action)."""
        pursuer_action = index // 4
        evader_action = index % 4
        return pursuer_action, evader_action
    
    def get_all_start_pairs(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all possible distinct start position pairs."""
        pairs = []
        for pursuer_pos in self.valid_positions:
            for evader_pos in self.valid_positions:
                if pursuer_pos != evader_pos:
                    pairs.append((pursuer_pos, evader_pos))
        return pairs

