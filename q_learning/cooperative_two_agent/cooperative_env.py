"""
Cooperative Two-Agent Environment
==================================

A 4x4 grid environment where two agents cooperate to visit two goal cells.
"""

import numpy as np
from typing import Tuple, List, Optional, Set


class CooperativeEnv:
    """Cooperative environment with two agents and two goal cells."""
    
    def __init__(self, size=4, goal1: Tuple[int, int] = (0, 0), goal2: Tuple[int, int] = (3, 3)):
        self.size = size
        self.goal1 = goal1
        self.goal2 = goal2
        
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
        
        # Get all valid positions (excluding goals)
        self.valid_positions = [(y, x) for y in range(size) for x in range(size)
                               if (y, x) != goal1 and (y, x) != goal2]
        
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
    
    def reset(self, agent1_pos: Optional[Tuple[int, int]] = None,
              agent2_pos: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[int, int], Tuple[int, int], Set[Tuple[int, int]]]:
        """
        Reset environment to random or specified positions.
        Returns: (agent1_pos, agent2_pos, visited_goals)
        """
        if agent1_pos is None or agent2_pos is None:
            # Sample random distinct positions
            available = self.valid_positions.copy()
            agent1_pos = tuple(available.pop(np.random.randint(len(available))))
            agent2_pos = tuple(available[np.random.randint(len(available))])
        
        if agent1_pos == agent2_pos:
            raise ValueError("Agents must start in distinct cells")
        
        visited_goals = set()
        return agent1_pos, agent2_pos, visited_goals
    
    def step(self, agent1_pos: Tuple[int, int], agent1_action: int,
             agent2_pos: Tuple[int, int], agent2_action: int,
             visited_goals: Set[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int], 
                                                          Set[Tuple[int, int]], float, bool]:
        """
        Take a step in the environment with collision handling.
        Returns: (new_agent1_pos, new_agent2_pos, new_visited_goals, reward, done)
        """
        # Compute intended positions
        intended_pos1 = self._apply_action(agent1_pos, agent1_action)
        intended_pos2 = self._apply_action(agent2_pos, agent2_action)
        
        # Handle collisions: if both try to go to the same cell
        if intended_pos1 == intended_pos2:
            # One agent goes, the other stays
            # Agent 1 goes, agent 2 stays
            new_agent1_pos = intended_pos1
            new_agent2_pos = agent2_pos
        elif intended_pos1 == agent2_pos and intended_pos2 == agent1_pos:
            # They try to swap positions - both stay
            new_agent1_pos = agent1_pos
            new_agent2_pos = agent2_pos
        else:
            # No collision, both move
            new_agent1_pos = intended_pos1
            new_agent2_pos = intended_pos2
        
        # Check goal visits
        new_visited_goals = visited_goals.copy()
        reward = -1.0  # Step penalty
        
        if new_agent1_pos == self.goal1 or new_agent2_pos == self.goal1:
            if self.goal1 not in new_visited_goals:
                new_visited_goals.add(self.goal1)
                reward += 100.0  # +100 for visiting first goal
        
        if new_agent1_pos == self.goal2 or new_agent2_pos == self.goal2:
            if self.goal2 not in new_visited_goals:
                new_visited_goals.add(self.goal2)
                reward += 100.0  # +100 for visiting second goal
        
        # Episode ends when both goals are visited
        done = len(new_visited_goals) == 2
        
        return new_agent1_pos, new_agent2_pos, new_visited_goals, reward, done
    
    def state_to_features(self, agent1_pos: Tuple[int, int], 
                         agent2_pos: Tuple[int, int],
                         visited_goals: Set[Tuple[int, int]] = None) -> np.ndarray:
        """Convert state to feature vector: [agent1_x, agent1_y, agent2_x, agent2_y, goal1_visited, goal2_visited]."""
        agent1_y, agent1_x = agent1_pos
        agent2_y, agent2_x = agent2_pos
        
        if visited_goals is None:
            goal1_visited = 0.0
            goal2_visited = 0.0
        else:
            goal1_visited = 1.0 if self.goal1 in visited_goals else 0.0
            goal2_visited = 1.0 if self.goal2 in visited_goals else 0.0
        
        return np.array([agent1_x, agent1_y, agent2_x, agent2_y, goal1_visited, goal2_visited], dtype=np.float32)
    
    def joint_action_to_index(self, action1: int, action2: int) -> int:
        """Convert joint action (action1, action2) to linear index (0-15)."""
        return action1 * 4 + action2
    
    def index_to_joint_action(self, index: int) -> Tuple[int, int]:
        """Convert linear index to joint action (action1, action2)."""
        action1 = index // 4
        action2 = index % 4
        return action1, action2
    
    def get_all_start_pairs(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all possible distinct start position pairs (excluding goals)."""
        pairs = []
        for agent1_pos in self.valid_positions:
            for agent2_pos in self.valid_positions:
                if agent1_pos != agent2_pos:
                    pairs.append((agent1_pos, agent2_pos))
        return pairs

