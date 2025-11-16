"""
Tabular Q-Learning Algorithm
=============================
"""

import numpy as np
from typing import Tuple


class QLearner:
    """Tabular Q-Learning agent."""
    
    def __init__(self, state_size: int, action_size: int, 
                 alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.15):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.qtable = np.zeros((state_size, action_size))
        
        # Track TD errors per episode
        self.episode_td_errors = []
    
    def choose_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            # Break ties randomly
            max_q = np.max(self.qtable[state, :])
            max_indices = np.where(self.qtable[state, :] == max_q)[0]
            return np.random.choice(max_indices)
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Update Q-value using Q-learning rule."""
        current_q = self.qtable[state, action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.qtable[next_state, :])
        
        td_error = target - current_q
        self.qtable[state, action] += self.alpha * td_error
        
        # Track TD error for this episode
        self.episode_td_errors.append(abs(td_error))
    
    def start_episode(self):
        """Called at the start of each episode."""
        self.episode_td_errors = []
    
    def end_episode(self) -> float:
        """Called at the end of each episode. Returns average TD error."""
        if self.episode_td_errors:
            avg_td = np.mean(self.episode_td_errors)
        else:
            avg_td = 0.0
        self.episode_td_errors = []  # Reset for next episode
        return avg_td
    
    def get_q_values(self, state: int) -> np.ndarray:
        """Get Q-values for a given state."""
        return self.qtable[state, :]
    
    def get_value(self, state: int) -> float:
        """Get value (max Q) for a given state."""
        return np.max(self.qtable[state, :])
    
    def get_policy(self, state: int) -> int:
        """Get greedy policy action for a given state."""
        return np.argmax(self.qtable[state, :])

