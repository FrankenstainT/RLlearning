"""
Deep Q-Network (DQN) Algorithm
==============================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List


class DQNNetwork(nn.Module):
    """DQN Network: 2 hidden layers of 64 neurons each."""
    
    def __init__(self, input_size: int = 2, output_size: int = 4, hidden_size: int = 64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNLearner:
    """Deep Q-Network agent."""
    
    def __init__(self, input_size: int = 2, action_size: int = 4,
                 learning_rate: float = 1e-3, gamma: float = 0.95, 
                 epsilon: float = 0.15, tau: float = 0.005,
                 batch_size: int = 64, buffer_size: int = 10000):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(input_size, action_size).to(self.device)
        self.target_network = DQNNetwork(input_size, action_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Track TD errors per episode
        self.episode_td_errors = []
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track TD error
        td_error = (target_q_values - current_q_values.squeeze()).abs().mean().item()
        self.episode_td_errors.append(td_error)
        
        # Soft update target network
        self._soft_update_target_network()
    
    def _soft_update_target_network(self):
        """Soft update target network using tau."""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                            self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                   (1.0 - self.tau) * target_param.data)
    
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
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value (max Q) for a given state."""
        q_values = self.get_q_values(state)
        return float(np.max(q_values))
    
    def get_policy(self, state: np.ndarray) -> int:
        """Get greedy policy action for a given state."""
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

