"""
Two-Agent DQN for Pursuer-Evader Game
======================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List
from dqn_learning import DQNNetwork, ReplayBuffer


class TwoAgentDQN:
    """Two-agent DQN system for pursuer-evader game."""
    
    def __init__(self, input_size: int = 4, action_size: int = 4,
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
        
        # Two separate DQNs
        self.pursuer_network = DQNNetwork(input_size, action_size).to(self.device)
        self.pursuer_target = DQNNetwork(input_size, action_size).to(self.device)
        self.pursuer_target.load_state_dict(self.pursuer_network.state_dict())
        self.pursuer_optimizer = optim.Adam(self.pursuer_network.parameters(), lr=learning_rate)
        self.pursuer_buffer = ReplayBuffer(buffer_size)
        
        self.evader_network = DQNNetwork(input_size, action_size).to(self.device)
        self.evader_target = DQNNetwork(input_size, action_size).to(self.device)
        self.evader_target.load_state_dict(self.evader_network.state_dict())
        self.evader_optimizer = optim.Adam(self.evader_network.parameters(), lr=learning_rate)
        self.evader_buffer = ReplayBuffer(buffer_size)
        
        # Track TD errors per episode
        self.pursuer_td_errors = []
        self.evader_td_errors = []
    
    def choose_action(self, state: np.ndarray, agent: str, training: bool = True) -> int:
        """Epsilon-greedy action selection for specified agent."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if agent == 'pursuer':
                    q_values = self.pursuer_network(state_tensor)
                else:
                    q_values = self.evader_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool, agent: str):
        """Store experience in replay buffer for specified agent."""
        if agent == 'pursuer':
            self.pursuer_buffer.push(state, action, reward, next_state, done)
        else:
            self.evader_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, agent: str):
        """Perform one training step from replay buffer for specified agent."""
        if agent == 'pursuer':
            buffer = self.pursuer_buffer
            network = self.pursuer_network
            target = self.pursuer_target
            optimizer = self.pursuer_optimizer
            td_errors = self.pursuer_td_errors
        else:
            buffer = self.evader_buffer
            network = self.evader_network
            target = self.evader_target
            optimizer = self.evader_optimizer
            td_errors = self.evader_td_errors
        
        if len(buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = target(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track TD error
        td_error = (target_q_values - current_q_values.squeeze()).abs().mean().item()
        td_errors.append(td_error)
        
        # Soft update target network
        self._soft_update_target_network(agent)
    
    def _soft_update_target_network(self, agent: str):
        """Soft update target network using tau."""
        if agent == 'pursuer':
            target = self.pursuer_target
            local = self.pursuer_network
        else:
            target = self.evader_target
            local = self.evader_network
        
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                   (1.0 - self.tau) * target_param.data)
    
    def start_episode(self):
        """Called at the start of each episode."""
        self.pursuer_td_errors = []
        self.evader_td_errors = []
    
    def end_episode(self) -> Tuple[float, float]:
        """Called at the end of each episode. Returns average TD errors."""
        pursuer_avg = np.mean(self.pursuer_td_errors) if self.pursuer_td_errors else 0.0
        evader_avg = np.mean(self.evader_td_errors) if self.evader_td_errors else 0.0
        self.pursuer_td_errors = []
        self.evader_td_errors = []
        return pursuer_avg, evader_avg
    
    def get_q_values(self, state: np.ndarray, agent: str) -> np.ndarray:
        """Get Q-values for a given state and agent."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if agent == 'pursuer':
                q_values = self.pursuer_network(state_tensor)
            else:
                q_values = self.evader_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def get_value(self, state: np.ndarray, agent: str) -> float:
        """Get value (max Q) for a given state and agent."""
        q_values = self.get_q_values(state, agent)
        return float(np.max(q_values))
    
    def get_policy(self, state: np.ndarray, agent: str) -> int:
        """Get greedy policy action for a given state and agent."""
        q_values = self.get_q_values(state, agent)
        return int(np.argmax(q_values))

