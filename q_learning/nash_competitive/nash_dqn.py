"""
Nash DQN for Competitive Two-Agent Game
=======================================

Single Q-network that outputs Q-values for all joint actions (16 values).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Dict, List
from scipy.optimize import linprog
import sys
import os
# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn_learning import ReplayBuffer


class NashDQNNetwork(nn.Module):
    """DQN Network for joint actions: 2 hidden layers of 128 neurons each."""
    
    def __init__(self, input_size: int = 4, output_size: int = 16, hidden_size: int = 128):
        super(NashDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def solve_nash_equilibrium(M: np.ndarray, jitter=1e-8, max_retries=2):
    """
    Solve Nash equilibrium using linear programming.
    M is a matrix where M[i, j] is the payoff for pursuer action i and evader action j.
    Returns (pursuer_policy, evader_policy, value) for the pursuer's perspective.
    """
    M = np.asarray(M, float)
    # sanitize
    if not np.all(np.isfinite(M)):
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    
    # center & scale for conditioning
    mu = float(np.mean(M))
    Ms = M - mu
    s = float(np.max(np.abs(Ms)))
    if s < 1e-12:
        # essentially constant matrix
        m, n = M.shape
        x = np.ones(m) / m
        y = np.ones(n) / n
        v = mu
        return x, y, v
    Ms /= s
    
    def _solve_raw(A):
        m, n = A.shape
        num = m + n + 1
        xs = slice(0, m)
        ys = slice(m, m + n)
        vidx = m + n
        
        c = np.zeros(num)
        c[vidx] = -1.0
        A_ub = []
        b_ub = []
        
        # v <= x^T A[:,j] for all j (evader actions)
        for j in range(n):
            row = np.zeros(num)
            row[xs] = -A[:, j]
            row[vidx] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
        
        # A[i,:] y <= v for all i (pursuer actions)
        for i in range(m):
            row = np.zeros(num)
            row[ys] = A[i, :]
            row[vidx] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
        
        A_ub = np.vstack(A_ub)
        b_ub = np.array(b_ub)
        A_eq = np.zeros((2, num))
        A_eq[0, xs] = 1.0
        A_eq[1, ys] = 1.0
        b_eq = np.array([1.0, 1.0])
        bounds = [(0, None)] * m + [(0, None)] * n + [(None, None)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
        return res
    
    # try solve, with jittered retries if needed
    A = Ms
    for attempt in range(max_retries + 1):
        res = _solve_raw(A)
        if res.status == 0 and np.isfinite(res.fun):
            z = res.x
            m, n = Ms.shape
            x = np.clip(z[:m], 0, None)
            y = np.clip(z[m:m + n], 0, None)
            v = float(z[m + n])
            sx, sy = x.sum(), y.sum()
            x = x / sx if sx > 0 else np.ones(m) / m
            y = y / sy if sy > 0 else np.ones(n) / n
            # rescale value back
            v = v * s + mu
            return x, y, v
        # add tiny jitter and retry
        A = Ms + np.random.default_rng().normal(scale=jitter, size=Ms.shape)
    
    # last resort: return uniform
    m, n = M.shape
    x = np.ones(m) / m
    y = np.ones(n) / n
    v = float(np.mean(M))
    return x, y, v


class NashDQN:
    """Nash DQN agent with shared Q-network for joint actions."""
    
    def __init__(self, input_size: int = 4, joint_action_size: int = 16,
                 learning_rate: float = 5e-4, gamma: float = 0.95, 
                 epsilon_start: float = 0.5, epsilon_end: float = 0.01, 
                 epsilon_decay: float = 0.995, tau: float = 0.01,
                 batch_size: int = 64, buffer_size: int = 50000):
        self.joint_action_size = joint_action_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.tau = tau
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Single shared network for joint actions
        self.q_network = NashDQNNetwork(input_size, joint_action_size).to(self.device)
        self.target_network = NashDQNNetwork(input_size, joint_action_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Track TD errors per episode
        self.episode_td_errors = []
        
        # Cache for Nash policies and values
        self._nash_cache = {}
    
    def _q_values_to_matrix(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q-values (16) to payoff matrix (4x4) for pursuer."""
        # Q-values are indexed as: pursuer_action * 4 + evader_action
        # Matrix M[i, j] = Q-value for pursuer action i, evader action j
        M = np.zeros((4, 4))
        for pursuer_action in range(4):
            for evader_action in range(4):
                joint_idx = pursuer_action * 4 + evader_action
                M[pursuer_action, evader_action] = q_values[joint_idx]
        return M
    
    def _solve_nash_for_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve Nash equilibrium for a given state. Returns (pursuer_policy, evader_policy, value)."""
        state_key = tuple(state)
        if state_key in self._nash_cache:
            return self._nash_cache[state_key]
        
        # Get Q-values
        q_values = self.get_q_values(state)
        
        # Convert to payoff matrix
        M = self._q_values_to_matrix(q_values)
        
        # Solve Nash equilibrium
        pursuer_policy, evader_policy, value = solve_nash_equilibrium(M)
        
        # Cache result
        self._nash_cache[state_key] = (pursuer_policy, evader_policy, value)
        return pursuer_policy, evader_policy, value
    
    def choose_joint_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, int]:
        """Epsilon-greedy joint action selection using Nash policies."""
        if training and np.random.random() < self.epsilon:
            # Random joint action
            pursuer_action = np.random.randint(4)
            evader_action = np.random.randint(4)
            return pursuer_action, evader_action
        else:
            # Use Nash equilibrium policies
            pursuer_policy, evader_policy, _ = self._solve_nash_for_state(state)
            
            # Sample from Nash policies
            pursuer_action = np.random.choice(4, p=pursuer_policy)
            evader_action = np.random.choice(4, p=evader_policy)
            return pursuer_action, evader_action
    
    def get_policy_distribution(self, state: np.ndarray) -> np.ndarray:
        """Get policy distribution over joint actions from Nash equilibrium."""
        pursuer_policy, evader_policy, _ = self._solve_nash_for_state(state)
        
        # Joint policy is product of marginal policies
        joint_policy = np.outer(pursuer_policy, evader_policy).flatten()
        return joint_policy
    
    def update(self, state: np.ndarray, joint_action_idx: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, joint_action_idx, reward, next_state, done)
    
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
        
        # Next Q values from target network - use Nash value instead of max
        with torch.no_grad():
            next_nash_values = []
            for i in range(len(next_states)):
                if dones[i]:
                    next_nash_values.append(0.0)
                else:
                    next_state = next_states[i].cpu().numpy()
                    _, _, nash_value = self._solve_nash_for_state_target(next_state)
                    next_nash_values.append(nash_value)
            next_nash_values = torch.FloatTensor(next_nash_values).to(self.device)
            target_q_values = rewards + (self.gamma * next_nash_values * ~dones)
        
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
        # Clear cache periodically to avoid stale solutions
        if len(self._nash_cache) > 1000:
            self._nash_cache.clear()
    
    def end_episode(self) -> float:
        """Called at the end of each episode. Returns average TD error."""
        if self.episode_td_errors:
            avg_td = np.mean(self.episode_td_errors)
        else:
            avg_td = 0.0
        self.episode_td_errors = []  # Reset for next episode
        return avg_td
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all joint actions."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def _solve_nash_for_state_target(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve Nash equilibrium using target network."""
        # Get Q-values from target network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.target_network(state_tensor).cpu().numpy().flatten()
        
        # Convert to payoff matrix
        M = self._q_values_to_matrix(q_values)
        
        # Solve Nash equilibrium
        pursuer_policy, evader_policy, value = solve_nash_equilibrium(M)
        return pursuer_policy, evader_policy, value
    
    def get_value(self, state: np.ndarray) -> float:
        """Get Nash equilibrium value for a given state."""
        _, _, value = self._solve_nash_for_state(state)
        return float(value)
    
    def get_policy(self, state: np.ndarray) -> Tuple[int, int]:
        """Get joint action from Nash equilibrium policies (argmax)."""
        pursuer_policy, evader_policy, _ = self._solve_nash_for_state(state)
        
        # Get argmax actions
        pursuer_action = int(np.argmax(pursuer_policy))
        evader_action = int(np.argmax(evader_policy))
        return pursuer_action, evader_action
    
    def get_pursuer_policy(self, state: np.ndarray) -> np.ndarray:
        """Get pursuer's Nash equilibrium policy."""
        pursuer_policy, _, _ = self._solve_nash_for_state(state)
        return pursuer_policy
    
    def get_evader_policy(self, state: np.ndarray) -> np.ndarray:
        """Get evader's Nash equilibrium policy."""
        _, evader_policy, _ = self._solve_nash_for_state(state)
        return evader_policy
    
    def snapshot_policies_and_values(self, env, all_states: List[np.ndarray]) -> Dict:
        """Snapshot current policies and values for all states."""
        policies = {}
        values = {}
        pursuer_policies = {}
        evader_policies = {}
        
        for state in all_states:
            state_key = tuple(state)
            pursuer_policy, evader_policy, value = self._solve_nash_for_state(state)
            policies[state_key] = np.outer(pursuer_policy, evader_policy).flatten()
            pursuer_policies[state_key] = pursuer_policy
            evader_policies[state_key] = evader_policy
            values[state_key] = value
        
        return {
            'policies': policies,
            'pursuer_policies': pursuer_policies,
            'evader_policies': evader_policies,
            'values': values
        }
    
    def clear_cache(self):
        """Clear the Nash equilibrium cache."""
        self._nash_cache.clear()

