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
from functools import partial
# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn_learning import ReplayBuffer

# Try to import parallel processing
# Note: With CUDA, we must use 'spawn' start method instead of 'fork'
# We ensure only numpy arrays are passed to workers (no CUDA tensors)
import platform
IS_WINDOWS = platform.system() == 'Windows'

# Check if CUDA is available
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except:
    CUDA_AVAILABLE = False

try:
    from multiprocessing import Pool, cpu_count, set_start_method, get_start_method
    HAS_MULTIPROCESSING = True
    
    # Set start method for multiprocessing
    # 'spawn' works with CUDA, 'fork' does not
    if CUDA_AVAILABLE and not IS_WINDOWS:
        # On Linux with CUDA, use 'spawn' instead of default 'fork'
        try:
            current_method = get_start_method(allow_none=True)
            if current_method != 'spawn':
                set_start_method('spawn', force=True)
        except RuntimeError:
            # Start method already set, that's fine
            pass
except ImportError:
    HAS_MULTIPROCESSING = False
    cpu_count = lambda: 1

# Try to import faster LP solvers
try:
    import highspy
    HAS_HIGHSPY = True
except ImportError:
    HAS_HIGHSPY = False

try:
    from scipy.optimize import linprog
    # Use HiGHS method if available (faster than default)
    LP_METHOD = "highs" if hasattr(linprog, '__defaults__') else "interior-point"
except:
    LP_METHOD = "interior-point"


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


def solve_nash_equilibrium_fast_4x4(M: np.ndarray):
    """
    Fast approximate Nash solver for 4x4 games using iterative best response.
    Much faster than LP for small games.
    """
    M = np.asarray(M, float)
    m, n = M.shape
    
    # Initialize with uniform
    x = np.ones(m) / m
    y = np.ones(n) / n
    
    # Iterative best response (typically converges in < 10 iterations for 4x4)
    for _ in range(20):  # Max iterations
        # Best response for pursuer (maximizer)
        pursuer_payoffs = M @ y
        best_pursuer = np.argmax(pursuer_payoffs)
        x_new = np.zeros(m)
        x_new[best_pursuer] = 1.0
        
        # Best response for evader (minimizer)
        evader_payoffs = x @ M
        best_evader = np.argmin(evader_payoffs)
        y_new = np.zeros(n)
        y_new[best_evader] = 1.0
        
        # Check convergence
        if np.allclose(x, x_new) and np.allclose(y, y_new):
            break
        
        # Soft update (mixing parameter for stability)
        x = 0.7 * x + 0.3 * x_new
        y = 0.7 * y + 0.3 * y_new
    
    # Compute value
    v = float(x @ M @ y)
    return x, y, v


def _solve_nash_for_q_values_helper(q_values: np.ndarray, use_fast_nash: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    """Helper function for parallel Nash solving (must be module-level for multiprocessing)."""
    M = _q_values_to_matrix_static(q_values)
    return solve_nash_equilibrium(M, use_fast_approx=use_fast_nash)


def _q_values_to_matrix_static(q_values: np.ndarray) -> np.ndarray:
    """Convert Q-values (16) to payoff matrix (4x4) - static version for multiprocessing."""
    M = np.zeros((4, 4))
    for pursuer_action in range(4):
        for evader_action in range(4):
            joint_idx = pursuer_action * 4 + evader_action
            M[pursuer_action, evader_action] = q_values[joint_idx]
    return M


def solve_nash_equilibrium(M: np.ndarray, jitter=1e-8, max_retries=2, use_fast_approx=False):
    """
    Solve Nash equilibrium using linear programming.
    M is a matrix where M[i, j] is the payoff for pursuer action i and evader action j.
    Returns (pursuer_policy, evader_policy, value) for the pursuer's perspective.
    
    Args:
        use_fast_approx: If True and M is 4x4, use fast iterative method instead of LP
    """
    M = np.asarray(M, float)
    m, n = M.shape
    
    # For 4x4 games, use fast approximate method if requested
    if use_fast_approx and m == 4 and n == 4:
        return solve_nash_equilibrium_fast_4x4(M)
    
    # sanitize
    if not np.all(np.isfinite(M)):
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    
    # center & scale for conditioning
    mu = float(np.mean(M))
    Ms = M - mu
    s = float(np.max(np.abs(Ms)))
    if s < 1e-12:
        # essentially constant matrix
        x = np.ones(m) / m
        y = np.ones(n) / n
        v = mu
        return x, y, v
    Ms /= s
    
    def _solve_raw(A):
        num = m + n + 1
        xs = slice(0, m)
        ys = slice(m, m + n)
        vidx = m + n
        
        c = np.zeros(num, dtype=np.float64)
        c[vidx] = -1.0
        
        # Pre-allocate constraint matrices for better performance
        A_ub = np.zeros((m + n, num), dtype=np.float64)
        b_ub = np.zeros(m + n, dtype=np.float64)
        
        # v <= x^T A[:,j] for all j (evader actions)
        for j in range(n):
            A_ub[j, xs] = -A[:, j]
            A_ub[j, vidx] = 1.0
            b_ub[j] = 0.0
        
        # A[i,:] y <= v for all i (pursuer actions)
        for i in range(m):
            A_ub[n + i, ys] = A[i, :]
            A_ub[n + i, vidx] = -1.0
            b_ub[n + i] = 0.0
        
        A_eq = np.zeros((2, num), dtype=np.float64)
        A_eq[0, xs] = 1.0
        A_eq[1, ys] = 1.0
        b_eq = np.array([1.0, 1.0], dtype=np.float64)
        bounds = [(0, None)] * m + [(0, None)] * n + [(None, None)]
        
        # Use fastest available method
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method=LP_METHOD,
                     options={'maxiter': 1000, 'presolve': True})
        return res
    
    # try solve, with jittered retries if needed
    A = Ms
    for attempt in range(max_retries + 1):
        res = _solve_raw(A)
        if res.status == 0 and np.isfinite(res.fun):
            z = res.x
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
                 batch_size: int = 64, buffer_size: int = 50000,
                 use_fast_nash: bool = True,
                 update_cache_after_training: bool = False,
                 num_workers: int = None):
        self.joint_action_size = joint_action_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.tau = tau
        self.batch_size = batch_size
        self.use_fast_nash = use_fast_nash  # Use fast approximate Nash for 4x4 games
        self.update_cache_after_training = update_cache_after_training  # Update cache after each training step
        self.num_workers = num_workers if num_workers is not None else (cpu_count() if HAS_MULTIPROCESSING else 1)
        
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
        
        # Cache for Nash policies and values (with size limit)
        # Separate caches for Q-network and target network
        self._nash_cache_q = {}  # Cache for main Q-network
        self._nash_cache_max_size = 5000  # Limit cache size to prevent memory issues


    
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
        """Solve Nash equilibrium for a given state using Q-network. Returns (pursuer_policy, evader_policy, value)."""
        state_key = tuple(state)
        
        # Check cache (but be aware it may be stale during training)
        if state_key in self._nash_cache_q:
            return self._nash_cache_q[state_key]
        
        # Get Q-values from Q-network
        q_values = self.get_q_values(state)
        
        # Convert to payoff matrix
        M = self._q_values_to_matrix(q_values)
        
        # Solve Nash equilibrium (use fast approx for 4x4 games if enabled)
        pursuer_policy, evader_policy, value = solve_nash_equilibrium(
            M, use_fast_approx=self.use_fast_nash)
        
        # Cache result (with size limit)
        if len(self._nash_cache_q) >= self._nash_cache_max_size:
            # Remove oldest entries (simple FIFO by clearing half)
            keys_to_remove = list(self._nash_cache_q.keys())[:self._nash_cache_max_size // 2]
            for key in keys_to_remove:
                del self._nash_cache_q[key]
        self._nash_cache_q[state_key] = (pursuer_policy, evader_policy, value)
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
        
        # Convert to tensors efficiently - use torch.from_numpy to avoid unnecessary copies
        states = torch.from_numpy(np.array([e[0] for e in batch], dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array([e[1] for e in batch], dtype=np.int64)).to(self.device)
        rewards = torch.from_numpy(np.array([e[2] for e in batch], dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array([e[3] for e in batch], dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array([e[4] for e in batch], dtype=bool)).to(self.device)
        
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
        
        # optimizer.step() updates ALL Q-network weights, making all cached Nash solutions stale
        # Clear Q-network cache immediately after weight update
        self._nash_cache_q.clear()
        
        # Track TD error
        td_error = (target_q_values - current_q_values.squeeze()).abs().mean().item()
        self.episode_td_errors.append(td_error)
        
        # Soft update target network
        self._soft_update_target_network()
        
        # Optionally update cache for all states after training step
        # This pre-computes Nash equilibria so action selection is faster
        if self.update_cache_after_training and hasattr(self, '_all_states_for_cache'):
            self._update_cache_for_all_states()
    
    def _soft_update_target_network(self):
        """Soft update target network using tau."""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                            self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                   (1.0 - self.tau) * target_param.data)
    
    def start_episode(self):
        """Called at the start of each episode."""
        self.episode_td_errors = []
        # Cache invalidation is handled in train_step based on training steps
    
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
            # Use torch.from_numpy for better performance (avoids copy)
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def _solve_nash_for_state_target(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve Nash equilibrium using target network."""
        # Get Q-values from target network
        with torch.no_grad():
            # Use torch.from_numpy for better performance (avoids copy)
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.target_network(state_tensor).cpu().numpy().flatten()
        
        # Convert to payoff matrix
        M = self._q_values_to_matrix(q_values)
        
        # Solve Nash equilibrium (use fast approx for 4x4 games if enabled)
        pursuer_policy, evader_policy, value = solve_nash_equilibrium(
            M, use_fast_approx=self.use_fast_nash)
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
    
    def _batch_compute_q_values(self, states: List[np.ndarray]) -> np.ndarray:
        """Batch compute Q-values for multiple states on GPU."""
        if not states:
            return np.array([])
        
        # Stack all states into a batch tensor
        states_tensor = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        
        # Forward pass on GPU (batch computation)
        with torch.no_grad():
            q_values_batch = self.q_network(states_tensor)  # [batch_size, 16]
        
        return q_values_batch.cpu().numpy()
    
    def _solve_nash_for_q_values(self, q_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve Nash equilibrium given Q-values (helper for parallel processing)."""
        return _solve_nash_for_q_values_helper(q_values, self.use_fast_nash)
    
    def _update_cache_for_all_states(self):
        """Update cache for all states in parallel (called after training step)."""
        if not hasattr(self, '_all_states_for_cache') or not self._all_states_for_cache:
            return
        
        all_states = self._all_states_for_cache
        
        # Batch compute all Q-values on GPU (very fast)
        q_values_batch = self._batch_compute_q_values(all_states)
        
        # Solve Nash equilibria
        # Note: Multiprocessing is disabled when CUDA is available or on Windows
        # We use sequential processing but still benefit from batch GPU computation above
        if HAS_MULTIPROCESSING and self.num_workers > 1 and len(all_states) > 10:
            # Parallel processing (use module-level function) - only on non-Windows
            solve_func = partial(_solve_nash_for_q_values_helper, use_fast_nash=self.use_fast_nash)
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(solve_func, q_values_batch)
        else:
            # Sequential processing (Windows or small batches)
            # Still fast because Q-values were computed in batch on GPU
            results = [self._solve_nash_for_q_values(qv) for qv in q_values_batch]
        
        # Update cache
        for state, (pursuer_policy, evader_policy, value) in zip(all_states, results):
            state_key = tuple(state)
            self._nash_cache_q[state_key] = (pursuer_policy, evader_policy, value)
    
    def set_all_states_for_cache(self, all_states: List[np.ndarray]):
        """Set the list of all states to cache (call this once at the start of training)."""
        self._all_states_for_cache = all_states
    
    def snapshot_policies_and_values(self, env, all_states: List[np.ndarray]) -> Dict:
        """Snapshot current policies and values for all states (parallelized version)."""
        policies = {}
        values = {}
        pursuer_policies = {}
        evader_policies = {}
        
        # Batch compute all Q-values on GPU (much faster than one-by-one)
        q_values_batch = self._batch_compute_q_values(all_states)
        
        # Solve Nash equilibria
        if HAS_MULTIPROCESSING and self.num_workers > 1 and len(all_states) > 10:
            # Parallel processing (use module-level function for pickling)
            solve_func = partial(_solve_nash_for_q_values_helper, use_fast_nash=self.use_fast_nash)
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(solve_func, q_values_batch)
        else:
            # Sequential processing (fallback)
            results = [self._solve_nash_for_q_values(qv) for qv in q_values_batch]
        
        # Organize results
        for state, (pursuer_policy, evader_policy, value) in zip(all_states, results):
            state_key = tuple(state)
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
        self._nash_cache_q.clear()

