"""
Pure Nash Q-Learning for Competitive Two-Agent Game
===================================================

Tabular Q-learning with Nash equilibrium action selection using linear programming.
No neural networks - pure tabular approach.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.optimize import linprog


def solve_nash_equilibrium_lp(M: np.ndarray, jitter=1e-8, max_retries=2):
    """
    Solve Nash equilibrium using linear programming.
    M is a matrix where M[i, j] is the payoff for pursuer action i and evader action j.
    Returns (pursuer_policy, evader_policy, value) for the pursuer's perspective.
    """
    M = np.asarray(M, float)
    m, n = M.shape
    
    # Sanitize
    if not np.all(np.isfinite(M)):
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Center & scale for conditioning
    mu = float(np.mean(M))
    Ms = M - mu
    s = float(np.max(np.abs(Ms)))
    if s < 1e-12:
        # Essentially constant matrix
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
        
        # Pre-allocate constraint matrices
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
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs',
                         options={'maxiter': 1000, 'presolve': True})
        except:
            # Fallback to interior-point if highs not available
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='interior-point',
                         options={'maxiter': 1000})
        return res
    
    # Try solve, with jittered retries if needed
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
            # Rescale value back
            v = v * s + mu
            return x, y, v
        # Add tiny jitter and retry
        A = Ms + np.random.default_rng().normal(scale=jitter, size=Ms.shape)
    
    # Last resort: return uniform
    x = np.ones(m) / m
    y = np.ones(n) / n
    v = float(np.mean(M))
    return x, y, v


class NashQLearning:
    """Pure Nash Q-Learning agent with tabular Q-table."""
    
    def __init__(self, state_size: int = 16,  # 4x4 grid = 16 states for each agent
                 joint_action_size: int = 16,  # 4 × 4 joint actions
                 learning_rate: float = 0.1,
                 gamma: float = 0.95,
                 epsilon_start: float = 0.5,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 use_decreasing_lr: bool = True,
                 optimistic_init: float = 0.0):
        """
        Initialize Nash Q-Learning agent.
        
        Args:
            state_size: Number of possible states (for each agent position)
            joint_action_size: Number of joint actions (4 × 4 = 16)
            learning_rate: Initial learning rate (alpha_0) - will decrease if use_decreasing_lr=True
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay factor per episode
            use_decreasing_lr: If True, use decreasing learning rate: alpha_t = alpha_0 / (1 + t/tau)
                               where tau=10000 slows decay while maintaining convergence
            optimistic_init: Initial Q-value (optimistic initialization). 
                             If > 0, encourages exploration of unvisited states.
                             Should be small (e.g., 0.1) relative to reward scale to avoid
                             biasing Nash equilibrium computation with constant matrices.
        """
        self.state_size = state_size
        self.joint_action_size = joint_action_size
        self.learning_rate_initial = learning_rate
        self.use_decreasing_lr = use_decreasing_lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.optimistic_init = optimistic_init
        
        # Track total updates for decreasing learning rate
        self.total_updates = 0
        
        # Q-table: Q[state_key][joint_action] = Q-value
        # state_key is a tuple (pursuer_y, pursuer_x, evader_y, evader_x)
        self.Q = {}
        
        # Track visit frequencies for each state
        self.visit_counts = {}  # state_key -> visit count
        
        # Track state-action visit counts for better coverage
        self.state_action_visits = {}  # (state_key, joint_action_idx) -> visit count
        
        # Track TD errors per episode
        self.episode_td_errors = []
        
        # Cache for Nash policies and values
        self._nash_cache = {}
    
    def _state_to_key(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert positions to state key."""
        pursuer_y, pursuer_x = pursuer_pos
        evader_y, evader_x = evader_pos
        return (pursuer_y, pursuer_x, evader_y, evader_x)
    
    def _get_q_values(self, state_key: Tuple[int, int, int, int]) -> np.ndarray:
        """Get Q-values for all joint actions in a state."""
        if state_key not in self.Q:
            # Optimistic initialization: initialize with positive value to encourage exploration
            if self.optimistic_init > 0:
                self.Q[state_key] = np.full(self.joint_action_size, self.optimistic_init, dtype=np.float64)
            else:
                self.Q[state_key] = np.zeros(self.joint_action_size, dtype=np.float64)
        return self.Q[state_key]
    
    def _get_learning_rate(self) -> float:
        """Get current learning rate (decreasing if enabled)."""
        if self.use_decreasing_lr:
            # Decreasing learning rate: alpha_t = alpha_0 / (1 + t / tau)
            # Using tau = 10000 to slow down decay while still satisfying convergence conditions
            # This satisfies: sum(alpha) = inf, sum(alpha^2) < inf
            # Slower decay allows more learning in later stages
            tau = 10000.0  # Decay scaling factor
            return self.learning_rate_initial / (1.0 + self.total_updates / tau)
        else:
            return self.learning_rate_initial
    
    def _q_values_to_matrix(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q-values (16) to payoff matrix (4x4) for pursuer."""
        M = np.zeros((4, 4))
        for pursuer_action in range(4):
            for evader_action in range(4):
                joint_idx = pursuer_action * 4 + evader_action
                M[pursuer_action, evader_action] = q_values[joint_idx]
        return M
    
    def _solve_nash_for_state(self, state_key: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve Nash equilibrium for a given state. Returns (pursuer_policy, evader_policy, value)."""
        # Check cache
        if state_key in self._nash_cache:
            return self._nash_cache[state_key]
        
        # Get Q-values
        q_values = self._get_q_values(state_key)
        
        # Convert to payoff matrix
        M = self._q_values_to_matrix(q_values)
        
        # Solve Nash equilibrium using LP
        pursuer_policy, evader_policy, value = solve_nash_equilibrium_lp(M)
        
        # Cache result
        self._nash_cache[state_key] = (pursuer_policy, evader_policy, value)
        return pursuer_policy, evader_policy, value
    
    def choose_joint_action(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                           training: bool = True) -> Tuple[int, int]:
        """Epsilon-greedy joint action selection using Nash policies."""
        state_key = self._state_to_key(pursuer_pos, evader_pos)
        
        if training and np.random.random() < self.epsilon:
            # Random joint action
            pursuer_action = np.random.randint(4)
            evader_action = np.random.randint(4)
            return pursuer_action, evader_action
        else:
            # Use Nash equilibrium policies
            pursuer_policy, evader_policy, _ = self._solve_nash_for_state(state_key)
            
            # Sample from Nash policies
            pursuer_action = np.random.choice(4, p=pursuer_policy)
            evader_action = np.random.choice(4, p=evader_policy)
            return pursuer_action, evader_action
    
    def update(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
               pursuer_action: int, evader_action: int,
               pursuer_reward: float, evader_reward: float,
               next_pursuer_pos: Tuple[int, int], next_evader_pos: Tuple[int, int],
               done: bool):
        """
        Update Q-values using Nash Q-Learning update rule.
        
        For Nash Q-Learning, we update using the Nash equilibrium value of the next state.
        We use the pursuer's reward for the Q-update (since Q represents pursuer's perspective).
        """
        state_key = self._state_to_key(pursuer_pos, evader_pos)
        next_state_key = self._state_to_key(next_pursuer_pos, next_evader_pos)
        
        # Track state visit
        self.visit_counts[state_key] = self.visit_counts.get(state_key, 0) + 1
        
        # Get joint action index
        joint_action_idx = pursuer_action * 4 + evader_action
        
        # Track state-action visit for better coverage analysis
        sa_key = (state_key, joint_action_idx)
        self.state_action_visits[sa_key] = self.state_action_visits.get(sa_key, 0) + 1
        
        # Get current Q-value
        q_values = self._get_q_values(state_key)
        current_q = q_values[joint_action_idx]
        
        # Compute target Q-value using Nash equilibrium value of next state
        if done:
            target_q = pursuer_reward
        else:
            # Get Nash equilibrium value for next state
            _, _, nash_value = self._solve_nash_for_state(next_state_key)
            target_q = pursuer_reward + self.gamma * nash_value
        
        # TD error
        td_error = target_q - current_q
        
        # Get current learning rate (decreasing if enabled)
        current_lr = self._get_learning_rate()
        
        # Update Q-value using Nash Q-Learning update
        q_values[joint_action_idx] = current_q + current_lr * td_error
        
        # Increment total updates for decreasing learning rate
        self.total_updates += 1
        
        # Track TD error
        self.episode_td_errors.append(abs(td_error))
        
        # Clear cache for this state (Q-value changed)
        if state_key in self._nash_cache:
            del self._nash_cache[state_key]
    
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
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_value(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int]) -> float:
        """Get Nash equilibrium value for a given state."""
        state_key = self._state_to_key(pursuer_pos, evader_pos)
        _, _, value = self._solve_nash_for_state(state_key)
        return float(value)
    
    def get_policy(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get joint action from Nash equilibrium policies (argmax)."""
        state_key = self._state_to_key(pursuer_pos, evader_pos)
        pursuer_policy, evader_policy, _ = self._solve_nash_for_state(state_key)
        
        # Get argmax actions
        pursuer_action = int(np.argmax(pursuer_policy))
        evader_action = int(np.argmax(evader_policy))
        return pursuer_action, evader_action
    
    def get_policy_distribution(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int]) -> np.ndarray:
        """Get policy distribution over joint actions from Nash equilibrium."""
        state_key = self._state_to_key(pursuer_pos, evader_pos)
        pursuer_policy, evader_policy, _ = self._solve_nash_for_state(state_key)
        
        # Joint policy is product of marginal policies
        joint_policy = np.outer(pursuer_policy, evader_policy).flatten()
        return joint_policy
    
    def get_pursuer_policy(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int]) -> np.ndarray:
        """Get pursuer's Nash equilibrium policy."""
        state_key = self._state_to_key(pursuer_pos, evader_pos)
        pursuer_policy, _, _ = self._solve_nash_for_state(state_key)
        return pursuer_policy
    
    def get_evader_policy(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int]) -> np.ndarray:
        """Get evader's Nash equilibrium policy."""
        state_key = self._state_to_key(pursuer_pos, evader_pos)
        _, evader_policy, _ = self._solve_nash_for_state(state_key)
        return evader_policy
    
    def snapshot_policies_and_values(self, env, all_states: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Dict:
        """Snapshot current policies and values for all states."""
        policies = {}
        values = {}
        pursuer_policies = {}
        evader_policies = {}
        
        for pursuer_pos, evader_pos in all_states:
            state_key = self._state_to_key(pursuer_pos, evader_pos)
            pursuer_policy, evader_policy, value = self._solve_nash_for_state(state_key)
            
            # Store policies and values
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
    
    def save_q_table(self, filepath: str):
        """Save the Q-table to a file."""
        import json
        import os
        
        # Convert Q-table to a serializable format
        q_table_dict = {}
        for state_key, q_values in self.Q.items():
            # Convert tuple key to string for JSON serialization
            state_key_str = f"{state_key[0]}_{state_key[1]}_{state_key[2]}_{state_key[3]}"
            q_table_dict[state_key_str] = q_values.tolist()
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(q_table_dict, f, indent=2)
        print(f"Saved Q-table to {filepath}")
    
    def save_visit_frequencies(self, filepath: str):
        """Save state visit frequencies to a file."""
        import json
        import os
        
        # Convert visit counts to a serializable format
        visit_dict = {}
        for state_key, count in self.visit_counts.items():
            # Convert tuple key to string for JSON serialization
            state_key_str = f"{state_key[0]}_{state_key[1]}_{state_key[2]}_{state_key[3]}"
            visit_dict[state_key_str] = count
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(visit_dict, f, indent=2)
        print(f"Saved visit frequencies to {filepath}")
    
    def compute_all_policies_and_values(self, env) -> Dict:
        """Compute policies and values for all possible states."""
        policies = {}
        pursuer_policies = {}
        evader_policies = {}
        values = {}
        
        for pursuer_pos in env.valid_positions:
            for evader_pos in env.valid_positions:
                # Skip if evader is at goal (terminal state)
                if hasattr(env, 'evader_goal') and evader_pos == env.evader_goal:
                    continue
                if pursuer_pos != evader_pos:
                    state_key = self._state_to_key(pursuer_pos, evader_pos)
                    pursuer_policy, evader_policy, value = self._solve_nash_for_state(state_key)
                    
                    # Store policies and values
                    policies[state_key] = np.outer(pursuer_policy, evader_policy).flatten().tolist()
                    pursuer_policies[state_key] = pursuer_policy.tolist()
                    evader_policies[state_key] = evader_policy.tolist()
                    values[state_key] = float(value)
        
        return {
            'policies': policies,
            'pursuer_policies': pursuer_policies,
            'evader_policies': evader_policies,
            'values': values
        }
    
    def save_policies_and_values(self, env, filepath: str):
        """Compute and save all policies and values to a file."""
        import json
        import os
        
        print("Computing policies and values for all states...")
        result = self.compute_all_policies_and_values(env)
        
        # Convert tuple keys to strings for JSON serialization
        def convert_keys(d):
            return {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in d.items()}
        
        result_serializable = {
            'policies': convert_keys(result['policies']),
            'pursuer_policies': convert_keys(result['pursuer_policies']),
            'evader_policies': convert_keys(result['evader_policies']),
            'values': convert_keys(result['values'])
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(result_serializable, f, indent=2)
        print(f"Saved policies and values to {filepath}")
    
    def analyze_state_coverage(self, env) -> Dict:
        """Analyze Q-table coverage and provide diagnostics."""
        total_states = 0
        visited_states = 0
        unvisited_states = []
        visit_frequencies = []
        
        # Analyze state-action coverage
        total_state_actions = 0
        visited_state_actions = 0
        state_action_visit_freqs = []
        
        for pursuer_pos in env.valid_positions:
            for evader_pos in env.valid_positions:
                # Skip if evader is at goal (terminal state)
                if hasattr(env, 'evader_goal') and evader_pos == env.evader_goal:
                    continue
                if pursuer_pos != evader_pos:
                    total_states += 1
                    state_key = self._state_to_key(pursuer_pos, evader_pos)
                    visit_count = self.visit_counts.get(state_key, 0)
                    visit_frequencies.append(visit_count)
                    
                    # Check state-action coverage
                    for joint_action_idx in range(self.joint_action_size):
                        total_state_actions += 1
                        sa_key = (state_key, joint_action_idx)
                        sa_visit_count = self.state_action_visits.get(sa_key, 0)
                        state_action_visit_freqs.append(sa_visit_count)
                        if sa_visit_count > 0:
                            visited_state_actions += 1
                    
                    q_values = self._get_q_values(state_key)
                    # Check if Q-values are all zeros or optimistic init (unvisited or never updated)
                    if (np.allclose(q_values, 0) or 
                        (self.optimistic_init > 0 and np.allclose(q_values, self.optimistic_init))) and visit_count == 0:
                        unvisited_states.append((pursuer_pos, evader_pos))
                    else:
                        visited_states += 1
        
        visit_frequencies = np.array(visit_frequencies)
        state_action_visit_freqs = np.array(state_action_visit_freqs)
        
        return {
            'total_states': total_states,
            'visited_states': visited_states,
            'unvisited_states': len(unvisited_states),
            'coverage': visited_states / total_states if total_states > 0 else 0.0,
            'unvisited_state_list': unvisited_states[:10],  # First 10 as sample
            'visit_statistics': {
                'total_visits': int(np.sum(visit_frequencies)),
                'mean_visits': float(np.mean(visit_frequencies)),
                'median_visits': float(np.median(visit_frequencies)),
                'min_visits': int(np.min(visit_frequencies)),
                'max_visits': int(np.max(visit_frequencies)),
                'std_visits': float(np.std(visit_frequencies))
            },
            'state_action_coverage': {
                'total_state_actions': total_state_actions,
                'visited_state_actions': visited_state_actions,
                'coverage': visited_state_actions / total_state_actions if total_state_actions > 0 else 0.0,
                'mean_visits_per_sa': float(np.mean(state_action_visit_freqs)),
                'median_visits_per_sa': float(np.median(state_action_visit_freqs)),
                'min_visits_per_sa': int(np.min(state_action_visit_freqs)),
                'max_visits_per_sa': int(np.max(state_action_visit_freqs))
            }
        }
    
    def clear_cache(self):
        """Clear Nash equilibrium cache."""
        self._nash_cache.clear()

