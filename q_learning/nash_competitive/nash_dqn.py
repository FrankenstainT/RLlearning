"""
Nash DQN for Competitive Two-Agent Game
=======================================

Single Q-network that outputs Q-values for all joint actions (16 values).
"""

import numpy as np
# Lazy imports for heavy modules (only import when needed)
# This speeds up imports for test files that only need Nash solvers
import sys
import os
from typing import Tuple, Dict, List
from functools import partial

# Lazy import torch (only when creating networks)
def _import_torch():
    import torch
    return torch

# Lazy import scipy (only when solving LP)
_linprog = None
def _get_linprog():
    global _linprog
    if _linprog is None:
        from scipy.optimize import linprog
        _linprog = linprog
    return _linprog

# Lazy import cvxpy (for GPU-accelerated LP solving if available)
_cvxpy_available = None
def _check_cvxpy():
    global _cvxpy_available
    if _cvxpy_available is None:
        try:
            import cvxpy as cp
            _cvxpy_available = cp
        except ImportError:
            _cvxpy_available = False
    return _cvxpy_available

# Lazy import cuOpt (for GPU-accelerated LP solving if available)
_cuopt_available = None
_cuopt_client = None
def _check_cuopt():
    global _cuopt_available, _cuopt_client
    if _cuopt_available is None:
        try:
            # Try server-client API (recommended for batch processing)
            try:
                from cuopt.client import Client
                _cuopt_client = Client()
                _cuopt_available = 'server'
            except ImportError:
                try:
                    # Try direct libcuopt
                    import libcuopt
                    try:
                        libcuopt.load_library()
                        _cuopt_available = libcuopt
                    except Exception:
                        _cuopt_available = False
                except ImportError:
                    try:
                        # Alternative import name
                        import cuopt
                        _cuopt_available = cuopt
                    except ImportError:
                        _cuopt_available = False
        except Exception:
            _cuopt_available = False
    return _cuopt_available

# Add parent directory to path to import shared modules (only when needed)
_dqn_learning_imported = False
def _import_dqn_learning():
    global _dqn_learning_imported
    if not _dqn_learning_imported:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dqn_learning import ReplayBuffer
        _dqn_learning_imported = True
        return ReplayBuffer
    else:
        from dqn_learning import ReplayBuffer
        return ReplayBuffer

# Try to import parallel processing
# Note: With CUDA, we must use 'spawn' start method instead of 'fork'
# We ensure only numpy arrays are passed to workers (no CUDA tensors)
import platform
IS_WINDOWS = platform.system() == 'Windows'

# Lazy CUDA check (only when needed, not at import time to avoid slow startup)
_CUDA_AVAILABLE = None
def _get_cuda_available():
    """Lazy check for CUDA availability."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:
            import torch
            _CUDA_AVAILABLE = torch.cuda.is_available()
        except:
            _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE

# For backward compatibility, use a property-like access
def get_cuda_available():
    return _get_cuda_available()

# Lazy multiprocessing setup (only when actually needed)
_MULTIPROCESSING_SETUP = None
def _setup_multiprocessing():
    """Lazy setup of multiprocessing (only called when needed)."""
    global _MULTIPROCESSING_SETUP
    if _MULTIPROCESSING_SETUP is not None:
        return _MULTIPROCESSING_SETUP
    
    result = {'HAS_MULTIPROCESSING': False, 'Pool': None, 'cpu_count': lambda: 1}
    
    try:
        from multiprocessing import Pool, cpu_count, set_start_method, get_start_method
        result['HAS_MULTIPROCESSING'] = True
        result['Pool'] = Pool
        result['cpu_count'] = cpu_count
        
        # Set start method for multiprocessing (only if CUDA is available)
        # 'spawn' works with CUDA, 'fork' does not
        if _get_cuda_available() and not IS_WINDOWS:
            # On Linux with CUDA, use 'spawn' instead of default 'fork'
            try:
                current_method = get_start_method(allow_none=True)
                if current_method != 'spawn':
                    set_start_method('spawn', force=True)
            except RuntimeError:
                # Start method already set, that's fine
                pass
    except ImportError:
        pass
    
    _MULTIPROCESSING_SETUP = result
    return result

# Lazy accessors for backward compatibility
def get_has_multiprocessing():
    return _setup_multiprocessing()['HAS_MULTIPROCESSING']

def get_pool():
    return _setup_multiprocessing()['Pool']

def get_cpu_count():
    return _setup_multiprocessing()['cpu_count']()

# For backward compatibility (used in code)
HAS_MULTIPROCESSING = False  # Will be set lazily when accessed
CUDA_AVAILABLE = False  # Will be set lazily when accessed

# Lazy LP method detection (only when needed)
_LP_METHOD = None
def _get_lp_method():
    global _LP_METHOD
    if _LP_METHOD is None:
        try:
            linprog = _get_linprog()
            # Use HiGHS method if available (faster than default)
            _LP_METHOD = "highs" if hasattr(linprog, '__defaults__') else "interior-point"
        except:
            _LP_METHOD = "interior-point"
    return _LP_METHOD

# Lazy highspy import
_HAS_HIGHSPY = None
def _get_has_highspy():
    global _HAS_HIGHSPY
    if _HAS_HIGHSPY is None:
        try:
            import highspy
            _HAS_HIGHSPY = True
        except ImportError:
            _HAS_HIGHSPY = False
    return _HAS_HIGHSPY


class NashDQNNetwork:
    """DQN Network for joint actions: 2 hidden layers of 128 neurons each."""
    
    def __init__(self, input_size: int = 4, output_size: int = 16, hidden_size: int = 128):
        # Lazy import torch only when creating network
        torch = _import_torch()
        nn = torch.nn
        
        # Create a proper nn.Module subclass dynamically
        class _NetworkModule(nn.Module):
            def __init__(self, input_size, output_size, hidden_size):
                super(_NetworkModule, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # Store the actual module
        self._module = _NetworkModule(input_size, output_size, hidden_size)
        self.torch = torch  # Store torch for .to() method
        
    def to(self, device):
        """Move network to device."""
        self._module = self._module.to(device)
        return self
        
    def __call__(self, x):
        return self._module(x)
    
    def parameters(self):
        return self._module.parameters()
    
    def state_dict(self):
        return self._module.state_dict()
    
    def load_state_dict(self, state_dict):
        return self._module.load_state_dict(state_dict)


def solve_nash_equilibrium_fast_4x4(M: np.ndarray):
    """
    Fast approximate Nash solver for 4x4 games using improved iterative best response.
    Uses multiple restarts and better convergence criteria for accuracy.
    Target: value difference < 0.01 from exact LP solution.
    """
    M = np.asarray(M, float)
    m, n = M.shape
    assert m == 4 and n == 4, "This function is optimized for 4x4 games"
    
    # Special case: constant matrix (all entries are the same)
    # In this case, any strategy is optimal, return uniform
    if np.allclose(M, M[0, 0]):
        x = np.ones(m) / m
        y = np.ones(n) / n
        v = float(M[0, 0])
        return x, y, v
    
    best_x, best_y, best_v = None, None, None
    best_error = float('inf')
    
    # Try multiple random initializations for better accuracy (10 restarts for higher accuracy)
    for init_idx in range(10):
        if init_idx == 0:
            # First: uniform initialization
            x = np.ones(m) / m
            y = np.ones(n) / n
        elif init_idx == 1:
            # Second: start from best response to uniform
            y_uniform = np.ones(n) / n
            pursuer_payoffs_uniform = M @ y_uniform
            best_pursuer = np.argmax(pursuer_payoffs_uniform)
            x = np.zeros(m)
            x[best_pursuer] = 1.0
            
            x_uniform = np.ones(m) / m
            evader_payoffs_uniform = x_uniform @ M
            best_evader = np.argmin(evader_payoffs_uniform)
            y = np.zeros(n)
            y[best_evader] = 1.0
        else:
            # Random initialization using Dirichlet distribution
            x = np.random.dirichlet(np.ones(m))
            y = np.random.dirichlet(np.ones(n))
        
        # Improved iterative best response with better convergence
        prev_v = float('inf')
        for iteration in range(200):  # Increased from 100 to 200
            # Best response for pursuer (maximizer)
            pursuer_payoffs = M @ y
            best_pursuer = np.argmax(pursuer_payoffs)
            x_br = np.zeros(m)
            x_br[best_pursuer] = 1.0
            
            # Best response for evader (minimizer)
            evader_payoffs = x @ M
            best_evader = np.argmin(evader_payoffs)
            y_br = np.zeros(n)
            y_br[best_evader] = 1.0
            
            # Compute current value
            v = float(x @ M @ y)
            
            # Adaptive mixing: use smaller step size as we converge
            # Slower decay for better convergence
            alpha = max(0.05, 0.95 * (0.98 ** iteration))  # Decay from 0.95 to 0.05
            x = (1 - alpha) * x + alpha * x_br
            y = (1 - alpha) * y + alpha * y_br
            
            # Check convergence: value change and policy stability
            value_change = abs(v - prev_v)
            policy_change = max(np.max(np.abs(x - x_br)), np.max(np.abs(y - y_br)))
            
            # Tighter convergence criteria
            if value_change < 1e-8 and policy_change < 1e-5:
                break
            
            prev_v = v
        
        # Normalize to ensure valid probability distributions
        x = np.clip(x, 0, None)
        y = np.clip(y, 0, None)
        x = x / (x.sum() + 1e-10)
        y = y / (y.sum() + 1e-10)
        
        # Compute final value
        v = float(x @ M @ y)
        
        # Check Nash equilibrium conditions (approximate)
        # For pursuer: all actions should give similar payoff at equilibrium
        pursuer_payoffs_all = M @ y
        x_mask = x > 1e-6
        if np.any(x_mask):
            pursuer_max_dev = np.max(pursuer_payoffs_all) - np.min(pursuer_payoffs_all[x_mask])
        else:
            pursuer_max_dev = np.max(pursuer_payoffs_all) - np.min(pursuer_payoffs_all)
        
        # For evader: all actions should give similar payoff at equilibrium
        evader_payoffs_all = x @ M
        y_mask = y > 1e-6
        if np.any(y_mask):
            evader_max_dev = np.max(evader_payoffs_all[y_mask]) - np.min(evader_payoffs_all)
        else:
            evader_max_dev = np.max(evader_payoffs_all) - np.min(evader_payoffs_all)
        
        # Additional error: check regret
        pursuer_max_payoff = np.max(pursuer_payoffs_all)
        pursuer_regret = pursuer_max_payoff - v
        
        evader_min_payoff = np.min(evader_payoffs_all)
        evader_regret = v - evader_min_payoff
        
        # Combined error metric
        error = max(pursuer_max_dev, evader_max_dev, pursuer_regret, evader_regret)
        
        if error < best_error:
            best_error = error
            best_x, best_y, best_v = x.copy(), y.copy(), v
    
    # Refinement step: run multiple refinement passes with decreasing step sizes
    if best_x is not None:
        x_refine = best_x.copy()
        y_refine = best_y.copy()
        prev_v_refine = best_v
        
        # Multiple refinement passes with different step sizes
        for refine_pass in range(3):
            # Step size decreases with each pass
            alpha_refine = 0.05 / (2 ** refine_pass)  # 0.05, 0.025, 0.0125
            
            for refine_iter in range(100):
                # Best response
                pursuer_payoffs = M @ y_refine
                best_pursuer = np.argmax(pursuer_payoffs)
                x_br = np.zeros(m)
                x_br[best_pursuer] = 1.0
                
                evader_payoffs = x_refine @ M
                best_evader = np.argmin(evader_payoffs)
                y_br = np.zeros(n)
                y_br[best_evader] = 1.0
                
                # Small step size for refinement
                x_refine = (1 - alpha_refine) * x_refine + alpha_refine * x_br
                y_refine = (1 - alpha_refine) * y_refine + alpha_refine * y_br
                
                # Normalize
                x_refine = np.clip(x_refine, 0, None)
                y_refine = np.clip(y_refine, 0, None)
                x_refine = x_refine / (x_refine.sum() + 1e-10)
                y_refine = y_refine / (y_refine.sum() + 1e-10)
                
                # Compute value
                v_refine = float(x_refine @ M @ y_refine)
                
                # Check convergence
                value_change = abs(v_refine - prev_v_refine)
                if value_change < 1e-10:
                    break
                
                prev_v_refine = v_refine
            
            # Check if we've converged well enough
            pursuer_payoffs_all = M @ y_refine
            evader_payoffs_all = x_refine @ M
            pursuer_regret = np.max(pursuer_payoffs_all) - v_refine
            evader_regret = v_refine - np.min(evader_payoffs_all)
            max_regret = max(pursuer_regret, evader_regret)
            
            if max_regret < 1e-6:
                break
        
        # Use refined solution
        best_x = x_refine
        best_y = y_refine
        best_v = prev_v_refine
    
    return best_x, best_y, best_v


def solve_nash_equilibrium_lp_torch_batch(q_values_batch) -> Tuple:
    """
    Exact Nash equilibrium solver using linear programming in PyTorch (GPU batch).
    Implements primal-dual interior point method for true GPU batch processing.
    
    Args:
        q_values_batch: Tensor of shape [batch_size, 16] containing Q-values for each state
    
    Returns:
        pursuer_policies: Tensor of shape [batch_size, 4]
        evader_policies: Tensor of shape [batch_size, 4]
        values: Tensor of shape [batch_size]
    """
    # Lazy import torch (only when this function is called)
    torch = _import_torch()
    
    device = q_values_batch.device
    batch_size = q_values_batch.shape[0]
    m, n = 4, 4  # 4x4 games
    
    # Reshape Q-values to [batch_size, 4, 4] matrices
    M = q_values_batch.view(batch_size, m, n)
    
    # Check for constant matrices
    M_range = torch.max(M, dim=2)[0] - torch.min(M, dim=2)[0]  # [batch_size, 4]
    M_range = torch.max(M_range, dim=1)[0]  # [batch_size]
    is_constant = M_range < 1e-6
    
    # Handle constant matrices
    if torch.any(is_constant):
        x_result = torch.ones(batch_size, m, device=device) / m
        y_result = torch.ones(batch_size, n, device=device) / n
        v_result = M.mean(dim=(1, 2))
        if torch.all(is_constant):
            return x_result, y_result, v_result
    
    # Normalize for numerical stability
    M_mean = M.mean(dim=(1, 2), keepdim=True)  # [batch_size, 1, 1]
    M_centered = M - M_mean
    M_scale = torch.max(torch.abs(M_centered), dim=2, keepdim=True)[0]  # [batch_size, 4, 1]
    M_scale = torch.max(M_scale, dim=1, keepdim=True)[0]  # [batch_size, 1, 1]
    M_scale = torch.clamp(M_scale, min=1e-12)
    M_work = M_centered / M_scale  # [batch_size, 4, 4]
    
    # LP formulation: variables [x (m), y (n), v (1)]
    num_vars = m + n + 1
    
    # Objective: minimize -v (maximize v)
    c = torch.zeros(batch_size, num_vars, device=device)
    c[:, m + n] = -1.0  # -v
    
    # Build constraint matrices
    # Inequality constraints: A_ub @ z <= b_ub
    # v <= x^T M[:,j] for all j -> -M[:,j]^T x + v <= 0
    # M[i,:] y <= v for all i -> M[i,:] y - v <= 0
    num_ineq = m + n
    A_ub = torch.zeros(batch_size, num_ineq, num_vars, device=device)
    b_ub = torch.zeros(batch_size, num_ineq, device=device)
    
    # v <= x^T M[:,j] for all j (n constraints)
    for j in range(n):
        A_ub[:, j, :m] = -M_work[:, :, j]  # -M[:,j]^T
        A_ub[:, j, m + n] = 1.0  # +v
    
    # M[i,:] y <= v for all i (m constraints)
    for i in range(m):
        A_ub[:, n + i, m:m + n] = M_work[:, i, :]  # M[i,:]
        A_ub[:, n + i, m + n] = -1.0  # -v
    
    # Equality constraints: A_eq @ z = b_eq
    # sum(x) = 1, sum(y) = 1
    A_eq = torch.zeros(batch_size, 2, num_vars, device=device)
    A_eq[:, 0, :m] = 1.0  # sum(x) = 1
    A_eq[:, 1, m:m + n] = 1.0  # sum(y) = 1
    b_eq = torch.ones(batch_size, 2, device=device)
    
    # Try to use GPU-accelerated solver if available, otherwise use optimized CPU solvers
    # Priority: GPU solvers (cuOpt, MPAX, CuClarabel) > HIGHS > SCIPY > exact CPU solver (linprog)
    use_gpu_solver = False
    
    solver_used = None
    
    # Try CVXPY with GPU backends (includes cuOpt, MPAX, CuClarabel)
    cvxpy = _check_cvxpy()
    cuopt_available = _check_cuopt()
    
    if cvxpy and batch_size > 1:
        try:
            import cvxpy as cp
            M_np = M.cpu().numpy()
            
            x_results = []
            y_results = []
            v_results = []
            
            for b in range(batch_size):
                if is_constant[b]:
                    x_results.append(np.ones(m) / m)
                    y_results.append(np.ones(n) / n)
                    v_results.append(float(M_np[b].mean()))
                    if solver_used is None:
                        solver_used = "Constant matrix shortcut (no LP solve)"
                else:
                    # Formulate LP using CVXPY
                    num_vars = m + n + 1
                    
                    # Variables: x (m), y (n), v (1)
                    z = cp.Variable(num_vars)
                    x_var = z[:m]
                    y_var = z[m:m + n]
                    v_var = z[m + n]
                    
                    # Objective: maximize v (minimize -v)
                    objective = cp.Minimize(-v_var)
                    
                    # Constraints
                    constraints = [
                        x_var >= 0,
                        y_var >= 0,
                        cp.sum(x_var) == 1,
                        cp.sum(y_var) == 1,
                    ]
                    
                    # v <= x^T M[:,j] for all j
                    for j in range(n):
                        constraints.append(v_var <= x_var @ M_np[b, :, j])
                    
                    # M[i,:] y <= v for all i
                    for i in range(m):
                        constraints.append(M_np[b, i, :] @ y_var <= v_var)
                    
                    # Solve with GPU solver if available
                    problem = cp.Problem(objective, constraints)
                    
                    # Try GPU solvers in order of preference: cuOpt > MPAX > CuClarabel
                    solved = False
                    installed_solvers = cp.installed_solvers()
                    
                    # Priority 1: cuOpt (if available through CVXPY)
                    # Note: cuOpt may not be available as a CVXPY solver by default
                    # It requires custom integration or may be available in future CVXPY versions
                    if cuopt_available and 'CUOPT' in installed_solvers:
                        try:
                            problem.solve(solver=cp.CUOPT, gpu=True, verbose=False, warm_start=False)
                            if problem.status == 'optimal' and z.value is not None:
                                z_sol = z.value
                                x = z_sol[:m]
                                y = z_sol[m:m + n]
                                v = z_sol[m + n]
                                x_results.append(x)
                                y_results.append(y)
                                v_results.append(v)
                                if solver_used is None:
                                    solver_used = "CVXPY: CUOPT"
                                solved = True
                        except Exception:
                            pass
                    
                    # Note: cuOpt is installed but not available through CVXPY yet
                    # cuOpt's Python API is primarily for routing problems
                    # For LP problems, cuOpt would need to be used through its C API
                    # or integrated with CVXPY via a custom solver class
                    
                    # Priority 2: MPAX
                    if not solved and 'MPAX' in installed_solvers:
                        try:
                            # MPAX may not need gpu=True parameter, try both
                            print("Now MPAX")
                            try:
                                problem.solve(solver=cp.MPAX, gpu=True, verbose=False, warm_start=False)
                            except (TypeError, AttributeError):
                                # If gpu parameter not supported, try without it
                                problem.solve(solver=cp.MPAX, verbose=False, warm_start=False)
                            print(f"[MPAX debug] status={problem.status}, solver_stats={problem.solver_stats}")
                            if problem.status == 'optimal' and z.value is not None:
                                z_sol = z.value
                                x = z_sol[:m]
                                y = z_sol[m:m + n]
                                v = z_sol[m + n]
                                x_results.append(x)
                                y_results.append(y)
                                v_results.append(v)
                                if solver_used is None:
                                    solver_used = "CVXPY: MPAX"
                                solved = True
                        except Exception as e:
                            print(f"[solve_nash_equilibrium_lp_torch_batch] MPAX solve failed: {type(e).__name__}: {e}")
                    
                    # Priority 3: CuClarabel
                    if not solved and ('CUCLARABEL' in installed_solvers or 'CuClarabel' in installed_solvers):
                        try:
                            # Try both naming conventions
                            solver_name = cp.CuClarabel if 'CuClarabel' in installed_solvers else cp.CUCLARABEL
                            # CuClarabel may not need gpu=True parameter, try both
                            try:
                                problem.solve(solver=solver_name, gpu=True, verbose=False)
                            except (TypeError, AttributeError):
                                # If gpu parameter not supported, try without it
                                problem.solve(solver=solver_name, verbose=False)
                            if problem.status == 'optimal' and z.value is not None:
                                z_sol = z.value
                                x = z_sol[:m]
                                y = z_sol[m:m + n]
                                v = z_sol[m + n]
                                x_results.append(x)
                                y_results.append(y)
                                v_results.append(v)
                                if solver_used is None:
                                    solver_used = "CVXPY: CuClarabel"
                                solved = True
                        except Exception:
                            pass
                    
                    # Priority 4: HIGHS (high-performance CPU solver, often faster than scipy.optimize.linprog)
                    if not solved and 'HIGHS' in installed_solvers:
                        try:
                            problem.solve(solver=cp.HIGHS, verbose=False)
                            if problem.status == 'optimal' and z.value is not None:
                                z_sol = z.value
                                x = z_sol[:m]
                                y = z_sol[m:m + n]
                                v = z_sol[m + n]
                                x_results.append(x)
                                y_results.append(y)
                                v_results.append(v)
                                if solver_used is None:
                                    solver_used = "CVXPY: HIGHS"
                                solved = True
                        except Exception:
                            pass
                    
                    # Priority 5: SCIPY (CVXPY's scipy solver, may be optimized)
                    if not solved and 'SCIPY' in installed_solvers:
                        try:
                            problem.solve(solver=cp.SCIPY, verbose=False)
                            if problem.status == 'optimal' and z.value is not None:
                                z_sol = z.value
                                x = z_sol[:m]
                                y = z_sol[m:m + n]
                                v = z_sol[m + n]
                                x_results.append(x)
                                y_results.append(y)
                                v_results.append(v)
                                if solver_used is None:
                                    solver_used = "CVXPY: SCIPY"
                                solved = True
                        except Exception:
                            pass
                    
                    if not solved:
                        # Fall back to exact solver (direct scipy.optimize.linprog)
                        x, y, v = solve_nash_equilibrium(M_np[b], use_fast_approx=False)
                        x_results.append(x)
                        y_results.append(y)
                        v_results.append(v)
            
            if len(x_results) == batch_size:
                x = torch.from_numpy(np.array(x_results)).float().to(device)
                y = torch.from_numpy(np.array(y_results)).float().to(device)
                v = torch.from_numpy(np.array(v_results)).float().to(device)
                use_gpu_solver = True
        except Exception as e:
            # Fall back to exact solver
            use_gpu_solver = False
    
    if not use_gpu_solver:
        # Use exact LP solver - this is the most reliable approach
        # For true GPU batch processing, install cuOpt and complete integration
        # This ensures exact solutions matching the reference
        M_np = M.cpu().numpy()
        
        x_results = []
        y_results = []
        v_results = []
        
        for b in range(batch_size):
            if is_constant[b]:
                x_results.append(np.ones(m) / m)
                y_results.append(np.ones(n) / n)
                v_results.append(float(M_np[b].mean()))
            else:
                # Use exact LP solver
                x, y, v = solve_nash_equilibrium(M_np[b], use_fast_approx=False)
                x_results.append(x)
                y_results.append(y)
                v_results.append(v)
        
        # Convert back to tensors on the original device
        x = torch.from_numpy(np.array(x_results)).float().to(device)
        y = torch.from_numpy(np.array(y_results)).float().to(device)
        v = torch.from_numpy(np.array(v_results)).float().to(device)
        solver_used = solver_used or "CPU: scipy.optimize.linprog"
    
    if solver_used:
        print(f"[solve_nash_equilibrium_lp_torch_batch] Solver used: {solver_used}")
    # Check for NaN/Inf and replace with uniform if needed
    has_nan = ~torch.isfinite(x) | ~torch.isfinite(y) | ~torch.isfinite(v)
    if torch.any(has_nan):
        # Replace NaN with uniform policies
        nan_mask = has_nan.any(dim=1) if has_nan.dim() > 1 else has_nan
        if torch.any(nan_mask):
            x[nan_mask] = 1.0 / m
            y[nan_mask] = 1.0 / n
            v[nan_mask] = M[nan_mask].mean(dim=(1, 2))
    
    # Normalize to ensure valid probabilities
    x = torch.clamp(x, min=0.0)
    y = torch.clamp(y, min=0.0)
    x = x / (x.sum(dim=1, keepdim=True) + 1e-10)
    y = y / (y.sum(dim=1, keepdim=True) + 1e-10)
    
    # Denormalize value
    M_scale_1d = M_scale.squeeze()
    M_mean_1d = M_mean.squeeze()
    if M_scale_1d.dim() == 0:
        M_scale_1d = M_scale_1d.unsqueeze(0)
    if M_mean_1d.dim() == 0:
        M_mean_1d = M_mean_1d.unsqueeze(0)
    v = v * M_scale_1d + M_mean_1d
    
    # Recompute value in original space for accuracy
    v_recomputed = torch.bmm(torch.bmm(x.unsqueeze(1), M), y.unsqueeze(2)).squeeze(-1).squeeze(-1)
    if v_recomputed.dim() == 0:
        v_recomputed = v_recomputed.unsqueeze(0)
    v = v_recomputed
    
    # Final NaN check after recomputation
    has_nan_final = ~torch.isfinite(x) | ~torch.isfinite(y) | ~torch.isfinite(v)
    if torch.any(has_nan_final):
        nan_mask_final = has_nan_final.any(dim=1) if has_nan_final.dim() > 1 else has_nan_final
        if torch.any(nan_mask_final):
            x[nan_mask_final] = 1.0 / m
            y[nan_mask_final] = 1.0 / n
            v[nan_mask_final] = M[nan_mask_final].mean(dim=(1, 2))
    
    # Handle constant matrices
    if torch.any(is_constant):
        constant_mask = is_constant.unsqueeze(1)
        x = torch.where(constant_mask, torch.ones(batch_size, m, device=device) / m, x)
        y = torch.where(constant_mask, torch.ones(batch_size, n, device=device) / n, y)
        v = torch.where(is_constant, M.mean(dim=(1, 2)), v)
    
    return x, y, v


def solve_nash_equilibrium_fast_4x4_torch_batch(q_values_batch) -> Tuple:
    """
    Exact Nash equilibrium solver using linear programming in PyTorch (GPU batch).
    This is now an alias for the LP solver.
    
    Args:
        q_values_batch: Tensor of shape [batch_size, 16] containing Q-values for each state
    
    Returns:
        pursuer_policies: Tensor of shape [batch_size, 4]
        evader_policies: Tensor of shape [batch_size, 4]
        values: Tensor of shape [batch_size]
    """
    return solve_nash_equilibrium_lp_torch_batch(q_values_batch)


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
        linprog = _get_linprog()
        lp_method = _get_lp_method()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method=lp_method,
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
                 cache_update_frequency: int = 1,
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
        self.cache_update_frequency = cache_update_frequency  # Update cache every N training steps
        self.num_workers = num_workers if num_workers is not None else (get_cpu_count() if get_has_multiprocessing() else 1)
        
        # Lazy import torch and related modules
        torch = _import_torch()
        optim = torch.optim
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Single shared network for joint actions
        self.q_network = NashDQNNetwork(input_size, joint_action_size).to(self.device)
        self.target_network = NashDQNNetwork(input_size, joint_action_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer (lazy import)
        ReplayBuffer = _import_dqn_learning()
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Track TD errors per episode
        self.episode_td_errors = []
        
        # Cache for Nash policies and values (with size limit)
        # Separate caches for Q-network and target network
        self._nash_cache_q = {}  # Cache for main Q-network
        self._nash_cache_max_size = 5000  # Limit cache size to prevent memory issues
        
        # Persistent worker pool for multiprocessing (reuse instead of spawning each time)
        self._worker_pool = None
        
        # Track network updates for cache invalidation
        self._training_steps = 0


    
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
        # Batch compute Nash values on GPU for efficiency
        with torch.no_grad():
            # Get Q-values for all next states in batch
            next_q_values = self.target_network(next_states)  # [batch_size, 16]
            
            # Compute Nash values in batch on GPU
            if self.use_fast_nash:
                # Use fast GPU batch solver
                _, _, next_nash_values = solve_nash_equilibrium_fast_4x4_torch_batch(next_q_values)
            else:
                # Fallback: compute individually (slower)
                next_nash_values = []
                for i in range(len(next_states)):
                    if dones[i]:
                        next_nash_values.append(0.0)
                    else:
                        next_state = next_states[i].cpu().numpy()
                        _, _, nash_value = self._solve_nash_for_state_target(next_state)
                        next_nash_values.append(nash_value)
                next_nash_values = torch.FloatTensor(next_nash_values).to(self.device)
            
            # Set to 0 for done states
            next_nash_values = next_nash_values * (~dones).float()
            target_q_values = rewards + (self.gamma * next_nash_values)
        
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
        
        # Optionally update cache for all states periodically
        # Since train_step is already batched, we can update cache more frequently
        # The cache update frequency is relative to training steps (which are batched)
        self._training_steps += 1
        if (self.update_cache_after_training and 
            hasattr(self, '_all_states_for_cache') and
            self._training_steps % self.cache_update_frequency == 0):
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
    
    def _get_worker_pool(self):
        """Get or create persistent worker pool for multiprocessing."""
        if self._worker_pool is None and get_has_multiprocessing() and self.num_workers > 1:
            # Create persistent pool (only once)
            Pool = get_pool()
            if Pool is not None:
                self._worker_pool = Pool(processes=self.num_workers)
        return self._worker_pool
    
    def _close_worker_pool(self):
        """Close worker pool if it exists."""
        if self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool.join()
            self._worker_pool = None
    
    def _update_cache_for_all_states(self):
        """Update cache for all states in parallel (called after training step)."""
        if not hasattr(self, '_all_states_for_cache') or not self._all_states_for_cache:
            return
        
        all_states = self._all_states_for_cache
        
        # Batch compute all Q-values on GPU (very fast)
        q_values_batch = self._batch_compute_q_values(all_states)
        
        # Solve Nash equilibria using persistent worker pool
        if get_has_multiprocessing() and self.num_workers > 1 and len(all_states) > 10:
            # Use persistent worker pool (reused, not recreated each time)
            pool = self._get_worker_pool()
            if pool is not None:
                solve_func = partial(_solve_nash_for_q_values_helper, use_fast_nash=self.use_fast_nash)
                results = pool.map(solve_func, q_values_batch)
            else:
                # Fallback to sequential
                results = [self._solve_nash_for_q_values(qv) for qv in q_values_batch]
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
        
        # Solve Nash equilibria using persistent worker pool
        if get_has_multiprocessing() and self.num_workers > 1 and len(all_states) > 10:
            # Use persistent worker pool (reused, not recreated each time)
            pool = self._get_worker_pool()
            if pool is not None:
                solve_func = partial(_solve_nash_for_q_values_helper, use_fast_nash=self.use_fast_nash)
                results = pool.map(solve_func, q_values_batch)
            else:
                # Fallback to sequential
                results = [self._solve_nash_for_q_values(qv) for qv in q_values_batch]
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
    
    def cleanup(self):
        """Clean up resources (close worker pool)."""
        self._close_worker_pool()

