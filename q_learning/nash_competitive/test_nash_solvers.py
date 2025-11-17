"""
Test and Compare Nash Equilibrium Solvers
==========================================

Compare solve_nash_equilibrium_fast_4x4 (fast approximate) 
with solve_nash_equilibrium (exact LP-based) on various test cases.
"""

import numpy as np
import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nash_dqn import solve_nash_equilibrium, solve_nash_equilibrium_fast_4x4, solve_nash_equilibrium_fast_4x4_torch_batch
import time
import torch


def generate_test_cases():
    """Generate various test cases for comparison."""
    test_cases = []
    
    # Test case 1: Random matrix
    np.random.seed(42)
    test_cases.append({
        'name': 'Random Matrix',
        'M': np.random.randn(4, 4) * 10
    })
    
    # Test case 2: Dominant strategy (pursuer has dominant action)
    M = np.array([
        [10, 5, 3, 2],
        [1, 2, 1, 1],
        [0, 1, 0, 0],
        [-1, 0, -1, -2]
    ])
    test_cases.append({
        'name': 'Pursuer Dominant Strategy',
        'M': M
    })
    
    # Test case 3: Evader dominant strategy
    M = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ])
    test_cases.append({
        'name': 'Evader Dominant Strategy',
        'M': M
    })
    
    # Test case 4: Pure Nash equilibrium
    M = np.array([
        [10, 0, 0, 0],
        [0, 8, 0, 0],
        [0, 0, 6, 0],
        [0, 0, 0, 4]
    ])
    test_cases.append({
        'name': 'Pure Nash Equilibrium',
        'M': M
    })
    
    # Test case 5: Mixed strategy required
    M = np.array([
        [3, 0, 2, 0],
        [0, 3, 0, 2],
        [2, 0, 3, 0],
        [0, 2, 0, 3]
    ])
    test_cases.append({
        'name': 'Mixed Strategy Required',
        'M': M
    })
    
    # Test case 6: Zero-sum game
    M = np.array([
        [1, -1, 2, -2],
        [-1, 1, -2, 2],
        [2, -2, 1, -1],
        [-2, 2, -1, 1]
    ])
    test_cases.append({
        'name': 'Zero-Sum Game',
        'M': M
    })
    
    # Test case 7: Constant matrix (degenerate case)
    test_cases.append({
        'name': 'Constant Matrix',
        'M': np.ones((4, 4)) * 5.0
    })
    
    # Test case 8: Large values
    test_cases.append({
        'name': 'Large Values',
        'M': np.random.randn(4, 4) * 100
    })
    
    # Test case 9: Small values
    test_cases.append({
        'name': 'Small Values',
        'M': np.random.randn(4, 4) * 0.01
    })
    
    # Test case 10: Negative values
    test_cases.append({
        'name': 'Negative Values',
        'M': np.random.randn(4, 4) * 10 - 20
    })
    
    return test_cases


def compare_solutions(x1, y1, v1, x2, y2, v2, M, tolerance=0.02):
    """Compare two Nash equilibrium solutions."""
    results = {
        'pursuer_policy_diff': np.linalg.norm(x1 - x2),
        'evader_policy_diff': np.linalg.norm(y1 - y2),
        'value_diff': abs(v1 - v2),
        'pursuer_policy_l1': np.sum(np.abs(x1 - x2)),
        'evader_policy_l1': np.sum(np.abs(y1 - y2)),
    }
    
    # Check if policies are valid (sum to 1, non-negative)
    results['pursuer_valid_1'] = abs(np.sum(x1) - 1.0) < 1e-6 and np.all(x1 >= -1e-6)
    results['pursuer_valid_2'] = abs(np.sum(x2) - 1.0) < 1e-6 and np.all(x2 >= -1e-6)
    results['evader_valid_1'] = abs(np.sum(y1) - 1.0) < 1e-6 and np.all(y1 >= -1e-6)
    results['evader_valid_2'] = abs(np.sum(y2) - 1.0) < 1e-6 and np.all(y2 >= -1e-6)
    
    # Check Nash equilibrium conditions (approximately)
    # For pursuer: x^T M y should be >= x'^T M y for any x'
    pursuer_payoff_1 = x1 @ M @ y1
    pursuer_payoff_2 = x2 @ M @ y2
    results['pursuer_payoff_1'] = pursuer_payoff_1
    results['pursuer_payoff_2'] = pursuer_payoff_2
    
    # For evader: x^T M y should be <= x^T M y' for any y'
    evader_payoff_1 = x1 @ M @ y1
    evader_payoff_2 = x2 @ M @ y2
    results['evader_payoff_1'] = evader_payoff_1
    results['evader_payoff_2'] = evader_payoff_2
    
    # Check if solutions are within tolerance
    results['within_tolerance'] = (

        results['value_diff'] < tolerance
    )
    
    return results


def run_comparison():
    """Run comparison tests."""
    test_cases = generate_test_cases()
    
    print("=" * 80)
    print("Nash Equilibrium Solver Comparison")
    print("=" * 80)
    print(f"\nTesting {len(test_cases)} test cases...\n")
    
    results_summary = {
        'total': len(test_cases),
        'within_tolerance': 0,
        'lp_times': [],
        'fast_times': [],
        'speedup': [],
        'max_pursuer_diff': 0,
        'max_evader_diff': 0,
        'max_value_diff': 0,
    }
    
    for i, test_case in enumerate(test_cases, 1):
        M = test_case['M']
        name = test_case['name']
        
        print(f"Test {i}/{len(test_cases)}: {name}")
        print(f"  Matrix:\n{M}\n")
        
        # Solve with LP (exact)
        start_time = time.perf_counter()
        x_lp, y_lp, v_lp = solve_nash_equilibrium(M.copy(), use_fast_approx=False)
        lp_time = time.perf_counter() - start_time
        results_summary['lp_times'].append(lp_time)
        
        # Solve with fast approximate
        start_time = time.perf_counter()
        x_fast, y_fast, v_fast = solve_nash_equilibrium_fast_4x4(M.copy())
        fast_time = time.perf_counter() - start_time
        results_summary['fast_times'].append(fast_time)
        
        if lp_time > 0:
            speedup = lp_time / fast_time
            results_summary['speedup'].append(speedup)
        
        # Compare solutions
        comparison = compare_solutions(x_lp, y_lp, v_lp, x_fast, y_fast, v_fast, M)
        
        print(f"  LP Solution:")
        print(f"    Pursuer policy: {x_lp}")
        print(f"    Evader policy: {y_lp}")
        print(f"    Value: {v_lp:.6f}")
        print(f"    Time: {lp_time*1000:.3f} ms")
        
        print(f"  Fast Solution:")
        print(f"    Pursuer policy: {x_fast}")
        print(f"    Evader policy: {y_fast}")
        print(f"    Value: {v_fast:.6f}")
        print(f"    Time: {fast_time*1000:.3f} ms")
        
        if lp_time > 0:
            print(f"  Speedup: {speedup:.2f}x")
        
        print(f"  Differences:")
        print(f"    Pursuer policy L2: {comparison['pursuer_policy_diff']:.6f}")
        print(f"    Evader policy L2: {comparison['evader_policy_diff']:.6f}")
        print(f"    Value difference: {comparison['value_diff']:.6f}")
        print(f"    Pursuer policy L1: {comparison['pursuer_policy_l1']:.6f}")
        print(f"    Evader policy L1: {comparison['evader_policy_l1']:.6f}")
        
        print(f"  Validity:")
        print(f"    LP pursuer valid: {comparison['pursuer_valid_1']}")
        print(f"    Fast pursuer valid: {comparison['pursuer_valid_2']}")
        print(f"    LP evader valid: {comparison['evader_valid_1']}")
        print(f"    Fast evader valid: {comparison['evader_valid_2']}")
        
        print(f"  Payoffs:")
        print(f"    LP: {comparison['pursuer_payoff_1']:.6f}")
        print(f"    Fast: {comparison['pursuer_payoff_2']:.6f}")
        
        if comparison['within_tolerance']:
            print(f"  ✓ Within tolerance (0.01)")
            results_summary['within_tolerance'] += 1
        else:
            print(f"  ✗ Outside tolerance (target: < 0.01)")
        
        # Update max differences
        results_summary['max_pursuer_diff'] = max(
            results_summary['max_pursuer_diff'], 
            comparison['pursuer_policy_diff']
        )
        results_summary['max_evader_diff'] = max(
            results_summary['max_evader_diff'],
            comparison['evader_policy_diff']
        )
        results_summary['max_value_diff'] = max(
            results_summary['max_value_diff'],
            comparison['value_diff']
        )
        
        print()
    
    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total test cases: {results_summary['total']}")
    print(f"Within tolerance (0.01): {results_summary['within_tolerance']}/{results_summary['total']}")
    print(f"  ({100*results_summary['within_tolerance']/results_summary['total']:.1f}%)")
    print()
    print("Maximum Differences:")
    print(f"  Pursuer policy L2: {results_summary['max_pursuer_diff']:.6f}")
    print(f"  Evader policy L2: {results_summary['max_evader_diff']:.6f}")
    print(f"  Value: {results_summary['max_value_diff']:.6f}")
    print()
    print("Performance:")
    if results_summary['lp_times']:
        avg_lp_time = np.mean(results_summary['lp_times']) * 1000
        avg_fast_time = np.mean(results_summary['fast_times']) * 1000
        print(f"  Average LP time: {avg_lp_time:.3f} ms")
        print(f"  Average fast time: {avg_fast_time:.3f} ms")
        if results_summary['speedup']:
            avg_speedup = np.mean(results_summary['speedup'])
            min_speedup = np.min(results_summary['speedup'])
            max_speedup = np.max(results_summary['speedup'])
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Speedup range: {min_speedup:.2f}x - {max_speedup:.2f}x")
    print("=" * 80)


def test_torch_batch_solver():
    """Test solve_nash_equilibrium_fast_4x4_torch_batch against linprog."""
    test_cases = generate_test_cases()
    
    print("=" * 80)
    print("Nash Equilibrium Solver Comparison (Torch Batch)")
    print("=" * 80)
    print(f"\nTesting {len(test_cases)} test cases...\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    results_summary = {
        'total': len(test_cases),
        'within_tolerance': 0,
        'lp_times': [],
        'torch_times': [],
        'speedup': [],
        'max_pursuer_diff': 0,
        'max_evader_diff': 0,
        'max_value_diff': 0,
    }
    
    for i, test_case in enumerate(test_cases, 1):
        M = test_case['M']
        name = test_case['name']
        
        print(f"Test {i}/{len(test_cases)}: {name}")
        print(f"  Matrix:\n{M}\n")
        
        # Solve with LP (exact)
        start_time = time.perf_counter()
        x_lp, y_lp, v_lp = solve_nash_equilibrium(M.copy(), use_fast_approx=False)
        lp_time = time.perf_counter() - start_time
        results_summary['lp_times'].append(lp_time)
        
        # Convert M to Q-values format (16 values)
        # Q-values are indexed as: pursuer_action * 4 + evader_action
        q_values = np.zeros(16)
        for pursuer_action in range(4):
            for evader_action in range(4):
                joint_idx = pursuer_action * 4 + evader_action
                q_values[joint_idx] = M[pursuer_action, evader_action]
        
        # Convert to batch tensor (batch_size=1 for single test)
        q_values_batch = torch.from_numpy(q_values).float().unsqueeze(0).to(device)
        
        # Solve with torch batch solver
        start_time = time.perf_counter()
        x_torch, y_torch, v_torch = solve_nash_equilibrium_fast_4x4_torch_batch(q_values_batch)
        torch_time = time.perf_counter() - start_time
        results_summary['torch_times'].append(torch_time)
        
        if lp_time > 0:
            speedup = lp_time / torch_time
            results_summary['speedup'].append(speedup)
        
        # Convert back to numpy
        x_torch_np = x_torch.cpu().numpy()[0]
        y_torch_np = y_torch.cpu().numpy()[0]
        v_torch_np = float(v_torch.cpu().numpy()[0])
        tolerance=.02
        # Compare solutions
        comparison = compare_solutions(x_lp, y_lp, v_lp, x_torch_np, y_torch_np, v_torch_np, M,tolerance)
        
        print(f"  LP Solution:")
        print(f"    Pursuer policy: {x_lp}")
        print(f"    Evader policy: {y_lp}")
        print(f"    Value: {v_lp:.6f}")
        print(f"    Time: {lp_time*1000:.3f} ms")
        
        print(f"  Torch Batch Solution:")
        print(f"    Pursuer policy: {x_torch_np}")
        print(f"    Evader policy: {y_torch_np}")
        print(f"    Value: {v_torch_np:.6f}")
        print(f"    Time: {torch_time*1000:.3f} ms")
        
        if lp_time > 0:
            print(f"  Speedup: {speedup:.2f}x")
        
        print(f"  Differences:")
        print(f"    Pursuer policy L2: {comparison['pursuer_policy_diff']:.6f}")
        print(f"    Evader policy L2: {comparison['evader_policy_diff']:.6f}")
        print(f"    Value difference: {comparison['value_diff']:.6f}")
        print(f"    Pursuer policy L1: {comparison['pursuer_policy_l1']:.6f}")
        print(f"    Evader policy L1: {comparison['evader_policy_l1']:.6f}")
        
        print(f"  Validity:")
        print(f"    LP pursuer valid: {comparison['pursuer_valid_1']}")
        print(f"    Torch pursuer valid: {comparison['pursuer_valid_2']}")
        print(f"    LP evader valid: {comparison['evader_valid_1']}")
        print(f"    Torch evader valid: {comparison['evader_valid_2']}")
        
        print(f"  Payoffs:")
        print(f"    LP: {comparison['pursuer_payoff_1']:.6f}")
        print(f"    Torch: {comparison['pursuer_payoff_2']:.6f}")
        
        if comparison['within_tolerance']:
            print(f"  ✓ Within tolerance ({tolerance})")
            results_summary['within_tolerance'] += 1
        else:
            print(f"  ✗ Outside tolerance (target: < {tolerance})")
        
        # Update max differences
        results_summary['max_pursuer_diff'] = max(
            results_summary['max_pursuer_diff'], 
            comparison['pursuer_policy_diff']
        )
        results_summary['max_evader_diff'] = max(
            results_summary['max_evader_diff'],
            comparison['evader_policy_diff']
        )
        results_summary['max_value_diff'] = max(
            results_summary['max_value_diff'],
            comparison['value_diff']
        )
        
        print()
    
    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total test cases: {results_summary['total']}")
    print(f"Within tolerance ({tolerance}): {results_summary['within_tolerance']}/{results_summary['total']}")
    print(f"  ({100*results_summary['within_tolerance']/results_summary['total']:.1f}%)")
    print()
    print("Maximum Differences:")
    print(f"  Pursuer policy L2: {results_summary['max_pursuer_diff']:.6f}")
    print(f"  Evader policy L2: {results_summary['max_evader_diff']:.6f}")
    print(f"  Value: {results_summary['max_value_diff']:.6f}")
    print()
    print("Performance:")
    if results_summary['lp_times']:
        avg_lp_time = np.mean(results_summary['lp_times']) * 1000
        avg_torch_time = np.mean(results_summary['torch_times']) * 1000
        print(f"  Average LP time: {avg_lp_time:.3f} ms")
        print(f"  Average torch time: {avg_torch_time:.3f} ms")
        if results_summary['speedup']:
            avg_speedup = np.mean(results_summary['speedup'])
            min_speedup = np.min(results_summary['speedup'])
            max_speedup = np.max(results_summary['speedup'])
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Speedup range: {min_speedup:.2f}x - {max_speedup:.2f}x")
    print("=" * 80)
    
    return results_summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--torch-batch":
        test_torch_batch_solver()
    else:
        run_comparison()

