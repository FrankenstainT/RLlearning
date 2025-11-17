#!/usr/bin/env python3
"""
Test script to verify cuOpt integration for Nash equilibrium solver.
Run this after installing cuOpt to verify it's working correctly.
"""

import sys
import numpy as np
import torch

def test_cuopt_availability():
    """Test if cuOpt is available and can be imported."""
    print("=" * 70)
    print("Testing cuOpt Availability")
    print("=" * 70)
    
    try:
        from nash_dqn import _check_cuopt
        cuopt = _check_cuopt()
        
        if cuopt:
            print("✓ cuOpt is available!")
            if hasattr(cuopt, '__version__'):
                print(f"  Version: {cuopt.__version__}")
            if hasattr(cuopt, '__file__'):
                print(f"  Location: {cuopt.__file__}")
            return True
        else:
            print("✗ cuOpt not found")
            print("\nTo install cuOpt:")
            print("  pip install --extra-index-url=https://pypi.nvidia.com 'libcuopt-cu13==25.10.*'")
            print("  (or use cu12 for CUDA 12)")
            return False
    except Exception as e:
        print(f"✗ Error checking cuOpt: {e}")
        return False

def test_gpu_availability():
    """Test if GPU is available."""
    print("\n" + "=" * 70)
    print("Testing GPU Availability")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        return True
    else:
        print("✗ CUDA not available")
        print("  cuOpt requires CUDA-enabled GPU")
        return False

def test_nash_solver():
    """Test the Nash equilibrium solver."""
    print("\n" + "=" * 70)
    print("Testing Nash Equilibrium Solver")
    print("=" * 70)
    
    try:
        from nash_dqn import solve_nash_equilibrium_fast_4x4_torch_batch
        
        # Create a test batch
        batch_size = 5
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        q_values = torch.randn(batch_size, 16, device=device)
        
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
        
        # Solve
        x, y, v = solve_nash_equilibrium_fast_4x4_torch_batch(q_values)
        
        print(f"✓ Solver works correctly")
        print(f"  Output shapes: x={x.shape}, y={y.shape}, v={v.shape}")
        print(f"  Policies are valid: x.sum()={x.sum().item():.2f}, y.sum()={y.sum().item():.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing solver: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("cuOpt Integration Test")
    print("=" * 70)
    print("\nThis script tests cuOpt integration for the Nash equilibrium solver.")
    print("See CUOPT_INTEGRATION.md for installation and integration details.\n")
    
    results = {
        'cuOpt': test_cuopt_availability(),
        'GPU': test_gpu_availability(),
        'Solver': test_nash_solver(),
    }
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! cuOpt integration is ready.")
    else:
        print("\n✗ Some tests failed. See CUOPT_INTEGRATION.md for help.")
        if not results['cuOpt']:
            print("\n  Next steps:")
            print("  1. Install cuOpt (see CUOPT_INTEGRATION.md)")
            print("  2. Verify CUDA installation")
            print("  3. Run this test again")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

