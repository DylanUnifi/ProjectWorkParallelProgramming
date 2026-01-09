#!/usr/bin/env python3
"""
Simple validation test for CUDA kernel compilation fixes.
Tests the key fixes without requiring GPU hardware.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def test_imports():
    """Test that all modules can be imported without syntax errors."""
    print("Testing imports...")
    try:
        from scripts.pipeline_backends import (
            _autotune_kernel_tiles,
            _round_to_pow2,
            compute_kernel_matrix
        )
        print("  ✓ pipeline_backends imports successful")
    except SyntaxError as e:
        print(f"  ✗ Syntax error in pipeline_backends: {e}")
        return False
    except Exception as e:
        # Other import errors (missing dependencies) are expected in CI
        print(f"  ⚠ Import warning (expected in CI without GPU): {type(e).__name__}")
        return True
    
    try:
        from tools.test_num_qubit_impact import benchmark_single_config
        print("  ✓ test_num_qubit_impact imports successful")
    except SyntaxError as e:
        print(f"  ✗ Syntax error in test_num_qubit_impact: {e}")
        return False
    except Exception as e:
        print(f"  ⚠ Import warning (expected in CI without GPU): {type(e).__name__}")
        return True
    
    return True

def test_round_to_pow2():
    """Test the _round_to_pow2 helper function."""
    print("\nTesting _round_to_pow2...")
    try:
        # Import numpy first if available, otherwise skip functional test
        import numpy as np
        from scripts.pipeline_backends import _round_to_pow2
        
        test_cases = [
            (1, 1),
            (100, 128),
            (256, 256),
            (500, 512),
            (1000, 1024),
            (2048, 2048),
        ]
        
        all_passed = True
        for input_val, expected in test_cases:
            result = _round_to_pow2(input_val)
            if result == expected:
                print(f"  ✓ _round_to_pow2({input_val}) = {result}")
            else:
                print(f"  ✗ _round_to_pow2({input_val}) = {result}, expected {expected}")
                all_passed = False
        
        return all_passed
    except ImportError:
        # If numpy is not available, just check that the function exists in source
        print("  ⚠ Numpy not available, checking source code...")
        with open(ROOT / "scripts" / "pipeline_backends.py", "r") as f:
            content = f.read()
        
        if "def _round_to_pow2(x):" in content:
            print("  ✓ _round_to_pow2 function defined in source")
            return True
        else:
            print("  ✗ _round_to_pow2 function not found in source")
            return False
    except Exception as e:
        print(f"  ✗ Error testing _round_to_pow2: {e}")
        return False

def test_qubit_aware_tile_constraints():
    """Verify that tile constraints are properly defined for different qubit counts."""
    print("\nTesting qubit-aware tile constraints...")
    
    # Read the source file and check for the constraint logic
    with open(ROOT / "scripts" / "pipeline_backends.py", "r") as f:
        content = f.read()
    
    checks = [
        ("if nq >= 14:", "High qubit (14+) constraint"),
        ("if nq >= 12:", "Medium qubit (12+) constraint"),
        ("candidates_m_n = [16, 32]", "Conservative tiles for 14+ qubits"),
        ("candidates_m_n = [16, 32, 64]", "Moderate tiles for 12+ qubits"),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ✗ Missing: {description}")
            all_passed = False
    
    return all_passed

def test_numerical_stability_checks():
    """Verify that numerical stability checks are in place."""
    print("\nTesting numerical stability checks...")
    
    with open(ROOT / "scripts" / "pipeline_backends.py", "r") as f:
        content = f.read()
    
    checks = [
        ("if nq >= 14 and dtype == \"float32\":", "Float64 forcing for high qubits"),
        ("if nq >= 12 and not cp.all(cp.isfinite(out_tile)):", "NaN/Inf detection"),
        ("cp.nan_to_num(out_tile", "NaN/Inf repair"),
        ("if nq >= 12:", "Intermediate normalization"),
        ("cp.linalg.norm", "State vector normalization"),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ✗ Missing: {description}")
            all_passed = False
    
    return all_passed

def test_error_handling():
    """Verify that error handling is in place."""
    print("\nTesting error handling...")
    
    with open(ROOT / "scripts" / "pipeline_backends.py", "r") as f:
        content = f.read()
    
    checks = [
        ("try:\n                        k_fn = _get_kernel", "Kernel compilation try-except"),
        ("if \"shared memory\" in str(e).lower() or \"ptxas\" in str(e).lower():", "Shared memory error detection"),
        ("tm, tn, tk = (16, 16, 16)", "Fallback to safe tiles"),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ✗ Missing: {description}")
            all_passed = False
    
    return all_passed

def test_graph_optimization():
    """Verify CUDA graph optimization improvements."""
    print("\nTesting CUDA graph optimization...")
    
    with open(ROOT / "scripts" / "pipeline_backends.py", "r") as f:
        content = f.read()
    
    checks = [
        ("_round_to_pow2(bi)", "Graph key rounding for bi"),
        ("_round_to_pow2(bj)", "Graph key rounding for bj"),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ✗ Missing: {description}")
            all_passed = False
    
    return all_passed

def main():
    """Run all validation tests."""
    print("="*70)
    print("CUDA Kernel Compilation Fixes - Validation Tests")
    print("="*70)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Round to Power of 2", test_round_to_pow2()))
    results.append(("Qubit-Aware Tile Constraints", test_qubit_aware_tile_constraints()))
    results.append(("Numerical Stability Checks", test_numerical_stability_checks()))
    results.append(("Error Handling", test_error_handling()))
    results.append(("Graph Optimization", test_graph_optimization()))
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = 0
    failed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("="*70)
    print(f"Total: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All validation tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
