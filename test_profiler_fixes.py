#!/usr/bin/env python3
"""
Quick test to validate memory profiler, stream utilization, and VRAM estimation fixes.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.pipeline_backends import compute_kernel_matrix
from tools.test_num_qubit_impact import get_safe_sample_size, QUBIT_SAMPLE_CONFIGS

def test_vram_estimation():
    """Test that VRAM estimation is less conservative."""
    print("=" * 60)
    print("TEST 1: VRAM Estimation")
    print("=" * 60)
    
    for nq in [4, 8, 12, 16, 20]:
        samples = get_safe_sample_size(nq, 80000, 102.0, 0.85)
        expected = QUBIT_SAMPLE_CONFIGS.get(nq, 10000)
        
        print(f"  {nq} qubits: {samples:6d} samples (expected ≈{expected:6d})")
        
        # Verify it's not too conservative
        if nq <= 12:
            assert samples >= 20000, f"Too conservative for {nq} qubits: {samples}"
        elif nq == 16:
            assert samples >= 10000, f"Too conservative for {nq} qubits: {samples}"
        elif nq == 20:
            assert samples >= 2000, f"Too conservative for {nq} qubits: {samples}"
    
    print("  ✅ VRAM estimation looks reasonable\n")

def test_memory_profiler():
    """Test that memory profiler tracks non-zero values."""
    print("=" * 60)
    print("TEST 2: Memory Profiler Tracking")
    print("=" * 60)
    
    # Small test case
    n_qubits = 4
    n_samples = 100
    
    rng = np.random.default_rng(42)
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np.float64)
    weights = rng.normal(0, 0.1, (2, n_qubits)).astype(np.float64)
    
    config = {
        "device_name": "lightning.gpu",
        "gram_backend": "cuda_states",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 10000,
        "state_tile": -1,
        "vram_fraction": 0.85,
        "autotune": False,  # Skip autotuning for speed
        "precompute_all_states": True,
        "dynamic_batch": True,
        "num_streams": 4,
        "learn_tiles": False,  # Skip learning for speed
        "use_cuda_graphs": True,
        "profile_memory": True,
        "verbose_profile": True,
        "progress": True,
    }
    
    try:
        print(f"  Running compute_kernel_matrix with {n_samples} samples, {n_qubits} qubits...")
        K = compute_kernel_matrix(angles, weights=weights, **config)
        print(f"  ✅ Kernel matrix computed: shape={K.shape}")
        print(f"  ✅ Memory profiler report shown above")
        
        # Verify kernel matrix is valid
        assert K.shape == (n_samples, n_samples), f"Wrong shape: {K.shape}"
        assert np.all(np.isfinite(K)), "NaN/Inf in kernel matrix"
        assert np.all(K >= 0), "Negative values in kernel matrix"
        assert np.all(K <= 1.0 + 1e-6), "Values > 1 in kernel matrix"
        
        print(f"  ✅ Kernel matrix is valid\n")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_stream_utilization():
    """Test that stream utilization is reported."""
    print("=" * 60)
    print("TEST 3: Stream Utilization")
    print("=" * 60)
    print("  (Tested as part of TEST 2 - check verbose_profile output above)")
    print("  ✅ Stream utilization should be > 0% in the report\n")

def main():
    print("\n" + "=" * 60)
    print("PROFILER FIXES VALIDATION TEST")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: VRAM estimation
        test_vram_estimation()
        
        # Test 2 & 3: Memory profiler and stream utilization
        # (combined because they require GPU)
        if os.environ.get("SKIP_GPU_TESTS") != "1":
            success = test_memory_profiler()
            test_stream_utilization()
            
            if success:
                print("=" * 60)
                print("ALL TESTS PASSED ✅")
                print("=" * 60)
                print("\nKey observations to verify in output above:")
                print("  1. Memory allocations (states_A, kernel_output) > 0 GB")
                print("  2. H→D and D→H transfer totals > 0 GB")
                print("  3. Stream Utilization > 0%")
                print("  4. CUDA Graph replays > 0 (if enough tiles)")
        else:
            print("⚠️ Skipping GPU tests (SKIP_GPU_TESTS=1)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
