#!/usr/bin/env python3
"""
Simple test to validate VRAM-aware precomputation logic.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.pipeline_backends import _can_precompute_all, _compute_max_precompute_size

def test_vram_check():
    """Test the VRAM checking function."""
    print("="*60)
    print("Testing VRAM-aware precomputation logic")
    print("="*60)
    
    # Test cases: (n_samples, n_qubits, dtype, expected_feasible)
    test_cases = [
        (100000, 8, np.float64, True),   # Small case - should fit
        (100000, 10, np.float64, True),  # Medium case - should fit
        (95000, 12, np.float64, True), # Large case - should fit
        (75000, 14, np.float64, True),  # Very large case
        (20000, 16, np.float64, None),  
        (30000, 16, np.float64, None),
    ]
    
    try:
        import cupy as cp
        device = cp.cuda.Device()
        total_vram_gb = device.mem_info[1] / 1e9
        available_vram_gb = device.mem_info[0] / 1e9
        print(f"\n💾 GPU VRAM: {total_vram_gb:.1f} GB total, {available_vram_gb:.1f} GB free\n")
    except Exception as e:
        print(f"\n⚠️ Cannot detect GPU: {e}")
        print("This test requires a CUDA-enabled GPU.\n")
        return
    
    print(f"{'Samples':<10} {'Qubits':<8} {'Dtype':<10} {'Can Precompute':<20} {'Max States':<12}")
    print("-"*60)
    
    for n_samples, n_qubits, dtype, expected in test_cases:
        can_precompute = _can_precompute_all(n_samples, n_qubits, dtype)
        max_states = _compute_max_precompute_size(0.95, n_qubits, dtype)
        
        dtype_str = "float64" if dtype == np.float64 else "float32"
        result_icon = "✅" if can_precompute else "❌"
        
        print(f"{n_samples:<10} {n_qubits:<8} {dtype_str:<10} {result_icon} {str(can_precompute):<17} {max_states:<12}")
        
        # Validate expected results for known cases
        if expected is not None and can_precompute != expected:
            print(f"   ⚠️ WARNING: Expected {expected}, got {can_precompute}")
    
    print("\n" + "="*60)
    print("✅ VRAM check test completed")
    print("="*60)

if __name__ == "__main__":
    test_vram_check()
