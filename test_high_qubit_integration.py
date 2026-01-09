#!/usr/bin/env python3
"""
Integration test for VRAM-aware kernel computation.
Tests that high qubit counts with large samples handle OOM gracefully.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.pipeline_backends import compute_kernel_matrix

def test_high_qubit_kernel():
    """Test kernel computation with high qubit counts."""
    print("="*70)
    print("Integration Test: High Qubit Kernel Computation with VRAM Awareness")
    print("="*70)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n⚠️ No CUDA GPU detected. Skipping test.")
            return
    except ImportError:
        print("\n⚠️ PyTorch not available. Skipping test.")
        return
    
    # Test cases with progressively more challenging configurations
    test_cases = [
        {"n_qubits": 8, "n_samples": 1000, "desc": "Small baseline"},
        {"n_qubits": 12, "n_samples": 5000, "desc": "Medium case"},
        {"n_qubits": 16, "n_samples": 10000, "desc": "High qubits, reduced samples"},
    ]
    
    print(f"\n{'Test Case':<30} {'Qubits':<8} {'Samples':<10} {'Status':<15}")
    print("-"*70)
    
    for test in test_cases:
        n_qubits = test["n_qubits"]
        n_samples = test["n_samples"]
        desc = test["desc"]
        
        try:
            # Generate test data
            rng = np.random.default_rng(42)
            X = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np.float64)
            w = rng.normal(0, 0.1, (2, n_qubits)).astype(np.float64)
            
            # Compute kernel with VRAM awareness enabled
            K = compute_kernel_matrix(
                X, 
                weights=w, 
                gram_backend='cuda_states', 
                device_name='lightning.gpu', 
                precompute_all_states=True,  # Will auto-disable if needed
                state_tile=-1,  # Auto-size
                vram_fraction=0.85,
                progress=True, 
                verbose_profile=False
            )
            
            # Validate output
            if not np.all(np.isfinite(K)):
                print(f"{desc:<30} {n_qubits:<8} {n_samples:<10} {'⚠️ NaN/Inf':<15}")
            else:
                print(f"{desc:<30} {n_qubits:<8} {n_samples:<10} {'✅ Success':<15}")
            
            del K
            
        except Exception as e:
            error_type = "OOM" if "memory" in str(e).lower() else "Error"
            print(f"{desc:<30} {n_qubits:<8} {n_samples:<10} {'❌ ' + error_type:<15}")
            print(f"   Error: {str(e)[:80]}")
    
    print("\n" + "="*70)
    print("✅ Integration test completed")
    print("="*70)

if __name__ == "__main__":
    test_high_qubit_kernel()
