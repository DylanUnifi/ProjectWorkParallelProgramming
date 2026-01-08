#!/usr/bin/env python3
"""
Standalone GPU Kernel Profiling Tool

This tool provides detailed profiling of quantum kernel matrix computation,
including memory usage, bandwidth analysis, and performance metrics.

Usage:
    python tools/profile_kernel.py --n-samples 1000 --n-qubits 6 --backend cuda_states
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_backends import compute_kernel_matrix


def generate_test_data(n_samples: int, n_qubits: int, n_layers: int = 2, seed: int = 42):
    """
    Generate synthetic test data for profiling.
    
    Args:
        n_samples: Number of data samples
        n_qubits: Number of qubits
        n_layers: Number of layers in the quantum circuit
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X, weights) where X is the input data and weights are circuit parameters
    """
    rng = np.random.RandomState(seed)
    
    # Generate random input data scaled to [0, 2π]
    X = rng.uniform(0, 2 * np.pi, size=(n_samples, n_qubits))
    
    # Generate random weights for quantum circuit
    # BasicEntanglerLayers expects shape (n_layers, n_qubits)
    weights = rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits))
    
    return X, weights


def main():
    parser = argparse.ArgumentParser(
        description="Profile GPU kernel matrix computation with detailed memory and performance analysis"
    )
    
    # Data parameters
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples (default: 500)")
    parser.add_argument("--n-qubits", type=int, default=6,
                        help="Number of qubits (default: 6)")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of circuit layers (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Backend parameters
    parser.add_argument("--backend", type=str, default="cuda_states",
                        choices=["cuda_states", "torch", "auto"],
                        help="Computation backend (default: cuda_states)")
    parser.add_argument("--device", type=str, default="lightning.gpu",
                        help="Quantum device (default: lightning.gpu)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64"],
                        help="Floating point precision (default: float32)")
    
    # Optimization parameters
    parser.add_argument("--state-tile", type=int, default=-1,
                        help="State tile size (-1 for auto, default: -1)")
    parser.add_argument("--vram-fraction", type=float, default=0.85,
                        help="VRAM fraction to use (default: 0.85)")
    parser.add_argument("--num-streams", type=int, default=4,
                        help="Number of CUDA streams (default: 4)")
    parser.add_argument("--no-autotune", action="store_true",
                        help="Disable kernel autotuning")
    parser.add_argument("--no-dynamic-batch", action="store_true",
                        help="Disable dynamic batch sizing")
    parser.add_argument("--no-cuda-graphs", action="store_true",
                        help="Disable CUDA graph optimization")
    parser.add_argument("--no-learn-tiles", action="store_true",
                        help="Disable tile size learning")
    parser.add_argument("--no-precompute", action="store_true",
                        help="Disable bulk state precomputation")
    
    # Profiling parameters
    parser.add_argument("--profile-memory", action="store_true",
                        help="Enable detailed memory profiling")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose profiling output")
    
    args = parser.parse_args()
    
    print("="*70)
    print("GPU KERNEL PROFILING TOOL")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Samples:       {args.n_samples}")
    print(f"  Qubits:        {args.n_qubits}")
    print(f"  Layers:        {args.n_layers}")
    print(f"  Backend:       {args.backend}")
    print(f"  Device:        {args.device}")
    print(f"  Precision:     {args.dtype}")
    print(f"  State Tile:    {'auto' if args.state_tile == -1 else args.state_tile}")
    print(f"  VRAM Fraction: {args.vram_fraction}")
    print(f"  Num Streams:   {args.num_streams}")
    print(f"\nOptimizations:")
    print(f"  Autotuning:       {'disabled' if args.no_autotune else 'enabled'}")
    print(f"  Dynamic Batch:    {'disabled' if args.no_dynamic_batch else 'enabled'}")
    print(f"  CUDA Graphs:      {'disabled' if args.no_cuda_graphs else 'enabled'}")
    print(f"  Tile Learning:    {'disabled' if args.no_learn_tiles else 'enabled'}")
    print(f"  Precompute:       {'disabled' if args.no_precompute else 'enabled'}")
    print(f"  Memory Profiling: {'enabled' if args.profile_memory else 'disabled'}")
    print("="*70)
    
    # Generate test data
    print("\n📊 Generating test data...")
    X, weights = generate_test_data(args.n_samples, args.n_qubits, args.n_layers, args.seed)
    print(f"   X shape: {X.shape}")
    print(f"   Weights shape: {weights.shape}")
    
    # Compute kernel matrix with profiling
    print("\n🚀 Computing kernel matrix...")
    try:
        K = compute_kernel_matrix(
            X, Y=None,
            weights=weights,
            device_name=args.device,
            gram_backend=args.backend,
            dtype=args.dtype,
            return_dtype=args.dtype,
            symmetric=True,
            progress=True,
            state_tile=args.state_tile,
            vram_fraction=args.vram_fraction,
            autotune=not args.no_autotune,
            precompute_all_states=not args.no_precompute,
            dynamic_batch=not args.no_dynamic_batch,
            num_streams=args.num_streams,
            learn_tiles=not args.no_learn_tiles,
            profile_memory=args.profile_memory,
            use_cuda_graphs=not args.no_cuda_graphs,
            verbose_profile=args.verbose
        )
        
        print(f"\n✅ Kernel matrix computed successfully!")
        print(f"   Output shape: {K.shape}")
        print(f"   Output dtype: {K.dtype}")
        print(f"   Min value: {K.min():.6f}")
        print(f"   Max value: {K.max():.6f}")
        print(f"   Mean value: {K.mean():.6f}")
        
        # Verify matrix properties
        if np.allclose(K, K.T, rtol=1e-5, atol=1e-5):
            print("   ✓ Matrix is symmetric")
        else:
            print("   ⚠ Matrix is not symmetric")
        
        if np.all(np.isfinite(K)):
            print("   ✓ All values are finite")
        else:
            print("   ⚠ Matrix contains NaN or Inf")
        
        print("\n" + "="*70)
        print("PROFILING COMPLETE")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during computation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
