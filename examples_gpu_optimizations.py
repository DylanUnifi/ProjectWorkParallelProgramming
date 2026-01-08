#!/usr/bin/env python3
"""
Example usage of GPU throughput optimizations.

This script demonstrates how to use the new optimization features
in pipeline_backends.py for maximum GPU performance.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scripts.pipeline_backends import compute_kernel_matrix
    import torch
    HAS_DEPS = True
except ImportError as e:
    print(f"⚠️  Dependencies not installed: {e}")
    print("This is an example script showing API usage.")
    HAS_DEPS = False

def example_automatic_optimization():
    """
    Example 1: Automatic optimization (recommended for most users).
    
    All optimizations enabled with automatic configuration:
    - VRAM-aware tile sizing
    - Kernel autotuning
    - Bulk state precomputation
    - Async dispatch
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Automatic Optimization (Recommended)")
    print("="*70 + "\n")
    
    if not HAS_DEPS:
        print("Code example (requires dependencies):")
    
    print("""
    # Generate test data
    n_samples = 10000
    n_qubits = 8
    n_layers = 2
    
    rng = np.random.default_rng(42)
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np.float32)
    weights = rng.normal(0, 0.1, (n_layers, n_qubits)).astype(np.float32)
    
    # Compute kernel with automatic optimizations
    K = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name="lightning.gpu",
        gram_backend="cuda_states",
        
        # NEW OPTIMIZATION PARAMETERS (all defaults):
        state_tile=-1,              # Auto-size based on available VRAM
        autotune=True,              # Auto-tune CUDA kernel parameters
        precompute_all_states=True, # Bulk precompute when possible
        vram_fraction=0.85,         # Use 85% of VRAM
        
        # Standard parameters:
        symmetric=True,
        dtype="float32",
        progress=True
    )
    
    print(f"Computed kernel matrix: {K.shape}")
    print(f"VRAM-aware sizing automatically selected optimal tile size")
    print(f"Kernel parameters auto-tuned for this GPU")
    """)
    
    if HAS_DEPS:
        try:
            # Actually run the example
            n_samples = 100  # Small for testing
            n_qubits = 4
            n_layers = 1
            
            rng = np.random.default_rng(42)
            angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np.float32)
            weights = rng.normal(0, 0.1, (n_layers, n_qubits)).astype(np.float32)
            
            K = compute_kernel_matrix(
                angles,
                weights=weights,
                device_name="lightning.gpu",
                gram_backend="cuda_states",
                state_tile=-1,
                autotune=True,
                precompute_all_states=True,
                vram_fraction=0.85,
                symmetric=True,
                dtype="float32",
                progress=False
            )
            
            print(f"\n✓ Successfully computed kernel: {K.shape}")
            print(f"  Min: {K.min():.6f}, Max: {K.max():.6f}, Mean: {K.mean():.6f}")
        except Exception as e:
            print(f"\n⚠️  Could not run example: {e}")

def example_memory_constrained():
    """
    Example 2: Memory-constrained configuration.
    
    For systems with limited VRAM or when processing very large datasets.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Memory-Constrained Configuration")
    print("="*70 + "\n")
    
    print("""
    # For systems with limited GPU memory
    K = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name="lightning.gpu",
        gram_backend="cuda_states",
        
        # Conservative memory usage:
        state_tile=2048,            # Manual smaller tile size
        vram_fraction=0.70,         # Only use 70% VRAM
        precompute_all_states=False,# Use tiled approach
        
        # Still enable autotuning:
        autotune=True,
        
        symmetric=True,
        dtype="float32"
    )
    
    print(f"Memory-efficient computation: {K.shape}")
    """)

def example_maximum_performance():
    """
    Example 3: Maximum performance configuration.
    
    For high-VRAM GPUs (e.g., 96GB RTX 6000) when maximum speed is needed.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Maximum Performance (96GB GPU)")
    print("="*70 + "\n")
    
    print("""
    # For high-end GPUs with abundant VRAM
    K = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name="lightning.gpu",
        gram_backend="cuda_states",
        
        # Aggressive optimization:
        state_tile=-1,              # Auto-size (will use large tiles)
        vram_fraction=0.90,         # Use 90% of VRAM
        autotune=True,              # Auto-tune kernels
        precompute_all_states=True, # Bulk precompute
        
        symmetric=True,
        dtype="float32",
        progress=True
    )
    
    print(f"Maximum performance computation: {K.shape}")
    print("Expected improvements:")
    print("  - 5-10x reduction in memory handoffs")
    print("  - 2-3x faster PCIe transfers")
    print("  - 20-40% lower latency")
    """)

def example_manual_tuning():
    """
    Example 4: Manual parameter tuning.
    
    For users who want fine-grained control.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Manual Parameter Tuning")
    print("="*70 + "\n")
    
    print("""
    # Fine-grained control over optimization parameters
    K = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name="lightning.gpu",
        gram_backend="cuda_states",
        
        # Manual configuration:
        state_tile=8192,            # Manual tile size
        autotune=False,             # Disable autotuning
        tile_m=64,                  # Manual CUDA tile M
        tile_n=64,                  # Manual CUDA tile N
        tile_k=32,                  # Manual CUDA tile K
        precompute_all_states=True, # Still use bulk precompute
        vram_fraction=0.85,
        
        symmetric=True,
        dtype="float32"
    )
    
    print(f"Manually tuned computation: {K.shape}")
    """)

def example_backward_compatibility():
    """
    Example 5: Backward compatibility.
    
    Old code continues to work without modification.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Backward Compatibility")
    print("="*70 + "\n")
    
    print("""
    # Old code (before optimizations) still works
    K = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name="lightning.gpu",
        gram_backend="cuda_states",
        tile_size=64,
        symmetric=True,
        dtype="float32"
    )
    
    # Automatically uses new optimizations with defaults:
    # - state_tile=-1 (auto-size)
    # - autotune=True
    # - precompute_all_states=True
    # - vram_fraction=0.85
    
    print("Old API works seamlessly with new optimizations!")
    """)

def example_benchmark_comparison():
    """
    Example 6: Benchmark old vs new implementation.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Performance Comparison")
    print("="*70 + "\n")
    
    print("""
    import time
    
    # Old implementation (disable optimizations)
    t0 = time.perf_counter()
    K_old = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name="lightning.gpu",
        gram_backend="cuda_states",
        state_tile=1024,            # Small fixed tile
        autotune=False,             # No autotuning
        precompute_all_states=False # No bulk precompute
    )
    time_old = time.perf_counter() - t0
    
    # New implementation (all optimizations)
    t0 = time.perf_counter()
    K_new = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name="lightning.gpu",
        gram_backend="cuda_states",
        state_tile=-1,              # Auto-size
        autotune=True,              # Autotuning
        precompute_all_states=True  # Bulk precompute
    )
    time_new = time.perf_counter() - t0
    
    speedup = time_old / time_new
    print(f"Old: {time_old:.3f}s")
    print(f"New: {time_new:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Results match: {np.allclose(K_old, K_new)}")
    """)

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GPU THROUGHPUT OPTIMIZATION - USAGE EXAMPLES")
    print("="*70)
    
    if not HAS_DEPS:
        print("\n⚠️  Note: Dependencies not installed.")
        print("Showing API usage examples without execution.\n")
    
    # Run examples
    example_automatic_optimization()
    example_memory_constrained()
    example_maximum_performance()
    example_manual_tuning()
    example_backward_compatibility()
    example_benchmark_comparison()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    print("Key Recommendations:")
    print("  1. Use automatic optimization (Example 1) for most cases")
    print("  2. Adjust vram_fraction if you encounter OOM errors")
    print("  3. Monitor performance with progress=True")
    print("  4. Check .cuda_kernel_autotune.json for cached results")
    print("\nFor detailed documentation, see GPU_OPTIMIZATIONS.md")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
