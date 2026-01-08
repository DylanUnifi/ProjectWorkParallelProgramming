#!/usr/bin/env python3
"""
Test: Impact of Tile Size and Sample Count on Performance
==========================================================

This test measures:
1. How tile_size affects throughput for each backend
2. How sample count (N) affects scaling (should be O(N²))
3. Optimal tile configurations for different workload sizes

Author: Dylan Fouepe
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pipeline_backends import compute_kernel_matrix

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Fixed qubit count for tile/sample tests
N_QUBITS = 10

# Sample sizes to test
SAMPLE_SIZES = [1000, 2000, 4000, 8000, 16000]

# Tile sizes to test
TILE_SIZES = {
    "cuda_states": {
        "state_tile": [512, 1024, 2048, 4096, 8192],
        "tile_size": [1000, 5000, 10000],  # Kernel tile
    },
    "torch": {
        "tile_size": [64, 128, 256, 512, 1024],
    },
    "numpy": {
        "tile_size": [32, 64, 128, 256, 512],
        "n_workers": [1, 2, 4, 8, 16],
    },
}

OUTPUT_DIR = ROOT / "benchmark_results"
OUTPUT_CSV = OUTPUT_DIR / "tile_samples_impact_results.csv"

# ═══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def reset_gpu():
    if HAS_TORCH:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_peak_vram() -> float:
    if HAS_TORCH:
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0

# ═══════════════════════════════════════════════════════════
# BENCHMARK FUNCTIONS
# ═══════════════════════════════════════════════════════════

def benchmark_config(n_samples: int, n_qubits: int, **kwargs) -> Dict:
    """Benchmark a single configuration."""
    rng = np.random.default_rng(42)
    dtype_str = kwargs.get("dtype", "float64")
    np_dtype = np.float32 if dtype_str == "float32" else np.float64
    
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np_dtype)
    weights = rng.normal(0, 0.1, (2, n_qubits)).astype(np_dtype)
    
    reset_gpu()
    
    # Warmup
    warmup_n = min(256, n_samples)
    _ = compute_kernel_matrix(angles[:warmup_n], weights=weights, **kwargs)
    
    if HAS_TORCH:
        torch.cuda.synchronize()
    
    reset_gpu()
    
    # Timed run
    if HAS_TORCH:
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    K = compute_kernel_matrix(angles, weights=weights, **kwargs)
    
    if HAS_TORCH:
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - t0
    peak_vram = get_peak_vram()
    
    n_pairs = n_samples * (n_samples + 1) // 2
    throughput = n_pairs / elapsed / 1e6
    
    del K
    
    return {
        "time_s": elapsed,
        "throughput_mpairs_s": throughput,
        "peak_vram_gb": peak_vram,
    }

def test_cuda_states_tile_impact():
    """Test state_tile impact for cuda_states backend."""
    print("\n" + "="*80)
    print("TEST 1: CUDA_STATES - state_tile Impact")
    print("="*80)
    
    results = []
    n_samples = 8000
    
    print(f"\n{'state_tile':<12} {'tile_size':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
    print("-"*70)
    
    for state_tile in TILE_SIZES["cuda_states"]["state_tile"]:
        for tile_size in TILE_SIZES["cuda_states"]["tile_size"]:
            try:
                res = benchmark_config(
                    n_samples, N_QUBITS,
                    device_name="lightning.gpu",
                    gram_backend="cuda_states",
                    dtype="float64",
                    symmetric=True,
                    state_tile=state_tile,
                    tile_size=tile_size,
                )
                
                results.append({
                    "backend": "cuda_states",
                    "n_samples": n_samples,
                    "n_qubits": N_QUBITS,
                    "state_tile": state_tile,
                    "tile_size": tile_size,
                    "n_workers": None,
                    **res,
                })
                
                print(f"{state_tile:<12} {tile_size:<12} {res['time_s']:<12.3f} "
                      f"{res['throughput_mpairs_s']:<12.3f} {res['peak_vram_gb']:<12.2f}")
                
            except Exception as e:
                print(f"{state_tile:<12} {tile_size:<12} ERROR: {str(e)[:40]}")
    
    # Find optimal
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\n✅ OPTIMAL: state_tile={best['state_tile']}, tile_size={best['tile_size']} "
              f"→ {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return results

def test_numpy_tile_workers_impact():
    """Test tile_size × n_workers for numpy backend."""
    print("\n" + "="*80)
    print("TEST 2: NUMPY - tile_size × n_workers Impact")
    print("="*80)
    
    results = []
    n_samples = 2000  # Smaller for CPU
    
    print(f"\n{'tile_size':<12} {'n_workers':<12} {'Time (s)':<12} {'Mpairs/s':<12}")
    print("-"*60)
    
    for tile_size in TILE_SIZES["numpy"]["tile_size"]:
        for n_workers in TILE_SIZES["numpy"]["n_workers"]:
            try:
                res = benchmark_config(
                    n_samples, N_QUBITS,
                    device_name="default.qubit",
                    gram_backend="numpy",
                    dtype="float64",
                    symmetric=True,
                    tile_size=tile_size,
                    n_workers=n_workers,
                )
                
                results.append({
                    "backend": "numpy",
                    "n_samples": n_samples,
                    "n_qubits": N_QUBITS,
                    "state_tile": None,
                    "tile_size": tile_size,
                    "n_workers": n_workers,
                    **res,
                })
                
                print(f"{tile_size:<12} {n_workers:<12} {res['time_s']:<12.3f} "
                      f"{res['throughput_mpairs_s']:<12.3f}")
                
            except Exception as e:
                print(f"{tile_size:<12} {n_workers:<12} ERROR: {str(e)[:40]}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\n✅ OPTIMAL: tile_size={best['tile_size']}, n_workers={best['n_workers']} "
              f"→ {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return results

def test_sample_scaling():
    """Test how performance scales with sample count."""
    print("\n" + "="*80)
    print("TEST 3: Sample Count Scaling (O(N²) verification)")
    print("="*80)
    
    results = []
    
    print(f"\n{'Backend':<15} {'N':<8} {'Time (s)':<12} {'Mpairs/s':<12} {'N²/Time':<12}")
    print("-"*70)
    
    # Use configurations from global BACKEND_CONFIGS
    test_backends = {
        "cuda_states": {
            **BACKEND_CONFIGS["cuda_states"],
            "state_tile": 4096
        },
        "numpy": {
            **BACKEND_CONFIGS["numpy"],
            "tile_size": 128,
            "n_workers": 4
        }
    }
    
    for backend_name, config in test_backends.items():
        sample_limits = SAMPLE_SIZES if backend_name == "cuda_states" else [s for s in SAMPLE_SIZES if s <= 4000]
        
        for n_samples in sample_limits:
            try:
                res = benchmark_config(n_samples, N_QUBITS, **config)
                
                n_squared_per_time = (n_samples ** 2) / res['time_s'] / 1e6
                
                results.append({
                    "backend": backend_name,
                    "n_samples": n_samples,
                    "n_qubits": N_QUBITS,
                    **res,
                    "n_squared_per_time": n_squared_per_time,
                })
                
                print(f"{backend_name:<15} {n_samples:<8} {res['time_s']:<12.3f} "
                      f"{res['throughput_mpairs_s']:<12.3f} {n_squared_per_time:<12.3f}")
                
            except Exception as e:
                print(f"{backend_name:<15} {n_samples:<8} ERROR: {str(e)[:40]}")
    
    return results

def run_all_tile_tests():
    """Run all tile and sample impact tests."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("TILE SIZE & SAMPLE COUNT IMPACT ANALYSIS")
    print("="*80)
    print(f"📊 Fixed configuration: {N_QUBITS} qubits")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("="*80)
    
    all_results = []
    
    # Run tests
    all_results.extend(test_cuda_states_tile_impact())
    all_results.extend(test_numpy_tile_workers_impact())
    all_results.extend(test_sample_scaling())
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 All results saved to: {OUTPUT_CSV}")
    
    return df

if __name__ == "__main__":
    try:
        df = run_all_tile_tests()
        print("\n✅ All tests completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)