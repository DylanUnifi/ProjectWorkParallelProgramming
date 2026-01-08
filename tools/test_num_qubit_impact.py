#!/usr/bin/env python3
"""
Test: Impact of Number of Qubits on Quantum Kernel Performance
==============================================================

This test measures how performance scales with qubit count across backends.
Key insight: State vector size = 2^n_qubits, so memory and compute scale exponentially.

Author: Dylan Fouepe
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pipeline_backends import compute_kernel_matrix

# Try to import torch for GPU memory monitoring
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Qubit range to test
QUBITS_RANGE = [4, 6, 8, 10, 12, 14, 16, 18, 20]

# Fixed sample size (large enough to amortize overhead)
N_SAMPLES = 2000

# Backend configurations with VALID parameters only
BACKEND_CONFIGS = {
    "cuda_states": {
        "device_name": "lightning.gpu",
        "gram_backend": "cuda_states",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 10000,
        "state_tile": 4096,  # Will be auto-adjusted for high qubit counts
    },
    "torch": {
        "device_name": "lightning.gpu",
        "gram_backend": "torch",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 512,
    },
    "numpy": {
        "device_name": "default.qubit",
        "gram_backend": "numpy",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 64,
        "n_workers": 4,
    },
}

# Qubit limits per backend (beyond this, backend becomes impractical)
BACKEND_QUBIT_LIMITS = {
    "cuda_states": 20,
    "torch": 16,
    "numpy": 10,
}

# Output configuration
OUTPUT_DIR = ROOT / "benchmark_results"
OUTPUT_CSV = OUTPUT_DIR / "qubit_impact_results.csv"

# ═══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def get_gpu_memory_info() -> Tuple[float, float]:
    """Get current and peak GPU memory usage in GB."""
    if HAS_TORCH:
        current = torch.cuda.memory_allocated() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return current, peak
    return 0.0, 0.0

def reset_gpu_memory():
    """Reset GPU memory stats and clear cache."""
    if HAS_TORCH:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def calculate_state_vector_size(n_qubits: int, dtype: str = "float64") -> float:
    """Calculate state vector size in GB for given qubit count."""
    dim = 2 ** n_qubits
    # Complex number: 2 floats
    bytes_per_element = 16 if dtype == "float64" else 8  # complex128 or complex64
    return dim * bytes_per_element / (1024**3)

def adjust_state_tile_for_qubits(n_qubits: int, base_tile: int = 4096) -> int:
    """Reduce state_tile for high qubit counts to avoid OOM."""
    if n_qubits >= 18:
        return min(base_tile, 512)
    elif n_qubits >= 16:
        return min(base_tile, 1024)
    elif n_qubits >= 14:
        return min(base_tile, 2048)
    return base_tile

# ═══════════════════════════════════════════════════════════
# BENCHMARK FUNCTIONS
# ═══════════════════════════════════════════════════════════

def benchmark_single_config(
    n_qubits: int,
    n_samples: int,
    backend_name: str,
    config: Dict,
    warmup: bool = True,
    repeats: int = 3,
) -> Optional[Dict]:
    """Run benchmark for a single configuration."""
    
    # Generate test data
    rng = np.random.default_rng(42)
    np_dtype = np.float32 if config.get("dtype") == "float32" else np.float64
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np_dtype)
    weights = rng.normal(0, 0.1, (2, n_qubits)).astype(np_dtype)
    
    # Adjust state_tile for high qubit counts
    run_config = config.copy()
    if "state_tile" in run_config:
        run_config["state_tile"] = adjust_state_tile_for_qubits(n_qubits, run_config["state_tile"])
    
    reset_gpu_memory()
    
    try:
        # Warmup run
        if warmup:
            _ = compute_kernel_matrix(angles[:min(256, n_samples)], weights=weights, **run_config)
            if HAS_TORCH:
                torch.cuda.synchronize()
        
        reset_gpu_memory()
        
        # Timed runs
        times = []
        for _ in range(repeats):
            if HAS_TORCH:
                torch.cuda.synchronize()
            
            t0 = time.perf_counter()
            K = compute_kernel_matrix(angles, weights=weights, **run_config)
            
            if HAS_TORCH:
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - t0)
            del K
        
        _, peak_vram = get_gpu_memory_info()
        
        # Calculate metrics
        mean_time = np.mean(times)
        std_time = np.std(times)
        n_pairs = n_samples * (n_samples + 1) // 2  # Symmetric matrix
        throughput = n_pairs / mean_time / 1e6
        
        return {
            "n_qubits": n_qubits,
            "backend": backend_name,
            "n_samples": n_samples,
            "time_s": mean_time,
            "time_std_s": std_time,
            "throughput_mpairs_s": throughput,
            "peak_vram_gb": peak_vram,
            "state_vector_size_gb": calculate_state_vector_size(n_qubits, config.get("dtype", "float64")),
            "state_tile": run_config.get("state_tile", "N/A"),
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)[:60]}")
        return None

def run_qubit_impact_test() -> pd.DataFrame:
    """Run the complete qubit impact test across all backends."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("TEST: Impact of Number of Qubits on Performance")
    print("="*80)
    print(f"📊 Configuration:")
    print(f"   - Qubits range: {QUBITS_RANGE}")
    print(f"   - Samples: {N_SAMPLES}")
    print(f"   - Backends: {list(BACKEND_CONFIGS.keys())}")
    print("="*80 + "\n")
    
    results = []
    
    for backend_name, config in BACKEND_CONFIGS.items():
        qubit_limit = BACKEND_QUBIT_LIMITS.get(backend_name, 20)
        applicable_qubits = [q for q in QUBITS_RANGE if q <= qubit_limit]
        
        print(f"\n🔧 Backend: {backend_name.upper()}")
        print(f"   Qubit range: {applicable_qubits}")
        print("-"*60)
        print(f"{'Qubits':<8} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12} {'State Vec':<12}")
        print("-"*60)
        
        for n_qubits in applicable_qubits:
            result = benchmark_single_config(
                n_qubits=n_qubits,
                n_samples=N_SAMPLES,
                backend_name=backend_name,
                config=config,
            )
            
            if result:
                results.append(result)
                print(f"{n_qubits:<8} {result['time_s']:<12.3f} {result['throughput_mpairs_s']:<12.3f} "
                      f"{result['peak_vram_gb']:<12.2f} {result['state_vector_size_gb']:<12.4f}")
            else:
                print(f"{n_qubits:<8} FAILED")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Results saved to: {OUTPUT_CSV}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Exponential Scaling Analysis")
    print("="*80)
    
    for backend in df['backend'].unique():
        backend_df = df[df['backend'] == backend].sort_values('n_qubits')
        if len(backend_df) >= 2:
            # Calculate scaling factor
            times = backend_df['time_s'].values
            qubits = backend_df['n_qubits'].values
            
            # Fit log(time) vs qubits to find scaling
            # Filter out invalid times before log transformation
            valid_mask = times > 0
            if valid_mask.sum() > 1:
                valid_times = times[valid_mask]
                valid_qubits = qubits[valid_mask]
                log_times = np.log2(valid_times)
                coeffs = np.polyfit(valid_qubits, log_times, 1)
                scaling_factor = 2 ** coeffs[0]  # How much time multiplies per qubit
                
                print(f"\n{backend}:")
                print(f"  - Time scaling per qubit: {scaling_factor:.2f}x")
                print(f"  - Fastest: {backend_df['throughput_mpairs_s'].max():.2f} Mpairs/s at {backend_df.loc[backend_df['throughput_mpairs_s'].idxmax(), 'n_qubits']} qubits")
    
    return df

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        df = run_qubit_impact_test()
        print("\n✅ Test completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)