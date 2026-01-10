#!/usr/bin/env python3
"""
Benchmark PRODUCTION - Comprehensive Performance Analysis
=========================================================

This script provides comprehensive benchmarking of ALL optimizations available
in both the cuda_states and torch backends.

Comprehensive benchmark sections:
1. Backend Comparison (Baseline) - Compare all backends with default optimizations
2. cuda_states Optimization Ablation - Test each optimization individually
3. cuda_states State Tile Optimization - Find optimal state_tile for different workloads
4. cuda_states VRAM Fraction Impact - Test impact of vram_fraction on performance  
5. cuda_states Stream Pool Impact - Test num_streams impact on throughput
6. torch Optimization Ablation - Test each torch optimization individually
7. torch Tile Size Optimization - Find optimal tile_size for torch backend
8. Memory Profiling - Run benchmarks with detailed memory profiling
9. Qubit Scaling Analysis - Measure exponential scaling with qubit count
10. Sample Scaling Analysis - Verify O(N²) scaling with sample count

CLI Arguments:
- --all: Run all benchmarks
- --backend-comparison: Compare all backends
- --cuda-states-ablation: Test cuda_states optimizations
- --cuda-states-state-tile: Test state_tile values
- --cuda-states-vram: Test VRAM fraction impact
- --cuda-states-streams: Test stream pool sizes
- --torch-ablation: Test torch optimizations
- --torch-tiles: Test torch tile sizes
- --memory-profiling: Run with memory profiling
- --qubit-scaling: Test scaling with qubit count
- --sample-scaling: Test scaling with sample count

Configuration:
- --n-samples: Number of samples (default: 20000)
- --n-qubits: Number of qubits (default: 16)
- --output-dir: Output directory (default: benchmark_results)
- --warmup-runs: Number of warmup runs (default: 1)
- --benchmark-runs: Number of benchmark runs (default: 3)
- --verbose: Print detailed information

Author: Dylan Fouepe
Date: 2026-01-08
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from scripts.pipeline_backends import compute_kernel_matrix
except ImportError:
    print("❌ Error: Cannot import 'scripts.pipeline_backends'.")
    print("Run this script from project root: python3 benchmark_production.py")
    sys.exit(1)

# Try to import torch for GPU memory monitoring
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Test configurations
QUBITS_RANGE = [8, 12, 16]
SAMPLE_SIZES = [10000, 20000, 30000]
N_SAMPLES_DEFAULT = 10000
N_QUBITS_DEFAULT = 16

# Backend configurations with VALID parameters only
BACKEND_CONFIGS = {
    "cuda_states": {
        "device_name": "lightning.gpu",
        "gram_backend": "cuda_states",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 5000,
        # cuda_states optimization parameters
        "state_tile": -1,
        "vram_fraction": 0.95,
        "autotune": True,
        "precompute_all_states": True,
        "dynamic_batch": True,
        "num_streams": 2,
        "learn_tiles": True,
        "use_cuda_graphs": True,
        "profile_memory": True,
        "verbose_profile": True,
    },
    #"torch": {
    #    "device_name": "lightning.gpu",
    #    "gram_backend": "torch",
    #    "dtype": "float64",
    #    "symmetric": True,
    #    "tile_size": 512,
        # Torch-specific optimizations
    #    "use_pinned_memory": True,
    #    "use_cuda_streams": True,
    #    "use_amp": False,
    #    "use_compile": False,
    #},
    "numpy": {
        "device_name": "default.qubit",
        "gram_backend": "numpy",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 128,
        "n_workers": 16,
    },
}

# Output configuration
OUTPUT_DIR = ROOT / "benchmark_results"
OUTPUT_CSV = OUTPUT_DIR / "production_benchmark.csv"
OUTPUT_PLOTS = OUTPUT_DIR / "production_benchmark.png"
OUTPUT_JSON = OUTPUT_DIR / "production_benchmark_summary.json"

# ═══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def format_time(seconds: float) -> str:
    """Format time in hours/minutes/seconds."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

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
    bytes_per_element = 16 if dtype == "float64" else 8
    return dim * bytes_per_element / (1024**3)

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
    
    rng = np.random.default_rng(42)
    np_dtype = np.float32 if config.get("dtype") == "float32" else np.float64
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np_dtype)
    weights = rng.normal(0, 0.1, (2, n_qubits)).astype(np_dtype)
    
    run_config = config.copy()
    
    # Adjust state_tile for high qubit counts
    if "state_tile" in run_config and n_qubits >= 14:
        run_config["state_tile"] = min(run_config["state_tile"], 2048)
    
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
        n_pairs = n_samples * (n_samples + 1) // 2
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
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)[:60]}")
        return None

def test_qubit_impact(backends: List[str] = None) -> pd.DataFrame:
    """Test impact of qubit count on performance."""
    
    if backends is None:
        backends = list(BACKEND_CONFIGS.keys())
    
    print("\n" + "="*80)
    print("TEST 1: Impact of Number of Qubits on Performance")
    print("="*80)
    
    results = []
    
    for backend_name in backends:
        if backend_name not in BACKEND_CONFIGS:
            continue
            
        config = BACKEND_CONFIGS[backend_name]
        qubit_limit = 16 
        applicable_qubits = [q for q in QUBITS_RANGE if q <= qubit_limit]
        
        print(f"\n🔧 Backend: {backend_name.upper()}")
        print(f"{'Qubits':<8} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
        print("-"*60)
        
        for n_qubits in applicable_qubits:
            result = benchmark_single_config(
                n_qubits=n_qubits,
                n_samples=N_SAMPLES_DEFAULT,
                backend_name=backend_name,
                config=config,
            )
            
            if result:
                results.append(result)
                print(f"{n_qubits:<8} {result['time_s']:<12.3f} {result['throughput_mpairs_s']:<12.3f} "
                      f"{result['peak_vram_gb']:<12.2f}")
    
    return pd.DataFrame(results)

def test_sample_scaling(backends: List[str] = None) -> pd.DataFrame:
    """Test how performance scales with sample count."""
    
    if backends is None:
        backends = list(BACKEND_CONFIGS.keys())
    
    print("\n" + "="*80)
    print("TEST 2: Sample Count Scaling (O(N²) verification)")
    print("="*80)
    
    results = []
    
    print(f"\n{'Backend':<15} {'N':<8} {'Time (s)':<12} {'Mpairs/s':<12}")
    print("-"*60)
    
    for backend_name in backends:
        if backend_name not in BACKEND_CONFIGS:
            continue
            
        config = BACKEND_CONFIGS[backend_name]
        sample_limits = SAMPLE_SIZES
        
        for n_samples in sample_limits:
            result = benchmark_single_config(
                n_qubits=N_QUBITS_DEFAULT,
                n_samples=n_samples,
                backend_name=backend_name,
                config=config,
            )
            
            if result:
                results.append(result)
                print(f"{backend_name:<15} {n_samples:<8} {result['time_s']:<12.3f} "
                      f"{result['throughput_mpairs_s']:<12.3f}")
    
    return pd.DataFrame(results)

def test_tile_optimization() -> pd.DataFrame:
    """Test tile size optimization for cuda_states."""
    
    print("\n" + "="*80)
    print("TEST 3: Tile Size Optimization (CUDA_STATES)")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    state_tiles = [2048, 4096, 8192, 16384, 32768, -1]
    
    print(f"\n{'state_tile':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
    print("-"*60)
    
    for state_tile in state_tiles:
        config = BACKEND_CONFIGS["cuda_states"].copy()
        config["state_tile"] = state_tile
        
        result = benchmark_single_config(
            n_qubits=N_QUBITS_DEFAULT,
            n_samples=n_samples,
            backend_name="cuda_states",
            config=config,
        )
        
        if result:
            result["state_tile"] = state_tile
            results.append(result)
            print(f"{state_tile:<12} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {result['peak_vram_gb']:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\n✅ OPTIMAL: state_tile={best['state_tile']} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_vram_fraction_impact() -> pd.DataFrame:
    """Test impact of VRAM fraction parameter for cuda_states."""
    
    print("\n" + "="*80)
    print("TEST 3B: VRAM Fraction Impact (CUDA_STATES)")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    vram_fractions = [0.5, 0.7, 0.85, 0.95]
    
    print(f"\n{'VRAM Frac':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
    print("-"*60)
    
    for vram_frac in vram_fractions:
        config = BACKEND_CONFIGS["cuda_states"].copy()
        config["vram_fraction"] = vram_frac
        config["state_tile"] = -1  # Auto-size based on VRAM fraction
        
        result = benchmark_single_config(
            n_qubits=N_QUBITS_DEFAULT,
            n_samples=n_samples,
            backend_name="cuda_states",
            config=config,
        )
        
        if result:
            result["vram_fraction"] = vram_frac
            results.append(result)
            print(f"{vram_frac:<12.2f} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {result['peak_vram_gb']:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\n✅ OPTIMAL: vram_fraction={best.get('vram_fraction', 0.85):.2f} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_stream_pool_impact() -> pd.DataFrame:
    """Test impact of num_streams parameter for cuda_states."""
    
    print("\n" + "="*80)
    print("TEST 3C: Stream Pool Impact (CUDA_STATES)")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    stream_counts = [1, 2, 4, 8]
    
    print(f"\n{'Streams':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
    print("-"*60)
    
    for num_streams in stream_counts:
        config = BACKEND_CONFIGS["cuda_states"].copy()
        config["num_streams"] = num_streams
        
        result = benchmark_single_config(
            n_qubits=N_QUBITS_DEFAULT,
            n_samples=n_samples,
            backend_name="cuda_states",
            config=config,
        )
        
        if result:
            result["num_streams"] = num_streams
            results.append(result)
            print(f"{num_streams:<12} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {result['peak_vram_gb']:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\n✅ OPTIMAL: num_streams={best.get('num_streams', 4)} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_optimization_ablation() -> pd.DataFrame:
    """Compare each optimization's individual contribution (cuda_states)."""
    
    print("\n" + "="*80)
    print("TEST 4: CUDA_STATES Optimization Ablation Study")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    
    # Define test configurations
    configs = {
        "All Optimizations": {
            "autotune": True,
            "precompute_all_states": True,
            "dynamic_batch": True,
            "use_cuda_graphs": True,
            "num_streams": 4,
        },
        "No Autotune": {
            "autotune": False,
            "precompute_all_states": True,
            "dynamic_batch": True,
            "use_cuda_graphs": True,
            "num_streams": 4,
        },
        "No Precompute": {
            "autotune": True,
            "precompute_all_states": False,
            "dynamic_batch": True,
            "use_cuda_graphs": True,
            "num_streams": 4,
        },
        "No Dynamic Batch": {
            "autotune": True,
            "precompute_all_states": True,
            "dynamic_batch": False,
            "use_cuda_graphs": True,
            "num_streams": 4,
        },
        "No CUDA Graphs": {
            "autotune": True,
            "precompute_all_states": True,
            "dynamic_batch": True,
            "use_cuda_graphs": False,
            "num_streams": 4,
        },
        "Baseline (No Opts)": {
            "autotune": False,
            "precompute_all_states": False,
            "dynamic_batch": False,
            "use_cuda_graphs": False,
            "num_streams": 1,
        },
    }
    
    print(f"\n{'Configuration':<22} {'Time (s)':<12} {'Mpairs/s':<12} {'Speedup':<10}")
    print("-"*70)
    
    baseline_time = None
    
    for config_name, opts in configs.items():
        config = BACKEND_CONFIGS["cuda_states"].copy()
        config.update(opts)
        
        result = benchmark_single_config(
            n_qubits=N_QUBITS_DEFAULT,
            n_samples=n_samples,
            backend_name="cuda_states",
            config=config,
        )
        
        if result:
            # Use baseline for speedup calculation
            if config_name == "Baseline (No Opts)":
                baseline_time = result['time_s']
            
            speedup = baseline_time / result['time_s'] if baseline_time and result['time_s'] > 0 else 1.0
            
            result["configuration"] = config_name
            result["speedup"] = speedup
            results.append(result)
            
            print(f"{config_name:<22} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {speedup:<10.2f}x")
    
    return pd.DataFrame(results)

def benchmark_with_profiling() -> pd.DataFrame:
    """Run with full memory profiling and report (cuda_states)."""
    
    print("\n" + "="*80)
    print("TEST 5: Memory Profiling Analysis")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    
    print("\nRunning cuda_states with full profiling enabled...")
    
    config = BACKEND_CONFIGS["cuda_states"].copy()
    config.update({
        "profile_memory": True,
        "verbose_profile": True,
    })
    
    result = benchmark_single_config(
        n_qubits=N_QUBITS_DEFAULT,
        n_samples=n_samples,
        backend_name="cuda_states",
        config=config,
    )
    
    if result:
        result["test"] = "memory_profiling"
        results.append(result)
        print(f"\n📊 Profiling completed: {result['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_torch_optimizations() -> pd.DataFrame:
    """Benchmark torch backend with different optimization flags."""
    
    print("\n" + "="*80)
    print("TEST 6: Torch Backend Optimizations")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    tile_size = 512
    
    configs = [
        {"name": "torch_baseline", "use_pinned_memory": False, "use_cuda_streams": False, "use_amp": False, "use_compile": False},
        {"name": "torch_pinned", "use_pinned_memory": True, "use_cuda_streams": False, "use_amp": False, "use_compile": False},
        {"name": "torch_streams", "use_pinned_memory": False, "use_cuda_streams": True, "use_amp": False, "use_compile": False},
        {"name": "torch_pinned+streams", "use_pinned_memory": True, "use_cuda_streams": True, "use_amp": False, "use_compile": False},
    ]
    
    print(f"\n{'Config':<25} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
    print("-"*70)
    
    for config_dict in configs:
        base_config = BACKEND_CONFIGS["torch"].copy()
        base_config["tile_size"] = tile_size
        # Update with optimization flags
        for key in ["use_pinned_memory", "use_cuda_streams", "use_amp", "use_compile"]:
            if key in config_dict:
                base_config[key] = config_dict[key]
        
        result = benchmark_single_config(
            n_qubits=N_QUBITS_DEFAULT,
            n_samples=n_samples,
            backend_name="torch",
            config=base_config,
        )
        
        if result:
            result["config_name"] = config_dict["name"]
            result["tile_size"] = tile_size
            results.append(result)
            print(f"{config_dict['name']:<25} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {result['peak_vram_gb']:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\n✅ BEST CONFIG: {best['config_name']} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_torch_tile_sizes() -> pd.DataFrame:
    """Test optimal tile_size for torch backend."""
    
    print("\n" + "="*80)
    print("TEST 7: Torch Tile Size Optimization")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    tile_sizes = [64, 128, 256, 512, 1024, 2048]
    
    print(f"\n{'tile_size':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
    print("-"*60)
    
    for tile_size in tile_sizes:
        config = BACKEND_CONFIGS["torch"].copy()
        config["tile_size"] = tile_size
        
        result = benchmark_single_config(
            n_qubits=N_QUBITS_DEFAULT,
            n_samples=n_samples,
            backend_name="torch",
            config=config,
        )
        
        if result:
            result["tile_size"] = tile_size
            results.append(result)
            print(f"{tile_size:<12} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {result['peak_vram_gb']:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\n✅ OPTIMAL: tile_size={best['tile_size']} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_backend_comparison() -> pd.DataFrame:
    """Compare all backends: cuda_states, torch, numpy."""
    
    print("\n" + "="*80)
    print("TEST 8: Backend Comparison")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    
    print(f"\n{'Backend':<15} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
    print("-"*60)
    
    for backend_name, config in BACKEND_CONFIGS.items():
        result = benchmark_single_config(
            n_qubits=N_QUBITS_DEFAULT,
            n_samples=n_samples,
            backend_name=backend_name,
            config=config,
        )
        
        if result:
            results.append(result)
            vram_str = f"{result['peak_vram_gb']:.2f}" if result['peak_vram_gb'] > 0 else "N/A"
            print(f"{backend_name:<15} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {vram_str:<12}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\n✅ FASTEST: {best['backend']} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════
# REPORTING AND VISUALIZATION
# ═══════════════════════════════════════════════════════════

def generate_plots(df_qubit: pd.DataFrame, df_sample: pd.DataFrame, df_tile: pd.DataFrame):
    """Generate comprehensive performance plots."""
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # --- PLOT 1: Qubit Scaling Comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    if not df_qubit.empty:
        for backend in df_qubit['backend'].unique():
            subset = df_qubit[df_qubit['backend'] == backend]
            ax1.plot(subset['n_qubits'], subset['throughput_mpairs_s'],
                    marker='o', linewidth=2.5, markersize=8, label=backend)
        
        ax1.set_title("Throughput vs Qubits", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Throughput (Mpairs/s)", fontsize=12)
        ax1.set_xlabel("Number of Qubits", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # --- PLOT 2: Qubit Scaling (Log Scale) ---
    ax2 = fig.add_subplot(gs[0, 1])
    if not df_qubit.empty:
        for backend in df_qubit['backend'].unique():
            subset = df_qubit[df_qubit['backend'] == backend]
            ax2.semilogy(subset['n_qubits'], subset['time_s'],
                        marker='s', linewidth=2.5, markersize=8, label=backend)
        
        ax2.set_title("Time vs Qubits (Log Scale)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Time (seconds) - Log Scale", fontsize=12)
        ax2.set_xlabel("Number of Qubits", fontsize=12)
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)
    
    # --- PLOT 3: Memory Usage ---
    ax3 = fig.add_subplot(gs[0, 2])
    if not df_qubit.empty and 'peak_vram_gb' in df_qubit.columns:
        gpu_data = df_qubit[df_qubit['backend'].isin(['cuda_states', 'torch'])]
        if not gpu_data.empty:
            for backend in gpu_data['backend'].unique():
                subset = gpu_data[gpu_data['backend'] == backend]
                ax3.plot(subset['n_qubits'], subset['peak_vram_gb'],
                        marker='d', linewidth=2.5, markersize=8, label=backend)
            
            ax3.set_title("GPU Memory Usage", fontsize=14, fontweight='bold')
            ax3.set_ylabel("Peak VRAM (GB)", fontsize=12)
            ax3.set_xlabel("Number of Qubits", fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # --- PLOT 4: Sample Scaling ---
    ax4 = fig.add_subplot(gs[1, 0])
    if not df_sample.empty:
        for backend in df_sample['backend'].unique():
            subset = df_sample[df_sample['backend'] == backend]
            ax4.loglog(subset['n_samples'], subset['time_s'],
                      marker='o', linewidth=2.5, markersize=8, label=backend)
        
        # Add O(N²) reference line
        if len(df_sample) > 0:
            # Use median time from first data point as reference
            first_backend_data = df_sample.groupby('backend').first()
            if not first_backend_data.empty:
                ref_n = first_backend_data['n_samples'].iloc[0]
                ref_t = first_backend_data['time_s'].iloc[0]
                x_ref = np.array([df_sample['n_samples'].min(), df_sample['n_samples'].max()])
                y_ref = (x_ref / ref_n)**2 * ref_t
                ax4.loglog(x_ref, y_ref, 'k--', linewidth=2, label='O(N²) reference', alpha=0.5)
        
        ax4.set_title("Sample Scaling (O(N²) verification)", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Time (seconds) - Log Scale", fontsize=12)
        ax4.set_xlabel("Number of Samples - Log Scale", fontsize=12)
        ax4.legend()
        ax4.grid(True, which="both", alpha=0.3)
    
    # --- PLOT 5: Optimization Ablation Study ---
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Use ablation results if available, otherwise use tile data as fallback
    if 'df_ablation' in dir() and not df_ablation.empty and 'configuration' in df_ablation.columns:
        # Sort by speedup for better visualization
        ablation_sorted = df_ablation.sort_values('speedup', ascending=True)
        
        # Create horizontal bar chart for ablation study
        colors = ['#2ca02c' if row['speedup'] >= 1.0 else '#d62728' 
                  for _, row in ablation_sorted.iterrows()]
        
        bars = ax5.barh(ablation_sorted['configuration'], ablation_sorted['speedup'],
                       color=colors, alpha=0.8, edgecolor='black')
        
        ax5.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        ax5.set_title("Optimization Ablation Study", fontsize=14, fontweight='bold')
        ax5.set_xlabel("Speedup vs Baseline", fontsize=12)
        ax5.set_ylabel("Configuration", fontsize=12)
        ax5.grid(axis='x', alpha=0.3)
        ax5.legend(loc='lower right')
        
        # Annotate bars with speedup values
        for bar, (_, row) in zip(bars, ablation_sorted.iterrows()):
            width = bar.get_width()
            ax5.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}×', ha='left', va='center', fontweight='bold', fontsize=9)
    
    elif not df_tile.empty and 'state_tile' in df_tile.columns:
        # Fallback to tile optimization if no ablation data
        ax5.bar(df_tile['state_tile'].astype(str), df_tile['throughput_mpairs_s'],
               color='steelblue', alpha=0.8, edgecolor='black')
        
        ax5.set_title("Tile Size Optimization", fontsize=14, fontweight='bold')
        ax5.set_ylabel("Throughput (Mpairs/s)", fontsize=12)
        ax5.set_xlabel("State Tile Size", fontsize=12)
        ax5.grid(axis='y', alpha=0.3)
        
        # Annotate bars
        for i, (idx, row) in enumerate(df_tile.iterrows()):
            height = row['throughput_mpairs_s']
            ax5.text(i, height, f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        ax5.text(0.5, 0.5, "No ablation/tile data available",
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    
    # --- PLOT 6: Speedup Comparison ---
    ax6 = fig.add_subplot(gs[1, 2])
    if not df_qubit.empty:
        # Calculate speedup for each qubit count
        speedup_data = []
        for n_qubits in df_qubit['n_qubits'].unique():
            subset = df_qubit[df_qubit['n_qubits'] == n_qubits]
            cpu_subset = subset[subset['backend'] == 'numpy']
            gpu_subset = subset[subset['backend'] == 'cuda_states']
            
            if not cpu_subset.empty and not gpu_subset.empty:
                cpu_time = cpu_subset['time_s'].values[0]
                gpu_time = gpu_subset['time_s'].values[0]
                # Validate both times are valid before calculating speedup
                if gpu_time > 0 and cpu_time > 0 and np.isfinite(cpu_time) and np.isfinite(gpu_time):
                    speedup = cpu_time / gpu_time
                    speedup_data.append({'n_qubits': n_qubits, 'speedup': speedup})
        
        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            bars = ax6.bar(speedup_df['n_qubits'].astype(str), speedup_df['speedup'],
                          color='green', alpha=0.8, edgecolor='black')
            
            ax6.set_title("GPU Speedup vs CPU", fontsize=14, fontweight='bold')
            ax6.set_ylabel("Speedup (× faster)", fontsize=12)
            ax6.set_xlabel("Number of Qubits", fontsize=12)
            ax6.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (1×)')
            ax6.legend()
            ax6.grid(axis='y', alpha=0.3)
            
            # Annotate bars
            for bar, (_, row) in zip(bars, speedup_df.iterrows()):
                height = bar.get_height()
                if np.isfinite(height):  # Only annotate valid values
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}×', ha='center', va='bottom',
                            fontweight='bold', fontsize=10)
    
    plt.savefig(OUTPUT_PLOTS, dpi=300, bbox_inches='tight')
    print(f"🖼️  Plots saved to: {OUTPUT_PLOTS}")

def generate_summary_report(df_all: pd.DataFrame):
    """Generate summary statistics and save to JSON."""
    
    import json
    
    if df_all.empty:
        print("⚠️  No data to generate summary report")
        return {}
    
    summary = {
        "total_tests": len(df_all),
        "backends_tested": df_all['backend'].unique().tolist(),
    }
    
    # Add ranges only if columns exist and have valid values
    if 'n_qubits' in df_all.columns and not df_all['n_qubits'].isna().all():
        summary["qubit_range"] = [int(df_all['n_qubits'].min()), int(df_all['n_qubits'].max())]
    
    if 'n_samples' in df_all.columns and not df_all['n_samples'].isna().all():
        summary["sample_range"] = [int(df_all['n_samples'].min()), int(df_all['n_samples'].max())]
    
    # Per-backend statistics
    for backend in df_all['backend'].unique():
        subset = df_all[df_all['backend'] == backend]
        summary[backend] = {
            "avg_throughput_mpairs_s": float(subset['throughput_mpairs_s'].mean()),
            "max_throughput_mpairs_s": float(subset['throughput_mpairs_s'].max()),
            "avg_time_s": float(subset['time_s'].mean()),
        }
        
        if 'peak_vram_gb' in subset.columns:
            summary[backend]["peak_vram_gb"] = float(subset['peak_vram_gb'].max())
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary report saved to: {OUTPUT_JSON}")
    
    return summary

# ═══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

def run_production_benchmark(tests: List[str] = None, backends: List[str] = None):
    """Run comprehensive production benchmark."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("PRODUCTION BENCHMARK - Comprehensive Performance Analysis")
    print("="*80)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🔧 Tests: {tests if tests else 'all'}")
    print(f"🔧 Backends: {backends if backends else 'all'}")
    print("="*80)
    
    all_results = []
    df_qubit = pd.DataFrame()
    df_sample = pd.DataFrame()
    df_tile = pd.DataFrame()
    df_torch_opt = pd.DataFrame()
    df_torch_tile = pd.DataFrame()
    df_comparison = pd.DataFrame()
    
    # Run tests based on selection
    if tests is None or 'qubit' in tests:
        df_qubit = test_qubit_impact(backends)
        all_results.append(df_qubit)
    
    if tests is None or 'sample' in tests:
        df_sample = test_sample_scaling(backends)
        all_results.append(df_sample)
    
    if tests is None or 'tile' in tests:
        df_tile = test_tile_optimization()
        all_results.append(df_tile)
    
    if tests is None or 'vram' in tests:
        df_vram = benchmark_vram_fraction_impact()
        all_results.append(df_vram)
    
    if tests is None or 'streams' in tests:
        df_streams = benchmark_stream_pool_impact()
        all_results.append(df_streams)
    
    if tests is None or 'ablation' in tests:
        df_ablation = benchmark_optimization_ablation()
        all_results.append(df_ablation)
    
    if tests is None or 'profile' in tests:
        df_profiling = benchmark_with_profiling()
        all_results.append(df_profiling)
    
    if tests is None or 'torch' in tests:
        df_torch_opt = benchmark_torch_optimizations()
        all_results.append(df_torch_opt)
    
    if tests is None or 'torch_tiles' in tests:
        df_torch_tile = benchmark_torch_tile_sizes()
        all_results.append(df_torch_tile)
    
    if tests is None or 'compare' in tests:
        df_comparison = benchmark_backend_comparison()
        all_results.append(df_comparison)
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    if not df_all.empty:
        # Save combined results
        df_all.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 All results saved to: {OUTPUT_CSV}")
        
        # Generate plots
        generate_plots(df_qubit, df_sample, df_tile)
        
        # Generate summary report
        summary = generate_summary_report(df_all)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        for backend, stats in summary.items():
            if isinstance(stats, dict):
                print(f"\n{backend.upper()}:")
                print(f"  - Avg Throughput: {stats['avg_throughput_mpairs_s']:.2f} Mpairs/s")
                print(f"  - Max Throughput: {stats['max_throughput_mpairs_s']:.2f} Mpairs/s")
                if 'peak_vram_gb' in stats:
                    print(f"  - Peak VRAM: {stats['peak_vram_gb']:.2f} GB")
    
    print("\n" + "="*80)
    print("🎉 Benchmark completed successfully!")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive GPU Backend Benchmark")
    
    # Test selection
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--backend-comparison", action="store_true", help="Run backend comparison benchmark")
    parser.add_argument("--cuda-states-ablation", action="store_true", help="Run cuda_states optimization ablation")
    parser.add_argument("--cuda-states-state-tile", action="store_true", help="Run cuda_states state tile optimization")
    parser.add_argument("--cuda-states-vram", action="store_true", help="Run cuda_states VRAM fraction test")
    parser.add_argument("--cuda-states-streams", action="store_true", help="Run cuda_states stream pool test")
    parser.add_argument("--torch-ablation", action="store_true", help="Run torch optimization ablation")
    parser.add_argument("--torch-tiles", action="store_true", help="Run torch tile size optimization")
    parser.add_argument("--memory-profiling", action="store_true", help="Run memory profiling benchmark")
    parser.add_argument("--qubit-scaling", action="store_true", help="Run qubit scaling analysis")
    parser.add_argument("--sample-scaling", action="store_true", help="Run sample scaling analysis")
    
    # Configuration
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT, help=f"Number of samples (default: {N_SAMPLES_DEFAULT})")
    parser.add_argument("--n-qubits", type=int, default=N_QUBITS_DEFAULT, help=f"Number of qubits (default: {N_QUBITS_DEFAULT})")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory for results")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs (default: 1)")
    parser.add_argument("--benchmark-runs", type=int, default=1, help="Number of benchmark runs (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    # Legacy support
    parser.add_argument('--tests', nargs='+', choices=['qubit', 'sample', 'tile', 'ablation', 'torch', 'compare', 'all'],
                       help="Legacy: Tests to run")
    parser.add_argument('--backends', nargs='+', choices=list(BACKEND_CONFIGS.keys()) + ['all'],
                       help="Legacy: Backends to test")
    
    args = parser.parse_args()
    
    # Map new args to tests list
    tests_to_run = []
    if args.all:
        tests_to_run = ['all']
    else:
        if args.backend_comparison:
            tests_to_run.append('compare')
        if args.cuda_states_ablation:
            tests_to_run.append('ablation')
        if args.cuda_states_state_tile:
            tests_to_run.append('tile')
        if args.cuda_states_vram:
            tests_to_run.append('vram')
        if args.cuda_states_streams:
            tests_to_run.append('streams')
        if args.torch_ablation:
            tests_to_run.append('torch')
        if args.torch_tiles:
            tests_to_run.append('torch_tiles')
        if args.memory_profiling:
            tests_to_run.append('profile')
        if args.qubit_scaling:
            tests_to_run.append('qubit')
        if args.sample_scaling:
            tests_to_run.append('sample')
    
    # Legacy support - merge with new args
    if args.tests:
        tests_to_run.extend(args.tests)
    
    # If no tests specified, run all
    if not tests_to_run:
        tests_to_run = ['all']
    
    tests = None if 'all' in tests_to_run else tests_to_run
    backends = None if not args.backends or 'all' in args.backends else args.backends
    
    try:
        run_production_benchmark(tests=tests, backends=backends)
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
