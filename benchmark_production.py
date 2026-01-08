#!/usr/bin/env python3
"""
Benchmark PRODUCTION - Comprehensive Performance Analysis
=========================================================

This script provides a comprehensive production benchmark integrating:
1. Qubit impact testing across backends
2. Tile size optimization
3. Sample scaling analysis  
4. Memory profiling
5. Detailed reporting with plots
6. Optimization ablation studies (NEW)
7. CUDA graph and stream utilization analysis (NEW)

New features:
- benchmark_optimization_ablation(): Compare individual optimization contributions
- benchmark_with_profiling(): Full memory profiling with detailed breakdown
- Support for --run-ablation and --profile-all flags

Author: Dylan Fouepe
Date: 2025-01-08
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
QUBITS_RANGE = [4, 6, 8, 10, 12, 14, 16]
SAMPLE_SIZES = [1000, 2000, 4000, 8000]
N_SAMPLES_DEFAULT = 4000
N_QUBITS_DEFAULT = 10

# Backend configurations with VALID parameters only
BACKEND_CONFIGS = {
    "cuda_states": {
        "device_name": "lightning.gpu",
        "gram_backend": "cuda_states",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 10000,
        # cuda_states optimization parameters
        "state_tile": -1,
        "vram_fraction": 0.85,
        "autotune": True,
        "precompute_all_states": True,
        "dynamic_batch": True,
        "num_streams": 4,
        "learn_tiles": True,
        "use_cuda_graphs": True,
        "profile_memory": False,
        "verbose_profile": False,
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
        qubit_limit = 16 if backend_name == "torch" else (10 if backend_name == "numpy" else 16)
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
        backends = ["cuda_states", "numpy"]
    
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
        sample_limits = SAMPLE_SIZES if backend_name == "cuda_states" else [s for s in SAMPLE_SIZES if s <= 4000]
        
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
    n_samples = 4000
    state_tiles = [512, 1024, 2048, 4096]
    
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

def benchmark_optimization_ablation() -> pd.DataFrame:
    """Compare each optimization's individual contribution."""
    
    print("\n" + "="*80)
    print("TEST 4: Optimization Ablation Study")
    print("="*80)
    
    results = []
    n_samples = 4000
    
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
    """Run with full memory profiling and report."""
    
    print("\n" + "="*80)
    print("TEST 5: Memory Profiling Analysis")
    print("="*80)
    
    results = []
    n_samples = 4000
    
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
    
    # --- PLOT 5: Tile Size Optimization ---
    ax5 = fig.add_subplot(gs[1, 1])
    if not df_tile.empty and 'state_tile' in df_tile.columns:
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
    df_ablation = pd.DataFrame()
    df_profiling = pd.DataFrame()
    
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
    
    if tests is None or 'ablation' in tests:
        df_ablation = benchmark_optimization_ablation()
        all_results.append(df_ablation)
    
    if tests is None or 'profile' in tests:
        df_profiling = benchmark_with_profiling()
        all_results.append(df_profiling)
    
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
    parser = argparse.ArgumentParser(description="Run production performance benchmark")
    parser.add_argument('--tests', nargs='+', 
                       choices=['qubit', 'sample', 'tile', 'ablation', 'profile', 'all'],
                       default=['all'], help="Tests to run (default: all)")
    parser.add_argument('--backends', nargs='+', choices=list(BACKEND_CONFIGS.keys()) + ['all'],
                       default=['all'], help="Backends to test (default: all)")
    parser.add_argument('--run-ablation', action='store_true',
                       help="Run optimization ablation study")
    parser.add_argument('--profile-all', action='store_true',
                       help="Enable memory profiling for all tests")
    
    args = parser.parse_args()
    
    # Handle convenience flags
    if args.run_ablation:
        if 'all' not in args.tests:
            args.tests.append('ablation')
        else:
            args.tests = ['ablation']
    
    if args.profile_all:
        if 'all' not in args.tests:
            args.tests.append('profile')
        else:
            args.tests = ['profile']
    
    tests = None if 'all' in args.tests else args.tests
    backends = None if 'all' in args.backends else args.backends
    
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
