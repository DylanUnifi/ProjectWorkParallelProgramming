#!/usr/bin/env python3
"""
Benchmark PRODUCTION - Comprehensive Performance Analysis
=========================================================

This script provides comprehensive benchmarking of the benchmark
suite, with a focus on the currently exposed cuda_states analyses and backend
comparison paths.

Comprehensive benchmark sections:
1. Backend Comparison (Baseline) - Compare all backends with default optimizations
2. cuda_states Optimization Ablation - Test each optimization individually
3. cuda_states State Tile Optimization - Find optimal state_tile for different workloads
4. cuda_states VRAM Fraction Impact - Test impact of vram_fraction on performance  
5. cuda_states Stream Pool Impact - Test num_streams impact on throughput
6. Memory Profiling - Run benchmarks with detailed memory profiling
7. Qubit Scaling Analysis - Measure exponential scaling with qubit count
8. Sample Scaling Analysis - Verify O(N²) scaling with sample count

CLI Arguments:
- --all: Run all benchmarks
- --backend-comparison: Compare all backends
- --cuda-states-ablation: Test cuda_states optimizations
- --cuda-states-state-tile: Test state_tile values
- --cuda-states-vram: Test VRAM fraction impact
- --cuda-states-streams: Test stream pool sizes
- --memory-profiling: Run with memory profiling
- --qubit-scaling: Test scaling with qubit count
- --sample-scaling: Test scaling with sample count

Configuration:
- --n-samples: Number of samples (default: 1024)
- --n-qubits: Number of qubits (default: 16)
- --output-dir: Output directory (default: benchmark_results)
- --warmup-runs: Number of warmup runs (default: 1)
- --benchmark-runs: Number of benchmark runs (default: 1)
- --verbose: Print detailed information

Author: Dylan Fouepe
Date: 2026-01-08
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
import traceback
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    print("Error: Cannot import 'scripts.pipeline_backends'.")
    print("Run this script from project root: python3 benchmark.py")
    sys.exit(1)

try:
    from data_loader.utils import load_dataset_by_name
except ImportError:
    load_dataset_by_name = None

# Try to import torch for GPU memory monitoring
try:
    torch = None
    import torch
    HAS_TORCH = torch.cuda.is_available()
    TORCH_GPU_COUNT = torch.cuda.device_count() if HAS_TORCH else 0
except ImportError:
    torch = None
    HAS_TORCH = False
    TORCH_GPU_COUNT = 0

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Test configurations
QUBITS_RANGE = [8, 12, 16]
SAMPLE_SIZES = [256, 512, 1024]
N_SAMPLES_DEFAULT = 1024
N_QUBITS_DEFAULT = 16
DEFAULT_PARALLEL_GPUS = 3
COMPARISON_VALIDATION_SAMPLES = 1024
WARMUP_RUNS_DEFAULT = 1
BENCHMARK_RUNS_DEFAULT = 1
CURRENT_DATASET_PROFILE = "custom"

DATASET_PROFILES = {
    "fashion": {
        "qubits_range": [6, 8, 12],
        "sample_sizes": [256, 512, 1024],
        "n_samples_default": 1024,
        "n_qubits_default": 12,
        "comparison_dataset_name": "fashion_mnist",
        "comparison_binary_classes": [0, 1],
        "comparison_validation_samples": 1024,
    },
    "cifar10": {
        "qubits_range": [8, 12, 16],
        "sample_sizes": [256, 512, 1024],
        "n_samples_default": 1024,
        "n_qubits_default": 16,
        "comparison_dataset_name": "cifar10",
        "comparison_binary_classes": [1, 6],
        "comparison_validation_samples": 1024,
    },
    "svhn": {
        "qubits_range": [8, 12, 16],
        "sample_sizes": [256, 512, 1024],
        "n_samples_default": 1024,
        "n_qubits_default": 16,
        "comparison_dataset_name": "svhn",
        "comparison_binary_classes": [1, 0],
        "comparison_validation_samples": 1024,
    },
}

# Backend configurations with VALID parameters only
BACKEND_CONFIGS = {
    "cuda_states": {
        "device_name": "lightning.gpu",
        "gram_backend": "cuda_states",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 2048,
        # cuda_states optimization parameters
        "state_tile": 2048,
        "vram_fraction": 0.50,
        "autotune": True,
        "precompute_all_states": True,
        "dynamic_batch": False,
        "num_streams": 1,
        "learn_tiles": True,
        "use_cuda_graphs": False,
        "profile_memory": False,
        "verbose_profile": False,
    },
    "torch": {
        "device_name": "lightning.gpu",
        "gram_backend": "torch",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 512,
        # Torch-specific optimizations
        "use_pinned_memory": False,
        "use_cuda_streams": False,
        "use_amp": False,
        "use_compile": False,
    },
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
OUTPUT_CSV = OUTPUT_DIR / "benchmark.csv"
OUTPUT_PLOTS = OUTPUT_DIR / "benchmark.png"
OUTPUT_PLOTS_SVG = OUTPUT_DIR / "benchmark.svg"
OUTPUT_JSON = OUTPUT_DIR / "benchmark_summary.json"
OUTPUT_BEST_CSV = OUTPUT_DIR / "benchmark_best_configs.csv"

BACKEND_COLORS = {
    "cuda_states": "#2ca02c",
    "torch": "#ff7f0e",
    "numpy": "#1f77b4",
}

def resolve_real_comparison_split_samples(profile_name: str) -> int:
    """Resolve a practical binary split size for the active dataset profile."""
    profile = DATASET_PROFILES.get(profile_name)
    if not profile:
        return COMPARISON_VALIDATION_SAMPLES

    fallback = int(profile.get("comparison_validation_samples", COMPARISON_VALIDATION_SAMPLES))
    dataset_name = profile.get("comparison_dataset_name")
    binary_classes = profile.get("comparison_binary_classes")

    if load_dataset_by_name is None or not dataset_name or not binary_classes:
        return fallback

    try:
        train_dataset, test_dataset = load_dataset_by_name(
            name=dataset_name,
            binary_classes=binary_classes,
            grayscale=True,
            root="./data",
        )
        real_count = int(len(train_dataset) + len(test_dataset))
        return min(real_count, fallback)
    except Exception:
        return fallback

def apply_dataset_profile(profile_name: str):
    """Apply dataset profile to benchmark globals."""
    global QUBITS_RANGE, SAMPLE_SIZES, N_SAMPLES_DEFAULT, N_QUBITS_DEFAULT, COMPARISON_VALIDATION_SAMPLES
    profile = DATASET_PROFILES.get(profile_name)
    if not profile:
        return

    QUBITS_RANGE = list(profile["qubits_range"])
    SAMPLE_SIZES = list(profile["sample_sizes"])
    N_SAMPLES_DEFAULT = int(profile["n_samples_default"])
    N_QUBITS_DEFAULT = int(profile["n_qubits_default"])
    COMPARISON_VALIDATION_SAMPLES = resolve_real_comparison_split_samples(profile_name)


def configure_output_paths(output_dir_name: str):
    """Configure output artifact paths from CLI selection."""
    global OUTPUT_DIR, OUTPUT_CSV, OUTPUT_PLOTS, OUTPUT_PLOTS_SVG, OUTPUT_JSON, OUTPUT_BEST_CSV
    OUTPUT_DIR = ROOT / output_dir_name
    OUTPUT_CSV = OUTPUT_DIR / "benchmark.csv"
    OUTPUT_PLOTS = OUTPUT_DIR / "benchmark.png"
    OUTPUT_PLOTS_SVG = OUTPUT_DIR / "benchmark.svg"
    OUTPUT_JSON = OUTPUT_DIR / "benchmark_summary.json"
    OUTPUT_BEST_CSV = OUTPUT_DIR / "benchmark_best_configs.csv"

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
    """Get peak GPU memory usage in GB (alloc, reserved) subtracting baselines.

    Callers may pass baseline values (in bytes) to subtract pre-existing allocations
    recorded before the timed workload started. When PyTorch is not available this
    returns two zeros.
    """
    # Backwards-compatible signature: allow callers to pass baselines by positional args
    # but default to zero when not provided.
    # NOTE: We accept baseline values as the first two args if passed; to keep the
    # simple call-site usage we check for those below.
    return (0.0, 0.0)

def reset_gpu_memory():
    """Reset GPU memory stats and clear cache."""
    if HAS_TORCH and torch is not None:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def get_gpu_peak_memory_gb(baseline_alloc: int = 0, baseline_reserved: int = 0) -> Tuple[float, float]:
    """Return (alloc_peak_gb, reserved_peak_gb) after subtracting baselines (bytes).

    Baselines should be measured with `torch.cuda.memory_allocated()` and
    `torch.cuda.memory_reserved()` immediately after clearing/resetting peak stats.
    If torch is not available returns (0.0, 0.0).
    """
    if HAS_TORCH and torch is not None:
        try:
            max_alloc = int(torch.cuda.max_memory_allocated())
        except Exception:
            max_alloc = 0
        try:
            max_reserved = int(torch.cuda.max_memory_reserved())
        except Exception:
            max_reserved = 0

        alloc = max(0, max_alloc - int(baseline_alloc)) / (1024 ** 3)
        reserved = max(0, max_reserved - int(baseline_reserved)) / (1024 ** 3)
        return alloc, reserved
    return 0.0, 0.0

def calculate_state_vector_size(n_qubits: int, dtype: str = "float64") -> float:
    """Calculate state vector size in GB for given qubit count."""
    dim = 2 ** n_qubits
    bytes_per_element = 16 if dtype == "float64" else 8
    return dim * bytes_per_element / (1024**3)

def configure_gpu_device(gpu_id: Optional[int]):
    """Pin the current process to a GPU device when available."""
    if gpu_id is None or gpu_id < 0:
        return

    if HAS_TORCH and torch is not None:
        try:
            torch.cuda.set_device(gpu_id)
        except Exception:
            pass

    try:
        import cupy as cp
        cp.cuda.Device(gpu_id).use()
    except Exception:
        pass

def _is_gpu_backend(backend_name: str) -> bool:
    return backend_name in {"cuda_states", "torch"}

def _assign_gpu_id(job_index: int, backend_name: str, parallel_gpus: int) -> int:
    if parallel_gpus <= 0 or not _is_gpu_backend(backend_name):
        return -1
    return job_index % parallel_gpus

def _resolve_backends(backends: Optional[List[str]] = None) -> List[str]:
    """Resolve and validate backend list against available configurations."""
    if backends is None:
        return list(BACKEND_CONFIGS.keys())
    return [b for b in backends if b in BACKEND_CONFIGS]


def _sample_count_for_backend(backend_name: str, requested: int) -> int:
    """Return a practical capped sample count for the given backend.

    This prevents extremely long/oom runs by applying conservative defaults
    per backend while preserving the user's requested value when it is small.
    """
    caps = {
        "cuda_states": 1024,
        "torch": 1024,
        "numpy": 1024,
    }
    cap = caps.get(backend_name)
    if cap is None:
        return int(requested)
    return int(min(requested, cap))

def _backend_runtime_config(backend_name: str, config: Dict) -> Dict:
    """Prepare backend-specific config compatible with pipeline_backends behavior."""
    cfg = config.copy()

    if backend_name == "cuda_states":
        cfg["gram_backend"] = "cuda_states"
        cfg["device_name"] = "lightning.gpu"
        return cfg

    if backend_name == "torch":
        cfg["gram_backend"] = "torch"
        cfg["device_name"] = "lightning.gpu"
        # Keep torch path conservative by default; optimization tests override these explicitly.
        cfg.setdefault("use_pinned_memory", False)
        cfg.setdefault("use_cuda_streams", False)
        cfg.setdefault("use_amp", False)
        cfg.setdefault("use_compile", False)
        return cfg

    # numpy / CPU fallback path
    cfg["gram_backend"] = "numpy"
    cfg["device_name"] = "default.qubit"
    cfg.pop("state_tile", None)
    cfg.pop("vram_fraction", None)
    cfg.pop("dynamic_batch", None)
    cfg.pop("num_streams", None)
    cfg.pop("learn_tiles", None)
    cfg.pop("use_cuda_graphs", None)
    cfg.pop("profile_memory", None)
    cfg.pop("verbose_profile", None)
    cfg.pop("use_pinned_memory", None)
    cfg.pop("use_cuda_streams", None)
    cfg.pop("use_amp", None)
    cfg.pop("use_compile", None)
    return cfg

def _benchmark_job_worker(job: Dict) -> Optional[Dict]:
    """Worker entry point for parallel benchmark execution."""
    return benchmark_single_config(
        n_qubits=job["n_qubits"],
        n_samples=job["n_samples"],
        backend_name=job["backend_name"],
        config=job["config"],
        warmup=job.get("warmup", False),
        warmup_runs=job.get("warmup_runs", 1),
        repeats=job.get("repeats", 1),
        gpu_id=job.get("gpu_id", -1),
    )

def run_jobs_parallel(jobs: List[Dict], parallel_gpus: int, max_workers: Optional[int] = None) -> List[Optional[Dict]]:
    """Execute benchmark jobs in parallel across available GPUs."""
    if not jobs:
        return []

    gpu_jobs = [j for j in jobs if _is_gpu_backend(j["backend_name"]) and parallel_gpus > 0]
    cpu_jobs = [j for j in jobs if not _is_gpu_backend(j["backend_name"]) or parallel_gpus <= 0]

    results: List[Optional[Dict]] = []

    if gpu_jobs:
        workers = max_workers if max_workers is not None else min(len(gpu_jobs), parallel_gpus)
        workers = max(1, workers)
        gpu_results: List[Optional[Dict]] = [None] * len(gpu_jobs)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            future_to_idx = {ex.submit(_benchmark_job_worker, job): idx for idx, job in enumerate(gpu_jobs)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                gpu_results[idx] = fut.result()
        results.extend(gpu_results)

    # Keep CPU jobs sequential to avoid oversubscribing CPU worker pools used by compute_kernel_matrix.
    for job in cpu_jobs:
        results.append(_benchmark_job_worker(job))

    return results

# ═══════════════════════════════════════════════════════════
# BENCHMARK FUNCTIONS
# ═══════════════════════════════════════════════════════════

def benchmark_single_config(
    n_qubits: int,
    n_samples: int,
    backend_name: str,
    config: Dict,
    warmup: bool = False,
    warmup_runs: int = 1,
    repeats: int = 1,
    gpu_id: int = -1,
    return_kernel: bool = False,
) -> Optional[Dict]:
    """Run benchmark for a single configuration."""
    
    rng = np.random.default_rng(42)
    np_dtype = np.float32 if config.get("dtype") == "float32" else np.float64
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np_dtype)
    weights = rng.normal(0, 0.1, (2, n_qubits)).astype(np_dtype)
    
    run_config = config.copy()
    torch_fallback_tried = False

    configure_gpu_device(gpu_id)
    
    # Adjust state_tile for high qubit counts
    if "state_tile" in run_config and n_qubits >= 14:
        run_config["state_tile"] = min(run_config["state_tile"], 2048)
    
    reset_gpu_memory()
    kernel_snapshot = None
    
    try:
        # Warmup run
        if warmup:
            for _ in range(max(1, warmup_runs)):
                _ = compute_kernel_matrix(angles[:min(256, n_samples)], weights=weights, **run_config)
                if HAS_TORCH and torch is not None:
                    torch.cuda.synchronize()
        
        # Reset peak counters and capture baseline allocations (in bytes)
        reset_gpu_memory()
        baseline_alloc = torch.cuda.memory_allocated() if (HAS_TORCH and torch is not None) else 0
        baseline_reserved = torch.cuda.memory_reserved() if (HAS_TORCH and torch is not None) else 0

        # Timed runs
        times = []
        for _ in range(repeats):
            if HAS_TORCH and torch is not None:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

            t0 = time.perf_counter()
            K = compute_kernel_matrix(angles, weights=weights, **run_config)

            if return_kernel:
                kernel_snapshot = np.asarray(K, dtype=np.float64)
            
            if HAS_TORCH and torch is not None:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

            times.append(time.perf_counter() - t0)
            del K

        alloc_peak_gb, reserved_peak_gb = get_gpu_peak_memory_gb(baseline_alloc, baseline_reserved)

        # Calculate metrics
        mean_time = np.mean(times)
        std_time = np.std(times)
        n_pairs = n_samples * (n_samples + 1) // 2
        throughput = n_pairs / mean_time / 1e6
        
        # Prefer reserved memory peak for strict measurement but expose both values
        peak_vram = reserved_peak_gb

        return {
            "n_qubits": n_qubits,
            "backend": backend_name,
            "n_samples": n_samples,
            "gpu_id": gpu_id,
            "time_s": mean_time,
            "time_std_s": std_time,
            "throughput_mpairs_s": throughput,
            "peak_vram_gb": peak_vram,
            "peak_vram_alloc_gb": alloc_peak_gb,
            "peak_vram_reserved_gb": reserved_peak_gb,
            "state_vector_size_gb": calculate_state_vector_size(n_qubits, config.get("dtype", "float64")),
            "kernel_matrix": kernel_snapshot if return_kernel else None,
        }
        
    except Exception as e:
        if backend_name == "torch" and not torch_fallback_tried:
            safe_config = config.copy()
            safe_config.update({
                "use_pinned_memory": False,
                "use_cuda_streams": False,
                "use_amp": False,
                "use_compile": False,
            })
            torch_fallback_tried = True
            print("  Warning: Torch backend failed with the current config; retrying with conservative flags.")
            try:
                reset_gpu_memory()
                run_config = safe_config
                if warmup:
                    for _ in range(max(1, warmup_runs)):
                        _ = compute_kernel_matrix(angles[:min(256, n_samples)], weights=weights, **run_config)
                        if HAS_TORCH and torch is not None:
                            torch.cuda.synchronize()

                reset_gpu_memory()
                # Reset peak counters and capture baseline allocations (in bytes)
                reset_gpu_memory()
                baseline_alloc = torch.cuda.memory_allocated() if (HAS_TORCH and torch is not None) else 0
                baseline_reserved = torch.cuda.memory_reserved() if (HAS_TORCH and torch is not None) else 0

                times = []
                for _ in range(repeats):
                    if HAS_TORCH and torch is not None:
                        torch.cuda.synchronize()

                    t0 = time.perf_counter()
                    K = compute_kernel_matrix(angles, weights=weights, **run_config)

                    if return_kernel:
                        kernel_snapshot = np.asarray(K, dtype=np.float64)

                    if HAS_TORCH and torch is not None:
                        torch.cuda.synchronize()

                    times.append(time.perf_counter() - t0)
                    del K

                # Capture peaks relative to baseline
                alloc_peak_gb, reserved_peak_gb = get_gpu_peak_memory_gb(baseline_alloc, baseline_reserved)

                mean_time = np.mean(times)
                std_time = np.std(times)
                n_pairs = n_samples * (n_samples + 1) // 2
                throughput = n_pairs / mean_time / 1e6

                return {
                    "n_qubits": n_qubits,
                    "backend": backend_name,
                    "n_samples": n_samples,
                    "gpu_id": gpu_id,
                    "time_s": mean_time,
                    "time_std_s": std_time,
                    "throughput_mpairs_s": throughput,
                    "peak_vram_gb": reserved_peak_gb,
                    "peak_vram_alloc_gb": alloc_peak_gb,
                    "peak_vram_reserved_gb": reserved_peak_gb,
                    "state_vector_size_gb": calculate_state_vector_size(n_qubits, config.get("dtype", "float64")),
                    "torch_fallback": True,
                    "kernel_matrix": kernel_snapshot if return_kernel else None,
                }
            except Exception as fallback_error:
                print(f"  ERROR: {fallback_error}")
                traceback.print_exc()
                return None

        print(f"  ERROR: {e}")
        traceback.print_exc()
        return None
    finally:
        if HAS_TORCH and torch is not None:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                import gc
                gc.collect()
            except Exception:
                pass

def _kernel_fidelity_metrics(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    """Compute numerical fidelity metrics between two kernel matrices."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape or ref.size == 0:
        return {}

    diff = cand - ref
    denom = np.linalg.norm(ref, ord='fro')
    rel_fro = np.linalg.norm(diff, ord='fro') / (denom + 1e-12)

    diag_ref = np.diag(ref)
    diag_cand = np.diag(cand)

    return {
        "kernel_mae": float(np.mean(np.abs(diff))),
        "kernel_max_abs_err": float(np.max(np.abs(diff))),
        "kernel_rel_fro_err": float(rel_fro),
        "kernel_diag_mae": float(np.mean(np.abs(diag_cand - diag_ref))),
    }

def test_qubit_impact(backends: Optional[List[str]] = None, parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
    """Test impact of qubit count on performance."""
    
    backends = _resolve_backends(backends)
    
    print("\n" + "="*80)
    print("TEST 1: Impact of Number of Qubits on Performance")
    print("="*80)
    
    results = []
    
    jobs = []
    for backend_name in backends:
        if backend_name not in BACKEND_CONFIGS:
            continue
        config = _backend_runtime_config(backend_name, BACKEND_CONFIGS[backend_name])
        for n_qubits in [q for q in QUBITS_RANGE if q <= 16]:
            jobs.append({
                "n_qubits": n_qubits,
                "n_samples": _sample_count_for_backend(backend_name, N_SAMPLES_DEFAULT),
                "backend_name": backend_name,
                "config": config,
                "gpu_id": _assign_gpu_id(len(jobs), backend_name, parallel_gpus),
                "warmup": warmup_runs > 0,
                "warmup_runs": warmup_runs,
                "repeats": benchmark_runs,
            })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    results = [r for r in raw if r]

    for backend_name in backends:
        subset = sorted([r for r in results if r["backend"] == backend_name], key=lambda x: x["n_qubits"])
        if not subset:
            continue
        print(f"\n🔧 Backend: {backend_name.upper()}")
        print(f"{'Qubits':<8} {'GPU':<6} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM A (GB)':<12} {'VRAM R (GB)':<12}")
        print("-"*82)
        for result in subset:
            gpu_label = result['gpu_id'] if result['gpu_id'] >= 0 else "CPU"
            alloc_vram = result.get('peak_vram_alloc_gb', 0.0)
            reserved_vram = result.get('peak_vram_reserved_gb', 0.0)
            print(f"{result['n_qubits']:<8} {str(gpu_label):<6} {result['time_s']:<12.3f} {result['throughput_mpairs_s']:<12.3f} "
                  f"{alloc_vram:<12.2f} {reserved_vram:<12.2f}")
    
    return pd.DataFrame(results)

def test_sample_scaling(backends: Optional[List[str]] = None, parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
    """Test how performance scales with sample count."""
    
    backends = _resolve_backends(backends)
    
    print("\n" + "="*80)
    print("TEST 2: Sample Count Scaling (O(N²) verification)")
    print("="*80)
    
    results = []
    
    print(f"\n{'Backend':<15} {'N':<8} {'Time (s)':<12} {'Mpairs/s':<12}")
    print("-"*60)
    
    jobs = []
    for backend_name in backends:
        if backend_name not in BACKEND_CONFIGS:
            continue
        config = _backend_runtime_config(backend_name, BACKEND_CONFIGS[backend_name])
        for n_samples in SAMPLE_SIZES:
            jobs.append({
                "n_qubits": N_QUBITS_DEFAULT,
                "n_samples": _sample_count_for_backend(backend_name, n_samples),
                "backend_name": backend_name,
                "config": config,
                "gpu_id": _assign_gpu_id(len(jobs), backend_name, parallel_gpus),
                "warmup": warmup_runs > 0,
                "warmup_runs": warmup_runs,
                "repeats": benchmark_runs,
            })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    results = [r for r in raw if r]

    for result in sorted(results, key=lambda x: (x['backend'], x['n_samples'])):
        print(f"{result['backend']:<15} {result['n_samples']:<8} {result['time_s']:<12.3f} "
              f"{result['throughput_mpairs_s']:<12.3f}")
    
    return pd.DataFrame(results)

def test_tile_optimization(parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
    """Test tile size optimization for cuda_states."""
    
    print("\n" + "="*80)
    print("TEST 3: Tile Size Optimization (CUDA_STATES)")
    print("="*80)
    
    results = []
    n_samples = COMPARISON_VALIDATION_SAMPLES
    state_tiles = [2048, 4096, 8192, 16384, 32768, -1]
    
    print(f"\n{'state_tile':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM A (GB)':<12} {'VRAM R (GB)':<12}")
    print("-"*76)
    
    jobs = []
    for state_tile in state_tiles:
        config = BACKEND_CONFIGS["cuda_states"].copy()
        config["state_tile"] = state_tile
        jobs.append({
            "n_qubits": N_QUBITS_DEFAULT,
            "n_samples": _sample_count_for_backend("cuda_states", n_samples),
            "backend_name": "cuda_states",
            "config": config,
            "gpu_id": _assign_gpu_id(len(jobs), "cuda_states", parallel_gpus),
            "state_tile": state_tile,
            "warmup": warmup_runs > 0,
            "warmup_runs": warmup_runs,
            "repeats": benchmark_runs,
        })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    for job, result in zip(jobs, raw):
        if result:
            result["state_tile"] = job["state_tile"]
            results.append(result)
            alloc_vram = result.get('peak_vram_alloc_gb', 0.0)
            reserved_vram = result.get('peak_vram_reserved_gb', 0.0)
            print(f"{job['state_tile']:<12} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {alloc_vram:<12.2f} {reserved_vram:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\nSuccess: OPTIMAL: state_tile={best['state_tile']} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_vram_fraction_impact(parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
    """Test impact of VRAM fraction parameter for cuda_states."""
    
    print("\n" + "="*80)
    print("TEST 3B: VRAM Fraction Impact (CUDA_STATES)")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    vram_fractions = [0.5, 0.7, 0.85, 0.95]
    
    print(f"\n{'VRAM Frac':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM A (GB)':<12} {'VRAM R (GB)':<12}")
    print("-"*76)
    
    jobs = []
    for vram_frac in vram_fractions:
        config = BACKEND_CONFIGS["cuda_states"].copy()
        config["vram_fraction"] = vram_frac
        config["state_tile"] = -1  # Auto-size based on VRAM fraction
        config["dynamic_batch"] = False
        config["num_streams"] = 1
        config["use_cuda_graphs"] = False
        jobs.append({
            "n_qubits": N_QUBITS_DEFAULT,
            "n_samples": _sample_count_for_backend("cuda_states", n_samples),
            "backend_name": "cuda_states",
            "config": config,
            "gpu_id": _assign_gpu_id(len(jobs), "cuda_states", parallel_gpus),
            "vram_fraction": vram_frac,
            "warmup": warmup_runs > 0,
            "warmup_runs": warmup_runs,
            "repeats": benchmark_runs,
        })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    for job, result in zip(jobs, raw):
        if result:
            result["vram_fraction"] = job["vram_fraction"]
            results.append(result)
            alloc_vram = result.get('peak_vram_alloc_gb', 0.0)
            reserved_vram = result.get('peak_vram_reserved_gb', 0.0)
            print(f"{job['vram_fraction']:<12.2f} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {alloc_vram:<12.2f} {reserved_vram:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\nSuccess: OPTIMAL: vram_fraction={best.get('vram_fraction', 0.85):.2f} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_stream_pool_impact(parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
    """Test impact of num_streams parameter for cuda_states."""
    
    print("\n" + "="*80)
    print("TEST 3C: Stream Pool Impact (CUDA_STATES)")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    stream_counts = [1]
    
    print(f"\n{'Streams':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM A (GB)':<12} {'VRAM R (GB)':<12}")
    print("-"*76)
    
    jobs = []
    for num_streams in stream_counts:
        config = BACKEND_CONFIGS["cuda_states"].copy()
        config["num_streams"] = num_streams
        jobs.append({
            "n_qubits": N_QUBITS_DEFAULT,
            "n_samples": _sample_count_for_backend("cuda_states", n_samples),
            "backend_name": "cuda_states",
            "config": config,
            "gpu_id": _assign_gpu_id(len(jobs), "cuda_states", parallel_gpus),
            "num_streams": num_streams,
            "warmup": warmup_runs > 0,
            "warmup_runs": warmup_runs,
            "repeats": benchmark_runs,
        })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    for job, result in zip(jobs, raw):
        if result:
            result["num_streams"] = job["num_streams"]
            results.append(result)
            alloc_vram = result.get('peak_vram_alloc_gb', 0.0)
            reserved_vram = result.get('peak_vram_reserved_gb', 0.0)
            print(f"{job['num_streams']:<12} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {alloc_vram:<12.2f} {reserved_vram:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\nSuccess: OPTIMAL: num_streams={best.get('num_streams', 4)} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_optimization_ablation(parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
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
            "dynamic_batch": False,
            "use_cuda_graphs": False,
            "num_streams": 1,
        },
        "No Autotune": {
            "autotune": False,
            "precompute_all_states": True,
            "dynamic_batch": False,
            "use_cuda_graphs": False,
            "num_streams": 1,
        },
        "No Precompute": {
            "autotune": True,
            "precompute_all_states": False,
            "dynamic_batch": False,
            "use_cuda_graphs": False,
            "num_streams": 1,
        },
        "No Dynamic Batch": {
            "autotune": True,
            "precompute_all_states": True,
            "dynamic_batch": False,
            "use_cuda_graphs": False,
            "num_streams": 1,
        },
        "No CUDA Graphs": {
            "autotune": True,
            "precompute_all_states": True,
            "dynamic_batch": False,
            "use_cuda_graphs": False,
            "num_streams": 1,
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
    
    jobs = []
    ordered_names = list(configs.keys())
    for config_name in ordered_names:
        opts = configs[config_name]
        config = BACKEND_CONFIGS["cuda_states"].copy()
        config.update(opts)
        jobs.append({
            "n_qubits": N_QUBITS_DEFAULT,
            "n_samples": _sample_count_for_backend("cuda_states", n_samples),
            "backend_name": "cuda_states",
            "config": config,
            "gpu_id": _assign_gpu_id(len(jobs), "cuda_states", parallel_gpus),
            "configuration": config_name,
            "warmup": warmup_runs > 0,
            "warmup_runs": warmup_runs,
            "repeats": benchmark_runs,
        })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    rows = []
    for job, result in zip(jobs, raw):
        if result:
            result["configuration"] = job["configuration"]
            rows.append(result)

    baseline_row = next((r for r in rows if r.get("configuration") == "Baseline (No Opts)"), None)
    baseline_time = baseline_row['time_s'] if baseline_row else None

    for row in rows:
        speedup = baseline_time / row['time_s'] if baseline_time and row['time_s'] > 0 else 1.0
        row["speedup"] = speedup
        results.append(row)
        print(f"{row['configuration']:<22} {row['time_s']:<12.3f} "
              f"{row['throughput_mpairs_s']:<12.3f} {speedup:<10.2f}x")
    
    return pd.DataFrame(results)

def benchmark_torch_optimization_ablation(parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
    """Compare torch optimization flag contributions."""
    
    print("\n" + "="*80)
    print("TEST 4B: TORCH Backend Optimization Ablation Study")
    print("="*80)
    torch_df = benchmark_torch_optimizations(
        parallel_gpus=parallel_gpus,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )

    if torch_df.empty:
        return torch_df

    normalized_rows = []
    print(f"\n{'Configuration':<25} {'Time (s)':<12} {'Mpairs/s':<12} {'Speedup':<10}")
    print("-"*70)

    if "config_name" in torch_df.columns:
        torch_df = torch_df.copy()
        torch_df["configuration"] = torch_df["config_name"]
    elif "configuration" not in torch_df.columns:
        torch_df = torch_df.copy()
        torch_df["configuration"] = torch_df.index.astype(str)

    baseline_row = next((row for _, row in torch_df.iterrows() if row.get("configuration") in {"torch_baseline", "Baseline (No Opts)"}), None)
    baseline_time = float(baseline_row["time_s"]) if baseline_row is not None and baseline_row.get("time_s") else None

    for _, row in torch_df.iterrows():
        speedup = baseline_time / row["time_s"] if baseline_time and row["time_s"] > 0 else 1.0
        normalized_row = row.to_dict()
        normalized_row["speedup"] = speedup
        normalized_row["configuration"] = normalized_row.get("configuration", normalized_row.get("config_name", "torch_config"))
        normalized_rows.append(normalized_row)
        print(f"{normalized_row['configuration']:<25} {normalized_row['time_s']:<12.3f} "
              f"{normalized_row['throughput_mpairs_s']:<12.3f} {speedup:<10.2f}x")

    return pd.DataFrame(normalized_rows)

def benchmark_with_profiling(warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
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
        warmup=warmup_runs > 0,
        warmup_runs=warmup_runs,
        repeats=benchmark_runs,
    )
    
    if result:
        result["test"] = "memory_profiling"
        results.append(result)
        print(f"\nProfiling completed: {result['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_torch_optimizations(parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
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
    
    print(f"\n{'Config':<25} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM A (GB)':<12} {'VRAM R (GB)':<12}")
    print("-"*86)
    
    jobs = []
    for config_dict in configs:
        base_config = BACKEND_CONFIGS["torch"].copy()
        base_config["tile_size"] = tile_size
        # Update with optimization flags
        for key in ["use_pinned_memory", "use_cuda_streams", "use_amp", "use_compile"]:
            if key in config_dict:
                base_config[key] = config_dict[key]
        jobs.append({
            "n_qubits": N_QUBITS_DEFAULT,
            "n_samples": _sample_count_for_backend("torch", n_samples),
            "backend_name": "torch",
            "config": base_config,
            "gpu_id": _assign_gpu_id(len(jobs), "torch", parallel_gpus),
            "config_name": config_dict["name"],
            "tile_size": tile_size,
            "warmup": warmup_runs > 0,
            "warmup_runs": warmup_runs,
            "repeats": benchmark_runs,
        })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    for job, result in zip(jobs, raw):
        if result:
            result["config_name"] = job["config_name"]
            result["tile_size"] = job["tile_size"]
            results.append(result)
            alloc_vram = result.get('peak_vram_alloc_gb', 0.0)
            reserved_vram = result.get('peak_vram_reserved_gb', 0.0)
            print(f"{job['config_name']:<25} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {alloc_vram:<12.2f} {reserved_vram:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\nSuccess: BEST CONFIG: {best['config_name']} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_torch_tile_sizes(parallel_gpus: int = 1, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
    """Test optimal tile_size for torch backend."""
    
    print("\n" + "="*80)
    print("TEST 7: Torch Tile Size Optimization")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    tile_sizes = [64, 128, 256, 512, 1024, 2048]
    
    print(f"\n{'tile_size':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM A (GB)':<12} {'VRAM R (GB)':<12}")
    print("-"*76)
    
    jobs = []
    for tile_size in tile_sizes:
        config = BACKEND_CONFIGS["torch"].copy()
        config["tile_size"] = tile_size
        jobs.append({
            "n_qubits": N_QUBITS_DEFAULT,
            "n_samples": _sample_count_for_backend("torch", n_samples),
            "backend_name": "torch",
            "config": config,
            "gpu_id": _assign_gpu_id(len(jobs), "torch", parallel_gpus),
            "tile_size": tile_size,
            "warmup": warmup_runs > 0,
            "warmup_runs": warmup_runs,
            "repeats": benchmark_runs,
        })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    for job, result in zip(jobs, raw):
        if result:
            result["tile_size"] = job["tile_size"]
            results.append(result)
            alloc_vram = result.get('peak_vram_alloc_gb', 0.0)
            reserved_vram = result.get('peak_vram_reserved_gb', 0.0)
            print(f"{job['tile_size']:<12} {result['time_s']:<12.3f} "
                  f"{result['throughput_mpairs_s']:<12.3f} {alloc_vram:<12.2f} {reserved_vram:<12.2f}")
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\nSuccess: OPTIMAL: tile_size={best['tile_size']} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

def benchmark_backend_comparison(parallel_gpus: int = 1, backends: Optional[List[str]] = None, warmup_runs: int = 1, benchmark_runs: int = 1) -> pd.DataFrame:
    """Compare all backends: cuda_states, torch, numpy."""
    
    print("\n" + "="*80)
    print("TEST 8: Backend Comparison")
    print("="*80)
    
    results = []
    n_samples = N_SAMPLES_DEFAULT
    
    print(f"\n{'Backend':<15} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM A (GB)':<12} {'VRAM R (GB)':<12}")
    print("-"*76)
    
    jobs = []
    selected_backends = _resolve_backends(backends)
    for backend_name in selected_backends:
        config = _backend_runtime_config(backend_name, BACKEND_CONFIGS[backend_name])
        jobs.append({
            "n_qubits": N_QUBITS_DEFAULT,
            "n_samples": _sample_count_for_backend(backend_name, n_samples),
            "backend_name": backend_name,
            "config": config,
            "gpu_id": _assign_gpu_id(len(jobs), backend_name, parallel_gpus),
            "warmup": warmup_runs > 0,
            "warmup_runs": warmup_runs,
            "repeats": benchmark_runs,
        })

    raw = run_jobs_parallel(jobs, parallel_gpus=parallel_gpus)
    for result in [r for r in raw if r]:
        results.append(result)
        alloc_vram = result.get('peak_vram_alloc_gb', 0.0)
        reserved_vram = result.get('peak_vram_reserved_gb', 0.0)
        print(f"{result['backend']:<15} {result['time_s']:<12.3f} "
              f"{result['throughput_mpairs_s']:<12.3f} {alloc_vram:<12.2f} {reserved_vram:<12.2f}")

    # Evaluate kernel fidelity against numpy reference to match training objective.
    if results:
        result_by_backend = {r['backend']: r for r in results}
        if 'numpy' in result_by_backend:
            print("\nKernel fidelity vs numpy reference (smaller is better):")
            print(f"{'Backend':<15} {'MAE':<12} {'MaxAbs':<12} {'RelFro':<12} {'DiagMAE':<12}")
            print("-"*70)

            ref_backend = 'numpy'
            validation_samples = COMPARISON_VALIDATION_SAMPLES
            ref_config = _backend_runtime_config(ref_backend, BACKEND_CONFIGS[ref_backend])
            ref_eval = benchmark_single_config(
                n_qubits=N_QUBITS_DEFAULT,
                n_samples=validation_samples,
                backend_name=ref_backend,
                config=ref_config,
                warmup=False,
                repeats=1,
                gpu_id=-1,
                return_kernel=True,
            )
            ref_kernel = ref_eval.get('kernel_matrix') if ref_eval else None

            if ref_kernel is not None:
                for backend_name in sorted(result_by_backend.keys()):
                    if backend_name == ref_backend:
                        result_by_backend[backend_name].update({
                            'kernel_mae': 0.0,
                            'kernel_max_abs_err': 0.0,
                            'kernel_rel_fro_err': 0.0,
                            'kernel_diag_mae': 0.0,
                        })
                        print(f"{backend_name:<15} {0.0:<12.3e} {0.0:<12.3e} {0.0:<12.3e} {0.0:<12.3e}")
                        continue

                    gpu_id = _assign_gpu_id(0, backend_name, parallel_gpus)
                    eval_res = benchmark_single_config(
                        n_qubits=N_QUBITS_DEFAULT,
                        n_samples=validation_samples,
                        backend_name=backend_name,
                        config=_backend_runtime_config(backend_name, BACKEND_CONFIGS[backend_name]),
                        warmup=False,
                        repeats=1,
                        gpu_id=gpu_id,
                        return_kernel=True,
                    )
                    cand_kernel = eval_res.get('kernel_matrix') if eval_res else None
                    if cand_kernel is None:
                        continue

                    fidelity = _kernel_fidelity_metrics(ref_kernel, cand_kernel)
                    result_by_backend[backend_name].update(fidelity)
                    print(
                        f"{backend_name:<15} "
                        f"{fidelity.get('kernel_mae', np.nan):<12.3e} "
                        f"{fidelity.get('kernel_max_abs_err', np.nan):<12.3e} "
                        f"{fidelity.get('kernel_rel_fro_err', np.nan):<12.3e} "
                        f"{fidelity.get('kernel_diag_mae', np.nan):<12.3e}"
                    )
    
    if results:
        best = max(results, key=lambda x: x['throughput_mpairs_s'])
        print(f"\nSuccess: FASTEST: {best['backend']} → {best['throughput_mpairs_s']:.3f} Mpairs/s")
    
    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════
# REPORTING AND VISUALIZATION
# ═══════════════════════════════════════════════════════════

def generate_plots(df_qubit: pd.DataFrame, df_sample: pd.DataFrame, df_tile: pd.DataFrame, df_ablation: pd.DataFrame, df_comparison: pd.DataFrame, df_torch_ablation: pd.DataFrame):
    """Generate comprehensive performance plots."""
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(22, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    def _wrap_label(label: str, width: int = 24) -> str:
        return textwrap.fill(str(label), width=width)
    
    # --- PLOT 1: Qubit Scaling Comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    if not df_qubit.empty:
        for backend in df_qubit['backend'].unique():
            subset = df_qubit[df_qubit['backend'] == backend].sort_values('n_qubits')
            color = BACKEND_COLORS.get(backend, None)
            ax1.plot(subset['n_qubits'], subset['throughput_mpairs_s'],
                    marker='o', linewidth=2.5, markersize=8, label=backend, color=color)
        
        ax1.set_title("Throughput vs Qubits", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Throughput (Mpairs/s)", fontsize=12)
        ax1.set_xlabel("Number of Qubits", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    elif not df_comparison.empty:
        comparison_sorted = df_comparison.sort_values('throughput_mpairs_s', ascending=False)
        colors = [BACKEND_COLORS.get(backend, '#666666') for backend in comparison_sorted['backend']]
        bars = ax1.bar(comparison_sorted['backend'], comparison_sorted['throughput_mpairs_s'], color=colors, alpha=0.85, edgecolor='black')
        ax1.set_title("Backend Throughput", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Throughput (Mpairs/s)", fontsize=12)
        ax1.set_xlabel("Backend", fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height,
                     f"{height:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # --- PLOT 2: Qubit Scaling (Log Scale) ---
    ax2 = fig.add_subplot(gs[0, 1])
    if not df_qubit.empty:
        for backend in df_qubit['backend'].unique():
            subset = df_qubit[df_qubit['backend'] == backend].sort_values('n_qubits')
            color = BACKEND_COLORS.get(backend, None)
            ax2.semilogy(subset['n_qubits'], subset['time_s'],
                        marker='s', linewidth=2.5, markersize=8, label=backend, color=color)
        
        ax2.set_title("Time vs Qubits (Log Scale)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Time (seconds) - Log Scale", fontsize=12)
        ax2.set_xlabel("Number of Qubits", fontsize=12)
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)
    elif not df_comparison.empty:
        comparison_sorted = df_comparison.sort_values('time_s', ascending=True)
        colors = [BACKEND_COLORS.get(backend, '#666666') for backend in comparison_sorted['backend']]
        bars = ax2.bar(comparison_sorted['backend'], comparison_sorted['time_s'], color=colors, alpha=0.85, edgecolor='black')
        ax2.set_title("Backend Runtime", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Time (seconds)", fontsize=12)
        ax2.set_xlabel("Backend", fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                     f"{height:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # --- PLOT 3: Memory Usage ---
    ax3 = fig.add_subplot(gs[0, 2])
    if not df_qubit.empty and 'peak_vram_gb' in df_qubit.columns:
        gpu_data = df_qubit[df_qubit['backend'].isin(['cuda_states', 'torch'])]
        if not gpu_data.empty:
            for backend in gpu_data['backend'].unique():
                subset = gpu_data[gpu_data['backend'] == backend].sort_values('n_qubits')
                color = BACKEND_COLORS.get(backend, None)
                ax3.plot(subset['n_qubits'], subset['peak_vram_gb'],
                        marker='d', linewidth=2.5, markersize=8, label=backend, color=color)
            
            ax3.set_title("GPU Memory Usage", fontsize=14, fontweight='bold')
            ax3.set_ylabel("Peak VRAM (GB)", fontsize=12)
            ax3.set_xlabel("Number of Qubits", fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    elif not df_comparison.empty:
        comparison_sorted = df_comparison.sort_values('backend')
        x = np.arange(len(comparison_sorted))
        width = 0.35
        ax3.bar(x - width / 2, comparison_sorted['peak_vram_alloc_gb'], width=width, label='Allocated', color='#9467bd', alpha=0.85, edgecolor='black')
        ax3.bar(x + width / 2, comparison_sorted['peak_vram_reserved_gb'], width=width, label='Reserved', color='#8c564b', alpha=0.85, edgecolor='black')
        ax3.set_xticks(x)
        ax3.set_xticklabels(comparison_sorted['backend'])
        ax3.set_title("Backend VRAM Footprint", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Peak VRAM (GB)", fontsize=12)
        ax3.set_xlabel("Backend", fontsize=12)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # --- PLOT 4: Sample Scaling ---
    ax4 = fig.add_subplot(gs[1, 0])
    scaling_notes = []
    if not df_sample.empty:
        for backend in df_sample['backend'].unique():
            subset = df_sample[df_sample['backend'] == backend].sort_values('n_samples')
            color = BACKEND_COLORS.get(backend, None)
            ax4.loglog(subset['n_samples'], subset['time_s'],
                      marker='o', linewidth=2.5, markersize=8, label=backend, color=color)

            # Fit slope in log-log space to quantify observed scaling.
            if len(subset) >= 3 and (subset['n_samples'] > 0).all() and (subset['time_s'] > 0).all():
                x = np.log10(subset['n_samples'].values)
                y = np.log10(subset['time_s'].values)
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
                scaling_notes.append(f"{backend}: slope={slope:.2f}, R²={r2:.2f}")

        # Add O(N²) reference line
        first_backend_data = df_sample.groupby('backend').first()
        if not first_backend_data.empty:
            ref_n = first_backend_data['n_samples'].iloc[0]
            ref_t = first_backend_data['time_s'].iloc[0]
            x_ref = np.array([df_sample['n_samples'].min(), df_sample['n_samples'].max()])
            y_ref = (x_ref / ref_n) ** 2 * ref_t
            ax4.loglog(x_ref, y_ref, 'k--', linewidth=2, label='O(N²) reference', alpha=0.5)

        ax4.set_title("Sample Scaling (O(N²) verification)", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Time (seconds) - Log Scale", fontsize=12)
        ax4.set_xlabel("Number of Samples - Log Scale", fontsize=12)
        ax4.legend()
        ax4.grid(True, which="both", alpha=0.3)

        if scaling_notes:
            wrapped_notes = [textwrap.fill(note, width=28, break_long_words=False) for note in scaling_notes]
            ax4.text(0.03, 0.97, "\n".join(wrapped_notes), transform=ax4.transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    elif not df_comparison.empty:
        comparison_sorted = df_comparison.sort_values('throughput_mpairs_s', ascending=False)
        bars = ax4.bar(comparison_sorted['backend'], comparison_sorted['throughput_mpairs_s'],
                       color=[BACKEND_COLORS.get(backend, '#666666') for backend in comparison_sorted['backend']],
                       alpha=0.85, edgecolor='black')
        ax4.set_title("Backend Comparison Throughput", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Throughput (Mpairs/s)", fontsize=12)
        ax4.set_xlabel("Backend", fontsize=12)
        ax4.grid(axis='y', alpha=0.3)
        for bar, (_, row) in zip(bars, comparison_sorted.iterrows()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height,
                     f"{height:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No sample or comparison data available",
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # --- PLOT 5: Optimization Ablation Study (CUDA_STATES + TORCH) ---
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Combine both ablations if available
    combined_ablations = []
    if not df_ablation.empty and 'configuration' in df_ablation.columns:
        cuda_ablations = df_ablation.copy()
        cuda_ablations['backend_name'] = 'cuda_states'
        combined_ablations.append(cuda_ablations)
    
    if not df_torch_ablation.empty and 'configuration' in df_torch_ablation.columns:
        torch_ablations = df_torch_ablation.copy()
        torch_ablations['backend_name'] = 'torch'
        combined_ablations.append(torch_ablations)
    
    if combined_ablations:
        ablation_data = pd.concat(combined_ablations, ignore_index=True)
        ablation_data['label'] = ablation_data['backend_name'] + " | " + ablation_data['configuration']
        ablation_data = ablation_data.sort_values(['backend_name', 'speedup'], ascending=[True, True])
        y_pos = np.arange(len(ablation_data))
        colors = [BACKEND_COLORS.get(backend, '#666666') for backend in ablation_data['backend_name']]

        ax5.barh(y_pos, ablation_data['speedup'].values, color=colors, alpha=0.85, edgecolor='black')
        
        ax5.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([_wrap_label(label, width=28) for label in ablation_data['label'].tolist()], fontsize=8)
        ax5.set_title("Optimization Ablation (CUDA_STATES & TORCH)", fontsize=14, fontweight='bold')
        ax5.set_xlabel("Speedup vs Baseline", fontsize=12)
        ax5.tick_params(axis='x', labelsize=9)
        ax5.grid(axis='x', alpha=0.3)
        ax5.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLORS.get('cuda_states', '#666666'), alpha=0.85, label='cuda_states'),
            plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLORS.get('torch', '#666666'), alpha=0.85, label='torch'),
        ], loc='lower right', fontsize=9)
    else:
        ax5.text(0.5, 0.5, "No ablation data available",
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    
    # --- PLOT 6: Backend Speedup vs Numpy ---
    ax6 = fig.add_subplot(gs[1, 2])
    speedup_data = []
    speed_x_col = None
    speed_x_label = ""

    # Prefer sample-scaling output so speedup reflects dataset-size variation.
    if not df_sample.empty and {'backend', 'n_samples', 'time_s'}.issubset(df_sample.columns):
        speed_x_col = 'n_samples'
        speed_x_label = "Number of Samples"
        for n_samples in sorted(df_sample['n_samples'].dropna().unique()):
            subset = df_sample[df_sample['n_samples'] == n_samples]
            numpy_times = subset[subset['backend'] == 'numpy']['time_s'].dropna()
            if numpy_times.empty:
                continue

            numpy_time = float(np.median(numpy_times.values))
            for gpu_backend in ['cuda_states', 'torch']:
                gpu_times = subset[subset['backend'] == gpu_backend]['time_s'].dropna()
                if gpu_times.empty:
                    continue

                gpu_time = float(np.median(gpu_times.values))
                if gpu_time > 0 and numpy_time > 0 and np.isfinite(numpy_time) and np.isfinite(gpu_time):
                    speedup_data.append({
                        'x_value': int(n_samples),
                        'backend': gpu_backend,
                        'speedup': numpy_time / gpu_time,
                    })

    # Fallback to qubit scaling when sample scaling is unavailable.
    elif not df_qubit.empty and {'backend', 'n_qubits', 'time_s'}.issubset(df_qubit.columns):
        speed_x_col = 'n_qubits'
        speed_x_label = "Number of Qubits"
        for n_qubits in sorted(df_qubit['n_qubits'].dropna().unique()):
            subset = df_qubit[df_qubit['n_qubits'] == n_qubits]
            numpy_times = subset[subset['backend'] == 'numpy']['time_s'].dropna()
            if numpy_times.empty:
                continue

            numpy_time = float(np.median(numpy_times.values))
            for gpu_backend in ['cuda_states', 'torch']:
                gpu_times = subset[subset['backend'] == gpu_backend]['time_s'].dropna()
                if gpu_times.empty:
                    continue

                gpu_time = float(np.median(gpu_times.values))
                if gpu_time > 0 and numpy_time > 0 and np.isfinite(numpy_time) and np.isfinite(gpu_time):
                    speedup_data.append({
                        'x_value': int(n_qubits),
                        'backend': gpu_backend,
                        'speedup': numpy_time / gpu_time,
                    })

    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        backends_list = sorted(speedup_df['backend'].unique())
        x_values = sorted(speedup_df['x_value'].unique())
        x = np.arange(len(x_values))
        width = 0.35

        for i, backend in enumerate(backends_list):
            backend_series = (
                speedup_df[speedup_df['backend'] == backend]
                .groupby('x_value', as_index=False)['speedup']
                .median()
                .set_index('x_value')
                .reindex(x_values)
            )
            color = BACKEND_COLORS.get(backend, '#666666')
            ax6.bar(
                x + i * width,
                backend_series['speedup'].values,
                width=width,
                label=backend,
                color=color,
                alpha=0.85,
                edgecolor='black',
            )

        speed_title = "Backend Speedup vs Numpy (Sample Scaling)" if speed_x_col == 'n_samples' else "Backend Speedup vs Numpy (Qubit Scaling)"
        ax6.set_title(speed_title, fontsize=14, fontweight='bold')
        ax6.set_ylabel("Speedup (× faster)", fontsize=12)
        ax6.set_xlabel(speed_x_label, fontsize=12)
        ax6.set_xticks(x + width * 0.5)
        ax6.set_xticklabels([str(v) for v in x_values])
        ax6.tick_params(axis='x', labelsize=9)
        ax6.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (1×)')
        ax6.legend(fontsize=9)
        ax6.grid(axis='y', alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "No speedup data available",
                 ha='center', va='center', transform=ax6.transAxes, fontsize=12)

    plt.savefig(OUTPUT_PLOTS, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_PLOTS_SVG, bbox_inches='tight')
    print(f"🖼️  Plots saved to: {OUTPUT_PLOTS}")

def generate_best_config_report(df_all: pd.DataFrame) -> pd.DataFrame:
    """Create a ranked best-configuration report by backend and save to CSV."""
    if df_all.empty:
        return pd.DataFrame()

    rows = []
    for backend in sorted(df_all['backend'].dropna().unique()):
        subset = df_all[df_all['backend'] == backend].copy()
        if subset.empty:
            continue

        subset['vram_safe'] = subset['peak_vram_gb'].replace(0, np.nan)
        subset['throughput_per_gb'] = subset['throughput_mpairs_s'] / subset['vram_safe']
        subset['throughput_per_gb'] = subset['throughput_per_gb'].replace([np.inf, -np.inf], np.nan)
        subset['cv_pct'] = (subset['time_std_s'] / subset['time_s']) * 100.0
        subset['cv_pct'] = subset['cv_pct'].replace([np.inf, -np.inf], np.nan)

        best_tp = subset.loc[subset['throughput_mpairs_s'].idxmax()]
        best_time = subset.loc[subset['time_s'].idxmin()]

        mem_eff_subset = subset.dropna(subset=['throughput_per_gb'])
        if not mem_eff_subset.empty:
            best_mem_eff = mem_eff_subset.loc[mem_eff_subset['throughput_per_gb'].idxmax()]
            best_mem_eff_value = float(best_mem_eff['throughput_per_gb'])
        else:
            best_mem_eff = best_tp
            best_mem_eff_value = float('nan')

        stable_subset = subset.dropna(subset=['cv_pct'])
        if not stable_subset.empty:
            best_stability = stable_subset.loc[stable_subset['cv_pct'].idxmin()]
            best_cv = float(best_stability['cv_pct'])
        else:
            best_stability = best_tp
            best_cv = float('nan')

        rows.append({
            'backend': backend,
            'best_throughput_mpairs_s': float(best_tp['throughput_mpairs_s']),
            'best_throughput_time_s': float(best_tp['time_s']),
            'best_time_s': float(best_time['time_s']),
            'best_time_throughput_mpairs_s': float(best_time['throughput_mpairs_s']),
            'best_mem_eff_mpairs_per_gb': best_mem_eff_value,
            'best_stability_cv_pct': best_cv,
            'stability_flag': 'unstable' if np.isfinite(best_cv) and best_cv > 10.0 else 'stable',
        })

    report_df = pd.DataFrame(rows).sort_values('best_throughput_mpairs_s', ascending=False)
    report_df.to_csv(OUTPUT_BEST_CSV, index=False)
    print(f"Best configuration report saved to: {OUTPUT_BEST_CSV}")
    return report_df

def generate_summary_report(df_all: pd.DataFrame):
    """Generate summary statistics and save to JSON."""
    
    import json
    
    if df_all.empty:
        print("Warning:  No data to generate summary report")
        return {}
    
    summary = {
        "total_tests": len(df_all),
        "backends_tested": df_all['backend'].unique().tolist(),
    }

    if 'dataset_profile' in df_all.columns and not df_all['dataset_profile'].dropna().empty:
        summary["dataset_profiles"] = sorted(df_all['dataset_profile'].dropna().unique().tolist())
    
    # Add ranges only if columns exist and have valid values
    if 'n_qubits' in df_all.columns and not df_all['n_qubits'].isna().all():
        summary["qubit_range"] = [int(df_all['n_qubits'].min()), int(df_all['n_qubits'].max())]
    
    if 'n_samples' in df_all.columns and not df_all['n_samples'].isna().all():
        summary["sample_range"] = [int(df_all['n_samples'].min()), int(df_all['n_samples'].max())]
    
    # Per-backend statistics (global)
    for backend in df_all['backend'].unique():
        subset = df_all[df_all['backend'] == backend]
        p95_time = float(np.percentile(subset['time_s'], 95)) if len(subset) > 0 else float('nan')
        median_time = float(subset['time_s'].median()) if len(subset) > 0 else float('nan')
        std_time = float(subset['time_s'].std(ddof=0)) if len(subset) > 1 else 0.0
        cv_pct = (std_time / median_time * 100.0) if median_time > 0 else float('nan')
        p95_tp = float(np.percentile(subset['throughput_mpairs_s'], 95)) if len(subset) > 0 else float('nan')
        median_tp = float(subset['throughput_mpairs_s'].median()) if len(subset) > 0 else float('nan')

        vram_series = subset['peak_vram_gb'].replace(0, np.nan) if 'peak_vram_gb' in subset.columns else pd.Series(dtype=float)
        mem_eff_series = subset['throughput_mpairs_s'] / vram_series if not vram_series.empty else pd.Series(dtype=float)
        mem_eff_series = mem_eff_series.replace([np.inf, -np.inf], np.nan)

        summary[backend] = {
            "avg_throughput_mpairs_s": float(subset['throughput_mpairs_s'].mean()),
            "max_throughput_mpairs_s": float(subset['throughput_mpairs_s'].max()),
            "median_throughput_mpairs_s": median_tp,
            "p95_throughput_mpairs_s": p95_tp,
            "avg_time_s": float(subset['time_s'].mean()),
            "median_time_s": median_time,
            "p95_time_s": p95_time,
            "time_std_s": std_time,
            "time_cv_pct": cv_pct,
            "stability_flag": "unstable" if np.isfinite(cv_pct) and cv_pct > 10.0 else "stable",
        }

        if 'kernel_mae' in subset.columns and not subset['kernel_mae'].dropna().empty:
            summary[backend]["kernel_mae"] = float(subset['kernel_mae'].median())
        if 'kernel_max_abs_err' in subset.columns and not subset['kernel_max_abs_err'].dropna().empty:
            summary[backend]["kernel_max_abs_err"] = float(subset['kernel_max_abs_err'].median())
        if 'kernel_rel_fro_err' in subset.columns and not subset['kernel_rel_fro_err'].dropna().empty:
            summary[backend]["kernel_rel_fro_err"] = float(subset['kernel_rel_fro_err'].median())
        if 'kernel_diag_mae' in subset.columns and not subset['kernel_diag_mae'].dropna().empty:
            summary[backend]["kernel_diag_mae"] = float(subset['kernel_diag_mae'].median())
        
        if 'peak_vram_gb' in subset.columns:
            summary[backend]["peak_vram_gb"] = float(subset['peak_vram_gb'].max())
            if not mem_eff_series.empty and not mem_eff_series.dropna().empty:
                summary[backend]["median_throughput_per_gb"] = float(mem_eff_series.median())

    family_summary = {}
    if 'test_family' in df_all.columns:
        for family_name in sorted(df_all['test_family'].dropna().unique()):
            fam_df = df_all[df_all['test_family'] == family_name]
            if fam_df.empty:
                continue

            per_backend = {}
            for backend in sorted(fam_df['backend'].dropna().unique()):
                sub = fam_df[fam_df['backend'] == backend]
                if sub.empty:
                    continue

                median_time = float(sub['time_s'].median())
                std_time = float(sub['time_s'].std(ddof=0)) if len(sub) > 1 else 0.0
                cv_pct = (std_time / median_time * 100.0) if median_time > 0 else float('nan')

                per_backend[backend] = {
                    "num_points": int(len(sub)),
                    "median_time_s": median_time,
                    "time_std_s": std_time,
                    "time_cv_pct": cv_pct,
                    "stability_flag": "unstable" if np.isfinite(cv_pct) and cv_pct > 10.0 else "stable",
                    "median_throughput_mpairs_s": float(sub['throughput_mpairs_s'].median()),
                    "max_throughput_mpairs_s": float(sub['throughput_mpairs_s'].max()),
                }

            if per_backend:
                family_summary[family_name] = per_backend

    if family_summary:
        summary["stability_by_test_family"] = family_summary
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary report saved to: {OUTPUT_JSON}")
    
    return summary

# ═══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

def run_benchmark(tests: Optional[List[str]] = None, backends: Optional[List[str]] = None, parallel_gpus: int = 1):
    """Run comprehensive benchmark."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("BENCHMARK - Comprehensive Performance Analysis")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"🔧 Tests: {tests if tests else 'all'}")
    print(f"🔧 Backends: {backends if backends else 'all'}")
    print(f"🧩 Parallel GPUs requested: {parallel_gpus}")
    print(f"Warmup runs: {WARMUP_RUNS_DEFAULT}")
    print(f"📏 Benchmark runs: {BENCHMARK_RUNS_DEFAULT}")
    print("="*80)
    
    all_results = []
    df_qubit = pd.DataFrame()
    df_sample = pd.DataFrame()
    df_tile = pd.DataFrame()
    df_ablation = pd.DataFrame()
    df_torch_ablation = pd.DataFrame()
    df_torch_opt = pd.DataFrame()
    df_torch_tile = pd.DataFrame()
    df_comparison = pd.DataFrame()
    
    # Run tests based on selection
    if tests is None or 'qubit' in tests:
        df_qubit = test_qubit_impact(backends, parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_qubit.empty:
            df_qubit['test_family'] = 'qubit_scaling'
            df_qubit['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_qubit)
    
    if tests is None or 'sample' in tests:
        df_sample = test_sample_scaling(backends, parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_sample.empty:
            df_sample['test_family'] = 'sample_scaling'
            df_sample['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_sample)
    
    if tests is None or 'tile' in tests:
        df_tile = test_tile_optimization(parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_tile.empty:
            df_tile['test_family'] = 'cuda_states_tile'
            df_tile['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_tile)
    
    if tests is None or 'vram' in tests:
        df_vram = benchmark_vram_fraction_impact(parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_vram.empty:
            df_vram['test_family'] = 'cuda_states_vram'
            df_vram['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_vram)
    
    if tests is None or 'streams' in tests:
        df_streams = benchmark_stream_pool_impact(parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_streams.empty:
            df_streams['test_family'] = 'cuda_states_streams'
            df_streams['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_streams)
    
    if tests is None or 'ablation' in tests:
        df_ablation = benchmark_optimization_ablation(parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_ablation.empty:
            df_ablation['test_family'] = 'cuda_states_ablation'
            df_ablation['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_ablation)
        
        df_torch_ablation = benchmark_torch_optimization_ablation(parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_torch_ablation.empty:
            df_torch_ablation['test_family'] = 'torch_ablation'
            df_torch_ablation['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_torch_ablation)
    
    if tests is None or 'profile' in tests:
        df_profiling = benchmark_with_profiling(warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_profiling.empty:
            df_profiling['test_family'] = 'memory_profiling'
            df_profiling['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_profiling)
    
    if tests is None or 'torch' in tests:
        if df_torch_ablation.empty:
            df_torch_opt = benchmark_torch_optimizations(parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        else:
            df_torch_opt = df_torch_ablation.copy()
        if not df_torch_opt.empty:
            df_torch_opt['test_family'] = 'torch_optimizations'
            df_torch_opt['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_torch_opt)
    
    if tests is None or 'torch_tiles' in tests:
        df_torch_tile = benchmark_torch_tile_sizes(parallel_gpus=parallel_gpus, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_torch_tile.empty:
            df_torch_tile['test_family'] = 'torch_tile_sizes'
            df_torch_tile['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_torch_tile)
    
    if tests is None or 'backend_comparison' in tests:
        df_comparison = benchmark_backend_comparison(parallel_gpus=parallel_gpus, backends=backends, warmup_runs=WARMUP_RUNS_DEFAULT, benchmark_runs=BENCHMARK_RUNS_DEFAULT)
        if not df_comparison.empty:
            df_comparison['test_family'] = 'backend_comparison'
            df_comparison['dataset_profile'] = CURRENT_DATASET_PROFILE
        all_results.append(df_comparison)
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    if not df_all.empty:
        # Save combined results
        df_all.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved: All results saved to: {OUTPUT_CSV}")
        
        # Generate plots
        generate_plots(df_qubit, df_sample, df_tile, df_ablation, df_comparison, df_torch_ablation)
        
        # Generate summary report
        summary = generate_summary_report(df_all)
        best_report = generate_best_config_report(df_all)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        for backend, stats in summary.items():
            if backend == "stability_by_test_family":
                print("\nSTABILITY BY TEST FAMILY:")
                for family_name, family_stats in stats.items():
                    print(f"  {family_name}:")
                    for fam_backend, fam_vals in family_stats.items():
                        cv = fam_vals.get('time_cv_pct', float('nan'))
                        status = fam_vals.get('stability_flag', 'n/a')
                        med_tp = fam_vals.get('median_throughput_mpairs_s', float('nan'))
                        n_pts = fam_vals.get('num_points', 0)
                        print(
                            f"    - {fam_backend}: CV={cv:.2f}% [{status}], "
                            f"median_tp={med_tp:.3f} Mpairs/s, points={n_pts}"
                        )
                continue

            if isinstance(stats, dict) and 'avg_throughput_mpairs_s' in stats:
                print(f"\n{backend.upper()}:")
                print(f"  - Avg Throughput: {stats['avg_throughput_mpairs_s']:.2f} Mpairs/s")
                print(f"  - Max Throughput: {stats['max_throughput_mpairs_s']:.2f} Mpairs/s")
                print(f"  - Median Throughput: {stats['median_throughput_mpairs_s']:.2f} Mpairs/s")
                print(f"  - P95 Time: {stats['p95_time_s']:.3f} s")
                print(f"  - Stability (CV): {stats['time_cv_pct']:.2f}% [{stats['stability_flag']}]")
                if 'peak_vram_gb' in stats:
                    print(f"  - Peak VRAM: {stats['peak_vram_gb']:.2f} GB")
                if 'median_throughput_per_gb' in stats:
                    print(f"  - Median Throughput/GB: {stats['median_throughput_per_gb']:.2f} Mpairs/s/GB")
                if 'kernel_rel_fro_err' in stats:
                    print(f"  - Kernel RelFro Error vs numpy: {stats['kernel_rel_fro_err']:.3e}")
                if 'kernel_max_abs_err' in stats:
                    print(f"  - Kernel MaxAbs Error vs numpy: {stats['kernel_max_abs_err']:.3e}")

        if not best_report.empty:
            print("\n" + "="*80)
            print("BEST CONFIGURATION RANKING")
            print("="*80)
            print(best_report.to_string(index=False, float_format=lambda x: f"{x:.3f}" if np.isfinite(x) else "nan"))
    
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
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Number of samples (overrides dataset-profile default when provided)")
    parser.add_argument("--n-qubits", type=int, default=None,
                       help="Number of qubits (overrides dataset-profile default when provided)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory for results")
    parser.add_argument("--qubits-range", type=int, nargs='+', default=None,
                       help="Explicit qubit sweep values for --qubit-scaling (e.g. --qubits-range 6 8 12 16)")
    parser.add_argument("--sample-sizes", type=int, nargs='+', default=None,
                       help="Explicit sample sweep values for --sample-scaling (e.g. --sample-sizes 1024 2048 4096)")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs (default: 1)")
    parser.add_argument("--benchmark-runs", type=int, default=1, help="Number of benchmark runs (default: 1)")
    parser.add_argument("--parallel-gpus", type=int, default=DEFAULT_PARALLEL_GPUS,
                       help=f"Number of GPUs for parallel execution (default: {DEFAULT_PARALLEL_GPUS})")
    parser.add_argument("--dataset-profile", type=str, default="custom", choices=["custom"] + sorted(DATASET_PROFILES.keys()),
                       help="Dataset profile to apply benchmark scales (fashion, cifar10, svhn)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    # Legacy support
    parser.add_argument('--tests', nargs='+', choices=['qubit', 'sample', 'tile', 'ablation', 'torch', 'backend_comparison', 'all'],
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
            tests_to_run.append('backend_comparison')
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
        CURRENT_DATASET_PROFILE = args.dataset_profile
        if args.dataset_profile != "custom":
            apply_dataset_profile(args.dataset_profile)

        # Allow explicit CLI overrides on top of profile defaults.
        if args.n_samples is not None:
            N_SAMPLES_DEFAULT = int(args.n_samples)
        if args.n_qubits is not None:
            N_QUBITS_DEFAULT = int(args.n_qubits)

        if args.qubits_range:
            QUBITS_RANGE = [int(q) for q in args.qubits_range if int(q) > 0]
            if not QUBITS_RANGE:
                raise ValueError("--qubits-range must contain at least one positive integer")

        if args.sample_sizes:
            SAMPLE_SIZES = [int(n) for n in args.sample_sizes if int(n) > 0]
            if not SAMPLE_SIZES:
                raise ValueError("--sample-sizes must contain at least one positive integer")

        WARMUP_RUNS_DEFAULT = max(0, int(args.warmup_runs))
        BENCHMARK_RUNS_DEFAULT = max(1, int(args.benchmark_runs))
        configure_output_paths(args.output_dir)

        available_gpus = TORCH_GPU_COUNT if HAS_TORCH else 0
        requested_gpus = max(0, int(args.parallel_gpus))
        parallel_gpus = min(requested_gpus, available_gpus) if available_gpus > 0 else 0

        if requested_gpus > 0 and parallel_gpus == 0:
            print("Warning: Parallel GPU execution requested but no CUDA GPUs are visible; running without GPU parallelism.")
        elif requested_gpus > parallel_gpus:
            print(f"Warning: Requested {requested_gpus} GPUs but only {parallel_gpus} available; using {parallel_gpus}.")

        run_benchmark(tests=tests, backends=backends, parallel_gpus=parallel_gpus)
    except KeyboardInterrupt:
        print("\nWarning: Benchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
