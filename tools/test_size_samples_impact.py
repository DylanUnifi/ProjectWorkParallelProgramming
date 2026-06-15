#!/usr/bin/env python3
"""Sample-scaling benchmark for quantum kernel backends.

This script measures how runtime scales with the number of samples while keeping
qubit count fixed. It mirrors the validated structure used by
``tools/test_num_qubit_impact.py``:
- normalized backend configs
- conservative warmup / single benchmark run defaults
- practical sample sizes
- CSV export for later analysis
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pipeline_backends import compute_kernel_matrix

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

DEFAULT_N_QUBITS = 16
DEFAULT_SAMPLE_SIZES = [256, 512, 1024, 2048, 4096]
DEFAULT_WARMUP = False
DEFAULT_REPEATS = 1

BACKEND_CONFIGS = {
    "cuda_states": {
        "device_name": "lightning.gpu",
        "gram_backend": "cuda_states",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 10000,
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
    "numpy": {
        "device_name": "default.qubit",
        "gram_backend": "numpy",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 128,
        "n_workers": 16,
    },
}

if HAS_TORCH:
    BACKEND_CONFIGS["torch"] = {
        "device_name": "lightning.gpu",
        "gram_backend": "torch",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 512,
        "use_pinned_memory": False,
        "use_cuda_streams": False,
        "use_amp": False,
        "use_compile": False,
    }

BACKEND_QUBIT_LIMITS = {
    "cuda_states": 20,
    "numpy": 16,
}

if HAS_TORCH:
    BACKEND_QUBIT_LIMITS["torch"] = 16

OUTPUT_DIR = ROOT / "benchmark_results"
OUTPUT_CSV = OUTPUT_DIR / "sample_impact_results.csv"

# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════

def reset_gpu_memory() -> None:
    """Reset GPU memory stats and clear cache when torch is available."""
    if HAS_TORCH and torch is not None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory_info() -> float:
    """Return peak GPU memory usage in GB."""
    if HAS_TORCH and torch is not None:
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def _backend_runtime_config(backend_name: str, config: Dict) -> Dict:
    """Prepare backend-specific config compatible with pipeline_backends."""
    cfg = config.copy()

    if backend_name == "cuda_states":
        cfg["gram_backend"] = "cuda_states"
        cfg["device_name"] = "lightning.gpu"
        return cfg

    if backend_name == "torch":
        cfg["gram_backend"] = "torch"
        cfg["device_name"] = "lightning.gpu"
        cfg.setdefault("use_pinned_memory", False)
        cfg.setdefault("use_cuda_streams", False)
        cfg.setdefault("use_amp", False)
        cfg.setdefault("use_compile", False)
        return cfg

    cfg["gram_backend"] = "numpy"
    cfg["device_name"] = "default.qubit"
    cfg.pop("state_tile", None)
    cfg.pop("vram_fraction", None)
    cfg.pop("autotune", None)
    cfg.pop("precompute_all_states", None)
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


def _adjust_sample_sizes_for_backend(backend_name: str, sample_sizes: List[int]) -> List[int]:
    """Keep sample sizes within practical limits for each backend."""
    qubit_limit = BACKEND_QUBIT_LIMITS.get(backend_name, DEFAULT_N_QUBITS)
    if DEFAULT_N_QUBITS > qubit_limit and backend_name in {"torch", "numpy"}:
        # If a user changes DEFAULT_N_QUBITS above a backend limit, keep the test sane.
        return [s for s in sample_sizes if s <= 2048] or [min(sample_sizes)]
    return sample_sizes


def benchmark_single_config(
    n_samples: int,
    n_qubits: int,
    backend_name: str,
    config: Dict,
    warmup: bool = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> Optional[Dict]:
    """Run a benchmark for a single backend/sample configuration."""
    rng = np.random.default_rng(42)
    np_dtype = np.float32 if config.get("dtype") == "float32" else np.float64
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np_dtype)
    weights = rng.normal(0, 0.1, (2, n_qubits)).astype(np_dtype)

    run_config = _backend_runtime_config(backend_name, config)
    reset_gpu_memory()

    try:
        if warmup:
            _ = compute_kernel_matrix(angles[:min(128, n_samples)], weights=weights, **run_config)
            if HAS_TORCH and torch is not None:
                torch.cuda.synchronize()

        reset_gpu_memory()

        times = []
        for _ in range(repeats):
            if HAS_TORCH and torch is not None:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            K = compute_kernel_matrix(angles, weights=weights, **run_config)

            if HAS_TORCH and torch is not None:
                torch.cuda.synchronize()

            times.append(time.perf_counter() - t0)
            del K

        peak_vram = get_gpu_memory_info()
        mean_time = float(np.mean(times))
        std_time = float(np.std(times))
        n_pairs = n_samples * (n_samples + 1) // 2
        throughput = n_pairs / mean_time / 1e6

        return {
            "n_samples": n_samples,
            "n_qubits": n_qubits,
            "backend": backend_name,
            "time_s": mean_time,
            "time_std_s": std_time,
            "throughput_mpairs_s": throughput,
            "peak_vram_gb": peak_vram,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_sample_impact_test(
    n_qubits: int = DEFAULT_N_QUBITS,
    sample_sizes: Optional[List[int]] = None,
    warmup: bool = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> pd.DataFrame:
    """Run the sample-scaling benchmark across all available backends."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    sample_sizes = sample_sizes or list(DEFAULT_SAMPLE_SIZES)

    print("=" * 80)
    print("TEST: Sample Scaling by Backend")
    print("=" * 80)
    print("Configuration:")
    print(f"   - Qubits: {n_qubits}")
    print(f"   - Sample sizes: {sample_sizes}")
    print(f"   - Backends: {list(BACKEND_CONFIGS.keys())}")
    print("=" * 80 + "\n")

    results: List[Dict] = []

    for backend_name, config in BACKEND_CONFIGS.items():
        applicable_sizes = _adjust_sample_sizes_for_backend(backend_name, sample_sizes)

        print(f"\nBackend: {backend_name.upper()}")
        print("-" * 60)
        print(f"{'Samples':<10} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
        print("-" * 60)

        for n_samples in applicable_sizes:
            result = benchmark_single_config(
                n_samples=n_samples,
                n_qubits=n_qubits,
                backend_name=backend_name,
                config=config,
                warmup=warmup,
                repeats=repeats,
            )

            if result:
                results.append(result)
                print(f"{n_samples:<10} {result['time_s']:<12.3f} {result['throughput_mpairs_s']:<12.2f} {result['peak_vram_gb']:<12.2f}")
            else:
                print(f"{n_samples:<10} FAILED")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")

    print("\n" + "=" * 80)
    print("SUMMARY: Sample Scaling Analysis")
    print("=" * 80)

    if not df.empty:
        for backend in df["backend"].unique():
            backend_df = df[df["backend"] == backend].sort_values("n_samples")
            if len(backend_df) >= 2:
                samples = backend_df["n_samples"].values
                times = backend_df["time_s"].values
                valid_mask = (times > 1e-6) & np.isfinite(times)
                if valid_mask.sum() > 1:
                    valid_times = times[valid_mask]
                    valid_samples = samples[valid_mask]
                    log_times = np.log2(valid_times)
                    coeffs = np.polyfit(np.log2(valid_samples), log_times, 1)
                    scaling_factor = 2 ** coeffs[0]
                    print(f"\n{backend}:")
                    print(f"  - Time scaling per sample doubling: {scaling_factor:.2f}x")
                    print(f"  - Fastest: {backend_df['throughput_mpairs_s'].max():.2f} Mpairs/s at {backend_df.loc[backend_df['throughput_mpairs_s'].idxmax(), 'n_samples']} samples")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample-scaling benchmark")
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS, help="Number of qubits to fix during the sample sweep")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES, help="List of sample sizes to benchmark")
    parser.add_argument("--warmup", action="store_true", default=DEFAULT_WARMUP, help="Enable a warmup run before timing")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Number of timed repeats per configuration")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sample_impact_test(
        n_qubits=args.n_qubits,
        sample_sizes=args.sample_sizes,
        warmup=args.warmup,
        repeats=args.repeats,
    )
