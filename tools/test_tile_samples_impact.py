#!/usr/bin/env python3
"""Tile-size impact benchmark for quantum kernel backends.

The script measures tile-related scaling across the available backends.
For ``cuda_states`` it sweeps ``state_tile``. For ``numpy`` and ``torch``
it sweeps ``tile_size``. Sample-scaling is covered by
``tools/test_size_samples_impact.py``.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
DEFAULT_SAMPLE_COUNT = 4096
DEFAULT_WARMUP = False
DEFAULT_REPEATS = 1

DEFAULT_CUDA_STATE_TILE_SIZES = [512, 1024, 2048, 4096, -1]
DEFAULT_GENERIC_TILE_SIZES = [64, 128, 256, 512, 1024, 2048]

BACKEND_CONFIGS = {
    "cuda_states": {
        "device_name": "lightning.gpu",
        "gram_backend": "cuda_states",
        "dtype": "float64",
        "symmetric": True,
        "tile_size": 5000,
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

OUTPUT_DIR = ROOT / "benchmark_results"
OUTPUT_CSV = OUTPUT_DIR / "tile_samples_impact_results.csv"


# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════


def reset_gpu_memory() -> None:
    """Clear CUDA cache and reset peak stats if torch is available."""
    if HAS_TORCH and torch is not None:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_vram_gb(baseline_allocated_gb: float = 0.0, baseline_reserved_gb: float = 0.0) -> float:
    """Return incremental peak GPU memory usage in GB.

    Prefer CUDA reserved memory for a stricter view of the torch footprint,
    while falling back to allocated memory if needed.
    """
    if HAS_TORCH and torch is not None:
        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        strict_peak_gb = max(peak_reserved_gb, peak_allocated_gb)
        baseline_gb = max(baseline_allocated_gb, baseline_reserved_gb)
        return max(0.0, strict_peak_gb - baseline_gb)
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


def _tile_sweep_for_backend(backend_name: str, tile_sizes: Optional[List[int]]) -> Tuple[str, List[int]]:
    """Return the parameter name and tile sweep appropriate for a backend."""
    if backend_name == "cuda_states":
        values = list(tile_sizes) if tile_sizes is not None else list(DEFAULT_CUDA_STATE_TILE_SIZES)
        return "state_tile", values

    if tile_sizes is not None:
        values = [value for value in tile_sizes if value > 0]
        return "tile_size", values or list(DEFAULT_GENERIC_TILE_SIZES)

    return "tile_size", list(DEFAULT_GENERIC_TILE_SIZES)


def _sample_count_for_backend(backend_name: str, requested_samples: int) -> int:
    """Keep the tile benchmark practical for slower backends."""
    if backend_name == "cuda_states":
        return min(requested_samples, 1024)
    if backend_name == "torch":
        return min(requested_samples, 1024)
    if backend_name == "numpy":
        return min(requested_samples, 512)
    return requested_samples


def _adjust_state_tile_for_qubits(n_qubits: int, state_tile: int) -> int:
    """Keep very large state tiles conservative for higher qubit counts."""
    if state_tile <= 0:
        return state_tile
    if n_qubits >= 18:
        return min(state_tile, 512)
    if n_qubits >= 16:
        return min(state_tile, 1024)
    if n_qubits >= 14:
        return min(state_tile, 2048)
    return state_tile


def benchmark_config(
    n_samples: int,
    n_qubits: int,
    backend_name: str,
    config: Dict,
    warmup: bool = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> Optional[Dict]:
    """Benchmark a single backend/workload configuration."""
    rng = np.random.default_rng(42)
    np_dtype = np.float32 if config.get("dtype") == "float32" else np.float64
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np_dtype)
    weights = rng.normal(0, 0.1, (2, n_qubits)).astype(np_dtype)

    run_config = _backend_runtime_config(backend_name, config)
    reset_gpu_memory()
    vram_baseline_allocated_gb = 0.0
    vram_baseline_reserved_gb = 0.0
    if HAS_TORCH and torch is not None:
        vram_baseline_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        vram_baseline_reserved_gb = torch.cuda.memory_reserved() / (1024**3)

    try:
        if warmup:
            _ = compute_kernel_matrix(angles[: min(128, n_samples)], weights=weights, **run_config)
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

        mean_time = float(np.mean(times))
        std_time = float(np.std(times))
        if HAS_TORCH and torch is not None:
            torch.cuda.synchronize()
        gc.collect()
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0
        peak_vram = 0.0
        if HAS_TORCH and torch is not None:
            peak_allocated_gb = max(
                0.0,
                (torch.cuda.max_memory_allocated() / (1024**3)) - vram_baseline_allocated_gb,
            )
            peak_reserved_gb = max(
                0.0,
                (torch.cuda.max_memory_reserved() / (1024**3)) - vram_baseline_reserved_gb,
            )
            peak_vram = max(peak_allocated_gb, peak_reserved_gb)
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
            "peak_vram_allocated_gb": peak_allocated_gb,
            "peak_vram_reserved_gb": peak_reserved_gb,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_tile_impact_test(
    n_samples: int = DEFAULT_SAMPLE_COUNT,
    n_qubits: int = DEFAULT_N_QUBITS,
    tile_sizes: Optional[List[int]] = None,
    warmup: bool = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> pd.DataFrame:
    """Measure tile impact across all available backends."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("TEST 1: Tile impact by backend")
    print("=" * 80)
    print("Configuration:")
    print(f"   - Qubits: {n_qubits}")
    print(f"   - Samples: {n_samples}")
    print(f"   - Backends: {list(BACKEND_CONFIGS.keys())}")
    print(f"   - Tile sizes: {tile_sizes if tile_sizes is not None else 'backend defaults'}")
    print("=" * 80 + "\n")

    results: List[Dict] = []

    for backend_name, base_config in BACKEND_CONFIGS.items():
        param_name, sweep_values = _tile_sweep_for_backend(backend_name, tile_sizes)
        backend_samples = _sample_count_for_backend(backend_name, n_samples)

        print(f"\nBackend: {backend_name.upper()}")
        print("-" * 70)
        if backend_samples != n_samples:
            print(f"Using {backend_samples} samples for this backend to keep the sweep practical.")
        print(f"{param_name:<12} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM alloc (GB)':<16} {'VRAM resv (GB)':<14}")
        print("-" * 70)

        for tile_value in sweep_values:
            try:
                run_config = base_config.copy()

                if backend_name == "cuda_states":
                    run_config["state_tile"] = _adjust_state_tile_for_qubits(n_qubits, tile_value)
                    if tile_value == -1:
                        run_config["state_tile"] = -1
                else:
                    run_config["tile_size"] = tile_value

                result = benchmark_config(
                    n_samples=backend_samples,
                    n_qubits=n_qubits,
                    backend_name=backend_name,
                    config=run_config,
                    warmup=warmup,
                    repeats=repeats,
                )

                if result:
                    result["tile_param"] = param_name
                    result["tile_value"] = tile_value
                    result["tile_size"] = run_config.get("tile_size")
                    result["state_tile"] = run_config.get("state_tile")
                    results.append(result)
                    print(
                        f"{tile_value:<12} {result['time_s']:<12.3f} "
                        f"{result['throughput_mpairs_s']:<12.3f} "
                        f"{result['peak_vram_allocated_gb']:<16.2f} {result['peak_vram_reserved_gb']:<14.2f}"
                    )
                else:
                    print(f"{tile_value:<12} FAILED")
            except Exception as e:
                print(f"{tile_value:<12} ERROR: {str(e)[:60]}")

    df = pd.DataFrame(results)
    if not df.empty:
        df["test"] = "tile_impact"
    return df


def write_results(df: pd.DataFrame, output_csv: Path = OUTPUT_CSV) -> None:
    """Persist results to CSV."""
    output_csv.parent.mkdir(exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile impact benchmark")
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--warmup", action="store_true", default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile_df = run_tile_impact_test(
        n_samples=args.n_samples,
        n_qubits=args.n_qubits,
        tile_sizes=args.tile_sizes,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    if not tile_df.empty:
        write_results(tile_df)
    else:
        print("No successful benchmark results were produced.")


if __name__ == "__main__":
    main()