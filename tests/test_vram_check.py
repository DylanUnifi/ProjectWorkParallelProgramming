#!/usr/bin/env python3
"""
Simple test to validate VRAM-aware precomputation logic.
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TESTS_DIR = Path(__file__).resolve().parent
LOG_FILE = TESTS_DIR / "test_vram_check.log"
SUMMARY_FILE = TESTS_DIR / "test_vram_check_results.json"

LOG = logging.getLogger("test_vram_check")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOG.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)

RESULTS = []


def _record_result(name, status, details=None):
    entry = {"name": name, "status": status}
    if details is not None:
        entry["details"] = details
    RESULTS.append(entry)
    LOG.info("%s %s", name, json.dumps(entry, sort_keys=True, default=str))
    return entry


def _write_summary():
    summary = {"results": RESULTS}
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    LOG.info("Wrote VRAM test summary to %s", SUMMARY_FILE)

from scripts.pipeline_backends import _can_precompute_all, _compute_max_precompute_size, compute_kernel_matrix

def test_vram_check():
    """Test the VRAM checking function."""
    LOG.info("="*60)
    LOG.info("Testing VRAM-aware precomputation logic")
    LOG.info("="*60)
    
    # Test cases: (n_samples_a, n_samples_b, n_qubits, dtype, expected_feasible)
    # Focus on 16 qubits to better expose the VRAM threshold behavior.
    test_cases = [
        (5000, None, 14, np.float64, None),
        (10000, 2000, 14, np.float64, None),
        (15000, 5000, 14, np.float64, None),
        (5000, None, 16, np.float64, None),
        (10000, 2000, 16, np.float64, None),
        (15000, 5000, 16, np.float64, None),
        (20000, 10000, 16, np.float64, None),
        (25000, 15000, 16, np.float64, None),
        (30000, 20000, 16, np.float64, None),
    ]
    
    try:
        import cupy as cp
        device = cp.cuda.Device()
        total_vram_gb = device.mem_info[1] / 1e9
        available_vram_gb = device.mem_info[0] / 1e9
        LOG.info("GPU VRAM: %.1f GB total, %.1f GB free", total_vram_gb, available_vram_gb)
    except Exception as e:
        LOG.warning("Cannot detect GPU: %s", e)
        LOG.warning("This test requires a CUDA-enabled GPU.")
        _record_result("gpu_detection", "skipped", {"reason": str(e)})
        _write_summary()
        return

    LOG.info("Focused 16-qubit sweep: the goal is to show where bulk precomputation becomes infeasible.")
    vram_scale = max(0.5, min(2.0, available_vram_gb / 48.0))
    LOG.info("Using VRAM scale factor %.2f from %.1f GB free memory", vram_scale, available_vram_gb)

    def scaled_samples(base_samples: int) -> int:
        return max(1000, int(round(base_samples * vram_scale)))

    test_cases = [
        (scaled_samples(5000), None, 14, np.float64, None),
        (scaled_samples(10000), scaled_samples(2000), 14, np.float64, None),
        (scaled_samples(15000), scaled_samples(5000), 14, np.float64, None),
        (scaled_samples(5000), None, 16, np.float64, None),
        (scaled_samples(10000), scaled_samples(2000), 16, np.float64, None),
        (scaled_samples(15000), scaled_samples(5000), 16, np.float64, None),
        (scaled_samples(20000), scaled_samples(10000), 16, np.float64, None),
        (scaled_samples(25000), scaled_samples(15000), 16, np.float64, None),
        (scaled_samples(30000), scaled_samples(20000), 16, np.float64, None),
    ]

    LOG.info("%s", f"{'Samples A':<10} {'Samples B':<10} {'Qubits':<8} {'Dtype':<10} {'Can Precompute':<20} {'Max States':<12}")
    LOG.info("%s", "-"*60)
    
    sweep_results = []
    for n_samples_a, n_samples_b, n_qubits, dtype, expected in test_cases:
        can_precompute = _can_precompute_all(n_samples_a, n_samples_b, n_qubits, dtype)
        max_states = _compute_max_precompute_size(0.95, n_qubits, dtype)
        
        dtype_str = "float64" if dtype == np.float64 else "float32"
        result_icon = "Yes" if can_precompute else "No"
        samples_b_str = "self" if n_samples_b is None else str(n_samples_b)
        
        row = {
            "samples_a": n_samples_a,
            "samples_b": samples_b_str,
            "n_qubits": n_qubits,
            "dtype": dtype_str,
            "can_precompute": bool(can_precompute),
            "max_states": int(max_states),
        }
        LOG.info(
            "%-10s %-10s %-8s %-10s %s %-17s %-12s",
            n_samples_a,
            samples_b_str,
            n_qubits,
            dtype_str,
            result_icon,
            str(can_precompute),
            max_states,
        )
        _record_result("vram_sweep", "passed", row)
        sweep_results.append(row)
        
        # Validate expected results for known cases
        if expected is not None and can_precompute != expected:
            LOG.warning("Expected %s, got %s", expected, can_precompute)
            _record_result("vram_expectation", "warning", {**row, "expected": expected})

    qubits_14 = [row for row in sweep_results if row["n_qubits"] == 14]
    qubits_16 = [row for row in sweep_results if row["n_qubits"] == 16]

    assert len(qubits_14) == 3, "Expected exactly 3 sweep cases at 14 qubits"
    assert qubits_14[0]["can_precompute"], "The smallest 14-qubit case should precompute"
    assert qubits_14[-1]["can_precompute"] == (available_vram_gb > 0), "14-qubit sweep should remain feasible on a live GPU"
    assert len(qubits_16) == 6, "Expected exactly 6 sweep cases at 16 qubits"
    assert qubits_16[0]["can_precompute"], "The smallest 16-qubit case should precompute"
    assert any(not row["can_precompute"] for row in qubits_16), "At least one 16-qubit case should hit the VRAM limit"
    assert qubits_16[-1]["can_precompute"] is False, "The largest 16-qubit case should be VRAM-limited"

    LOG.info("%s", "\n" + "="*60)
    LOG.info("Backend smoke tests (cuda_states / torch / numpy)")
    LOG.info("%s", "="*60)

    smoke_X = np.random.uniform(-1, 1, (6, 4)).astype(np.float64)
    smoke_W = np.random.normal(0, 0.1, (2, 4)).astype(np.float64)

    smoke_backends = [
        (
            "cuda_states",
            {
                "device_name": "lightning.gpu",
                "dtype": "float64",
                "state_tile": -1,
                "precompute_all_states": True,
                "vram_fraction": 0.95,
            },
        ),
        (
            "torch",
            {
                "device_name": "lightning.gpu",
                "dtype": "float64",
                "use_pinned_memory": False,
                "use_cuda_streams": False,
                "use_amp": False,
                "use_compile": False,
            },
        ),
        (
            "numpy",
            {
                "device_name": "lightning.qubit",
                "dtype": "float64",
            },
        ),
    ]

    LOG.info("%s", f"{'Backend':<12} {'Status':<12} {'Shape':<12} {'Finite':<10}")
    LOG.info("%s", "-"*60)
    smoke_results = []
    for backend_name, backend_kwargs in smoke_backends:
        try:
            kernel = compute_kernel_matrix(
                smoke_X,
                weights=smoke_W,
                gram_backend=backend_name,
                symmetric=True,
                progress=False,
                **backend_kwargs,
            )
            finite = bool(np.all(np.isfinite(kernel)))
            LOG.info("%-12s %-12s %-12s %-10s", backend_name, "ok", str(kernel.shape), str(finite))
            smoke_results.append({"backend": backend_name, "finite": finite, "shape": list(kernel.shape)})
            _record_result(
                "backend_smoke",
                "passed",
                {"backend": backend_name, "shape": list(kernel.shape), "finite": finite},
            )
        except Exception as e:
            LOG.warning("%-12s %-12s %-12s %s", backend_name, "skipped", "-", type(e).__name__)
            smoke_results.append({"backend": backend_name, "finite": False, "skipped": True, "error": type(e).__name__})
            _record_result(
                "backend_smoke",
                "skipped",
                {"backend": backend_name, "error": type(e).__name__, "message": str(e)},
            )

    assert len(smoke_results) == 3, "Expected smoke coverage for exactly three backends"
    assert all(not result.get("skipped", False) for result in smoke_results), "All backend smoke tests should run successfully"
    assert all(result["finite"] for result in smoke_results), "All backend smoke outputs should be finite"
    
    LOG.info("%s", "\n" + "="*60)
    LOG.info("VRAM check test completed")
    LOG.info("%s", "="*60)
    _write_summary()

if __name__ == "__main__":
    test_vram_check()
