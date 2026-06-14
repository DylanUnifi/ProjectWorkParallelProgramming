"""Enhanced NaN/Inf checker for multiple backends.

This script runs `compute_kernel_matrix` for several backends
and logs detailed numeric stability statistics to console and
to `tests/check_nan_results.json`.
"""

import json
import time
import traceback
import logging
import os
import numpy as np
from scripts.pipeline_backends import compute_kernel_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger("check_nan")


def summarize_array_stats(arr: np.ndarray):
    stats = {}
    stats["shape"] = arr.shape
    stats["dtype"] = str(arr.dtype)
    stats["min"] = float(np.nanmin(arr)) if np.any(~np.isnan(arr)) else None
    stats["max"] = float(np.nanmax(arr)) if np.any(~np.isnan(arr)) else None
    stats["mean"] = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else None
    stats["std"] = float(np.nanstd(arr)) if np.any(~np.isnan(arr)) else None
    stats["median"] = float(np.nanmedian(arr)) if np.any(~np.isnan(arr)) else None
    stats["n_nans"] = int(np.isnan(arr).sum())
    stats["n_infs"] = int(np.isinf(arr).sum())
    return stats


def test_backend(X, W, backend_name, backend_kwargs):
    LOG.info(f"Testing backend: %s", backend_name)
    start = time.perf_counter()
    result = {
        "backend": backend_name,
        "backend_kwargs": backend_kwargs,
        "error": None,
        "elapsed_s": None,
        "stats": None,
    }
    try:
        K = compute_kernel_matrix(X, weights=W, gram_backend=backend_name, **backend_kwargs)
        elapsed = time.perf_counter() - start
        result["elapsed_s"] = elapsed
        stats = summarize_array_stats(K)
        result["stats"] = stats
        LOG.info("Backend %s finished in %.3fs — min %.6f max %.6f n_nans %d n_infs %d",
                 backend_name, elapsed, stats["min"] if stats["min"] is not None else float("nan"),
                 stats["max"] if stats["max"] is not None else float("nan"),
                 stats["n_nans"], stats["n_infs"])
    except Exception as e:
        elapsed = time.perf_counter() - start
        result["elapsed_s"] = elapsed
        result["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        LOG.exception("Backend %s raised an exception", backend_name)
    return result


def main():
    LOG.info("Generating test data...")
    X = np.random.uniform(-3, 3, (100, 10)).astype(np.float64)
    W = np.random.normal(0, 0.1, (2, 10)).astype(np.float64)

    # Prepare backend configurations. Some backends may be skipped if not available.
    backends = []

    # cuda_states: keep original parameters
    backends.append(("cuda_states", {"device_name": "lightning.gpu", "dtype": "float64"}))

    # torch backend: detect torch availability and map to a PennyLane device
    try:
        import torch
        if torch.cuda.is_available():
            pl_dev = "lightning.gpu"
        else:
            pl_dev = "lightning.qubit"
        backends.append(("torch", {"device_name": pl_dev, "dtype": "float64"}))
    except Exception:
        LOG.warning("torch not available — skipping torch backend")

    # numpy backend: CPU-only — use a PennyLane CPU device name
    backends.append(("numpy", {"device_name": "lightning.qubit", "dtype": "float64"}))

    results = []
    for name, kwargs in backends:
        res = test_backend(X, W, name, kwargs)
        results.append(res)

    out_path = os.path.join(os.path.dirname(__file__), "check_nan_results.json")
    try:
        with open(out_path, "w") as f:
            json.dump({"timestamp": time.time(), "results": results}, f, indent=2)
        LOG.info("Wrote results to %s", out_path)
    except Exception:
        LOG.exception("Failed to write results file %s", out_path)

    # Print concise summary
    print("\nSummary:")
    for r in results:
        b = r["backend"]
        if r["error"]:
            print(f"- {b}: ERROR {r['error']['type']}: {r['error']['message']}")
        else:
            s = r["stats"]
            print(f"- {b}: min={s['min']:.6f} max={s['max']:.6f} mean={s['mean']:.6f} n_nans={s['n_nans']} n_infs={s['n_infs']} time={r['elapsed_s']:.3f}s")


if __name__ == "__main__":
    main()
