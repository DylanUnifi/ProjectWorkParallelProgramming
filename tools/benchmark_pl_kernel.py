import cupy as cp
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
x = cp.ones((1,), dtype=cp.float32); del x
cp.cuda.runtime.deviceSynchronize()

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- ensure CUDA headers for NVRTC when launched from IDE (CLion/PyCharm) ---
import os
os.environ.setdefault("CUPY_NVRTC_OPTIONS", "-I/usr/local/cuda/include")
os.environ["CPATH"] = "/usr/local/cuda/include:" + os.environ.get("CPATH", "")
os.environ["CPLUS_INCLUDE_PATH"] = "/usr/local/cuda/include:" + os.environ.get("CPLUS_INCLUDE_PATH", "")
# -----------------------------------------------------------------------------

# tools/benchmark_pl_kernel.py
import argparse, itertools, os, time
import numpy as np
from scripts.pipeline_backends import compute_kernel_matrix

def pairs_count(n: int, symmetric: bool) -> int:
    return n * (n + 1) // 2 if symmetric else n * n

def run_once(
    n_samples, n_qubits, tile_size, workers, device_name,
    symmetric, repeats, seed, layers, dtype, return_dtype, gram_backend
):
    rng = np.random.default_rng(seed)
    np_dtype = np.float32 if dtype == "float32" else np.float64
    angles = rng.uniform(low=-np.pi, high=np.pi, size=(n_samples, n_qubits)).astype(np_dtype)

    # poids entangler (couches réglables)
    weights = rng.normal(loc=0.0, scale=0.1, size=(layers, n_qubits)).astype(np_dtype)

    # dtype de sortie effectif (si return_dtype est None → identique à dtype)
    return_dtype_eff = return_dtype if return_dtype in ("float32", "float64") else dtype

    # warmup
    _ = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name=device_name,
        tile_size=tile_size,
        symmetric=symmetric,
        n_workers=workers,
        dtype=dtype,
        return_dtype=return_dtype,
        gram_backend=gram_backend,
    )

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = compute_kernel_matrix(
            angles,
            weights=weights,
            device_name=device_name,
            tile_size=tile_size,
            symmetric=symmetric,
            n_workers=workers,
            dtype=dtype,
            return_dtype=return_dtype,
            gram_backend=gram_backend,  # <-- IMPORTANT: passer aussi ici
        )
        times.append(time.perf_counter() - t0)

    t_mean = float(np.mean(times))
    t_std  = float(np.std(times))
    n_pairs = pairs_count(n_samples, symmetric)
    throughput = n_pairs / t_mean  # pairs/s

    return dict(
        device=device_name,
        n_samples=n_samples,
        n_qubits=n_qubits,
        tile_size=tile_size,
        workers=workers,
        symmetric=symmetric,
        layers=layers,
        dtype=dtype,
        return_dtype=return_dtype_eff,   # valeur effective écrite/affichée
        gram_backend=gram_backend,       # NEW: traçabilité du GEMM
        time_s=round(t_mean, 4),
        std_s=round(t_std, 4),
        pairs=n_pairs,
        Mpairs_per_s=round(throughput / 1e6, 3),
    )

def main():
    parser = argparse.ArgumentParser(description="Benchmark PennyLane fidelity-kernel (parallel, tiled).")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--tile-size", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--device", type=str, nargs="+", default=["lightning.qubit", "lightning.gpu"])
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32","float64"], default="float64")
    parser.add_argument("--return-dtype", choices=["float32", "float64"], default=None)
    parser.add_argument("--gram-backend", choices=["auto", "numpy", "cupy", "torch"], default="auto")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    grid = list(itertools.product(args.device, args.qubits, args.tile_size, args.workers))
    results = []

    print(f"\nBenchmark kernel | N={args.samples} | symmetric={args.symmetric} | devices={','.join(args.device)}")
    print(f"Grid size: {len(grid)} combinations\n")
    print(f"{'device':<16} {'nq':>3} {'tile':>4} {'wrk':>3} {'L':>2} {'dtype':>7} {'ret':>7} {'gemm':>6}  {'time(s)':>8}  {'±sd':>6}  {'Mpairs/s':>9}")
    print("-"*112)

    for (dev, nq, tile, wrk) in grid:
        # GPU: force workers=1 (évite sur-parallélisme inutile/nocif)
        if "gpu" in dev.lower() and wrk != 1:
            continue
        try:
            r = run_once(
                n_samples=args.samples,
                n_qubits=nq,
                tile_size=tile,
                workers=wrk,
                device_name=dev,
                symmetric=args.symmetric,
                repeats=args.repeats,
                seed=args.seed,
                layers=args.layers,
                dtype=args.dtype,
                return_dtype=args.return_dtype,
                gram_backend=args.gram_backend,
            )
            results.append(r)
            print(f"{r['device']:<16} {r['n_qubits']:>3} {r['tile_size']:>4} {r['workers']:>3} {r['layers']:>2} {r['dtype']:>7} {r['return_dtype']:>7} {r['gram_backend']:>6}  {r['time_s']:>8.3f}  {r['std_s']:>6.3f}  {r['Mpairs_per_s']:>9.3f}")
        except Exception as e:
            print(f"{dev:<16} {nq:>3} {tile:>4} {wrk:>3}  ERROR: {e}")

    # === GPU vs CPU comparison ===
    if any(r["device"]=="lightning.qubit" for r in results) and \
    any(r["device"]=="lightning.gpu" for r in results):

        best_cpu = max(
            (r for r in results if r["device"]=="lightning.qubit"),
            key=lambda r: r["Mpairs_per_s"]
        )
        best_gpu = max(
            (r for r in results if r["device"]=="lightning.gpu"),
            key=lambda r: r["Mpairs_per_s"]
        )

        speedup = best_gpu["Mpairs_per_s"] / best_cpu["Mpairs_per_s"]

        print("\n=== CPU vs GPU comparison ===")
        print(f"Best CPU  : {best_cpu['Mpairs_per_s']} Mpairs/s "
            f"({best_cpu['n_qubits']}q tile={best_cpu['tile_size']})")
        print(f"Best GPU  : {best_gpu['Mpairs_per_s']} Mpairs/s "
            f"({best_gpu['n_qubits']}q tile={best_gpu['tile_size']})")

        print(f"\nGPU is {speedup:.2f}× faster than CPU\n")


    if results:
        import csv
        # best par device
        best_per_device = {}
        for r in results:
            dev = r["device"]
            if dev not in best_per_device or r["Mpairs_per_s"] > best_per_device[dev]["Mpairs_per_s"]:
                best_per_device[dev] = r

        print("\n=== Best per device ===")
        for dev, r in best_per_device.items():
            print(f"{dev:<16} -> {r['n_qubits']}q, tile {r['tile_size']}, wrk {r['workers']}, L {r['layers']}, dtype {r['dtype']}, ret {r['return_dtype']}, gemm {r['gram_backend']} : {r['Mpairs_per_s']} Mpairs/s")

        if "lightning.qubit" in best_per_device and "lightning.gpu" in best_per_device:
            cpu = best_per_device["lightning.qubit"]["Mpairs_per_s"]
            gpu = best_per_device["lightning.gpu"]["Mpairs_per_s"]
            if cpu > 0:
                print(f"\nGPU/CPU speedup = {round(gpu/cpu, 2)}x")

        # CSV avec ordre de colonnes stable (incluant layers/dtype/return_dtype/gram_backend)
        fieldnames = [
            "device", "n_samples", "n_qubits", "tile_size", "workers",
            "symmetric", "layers", "dtype", "return_dtype", "gram_backend",
            "time_s", "std_s", "pairs", "Mpairs_per_s",
        ]
        out = "pl_kernel_benchmark.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                row = {k: r.get(k, "") for k in fieldnames}
                w.writerow(row)
        print(f"\nSaved CSV -> {out}")

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
