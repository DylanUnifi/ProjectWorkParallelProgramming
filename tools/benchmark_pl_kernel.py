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
import numpy as np#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, itertools, os, sys, time, csv
from pathlib import Path
import numpy as np

# --- project root in sys.path (if launched from IDE) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- limit BLAS threads (parent) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --- only import CuPy / set allocators if needed later ---
def _maybe_setup_cupy():
    try:
        import cupy as cp  # type: ignore
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        _ = cp.ones((1,), dtype=cp.float32); del _
        cp.cuda.runtime.deviceSynchronize()
        os.environ.setdefault("CUPY_NVRTC_OPTIONS", "-I/usr/local/cuda/include")
        for key in ("CPATH", "CPLUS_INCLUDE_PATH"):
            os.environ[key] = "/usr/local/cuda/include:" + os.environ.get(key, "")
    except Exception:
        pass

# --- unified API import ---
from scripts.pipeline_backends import compute_kernel_matrix


def pairs_count(n: int, symmetric: bool) -> int:
    return n * (n + 1) // 2 if symmetric else n * n


def _rel_err(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + eps
    return float(num / den)


def _check_symmetry(K: np.ndarray, tol: float = 1e-5) -> bool:
    if K.shape[0] != K.shape[1]:
        return False
    return float(np.max(np.abs(K - K.T))) <= tol


def _check_psd(K: np.ndarray, tol: float = -1e-6) -> bool:
    try:
        w = np.linalg.eigvalsh((K + K.T) * 0.5)
        return float(np.min(w)) >= tol
    except Exception:
        return True


def run_once(
    X, W, n_qubits, tile_size, workers, device_name,
    symmetric, repeats, dtype, return_dtype, backend,
    progress=False, desc="Gram", ref_K=None,
    state_tile=None, tile_m="auto", tile_n="auto", tile_k="auto"
):
    return_dtype_eff = return_dtype if return_dtype in ("float32", "float64") else dtype
    times, last_K = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_K = compute_kernel_matrix(
            X,
            weights=W,
            device_name=device_name,
            tile_size=tile_size,
            symmetric=symmetric,
            n_workers=workers,
            dtype=dtype,
            return_dtype=return_dtype,
            gram_backend=backend,
            progress=progress,
            desc=desc,
            # new backend params (ignored by legacy backends)
            state_tile=(state_tile if state_tile is not None else 128),
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    t_mean = float(np.mean(times)) if times else float("nan")
    t_std  = float(np.std(times)) if times else float("nan")
    n_pairs = pairs_count(X.shape[0], symmetric)
    throughput = (n_pairs / t_mean) if (t_mean and t_mean > 0 and not np.isnan(t_mean)) else 0.0

    sym_ok = _check_symmetry(last_K) if (symmetric and last_K is not None) else True
    psd_ok = _check_psd(last_K) if (symmetric and last_K is not None) else True
    rel_err = None
    if ref_K is not None and last_K is not None and last_K.shape == ref_K.shape:
        rel_err = _rel_err(last_K.astype(np.float64, copy=False), ref_K)

    return dict(
        device=device_name,
        gram_backend=backend,
        n_samples=X.shape[0],
        n_qubits=n_qubits,
        tile_size=tile_size,
        state_tile=(state_tile if state_tile is not None else ""),
        workers=workers,
        symmetric=symmetric,
        dtype=dtype,
        return_dtype=return_dtype_eff,
        time_s=(None if np.isnan(t_mean) else round(t_mean, 4)),
        std_s=(None if np.isnan(t_std) else round(t_std, 4)),
        pairs=n_pairs,
        Mpairs_per_s=(None if (t_mean is None or t_mean == 0 or np.isnan(t_mean)) else round(throughput / 1e6, 3)),
        symmetry_ok=int(bool(sym_ok)),
        psd_ok=int(bool(psd_ok)),
        rel_err_vs_ref=(None if rel_err is None else round(rel_err, 6)),
    ), last_K


def parse_samples(args):
    if args.samples_range:
        start, stop, step = map(int, args.samples_range.split(":"))
        return list(range(start, stop + (1 if step > 0 else -1), step))
    if args.samples:
        return args.samples
    return [512]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Q-kernel (PennyLane + backends numpy/cupy/torch/cuda_ry/cuda_states)."
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[512])
    parser.add_argument("--samples-range", type=str, default=None)  # start:stop:step
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--tile-size", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Legacy matmul tile (rows per block) for numpy/cupy/torch legacy paths.")
    parser.add_argument("--state-tile", type=int, nargs="+", default=[128, 256, 512],
                        help="For cuda_states: number of examples per state block.")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--device", type=str, nargs="+", default=["lightning.qubit", "lightning.gpu"])
    parser.add_argument("--backend", type=str, nargs="+", default=["auto"],
                        help="Backends: numpy cupy torch cuda_ry cuda_states auto, or 'all'.")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--return-dtype", choices=["float32", "float64"], default=None)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--csv", type=str, default="pl_kernel_benchmark.csv")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--desc", type=str, default="Gram")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--ref-backend", dest="ref_backend", type=str, default="numpy",
                        choices=["numpy", "cupy", "torch", "auto", "cuda_ry", "cuda_states"])
    parser.add_argument("--gpu-tile-eq-samples", action="store_true",
                        help="GPU+torch legacy: force tile_size == n_samples")
    parser.add_argument("--gpu-state-eq-samples", action="store_true",
                        help="cuda_states: force state_tile == n_samples")
    parser.add_argument("--tile-m", type=str, default="auto")
    parser.add_argument("--tile-n", type=str, default="auto")
    parser.add_argument("--tile-k", type=str, default="auto")
    args = parser.parse_args()

    # Expand backends
    if len(args.backend) == 1 and args.backend[0].lower() == "all":
        args.backend = ["numpy", "cupy", "torch", "cuda_ry", "cuda_states", "auto"]
    else:
        args.backend = [b.lower() for b in args.backend]

    # CuPy setup if needed
    if any(b in ("cupy", "cuda_ry", "cuda_states") for b in args.backend) or args.ref_backend in ("cupy", "cuda_ry", "cuda_states"):
        _maybe_setup_cupy()

    # Safer start method for PL + torch/cuda
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    samples_list = parse_samples(args)

    print(f"\nBenchmark kernel | samples={samples_list} | symmetric={args.symmetric} | devices={','.join(args.device)} | backends={','.join(args.backend)}")
    grid = list(itertools.product(args.device, args.backend, samples_list, args.qubits, args.tile_size, args.workers))
    print(f"Grid size (base, before state-tiling expansion): {len(grid)} combinations\n")

    # ASCII header (avoid � which can break some terminals)
    header = f"{'device':<16} {'backend':<11} {'N':>6} {'nq':>3} {'tile':>6} {'sTile':>6} {'wrk':>3} {'dtype':>7} {'ret':>7}  {'time(s)':>8}  {'+/-sd':>6}  {'Mpairs/s':>9}  {'sym':>3}  {'psd':>3}  {'rel_err':>8}"
    print(header)
    print("-" * len(header))

    # Optional verification
    if args.verify:
        try:
            rng = np.random.default_rng(0)
            X = rng.uniform(-np.pi, np.pi, size=(8, 6)).astype(np.float32)
            W = rng.normal(0, 0.1, size=(args.layers, 6)).astype(np.float32)
            K = compute_kernel_matrix(
                X, weights=W, device_name=args.device[0],
                tile_size=4, symmetric=True, n_workers=1,
                dtype="float32", return_dtype="float32",
                gram_backend=args.backend[0], progress=False, desc="verify",
                state_tile=8
            )
            ok = (K.shape == (8,8)) and np.allclose(np.diag(K), 1.0, atol=1e-3) and np.allclose(K, K.T, atol=1e-5)
            print(f"[verify] small-case on {args.device[0]} / backend={args.backend[0]}: {'OK' if ok else 'FAIL'}\n")
        except Exception as e:
            print(f"[verify] failed: {e}\n")

    results = []

    # Shared (X,W) per (nq, N)
    data_by_key = {}
    rng_master = np.random.default_rng(args.seed)
    for nq in sorted(set(args.qubits)):
        for N in sorted(set(samples_list)):
            dt = np.float32 if args.dtype == "float32" else np.float64
            X = rng_master.uniform(-np.pi, np.pi, size=(N, nq)).astype(dt)
            W = rng_master.normal(0, 0.1, size=(args.layers, nq)).astype(dt)
            data_by_key[(nq, N)] = (X, W)

    # Reference per (nq, N)
    ref_by_key, ref_tag = {}, None
    try:
        for nq in sorted(set(args.qubits)):
            for N in sorted(set(samples_list)):
                Xr, Wr = data_by_key[(nq, N)]
                ref_device = "lightning.qubit" if args.ref_backend == "numpy" else "lightning.gpu"
                ref_by_key[(nq, N)] = compute_kernel_matrix(
                    Xr, weights=Wr, device_name=ref_device,
                    tile_size=min(512, N),
                    symmetric=args.symmetric, n_workers=max(1, (os.cpu_count() or 2) - 1),
                    dtype="float64", return_dtype="float64", gram_backend=args.ref_backend,
                    progress=False, desc=f"ref(N={N},q={nq})",
                    state_tile=min(512, N)
                )
        ref_tag = f"{args.ref_backend.upper()}64"
    except Exception:
        ref_by_key, ref_tag = {}, None

    # Main loop
    for (dev, backend, N, nq, tile, wrk) in grid:
        if "gpu" in dev.lower() and wrk != 1:
            continue

        eff_tile = tile
        if args.gpu_tile_eq_samples and ("gpu" in dev.lower()) and (backend == "torch"):
            eff_tile = N

        state_tile_list = args.state_tile if backend == "cuda_states" else [None]
        if backend == "cuda_states" and args.gpu_state_eq_samples and "gpu" in dev.lower():
            state_tile_list = [N]

        for s_tile in state_tile_list:
            # Warmup
            if not args.no_warmup:
                try:
                    Xw, Ww = data_by_key[(nq, N)]
                    n_w = min(64, N)
                    _ = compute_kernel_matrix(
                        Xw[:n_w], weights=Ww, device_name=dev, tile_size=min(eff_tile, 64),
                        symmetric=args.symmetric, n_workers=1 if "gpu" in dev.lower() else min(wrk, os.cpu_count() or 1),
                        dtype=args.dtype, return_dtype=args.return_dtype, gram_backend=backend,
                        progress=False, desc=args.desc,
                        state_tile=(n_w if backend == "cuda_states" else 128),
                        tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k
                    )
                except Exception:
                    pass

            # Run
            try:
                X, W = data_by_key[(nq, N)]
                ref_K = ref_by_key.get((nq, N))
                r, _ = run_once(
                    X=X, W=W, n_qubits=nq,
                    tile_size=eff_tile, workers=wrk, device_name=dev,
                    symmetric=args.symmetric, repeats=args.repeats,
                    dtype=args.dtype, return_dtype=args.return_dtype,
                    backend=backend, progress=args.progress, desc=args.desc,
                    ref_K=ref_K, state_tile=s_tile,
                    tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k
                )
                results.append(r)
                print(f"{r['device']:<16} {r['gram_backend']:<11} {r['n_samples']:>6} {r['n_qubits']:>3} {r['tile_size']:>6} {str(r['state_tile']):>6} {r['workers']:>3} "
                      f"{r['dtype']:>7} {r['return_dtype']:>7}  {str(r['time_s']):>8}  {str(r['std_s']):>6}  {str(r['Mpairs_per_s']):>9}  "
                      f"{r['symmetry_ok']:>3}  {r['psd_ok']:>3}  {str(r['rel_err_vs_ref']):>8}")
            except Exception as e:
                print(f"{dev:<16} {backend:<11} {N:>6} {nq:>3} {eff_tile:>6} {str(s_tile):>6} {wrk:>3}  ERROR: {e}")
                results.append({
                    "device": dev, "gram_backend": backend, "n_samples": N, "n_qubits": nq, "tile_size": eff_tile,
                    "state_tile": s_tile if backend == "cuda_states" else "",
                    "workers": wrk, "symmetric": args.symmetric,
                    "dtype": args.dtype, "return_dtype": args.return_dtype or args.dtype,
                    "time_s": None, "std_s": None, "pairs": None,
                    "Mpairs_per_s": None, "symmetry_ok": None, "psd_ok": None, "rel_err_vs_ref": None,
                    "error": str(e),
                })

    if not results:
        return

    # Best per device+backend
    best = {}
    for r in results:
        key = (r["device"], r["gram_backend"])
        if r.get("Mpairs_per_s") is None:
            continue
        if key not in best or (r["Mpairs_per_s"] > best[key]["Mpairs_per_s"]):
            best[key] = r

    print("\n=== Best per (device, backend) ===")
    for (dev, be), r in best.items():
        print(f"{dev:<16} / {be:<11} -> N {r['n_samples']}, {r['n_qubits']}q, tile {r['tile_size']}, sTile {r['state_tile']}, wrk {r['workers']}, "
              f"dtype {r['dtype']}, ret {r['return_dtype']} : {r['Mpairs_per_s']} Mpairs/s")

    # CSV dump (UTF-8)
    fieldnames = [
        "device", "gram_backend", "n_samples", "n_qubits", "tile_size", "state_tile", "workers",
        "symmetric", "dtype", "return_dtype",
        "time_s", "std_s", "pairs", "Mpairs_per_s",
        "symmetry_ok", "psd_ok", "rel_err_vs_ref", "error"
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            if "error" not in r:
                r["error"] = ""
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nSaved CSV -> {args.csv}")
    print("Note: cuda_states autotunes RawKernel tiles when --tile-m/n/k=auto (macros not reported in CSV).")
    if 'ref_tag' in locals() and ref_tag:
        print(f"Reference for relative error: {ref_tag}")

if __name__ == "__main__":
    main()

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
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--tile-size", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--device", type=str, nargs="+", default=["lightning.qubit"])
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32","float64"], default="float32")
    parser.add_argument("--return-dtype", choices=["float32", "float64"], default="float32")
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
