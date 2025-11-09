#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark fidelity kernel: cuda_states vs torch
- Génère des angles et des poids aléatoires (réplicables)
- Simule les états quantiques TOUJOURS avec 'lightning.qubit'
- Compare uniquement les backends 'cuda_states' et 'torch'
- Fait varier la taille du dataset (N)
"""

import os, sys, time, csv, itertools, argparse
from pathlib import Path
import numpy as np

# --- project root in sys.path (IDE safe) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Limiter les threads BLAS pour stabiliser le bench CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Import API unifiée
from scripts.pipeline_backends import compute_kernel_matrix


def _maybe_setup_cupy_env():
    """Prépare les includes CUDA pour les backends CuPy si nécessaire."""
    os.environ.setdefault("CUPY_NVRTC_OPTIONS", "-I/usr/local/cuda/include")
    os.environ["CPATH"] = "/usr/local/cuda/include:" + os.environ.get("CPATH", "")
    os.environ["CPLUS_INCLUDE_PATH"] = "/usr/local/cuda/include:" + os.environ.get("CPLUS_INCLUDE_PATH", "")
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        pass


def pairs_count(n: int, symmetric: bool) -> int:
    return n * (n + 1) // 2 if symmetric else n * n


def parse_samples(args) -> list[int]:
    if args.samples_range:
        start, stop, step = map(int, args.samples_range.split(":"))
        if step == 0:
            raise ValueError("samples-range step must be != 0")
        return list(range(start, stop + (1 if step > 0 else -1), step))
    return args.samples


def run_once(
    X: np.ndarray,
    W: np.ndarray,
    symmetric: bool,
    repeats: int,
    dtype: str,
    return_dtype: str | None,
    backend: str,
) -> tuple[dict, np.ndarray]:
    """Exécute compute_kernel_matrix plusieurs fois et renvoie métriques + dernière matrice."""
    times = []
    last_K = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_K = compute_kernel_matrix(
            X,
            weights=W,
            device_name="lightning.qubit",      # << TOUJOURS CPU pour la simulation d'états
            tile_size=min(512, X.shape[0]),     # sans impact sur cuda_states (param hérité)
            symmetric=symmetric,
            n_workers=1,                        # sûr et stable (spawn inutile ici)
            dtype=dtype,
            return_dtype=return_dtype,
            gram_backend=backend,               # << backend GEMM comparé: torch vs cuda_states
            progress=False,
            desc=f"bench[{backend}]",
            # paramètres additionnels tolérés (ignorés si non supportés)
            state_tile=min(512, X.shape[0]),    # utile pour cuda_states
            tile_m="auto", tile_n="auto", tile_k="auto",
        )
        times.append(time.perf_counter() - t0)

    t_mean = float(np.mean(times))
    t_std = float(np.std(times))
    n_pairs = pairs_count(X.shape[0], symmetric)
    throughput = n_pairs / t_mean if t_mean > 0 else 0.0

    metrics = dict(
        device="lightning.qubit",  # device de simulation d'états
        gram_backend=backend,
        n_samples=X.shape[0],
        n_qubits=X.shape[1],
        symmetric=int(symmetric),
        dtype=dtype,
        return_dtype=(return_dtype if return_dtype else dtype),
        time_s=round(t_mean, 4),
        std_s=round(t_std, 4),
        pairs=n_pairs,
        Mpairs_per_s=round(throughput / 1e6, 3),
    )
    return metrics, last_K


def main():
    parser = argparse.ArgumentParser(description="Benchmark cuda_states vs torch (états simulés avec lightning.qubit).")
    # Variation de N
    parser.add_argument("--samples", type=int, nargs="+", default=[256, 512, 1024, 2048],
                        help="Tailles N à tester (liste).")
    parser.add_argument("--samples-range", type=str, default=None,
                        help="Format: start:stop:step (ex: 256:4096:256).")
    # Qubits et couches
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--layers", type=int, default=2)
    # Kernel options
    parser.add_argument("--symmetric", action="store_true", help="Utiliser un noyau symétrique (train).")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--return-dtype", choices=["float32", "float64"], default=None)
    # Sortie CSV
    parser.add_argument("--csv", type=str, default="pl_kernel_benchmark_states_vs_torch.csv")
    args = parser.parse_args()

    # Backends à comparer: STRICTEMENT ceux demandés
    backends = ["torch", "cuda_states"]

    # Setup CuPy env si cuda_states est présent
    if "cuda_states" in backends:
        _maybe_setup_cupy_env()

    # Reproductibilité
    rng = np.random.default_rng(args.seed)
    samples_list = parse_samples(args)

    grid = list(itertools.product(samples_list, args.qubits, backends))
    print(f"\nBenchmark kernel | states on 'lightning.qubit' | "
          f"backends={backends} | N in {samples_list} | qubits in {args.qubits} | symmetric={args.symmetric}\n")

    header = f"{'backend':<11} {'N':>6} {'nq':>3} {'dtype':>7} {'ret':>7}  {'time(s)':>8}  {'±sd':>6}  {'Mpairs/s':>9}"
    print(header)
    print("-" * len(header))

    results, last_ref = [], None
    for (N, nq, backend) in grid:
        # Données partagées (mêmes pour torch et cuda_states aux mêmes (N,nq))
        dt = np.float32 if args.dtype == "float32" else np.float64
        X = rng.uniform(-np.pi, np.pi, size=(N, nq)).astype(dt)
        W = rng.normal(0, 0.1, size=(args.layers, nq)).astype(dt)

        # (Optionnel) pré-échauffement ultra-léger
        try:
            _ = compute_kernel_matrix(
                X[:min(64, N)],
                weights=W,
                device_name="lightning.qubit",
                tile_size=min(64, N),
                symmetric=args.symmetric,
                n_workers=1,
                dtype=args.dtype,
                return_dtype=args.return_dtype,
                gram_backend=backend,
                progress=False,
                desc="warmup",
                state_tile=min(64, N),
                tile_m="auto", tile_n="auto", tile_k="auto",
            )
        except Exception:
            pass

        # Mesure
        r, _ = run_once(
            X=X, W=W, symmetric=args.symmetric, repeats=args.repeats,
            dtype=args.dtype, return_dtype=args.return_dtype, backend=backend,
        )
        results.append(r)
        print(f"{r['gram_backend']:<11} {r['n_samples']:>6} {r['n_qubits']:>3} {r['dtype']:>7} {r['return_dtype']:>7}  "
              f"{r['time_s']:>8.3f}  {r['std_s']:>6.3f}  {r['Mpairs_per_s']:>9.3f}")

    # Résumé “best per backend”
    best = {}
    for r in results:
        key = r["gram_backend"]
        if r.get("Mpairs_per_s") is None:
            continue
        if key not in best or r["Mpairs_per_s"] > best[key]["Mpairs_per_s"]:
            best[key] = r

    print("\n=== Best per backend (états: lightning.qubit) ===")
    for be, r in best.items():
        print(f"{be:<11} -> N {r['n_samples']}, {r['n_qubits']}q, dtype {r['dtype']}, ret {r['return_dtype']} : "
              f"{r['Mpairs_per_s']} Mpairs/s (time {r['time_s']} s)")

    # Dump CSV
    fieldnames = [
        "device", "gram_backend", "n_samples", "n_qubits",
        "symmetric", "dtype", "return_dtype",
        "time_s", "std_s", "pairs", "Mpairs_per_s",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nSaved CSV -> {args.csv}\n")


if __name__ == "__main__":
    # Sécurité multiprocessing si jamais on l’ajoute plus tard
    try:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
