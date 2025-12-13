# tools/benchmark_pl_kernel.py
import argparse
import itertools
import os
import time
import sys
import csv
import numpy as np
from pathlib import Path

# Ajout du root au path pour les imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_backends import compute_kernel_matrix

# Tentative d'import torch pour le monitoring mémoire
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def pairs_count(n: int, symmetric: bool) -> int:
    return n * (n + 1) // 2 if symmetric else n * n

def get_peak_memory_gb():
    """Retourne le pic mémoire GPU en GB si Torch est dispo."""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0

def reset_peak_memory():
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def run_once(
    n_samples, n_qubits, tile_size, state_tile, workers, device_name,
    symmetric, repeats, seed, layers, dtype, return_dtype, gram_backend,
    angle_scale, embed_mode
):
    rng = np.random.default_rng(seed)
    np_dtype = np.float32 if dtype == "float32" else np.float64
    
    # Génération données
    angles = rng.uniform(low=-np.pi, high=np.pi, size=(n_samples, n_qubits)).astype(np_dtype)
    weights = rng.normal(loc=0.0, scale=0.1, size=(layers, n_qubits)).astype(np_dtype)

    # Paramètres communs
    kwargs = dict(
        weights=weights,
        device_name=device_name,
        tile_size=tile_size,
        symmetric=symmetric,
        n_workers=workers,
        dtype=dtype,
        return_dtype=return_dtype,
        gram_backend=gram_backend,
        angle_scale=angle_scale,
        embed_mode=embed_mode,
        state_tile=state_tile  # Important pour cuda_states
    )

    # --- Warmup ---
    # On fait un petit warmup pour compiler les kernels JIT/CuPy sans fausser le temps
    try:
        # Petit sous-ensemble pour le warmup pour aller vite
        warmup_n = min(n_samples, 256)
        _ = compute_kernel_matrix(angles[:warmup_n], **{**kwargs, "symmetric": symmetric})
        if HAS_TORCH and "gpu" in device_name:
            torch.cuda.synchronize()
    except Exception as e:
        print(f"⚠️ Warmup failed: {e}")

    # --- Benchmark ---
    reset_peak_memory()
    times = []
    
    for _ in range(repeats):
        # Force le nettoyage mémoire avant chaque run
        if HAS_TORCH and "gpu" in device_name:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        t0 = time.perf_counter()
        
        _ = compute_kernel_matrix(angles, **kwargs)
        
        if HAS_TORCH and "gpu" in device_name:
            torch.cuda.synchronize() # Attendre la fin réelle du GPU
            
        times.append(time.perf_counter() - t0)

    # Stats
    t_mean = float(np.mean(times))
    t_std  = float(np.std(times))
    n_pairs = pairs_count(n_samples, symmetric)
    throughput = n_pairs / t_mean if t_mean > 0 else 0
    max_vram = get_peak_memory_gb()

    return dict(
        device=device_name,
        n_samples=n_samples,
        n_qubits=n_qubits,
        tile_size=tile_size,
        state_tile=state_tile,
        workers=workers,
        symmetric=symmetric,
        layers=layers,
        dtype=dtype,
        gram_backend=gram_backend,
        time_s=round(t_mean, 4),
        std_s=round(t_std, 4),
        Mpairs_per_s=round(throughput / 1e6, 3),
        max_vram_gb=round(max_vram, 3)
    )

def main():
    parser = argparse.ArgumentParser(description="Benchmark PennyLane fidelity-kernel.")
    
    # Grid search params
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--qubits", type=int, nargs="+", default=[8, 10])
    parser.add_argument("--tile-size", type=int, nargs="+", default=[128])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--device", type=str, nargs="+", default=["lightning.qubit", "lightning.gpu"])
    
    # Backend params
    parser.add_argument("--gram-backend", choices=["auto", "numpy", "cupy", "torch", "cuda_states"], default="auto")
    parser.add_argument("--dtype", choices=["float32","float64"], default="float32")
    parser.add_argument("--return-dtype", choices=["float32", "float64"], default=None)
    parser.add_argument("--state-tile", type=int, default=4096, help="Taille du batch d'états (GPU)")
    
    # Kernel details
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--embed-mode", type=str, default="ryrz")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # Configuration CPU
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Construction de la grille
    # On filtre : pas de workers > 1 pour le GPU
    grid = []
    for dev, nq, tile, wrk in itertools.product(args.device, args.qubits, args.tile_size, args.workers):
        if "gpu" in dev.lower() and wrk > 1:
            continue # GPU est toujours worker=1 (streaming)
        grid.append((dev, nq, tile, wrk))

    print(f"\n{'='*100}")
    print(f"🚀 BENCHMARK KERNEL QUANTIQUE")
    print(f"Backend: {args.gram_backend} | Precision: {args.dtype} | Samples: {args.samples}")
    print(f"{'='*100}\n")
    
    headers = f"{'device':<16} {'nq':>3} {'tile':>5} {'wrk':>3} {'backend':>12} {'VRAM(GB)':>9} {'time(s)':>8} {'Mpairs/s':>9}"
    print(headers)
    print("-" * len(headers))

    results = []

    for (dev, nq, tile, wrk) in grid:
        try:
            r = run_once(
                n_samples=args.samples,
                n_qubits=nq,
                tile_size=tile,
                state_tile=args.state_tile,
                workers=wrk,
                device_name=dev,
                symmetric=args.symmetric,
                repeats=args.repeats,
                seed=args.seed,
                layers=args.layers,
                dtype=args.dtype,
                return_dtype=args.return_dtype,
                gram_backend=args.gram_backend,
                angle_scale=args.angle_scale,
                embed_mode=args.embed_mode
            )
            results.append(r)
            print(f"{r['device']:<16} {r['n_qubits']:>3} {r['tile_size']:>5} {r['workers']:>3} {r['gram_backend']:>12} {r['max_vram_gb']:>9.2f} {r['time_s']:>8.3f} {r['Mpairs_per_s']:>9.2f}")
            
        except Exception as e:
            # Afficher l'erreur sans crasher tout le bench
            err_msg = str(e).split('\n')[0]
            print(f"{dev:<16} {nq:>3} {tile:>5} {wrk:>3} ERROR: {err_msg}")

    # === Export CSV ===
    if results:
        fieldnames = [
            "device", "n_samples", "n_qubits", "tile_size", "state_tile", "workers",
            "symmetric", "layers", "dtype", "gram_backend",
            "time_s", "std_s", "Mpairs_per_s", "max_vram_gb"
        ]
        
        filename = f"bench_results_{args.gram_backend}_{args.dtype}.csv"
        with open(filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\n✅ Résultats sauvegardés dans : {filename}")

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()