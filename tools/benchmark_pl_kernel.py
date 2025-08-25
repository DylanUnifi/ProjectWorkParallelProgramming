# tools/benchmark_pl_kernel.py
import argparse, itertools, math, os, time
import numpy as np

# PennyLane-only kernel (ton fichier simplifié)
from scripts.pipeline_backends import compute_kernel_matrix

def pairs_count(n: int, symmetric: bool) -> int:
    return n * (n + 1) // 2 if symmetric else n * n

def run_once(n_samples, n_qubits, tile_size, workers, device_name, symmetric, repeats, seed):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(low=-np.pi, high=np.pi, size=(n_samples, n_qubits)).astype(np.float64)

    # poids entangler (fixes, mais réalistes)
    n_layers = 2
    weights = rng.normal(loc=0.0, scale=0.1, size=(n_layers, n_qubits)).astype(np.float64)

    # warmup (facultatif)
    _ = compute_kernel_matrix(
        angles,
        weights=weights,
        device_name=device_name,
        tile_size=tile_size,
        symmetric=symmetric,
        n_workers=workers,
    )

    times = []
    for r in range(repeats):
        t0 = time.perf_counter()
        _ = compute_kernel_matrix(
            angles,
            weights=weights,
            device_name=device_name,
            tile_size=tile_size,
            symmetric=symmetric,
            n_workers=workers,
        )
        dt = time.perf_counter() - t0
        times.append(dt)

    t_mean = np.mean(times)
    t_std  = np.std(times)
    n_pairs = pairs_count(n_samples, symmetric)
    throughput = n_pairs / t_mean  # pairs / s

    return dict(
        n_samples=n_samples,
        n_qubits=n_qubits,
        tile_size=tile_size,
        workers=workers,
        device=device_name,
        symmetric=symmetric,
        time_s=round(t_mean, 4),
        std_s=round(t_std, 4),
        pairs=n_pairs,
        Mpairs_per_s=round(throughput / 1e6, 3),
    )

def main():
    parser = argparse.ArgumentParser(description="Benchmark PennyLane fidelity-kernel (parallel, tiled).")
    parser.add_argument("--samples", type=int, default=512, help="N (angles rows)")
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8], help="List of n_qubits")
    parser.add_argument("--tile-size", type=int, nargs="+", default=[32, 64, 128], help="Tile sizes to try")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8], help="#processes (0 => cpu_count-1)")
    parser.add_argument("--device", type=str, default="lightning.qubit", help="PennyLane device")
    parser.add_argument("--symmetric", action="store_true", help="Compute symmetric K (Y=None)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per config")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")  # utile pour éviter sur-parallélisme interne

    grid = list(itertools.product(args.qubits, args.tile_size, args.workers))
    results = []

    print(f"\nBenchmark kernel | N={args.samples} | symmetric={args.symmetric} | device={args.device}")
    print(f"Grid size: {len(grid)} combinations\n")
    print(f"{'nq':>3} {'tile':>4} {'wrk':>3}  {'time(s)':>8}  {'±sd':>6}  {'Mpairs/s':>9}")
    print("-"*50)

    for (nq, tile, wrk) in grid:
        try:
            r = run_once(
                n_samples=args.samples,
                n_qubits=nq,
                tile_size=tile,
                workers=wrk,
                device_name=args.device,
                symmetric=args.symmetric,
                repeats=args.repeats,
                seed=args.seed,
            )
            results.append(r)
            print(f"{nq:>3} {tile:>4} {wrk:>3}  {r['time_s']:>8.3f}  {r['std_s']:>6.3f}  {r['Mpairs_per_s']:>9.3f}")
        except Exception as e:
            print(f"{nq:>3} {tile:>4} {wrk:>3}  ERROR: {e}")

    # tri et résumé meilleur score
    if results:
        best = max(results, key=lambda x: x["Mpairs_per_s"])
        print("\nBest configuration:")
        print(best)

        # CSV sommaire
        import csv
        out = "pl_kernel_benchmark.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print(f"\nSaved CSV -> {out}")

if __name__ == "__main__":
    # important: for multiprocessing on some platforms
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
