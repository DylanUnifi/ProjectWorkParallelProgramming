

import numpy as np
import time
import itertools
from scripts.pipeline_backends import compute_kernel_matrix

def benchmark_config(backend, n_samples, n_qubits, **kwargs):
    """Benchmark une config spécifique."""
    rng = np.random.default_rng(42)
    angles = rng.uniform(-np.pi, np.pi, (n_samples, n_qubits)).astype(np.float32)
    weights = rng.normal(0, 0.1, (2, n_qubits)).astype(np.float32)
    
    # Warmup
    _ = compute_kernel_matrix(angles, weights=weights, **kwargs)
    
    # Timed run
    t0 = time.perf_counter()
    K = compute_kernel_matrix(angles, weights=weights, **kwargs)
    elapsed = time.perf_counter() - t0
    
    n_pairs = n_samples * (n_samples + 1) // 2
    throughput = n_pairs / elapsed / 1e6
    
    return {
        "time": elapsed,
        "throughput": throughput,
        "kernel": K,
    }

def test_numpy_ncores():
    n = 24
    """Test impact tile_size sur n cores."""
    print("\n" + "="*80)
    print(f"TEST 1: NUMPY avec {n} CORES - Impact du tile_size")
    print("="*80 + "\n")
    
    n_samples = [10000, 20000, 30000, 40000, 50000]
    n_qubits = 8
    tile_size = 192
    
    print(f"{'tile_size':<12} {'Time (s)':<12} {'Mpairs/s':<12} {'Efficiency':<12}")
    print("-" * 60)
    
    results = []
    for n_sample in n_samples:
        res = benchmark_config(
            "numpy",
            n_sample, n_qubits,
            device_name="lightning.qubit",
            tile_size=tile_size,
            symmetric=True,
            n_workers=n,
            dtype="float32",
            gram_backend="numpy",
        )
        results.append((n_sample, res))
        
        # Efficiency = throughput / theoretical_peak
        # Avec 96 cores, peak théorique ~25-30 Mpairs/s
        efficiency = res["throughput"] / 25.0 * 100
        
        print(f"{n_sample:<12} {res['time']:<12.2f} {res['throughput']:<12.3f} {efficiency:<12.1f}%")
    
    # Trouve optimal
    best = max(results, key=lambda x: x[1]["throughput"])
    print(f"\n✅ OPTIMAL: tile_size={best[0]} → {best[1]['throughput']:.3f} Mpairs/s")
    
    return results

def test_cuda_states_massive_vram():
    print("\n" + "="*80)
    print("TEST 2: CUDA_STATES")
    print("="*80 + "\n")
    
    configs = [
        # (N, nq, state_tile, autotune, precompute_all_states, vram_fraction)
        (10000, 8, -1, True, True, 0.85),
        (20000, 8, -1, True, True, 0.85),
        (30000, 8, -1, True, True, 0.85),
    ]
    
    print(f"{'N':<8} {'nq':<4} {'state_tile':<12} {'tile_m':<8} {'Time (s)':<12} {'Mpairs/s':<12} {'VRAM (GB)':<12}")
    print("-" * 90)
    
    for n, nq, state_tile, autotune, precompute_all_states, vram_fraction in configs:
        try:
            res = benchmark_config(
                "cuda_states",
                n, nq,
                device_name="lightning.gpu",
                symmetric=True,
                dtype="float32",
                gram_backend="cuda_states",
                state_tile=state_tile,
                autotune=autotune,
                precompute_all_states=precompute_all_states,
                vram_fraction=vram_fraction,
                progress=True,
            )
            
            # Estime VRAM usage
            dim = 2 ** nq
            vram_states = n * dim * 8 * 2 / 1024**3  # 2 copies, complex64
            vram_kernel = n * n * 4 / 1024**3  # float32
            vram_total = vram_states + vram_kernel
            
            print(f"{n:<8} {nq:<4} {state_tile:<12} {autotune:<8} {res['time']:<12.2f} "
                  f"{res['throughput']:<12.3f} {vram_total:<12.1f}")
        
        except Exception as e:
            print(f"{n:<8} {nq:<4} {state_tile:<12} {autotune:<8} ERROR: {e}")
        

def test_tensorcore_blackwell():
    """Test Tensor Cores sur Blackwell."""
    print("\n" + "="*80)
    print("TEST 3: TENSORCORE sur Blackwell - FP16 vs BF16")
    print("="*80 + "\n")
    
    n_samples = 15000
    n_qubits = 8
    
    # Baseline FP32
    print("Running FP32 baseline...")
    res_fp32 = benchmark_config(
        "cuda_states",
        n_samples, n_qubits,
        device_name="lightning.gpu",
        symmetric=True,
        dtype="float32",
        gram_backend="cuda_states",
        state_tile=2048,
        tile_m=32, tile_n=32, tile_k=32,
    )
    
    # FP16
    print("Running FP16 Tensor Cores...")
    try:
        res_fp16 = benchmark_config(
            "tensorcore",
            n_samples, n_qubits,
            device_name="lightning.gpu",
            symmetric=True,
            dtype="float32",
            gram_backend="tensorcore",
            state_tile=4096,  # ⭐ Gros batch sur Blackwell
            tensorcore_precision="fp16",
        )
        speedup_fp16 = res_fp32["time"] / res_fp16["time"]
        rel_err_fp16 = np.max(np.abs(res_fp16["kernel"] - res_fp32["kernel"])) / np.max(np.abs(res_fp32["kernel"]))
    except Exception as e:
        print(f"FP16 failed: {e}")
        res_fp16 = None
    
    # BF16
    print("Running BF16 Tensor Cores...")
    try:
        res_bf16 = benchmark_config(
            "tensorcore",
            n_samples, n_qubits,
            device_name="lightning.gpu",
            symmetric=True,
            dtype="float32",
            gram_backend="tensorcore",
            state_tile=4096,
            tensorcore_precision="bf16",
        )
        speedup_bf16 = res_fp32["time"] / res_bf16["time"]
        rel_err_bf16 = np.max(np.abs(res_bf16["kernel"] - res_fp32["kernel"])) / np.max(np.abs(res_fp32["kernel"]))
    except Exception as e:
        print(f"BF16 failed: {e}")
        res_bf16 = None
    
    print(f"\n{'Backend':<15} {'Time (s)':<12} {'Mpairs/s':<12} {'Speedup':<10} {'Rel Error':<12}")
    print("-" * 70)
    print(f"{'FP32':<15} {res_fp32['time']:<12.2f} {res_fp32['throughput']:<12.3f} {'1.00x':<10} {'-':<12}")
    
    if res_fp16:
        print(f"{'FP16 TensorCore':<15} {res_fp16['time']:<12.2f} {res_fp16['throughput']:<12.3f} "
              f"{speedup_fp16:<10.2f}x {rel_err_fp16:<12.2e}")
    
    if res_bf16:
        print(f"{'BF16 TensorCore':<15} {res_bf16['time']:<12.2f} {res_bf16['throughput']:<12.3f} "
              f"{speedup_bf16:<10.2f}x {rel_err_bf16:<12.2e}")
    
    if res_bf16 and res_fp16:
        winner = "BF16" if res_bf16["throughput"] > res_fp16["throughput"] else "FP16"
        print(f"\n🏆 Winner on Blackwell: {winner}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace("tools/test_tile_impact_monster.py", ""))
    
    # Run all tests
    test_numpy_ncores()
    # test_cuda_states_massive_vram()
    #test_tensorcore_blackwell()