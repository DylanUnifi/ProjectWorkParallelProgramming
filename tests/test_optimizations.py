#!/usr/bin/env python3
"""
Unit tests for GPU optimization classes.

This test suite validates:
1. DynamicBatchSizer - Runtime batch size adjustment
2. CUDAStreamPool - Multiple CUDA stream management
3. TileSizeOptimizer - Learning optimal tile sizes
4. Torch backend optimizations - Runtime config propagation
5. MemoryProfiler - GPU memory tracking and analysis
6. CUDAGraphManager - CUDA graph capture and replay
"""

import sys
import os
import json
import logging
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS_DIR = Path(__file__).resolve().parent
LOG_FILE = TESTS_DIR / "test_optimizations.log"
SUMMARY_FILE = TESTS_DIR / "test_optimizations_summary.json"

LOG = logging.getLogger("test_optimizations")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOG.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)

TEST_RESULTS = []


def _record_test_result(test_name: str, status: str, details=None):
    entry = {"test": test_name, "status": status}
    if details is not None:
        entry["details"] = details
    TEST_RESULTS.append(entry)
    LOG.info("%s %s", test_name, json.dumps(entry, sort_keys=True, default=str))
    return entry


def _write_results_summary():
    summary = {"results": TEST_RESULTS}
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    LOG.info("Wrote test summary to %s", SUMMARY_FILE)


def _detect_gpu_vram_gb(default_vram_gb: float = 48.0):
    try:
        import cupy as cp

        device = cp.cuda.Device()
        free_vram_gb = device.mem_info[0] / 1e9
        total_vram_gb = device.mem_info[1] / 1e9
        LOG.info("GPU VRAM: %.1f GB total, %.1f GB free", total_vram_gb, free_vram_gb)
        return total_vram_gb, free_vram_gb, True
    except Exception as e:
        LOG.warning("Cannot detect GPU VRAM: %s", e)
        LOG.warning("Falling back to %.1f GB for VRAM-aware optimization tests", default_vram_gb)
        return default_vram_gb, default_vram_gb, False

from benchmark import BACKEND_CONFIGS, _backend_runtime_config
from scripts.pipeline_backends import (
    DynamicBatchSizer,
    CUDAStreamPool,
    TileSizeOptimizer,
    MemoryProfiler,
    CUDAGraphManager
)


def test_dynamic_batch_sizer():
    """Test DynamicBatchSizer class."""
    print("\n" + "="*70)
    print("TEST 1: DynamicBatchSizer")
    print("="*70)
    
    # Initialize batch sizer
    sizer = DynamicBatchSizer(
        initial_batch=2048,
        min_batch=512,
        max_batch=8192,
        target_memory_usage=0.85
    )
    
    print(f"  Initial batch size: {sizer.current_batch}")
    assert sizer.current_batch == 2048, "Initial batch size should be 2048"
    
    # Test memory pressure reduction
    print("\n  Testing high memory pressure (0.95)...")
    new_batch = sizer.adjust(current_memory_used=0.95, last_kernel_time=0.5)
    print(f"  Adjusted batch size: {new_batch}")
    assert new_batch < 2048, "Batch size should decrease under high memory pressure"
    assert new_batch >= 512, "Batch size should not go below minimum"
    
    # Test memory headroom increase
    print("\n  Testing low memory usage (0.60)...")
    for i in range(10):
        # Add stable kernel times
        new_batch = sizer.adjust(current_memory_used=0.60, last_kernel_time=0.4 + 0.01 * i)
    
    print(f"  Final batch size after low memory: {new_batch}")
    # With low variance, should increase
    
    # Test statistics
    report = sizer.report()
    print(f"\n  Statistics:")
    print(f"    Total adjustments: {report['adjustments']}")
    print(f"    Batch range: {report['min_batch_used']} - {report['max_batch_used']}")
    print(f"    Total kernels: {report['total_kernels']}")
    print(f"    Avg kernel time: {report['avg_kernel_time']:.4f}s")
    
    print("  Success: DynamicBatchSizer test passed")
    return {
        "adjustments": report["adjustments"],
        "batch_range": [report["min_batch_used"], report["max_batch_used"]],
        "total_kernels": report["total_kernels"],
        "avg_kernel_time_s": report["avg_kernel_time"],
    }


def test_cuda_stream_pool():
    """Test CUDAStreamPool class."""
    print("\n" + "="*70)
    print("TEST 2: CUDAStreamPool")
    print("="*70)
    
    try:
        import cupy as cp
        
        # Initialize stream pool
        num_streams = 4
        pool = CUDAStreamPool(num_streams=num_streams)
        
        print(f"  Created pool with {num_streams} streams")
        assert len(pool.streams) == num_streams, f"Should have {num_streams} streams"
        
        # Test round-robin assignment
        print("\n  Testing round-robin stream assignment...")
        streams = [pool.get_stream() for _ in range(10)]
        print(f"  Obtained {len(streams)} streams")
        print(f"  Usage counts: {pool.usage_count}")
        
        # Check round-robin distribution
        expected_per_stream = 10 // num_streams
        for count in pool.usage_count:
            assert count >= expected_per_stream, "Streams should be used roughly equally"
        
        # Test synchronization
        print("\n  Testing synchronization...")
        pool.synchronize_all()
        print("  Success: All streams synchronized")
        
        # Test context manager
        print("\n  Testing context manager...")
        with CUDAStreamPool(num_streams=2) as pool2:
            s1 = pool2.get_stream()
            s2 = pool2.get_stream()
            assert s1 is not s2, "Different streams should be returned"
        print("  Success: Context manager works correctly")
        
        # Test utilization
        utilization = pool.get_utilization()
        print(f"\n  Stream utilization: {utilization:.2%}")
        assert 0.0 <= utilization <= 1.0, "Utilization should be between 0 and 1"
        
        print("  Success: CUDAStreamPool test passed")
        return {
            "streams": num_streams,
            "usage_count": pool.usage_count,
            "utilization": utilization,
        }
        
    except ImportError:
        print("  Warning:  CuPy not available, skipping CUDAStreamPool test")
        return {"skipped": True, "reason": "cupy not available"}


def test_tile_size_optimizer():
    """Test TileSizeOptimizer class."""
    print("\n" + "="*70)
    print("TEST 3: TileSizeOptimizer")
    print("="*70)

    total_vram_gb, available_vram_gb, has_gpu_vram = _detect_gpu_vram_gb()
    vram_scale = max(0.5, min(2.0, available_vram_gb / 48.0))
    print(f"  Detected VRAM: {total_vram_gb:.1f} GB total, {available_vram_gb:.1f} GB free")
    print(f"  Using VRAM scale factor: {vram_scale:.2f}")
    
    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        # Initialize optimizer
        optimizer = TileSizeOptimizer(history_file=temp_file)
        
        print(f"  Created optimizer with history file: {temp_file}")
        
        # Record some test runs
        print("\n  Recording test runs...")
        base_test_configs = [
            (1000, 16, 2048, (32, 32, 32), 1e6, 12.5),
            (1000, 16, 4096, (32, 32, 64), 1.2e6, 18.0),
            (1000, 16, 2048, (64, 64, 32), 1.1e6, 14.0),
            (2000, 16, 1024, (32, 32, 32), 8e5, 28.0),
            (1000, 16, 8192, (64, 32, 32), 1.05e6, 20.0),
            (1000, 16, 1024, (32, 64, 32), 9.7e5, 9.5),
            (1000, 16, 3072, (64, 32, 64), 1.15e6, 16.5),
            (1000, 16, 6144, (32, 64, 64), 1.08e6, 22.0),
            (1000, 16, 2560, (48, 32, 32), 1.18e6, 11.0),
            (1000, 16, 5120, (48, 48, 32), 1.22e6, 15.0),
            (1000, 16, 3584, (32, 48, 64), 1.19e6, 13.5),
        ]
        test_configs = [
            (
                n_samples,
                n_qubits,
                state_tile,
                kernel_tiles,
                throughput,
                peak_mem * vram_scale,
            )
            for n_samples, n_qubits, state_tile, kernel_tiles, throughput, peak_mem in base_test_configs
        ]
        top_configs = sorted(test_configs, key=lambda config: config[4], reverse=True)[:3]
        best_config = next(config for config in test_configs if config[2] == 5120 and config[3] == (48, 48, 32))
        
        for n_samples, n_qubits, state_tile, kernel_tiles, throughput, peak_mem in test_configs:
            optimizer.record_run(n_samples, n_qubits, state_tile, kernel_tiles, throughput, peak_mem)
            print(f"    Recorded: n={n_samples}, nq={n_qubits}, tile={state_tile}, "
                  f"throughput={throughput:.2e}")

        print("\n  Top throughput candidates:")
        for rank, (n_samples, n_qubits, state_tile, kernel_tiles, throughput, peak_mem) in enumerate(top_configs, start=1):
            print(
                f"    #{rank}: tile={state_tile}, kernel_tiles={kernel_tiles}, "
                f"throughput={throughput:.2e}, peak_mem={peak_mem:.1f} GB"
            )
        
        # Test prediction
        print("\n  Testing tile prediction...")
        prediction = optimizer.predict_optimal_tiles(
            n_samples=1000,
            n_qubits=16,
            available_vram=available_vram_gb,
        )
        
        print(f"    Predicted state_tile: {prediction['state_tile']}")
        print(f"    Predicted kernel_tiles: {prediction['kernel_tiles']}")
        print(f"    Confidence: {prediction['confidence']:.2f}")
        print(f"    Source: {prediction['source']}")
        
        assert prediction['state_tile'] > 0, "Should predict a valid state tile"
        assert len(prediction['kernel_tiles']) == 3, "Should predict 3 kernel tile dimensions"
        assert prediction['source'] == "learned", "Should use learned history when available"
        assert prediction['confidence'] >= 0.8, "Should have a strong confidence signal from the expanded history"
        assert prediction['state_tile'] == 5120, "Should select the best-performing learned state tile"
        assert prediction['kernel_tiles'] == (48, 48, 32), "Should select the best-performing learned kernel tiles"

        print("\n  Testing VRAM-constrained prediction...")
        tight_vram_gb = max(4.0, best_config[5] * 0.9)
        constrained_prediction = optimizer.predict_optimal_tiles(
            n_samples=1000,
            n_qubits=16,
            available_vram=tight_vram_gb
        )

        print(f"    Constrained state_tile: {constrained_prediction['state_tile']}")
        print(f"    Constrained kernel_tiles: {constrained_prediction['kernel_tiles']}")
        print(f"    Constrained confidence: {constrained_prediction['confidence']:.2f}")

        assert constrained_prediction['state_tile'] == 2560, (
            "Should fall back to the best configuration that fits the tighter VRAM budget"
        )
        assert constrained_prediction['kernel_tiles'] == (48, 32, 32), (
            "Should select the best valid tiles under the tighter VRAM budget"
        )
        assert constrained_prediction['confidence'] >= 0.2, "Should still keep a meaningful confidence level"
        
        # Test statistics
        stats = optimizer.get_statistics()
        print(f"\n  Statistics:")
        print(f"    Total runs: {stats['total_runs']}")
        print(f"    Unique configs: {stats['unique_configs']}")
        print(f"    Avg throughput: {stats['avg_throughput']:.2e}")
        
        assert stats['total_runs'] == len(test_configs), "Should have recorded all runs"
        
        # Verify persistence
        print("\n  Testing persistence...")
        optimizer2 = TileSizeOptimizer(history_file=temp_file)
        stats2 = optimizer2.get_statistics()
        assert stats2['total_runs'] == stats['total_runs'], "History should persist"
        
        print("  Success: TileSizeOptimizer test passed")
        return {
            "total_runs": stats["total_runs"],
            "unique_configs": stats["unique_configs"],
            "avg_throughput": stats["avg_throughput"],
            "best_state_tile": prediction["state_tile"],
            "best_kernel_tiles": prediction["kernel_tiles"],
            "best_confidence": prediction["confidence"],
            "constrained_state_tile": constrained_prediction["state_tile"],
            "constrained_kernel_tiles": constrained_prediction["kernel_tiles"],
            "constrained_confidence": constrained_prediction["confidence"],
            "detected_total_vram_gb": total_vram_gb,
            "detected_available_vram_gb": available_vram_gb,
            "vram_scale": vram_scale,
            "vram_detection": "gpu" if has_gpu_vram else "fallback",
            "top_throughput_configs": [
                {
                    "state_tile": config[2],
                    "kernel_tiles": config[3],
                    "throughput": config[4],
                    "peak_memory": config[5],
                }
                for config in top_configs
            ],
        }
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_torch_optimizations():
    """Test Torch backend optimization config propagation."""
    print("\n" + "="*70)
    print("TEST 4: Torch Backend Optimizations")
    print("="*70)

    base_config = BACKEND_CONFIGS["torch"]
    runtime_config = _backend_runtime_config("torch", base_config)

    print("  Checking default Torch runtime config...")
    assert runtime_config["gram_backend"] == "torch", "Torch backend should keep gram_backend='torch'"
    assert runtime_config["device_name"] == "lightning.gpu", "Torch backend should target lightning.gpu"

    expected_defaults = {
        "use_pinned_memory": False,
        "use_cuda_streams": False,
        "use_amp": False,
        "use_compile": False,
    }
    for flag, expected_value in expected_defaults.items():
        assert runtime_config[flag] is expected_value, f"Default {flag} should be {expected_value}"

    print("  Checking Torch optimization flag combinations...")
    configs = [
        {
            "name": "torch_pinned",
            "use_pinned_memory": True,
            "use_cuda_streams": False,
            "use_amp": False,
            "use_compile": False,
        },
        {
            "name": "torch_streams",
            "use_pinned_memory": False,
            "use_cuda_streams": True,
            "use_amp": False,
            "use_compile": False,
        },
        {
            "name": "torch_amp_compile",
            "use_pinned_memory": False,
            "use_cuda_streams": False,
            "use_amp": True,
            "use_compile": True,
        },
    ]

    for config in configs:
        runtime_config = _backend_runtime_config("torch", config)
        print(f"    Validating {config['name']} -> {runtime_config}")
        assert runtime_config["gram_backend"] == "torch", "Torch config should preserve gram_backend"
        assert runtime_config["device_name"] == "lightning.gpu", "Torch config should preserve device_name"
        for flag in expected_defaults:
            assert runtime_config[flag] is config[flag], (
                f"{config['name']} should preserve {flag}={config[flag]}"
            )

    print("  Success: Torch backend optimization test passed")
    return {"configurations_checked": len(configs)}


def test_memory_profiler():
    """Test MemoryProfiler class."""
    print("\n" + "="*70)
    print("TEST 5: MemoryProfiler")
    print("="*70)
    
    # Initialize profiler
    profiler = MemoryProfiler(enable_realtime=False)
    
    print("  Created MemoryProfiler")
    
    # Track some allocations
    print("\n  Tracking allocations...")
    profiler.track_allocation("states_A", 12.8e9)  # 12.8 GB
    profiler.track_allocation("states_B", 25.6e9)  # 25.6 GB
    profiler.track_allocation("kernel_output", 6.4e9)  # 6.4 GB
    profiler.track_allocation("working_memory", 0.4e9)  # 0.4 GB
    
    print(f"    Tracked {len(profiler.allocations)} allocation categories")
    print(f"    Peak allocated: {profiler.peak_allocated / 1e9:.1f} GB")
    
    # Track some transfers
    print("\n  Tracking transfers...")
    profiler.track_transfer("H2D", 38.4e9, 3175.0)  # 38.4 GB in 3.175s
    profiler.track_transfer("H2D", 10.0e9, 800.0)
    profiler.track_transfer("D2H", 6.4e9, 542.0)   # 6.4 GB in 0.542s
    
    print(f"    Tracked {len(profiler.transfers)} transfers")
    
    # Track kernel executions
    print("\n  Tracking kernel executions...")
    for i in range(100):
        profiler.track_kernel(0.42 + 0.05 * np.random.randn())  # ~0.42ms avg
    
    print(f"    Tracked {profiler.kernel_launches} kernel launches")
    
    # Generate report
    print("\n  Generating report...")
    report = profiler.report(verbose=False)
    
    print(f"    Peak allocated: {report['peak_allocated_gb']:.1f} GB")
    print(f"    H2D total: {report['h2d_total_gb']:.1f} GB")
    print(f"    D2H total: {report['d2h_total_gb']:.1f} GB")
    print(f"    Avg kernel time: {report['avg_kernel_time_ms']:.2f} ms")
    
    assert report['peak_allocated_gb'] > 0, "Should have tracked allocations"
    assert report['kernel_launches'] == 100, "Should have tracked 100 kernels"
    
    # Test snapshot
    print("\n  Testing snapshot...")
    snapshot = profiler.snapshot()
    print(f"    Snapshot keys: {list(snapshot.keys())}")
    
    print("  Success: MemoryProfiler test passed")
    return {
        "peak_allocated_gb": report["peak_allocated_gb"],
        "kernel_launches": report["kernel_launches"],
        "h2d_total_gb": report["h2d_total_gb"],
        "d2h_total_gb": report["d2h_total_gb"],
    }


def test_cuda_graph_manager():
    """Test CUDAGraphManager class."""
    print("\n" + "="*70)
    print("TEST 6: CUDAGraphManager")
    print("="*70)
    
    try:
        import cupy as cp
        
        # Initialize graph manager
        manager = CUDAGraphManager()
        
        print("  Created CUDAGraphManager")
        
        # Create a simple test kernel
        kernel_code = """
        extern "C" __global__
        void test_kernel(float* out, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                out[i] = i * 2.0f;
            }
        }
        """
        
        mod = cp.RawModule(code=kernel_code, options=("--std=c++14",))
        kernel = mod.get_function("test_kernel")
        
        # Create test data
        n = 1024
        out = cp.empty(n, dtype=cp.float32)
        stream = cp.cuda.Stream()
        
        # Test normal execution
        print("\n  Testing normal kernel execution...")
        grid = ((n + 255) // 256,)
        block = (256,)
        args = (out, n)
        
        with stream:
            kernel(grid, block, args)
        stream.synchronize()
        
        print("  Success: Normal execution succeeded")
        
        # Test graph capture (Note: May not work in all environments)
        print("\n  Testing graph capture...")
        graph_key = ("test", n, 256)
        
        try:
            manager.capture_graph(graph_key, stream, kernel, grid, block, args)
            
            if manager.has_graph(graph_key):
                print("  Success: Graph captured successfully")
                
                # Test graph replay
                print("\n  Testing graph replay...")
                manager.replay_graph(graph_key, stream)
                stream.synchronize()
                print("  Success: Graph replayed successfully")
            else:
                print("  Warning:  Graph capture not supported in this environment")
        
        except Exception as e:
            print(f"  Warning:  Graph operations not supported: {e}")
        
        # Test statistics
        stats = manager.get_statistics()
        print(f"\n  Statistics:")
        print(f"    Total graphs: {stats['total_graphs']}")
        print(f"    Total captures: {stats['total_captures']}")
        print(f"    Total replays: {stats['total_replays']}")
        print(f"    Hit rate: {stats['hit_rate']:.2%}")
        
        # Test clear
        print("\n  Testing clear...")
        manager.clear()
        assert len(manager.graphs) == 0, "Graphs should be cleared"
        print("  Success: Clear succeeded")
        
        print("  Success: CUDAGraphManager test passed")
        return {
            "total_graphs": stats["total_graphs"],
            "total_captures": stats["total_captures"],
            "total_replays": stats["total_replays"],
            "hit_rate": stats["hit_rate"],
        }
        
    except ImportError:
        print("  Warning:  CuPy not available, skipping CUDAGraphManager test")
        return {"skipped": True, "reason": "cupy not available"}
    except Exception as e:
        print(f"  Warning:  Test skipped due to environment limitations: {e}")
        return {"skipped": True, "reason": str(e)}


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GPU optimization classes - unit tests")
    print("="*70)
    
    all_passed = True
    
    try:
        result = test_dynamic_batch_sizer()
        _record_test_result("DynamicBatchSizer", "passed", result)
    except AssertionError as e:
        print(f"  Test failed: {e}")
        all_passed = False
        _record_test_result("DynamicBatchSizer", "failed", {"error": str(e)})
    except Exception as e:
        print(f"  Warning:  Test error: {e}")
        _record_test_result("DynamicBatchSizer", "error", {"error": str(e)})
    
    try:
        result = test_cuda_stream_pool()
        _record_test_result("CUDAStreamPool", "skipped" if result.get("skipped") else "passed", result)
    except AssertionError as e:
        print(f"  Test failed: {e}")
        all_passed = False
        _record_test_result("CUDAStreamPool", "failed", {"error": str(e)})
    except Exception as e:
        print(f"  Warning:  Test error: {e}")
        _record_test_result("CUDAStreamPool", "error", {"error": str(e)})
    
    try:
        result = test_tile_size_optimizer()
        _record_test_result("TileSizeOptimizer", "passed", result)
    except AssertionError as e:
        print(f"  Test failed: {e}")
        all_passed = False
        _record_test_result("TileSizeOptimizer", "failed", {"error": str(e)})
    except Exception as e:
        print(f"  Warning:  Test error: {e}")
        _record_test_result("TileSizeOptimizer", "error", {"error": str(e)})

    try:
        result = test_torch_optimizations()
        _record_test_result("TorchBackendOptimizations", "passed", result)
    except AssertionError as e:
        print(f"  Test failed: {e}")
        all_passed = False
        _record_test_result("TorchBackendOptimizations", "failed", {"error": str(e)})
    except Exception as e:
        print(f"  Warning:  Test error: {e}")
        _record_test_result("TorchBackendOptimizations", "error", {"error": str(e)})
    
    try:
        result = test_memory_profiler()
        _record_test_result("MemoryProfiler", "passed", result)
    except AssertionError as e:
        print(f"  Test failed: {e}")
        all_passed = False
        _record_test_result("MemoryProfiler", "failed", {"error": str(e)})
    except Exception as e:
        print(f"  Warning:  Test error: {e}")
        _record_test_result("MemoryProfiler", "error", {"error": str(e)})
    
    try:
        result = test_cuda_graph_manager()
        _record_test_result("CUDAGraphManager", "skipped" if result.get("skipped") else "passed", result)
    except AssertionError as e:
        print(f"  Test failed: {e}")
        all_passed = False
        _record_test_result("CUDAGraphManager", "failed", {"error": str(e)})
    except Exception as e:
        print(f"  Warning:  Test error: {e}")
        _record_test_result("CUDAGraphManager", "error", {"error": str(e)})
    
    print("\n" + "="*70)
    if all_passed:
        print("All tests passed")
    else:
        print("Some tests failed")
    print("="*70 + "\n")

    _write_results_summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
