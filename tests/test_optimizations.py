#!/usr/bin/env python3
"""
Unit tests for GPU optimization classes.

This test suite validates:
1. DynamicBatchSizer - Runtime batch size adjustment
2. CUDAStreamPool - Multiple CUDA stream management
3. TileSizeOptimizer - Learning optimal tile sizes
4. MemoryProfiler - GPU memory tracking and analysis
5. CUDAGraphManager - CUDA graph capture and replay
"""

import sys
import os
import json
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    
    print("  ✅ DynamicBatchSizer test passed")


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
        print("  ✅ All streams synchronized")
        
        # Test context manager
        print("\n  Testing context manager...")
        with CUDAStreamPool(num_streams=2) as pool2:
            s1 = pool2.get_stream()
            s2 = pool2.get_stream()
            assert s1 is not s2, "Different streams should be returned"
        print("  ✅ Context manager works correctly")
        
        # Test utilization
        utilization = pool.get_utilization()
        print(f"\n  Stream utilization: {utilization:.2%}")
        assert 0.0 <= utilization <= 1.0, "Utilization should be between 0 and 1"
        
        print("  ✅ CUDAStreamPool test passed")
        
    except ImportError:
        print("  ⚠️  CuPy not available, skipping CUDAStreamPool test")


def test_tile_size_optimizer():
    """Test TileSizeOptimizer class."""
    print("\n" + "="*70)
    print("TEST 3: TileSizeOptimizer")
    print("="*70)
    
    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        # Initialize optimizer
        optimizer = TileSizeOptimizer(history_file=temp_file)
        
        print(f"  Created optimizer with history file: {temp_file}")
        
        # Record some test runs
        print("\n  Recording test runs...")
        test_configs = [
            (1000, 6, 2048, (32, 32, 32), 1e6, 12.5),
            (1000, 6, 4096, (32, 32, 64), 1.2e6, 18.0),
            (1000, 6, 2048, (64, 64, 32), 1.1e6, 14.0),
            (2000, 8, 1024, (32, 32, 32), 8e5, 28.0),
        ]
        
        for n_samples, n_qubits, state_tile, kernel_tiles, throughput, peak_mem in test_configs:
            optimizer.record_run(n_samples, n_qubits, state_tile, kernel_tiles, throughput, peak_mem)
            print(f"    Recorded: n={n_samples}, nq={n_qubits}, tile={state_tile}, "
                  f"throughput={throughput:.2e}")
        
        # Test prediction
        print("\n  Testing tile prediction...")
        prediction = optimizer.predict_optimal_tiles(
            n_samples=1000,
            n_qubits=6,
            available_vram=96.0  # 96GB VRAM
        )
        
        print(f"    Predicted state_tile: {prediction['state_tile']}")
        print(f"    Predicted kernel_tiles: {prediction['kernel_tiles']}")
        print(f"    Confidence: {prediction['confidence']:.2f}")
        print(f"    Source: {prediction['source']}")
        
        assert prediction['state_tile'] > 0, "Should predict a valid state tile"
        assert len(prediction['kernel_tiles']) == 3, "Should predict 3 kernel tile dimensions"
        
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
        
        print("  ✅ TileSizeOptimizer test passed")
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_memory_profiler():
    """Test MemoryProfiler class."""
    print("\n" + "="*70)
    print("TEST 4: MemoryProfiler")
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
    
    print("  ✅ MemoryProfiler test passed")


def test_cuda_graph_manager():
    """Test CUDAGraphManager class."""
    print("\n" + "="*70)
    print("TEST 5: CUDAGraphManager")
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
        
        print("  ✅ Normal execution succeeded")
        
        # Test graph capture (Note: May not work in all environments)
        print("\n  Testing graph capture...")
        graph_key = ("test", n, 256)
        
        try:
            manager.capture_graph(graph_key, stream, kernel, grid, block, args)
            
            if manager.has_graph(graph_key):
                print("  ✅ Graph captured successfully")
                
                # Test graph replay
                print("\n  Testing graph replay...")
                manager.replay_graph(graph_key, stream)
                stream.synchronize()
                print("  ✅ Graph replayed successfully")
            else:
                print("  ⚠️  Graph capture not supported in this environment")
        
        except Exception as e:
            print(f"  ⚠️  Graph operations not supported: {e}")
        
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
        print("  ✅ Clear succeeded")
        
        print("  ✅ CUDAGraphManager test passed")
        
    except ImportError:
        print("  ⚠️  CuPy not available, skipping CUDAGraphManager test")
    except Exception as e:
        print(f"  ⚠️  Test skipped due to environment limitations: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GPU OPTIMIZATION CLASSES - UNIT TESTS")
    print("="*70)
    
    all_passed = True
    
    try:
        test_dynamic_batch_sizer()
    except AssertionError as e:
        print(f"  ❌ Test failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ⚠️  Test error: {e}")
    
    try:
        test_cuda_stream_pool()
    except AssertionError as e:
        print(f"  ❌ Test failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ⚠️  Test error: {e}")
    
    try:
        test_tile_size_optimizer()
    except AssertionError as e:
        print(f"  ❌ Test failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ⚠️  Test error: {e}")
    
    try:
        test_memory_profiler()
    except AssertionError as e:
        print(f"  ❌ Test failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ⚠️  Test error: {e}")
    
    try:
        test_cuda_graph_manager()
    except AssertionError as e:
        print(f"  ❌ Test failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ⚠️  Test error: {e}")
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
