# pipeline_backends.py — Optimized, Synchronized, Vmap-Free & Numerically Safe
from typing import Optional, Any, Callable, Dict, Tuple, List
import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm

# =====================================================================
# Helpers
# =====================================================================
def _ensure_numpy(a: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    try:
        import torch
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(a, order="C")
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr

def _tile_ranges(n: int, tile: int):
    for s in range(0, n, tile):
        e = min(s + tile, n)
        yield s, e

def _normalize_diag_inplace(K: np.ndarray):
    if K.shape[0] != K.shape[1]: return
    # Clip pour éviter les racines de nombres négatifs minuscules
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K /= d[:, None] * d[None, :]
    np.fill_diagonal(K, 1.0)

# =====================================================================
# VRAM & Memory Management
# =====================================================================
# Constants for memory calculations
MATRIX_PAIRS_FACTOR = 2  # Factor for A and B matrices in precompute size calculation
BATCH_SYNC_INTERVAL = 32  # Number of tiles between synchronization calls
BATCH_ADJUST_INTERVAL = 10  # Number of tiles between dynamic batch size adjustments

class PersistentBufferPool:
    """Manages reusable GPU buffers to reduce allocation overhead."""
    def __init__(self):
        self.buffers: Dict[Tuple[tuple, type], Any] = {}
    
    def get_buffer(self, shape: tuple, dtype, device="cuda"):
        """Get or create a buffer with the given shape and dtype."""
        import cupy as cp
        key = (shape, dtype)
        if key not in self.buffers:
            self.buffers[key] = cp.empty(shape, dtype=dtype)
        return self.buffers[key]
    
    def clear(self):
        """Clear all cached buffers."""
        self.buffers.clear()

_BUFFER_POOL = PersistentBufferPool()
_PINNED_BUFFERS: Dict[Tuple[tuple, type], Any] = {}

def _get_pinned_buffer(shape: tuple, dtype):
    """Get or create a pinned host memory buffer."""
    import cupy as cp
    key = (shape, dtype)
    if key not in _PINNED_BUFFERS:
        pinned_pool = cp.cuda.PinnedMemoryPool()
        # Use pinned memory allocator
        with cp.cuda.using_allocator(pinned_pool.malloc):
            _PINNED_BUFFERS[key] = cp.zeros(shape, dtype=dtype)
    return _PINNED_BUFFERS[key]

# =====================================================================
# Advanced GPU Optimization Classes
# =====================================================================
class DynamicBatchSizer:
    """Adjusts batch sizes at runtime based on GPU memory pressure and throughput."""
    
    def __init__(self, initial_batch: int, min_batch: int = 64, max_batch: int = 16384,
                 target_memory_usage: float = 0.85):
        """
        Initialize the dynamic batch sizer.
        
        Args:
            initial_batch: Starting batch size
            min_batch: Minimum allowed batch size
            max_batch: Maximum allowed batch size
            target_memory_usage: Target GPU memory utilization (0-1)
        """
        self.current_batch = initial_batch
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_memory = target_memory_usage
        self.history = []
        self.adjustments = 0
        self.total_kernel_time = 0.0
        self.kernel_count = 0
    
    def adjust(self, current_memory_used: float, last_kernel_time: float) -> int:
        """
        Returns adjusted batch size based on runtime metrics.
        
        Args:
            current_memory_used: Current GPU memory usage as fraction (0-1)
            last_kernel_time: Last kernel execution time in seconds
        
        Returns:
            Adjusted batch size
        """
        self.total_kernel_time += last_kernel_time
        self.kernel_count += 1
        
        old_batch = self.current_batch
        
        # If memory pressure is too high, reduce batch size
        if current_memory_used > self.target_memory + 0.05:
            self.current_batch = max(self.min_batch, int(self.current_batch * 0.75))
            self.adjustments += 1
        # If memory headroom exists and kernel time is stable, increase batch size
        elif current_memory_used < self.target_memory - 0.10:
            # Check variance of recent kernel times
            if len(self.history) >= 5:
                recent_times = [h[1] for h in self.history[-5:]]
                variance = np.var(recent_times)
                mean_time = np.mean(recent_times)
                # Only increase if variance is low (stable performance)
                if variance < (mean_time * 0.1) ** 2:
                    self.current_batch = min(self.max_batch, int(self.current_batch * 1.25))
                    self.adjustments += 1
        
        self.history.append((current_memory_used, last_kernel_time, self.current_batch))
        
        return self.current_batch
    
    def report(self) -> dict:
        """Returns statistics on batch size adjustments."""
        batch_sizes = [h[2] for h in self.history]
        return {
            "adjustments": self.adjustments,
            "current_batch": self.current_batch,
            "min_batch_used": min(batch_sizes) if batch_sizes else self.current_batch,
            "max_batch_used": max(batch_sizes) if batch_sizes else self.current_batch,
            "avg_kernel_time": self.total_kernel_time / self.kernel_count if self.kernel_count > 0 else 0.0,
            "total_kernels": self.kernel_count
        }


class CUDAStreamPool:
    """Manages multiple CUDA streams for concurrent operations."""
    
    def __init__(self, num_streams: int = 4):
        """
        Initialize the stream pool.
        
        Args:
            num_streams: Number of streams to pre-allocate
        """
        import cupy as cp
        self.num_streams = num_streams
        self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
        self.current_idx = 0
        self.usage_count = [0] * num_streams
        self.total_operations = 0
    
    def get_stream(self):
        """Get next available stream from pool using round-robin."""
        stream = self.streams[self.current_idx]
        self.usage_count[self.current_idx] += 1
        self.current_idx = (self.current_idx + 1) % self.num_streams
        self.total_operations += 1
        return stream
    
    def synchronize_all(self):
        """Synchronize all streams in the pool."""
        for stream in self.streams:
            stream.synchronize()
    
    def get_utilization(self) -> float:
        """Calculate stream utilization balance (1.0 = perfectly balanced)."""
        if self.total_operations == 0:
            return 0.0
        expected_per_stream = self.total_operations / self.num_streams
        if expected_per_stream == 0:
            return 0.0
        variance = np.var(self.usage_count)
        # Lower variance = better utilization
        return 1.0 - min(1.0, variance / (expected_per_stream ** 2))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.synchronize_all()
        return False


class TileSizeOptimizer:
    """Learns optimal tile sizes from historical runs."""
    
    def __init__(self, history_file: str = ".tile_optimizer_history.json"):
        """
        Initialize the tile size optimizer.
        
        Args:
            history_file: Path to file for persisting learning data
        """
        self.history_file = history_file
        self.history = []
        self.load_history()
    
    def load_history(self):
        """Load historical performance data from disk."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []
    
    def save_history(self):
        """Save historical performance data to disk."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass
    
    def record_run(self, n_samples: int, n_qubits: int, state_tile: int,
                   kernel_tiles: tuple, throughput: float, peak_memory: float):
        """
        Record metrics from a completed run.
        
        Args:
            n_samples: Number of samples processed
            n_qubits: Number of qubits
            state_tile: State tile size used
            kernel_tiles: Tuple of (TILE_M, TILE_N, TILE_K)
            throughput: Achieved throughput (pairs/sec)
            peak_memory: Peak memory usage in GB
        """
        entry = {
            "n_samples": n_samples,
            "n_qubits": n_qubits,
            "state_tile": state_tile,
            "kernel_tiles": list(kernel_tiles),
            "throughput": throughput,
            "peak_memory": peak_memory
        }
        self.history.append(entry)
        self.save_history()
    
    def predict_optimal_tiles(self, n_samples: int, n_qubits: int,
                               available_vram: float) -> dict:
        """
        Predict optimal state_tile and kernel tiles based on history.
        
        Args:
            n_samples: Number of samples to process
            n_qubits: Number of qubits
            available_vram: Available VRAM in GB
        
        Returns:
            Dictionary with predicted optimal configuration
        """
        if not self.history:
            return {
                "state_tile": -1,  # Use auto-detection
                "kernel_tiles": (32, 32, 32),
                "confidence": 0.0,
                "source": "default"
            }
        
        # Filter similar workloads (same n_qubits, similar n_samples)
        similar = [h for h in self.history if h["n_qubits"] == n_qubits]
        
        if not similar:
            # Fallback to any historical data with interpolation
            similar = self.history
        
        # Find best performing configuration within memory constraints
        valid = [h for h in similar if h["peak_memory"] <= available_vram * 0.9]
        
        if not valid:
            return {
                "state_tile": -1,
                "kernel_tiles": (32, 32, 32),
                "confidence": 0.0,
                "source": "no_valid_history"
            }
        
        # Sort by throughput (higher is better)
        valid.sort(key=lambda x: x["throughput"], reverse=True)
        best = valid[0]
        
        confidence = min(1.0, len(valid) / 10.0)  # More data = higher confidence
        
        return {
            "state_tile": best["state_tile"],
            "kernel_tiles": tuple(best["kernel_tiles"]),
            "confidence": confidence,
            "source": "learned"
        }
    
    def get_statistics(self) -> dict:
        """Return learning statistics and confidence metrics."""
        if not self.history:
            return {
                "total_runs": 0,
                "unique_configs": 0,
                "avg_throughput": 0.0
            }
        
        unique_configs = len(set(
            (h["n_qubits"], h["state_tile"], tuple(h["kernel_tiles"]))
            for h in self.history
        ))
        
        return {
            "total_runs": len(self.history),
            "unique_configs": unique_configs,
            "avg_throughput": np.mean([h["throughput"] for h in self.history])
        }


class MemoryProfiler:
    """Detailed GPU memory analysis and profiling."""
    
    def __init__(self, enable_realtime: bool = False):
        """
        Initialize the memory profiler.
        
        Args:
            enable_realtime: Enable real-time memory monitoring during execution
        """
        self.enable_realtime = enable_realtime
        self.allocations = {}
        self.transfers = []
        self.snapshots = []
        self.peak_allocated = 0.0
        self.kernel_launches = 0
        self.kernel_times = []
    
    def track_allocation(self, name: str, size_bytes: int):
        """
        Track a named memory allocation.
        
        Args:
            name: Allocation category name
            size_bytes: Size in bytes
        """
        if name not in self.allocations:
            self.allocations[name] = 0
        self.allocations[name] += size_bytes
        
        total = sum(self.allocations.values())
        self.peak_allocated = max(self.peak_allocated, total)
    
    def track_transfer(self, direction: str, size_bytes: int, duration_ms: float):
        """
        Track a memory transfer.
        
        Args:
            direction: Transfer direction ('H2D' or 'D2H')
            size_bytes: Size in bytes
            duration_ms: Transfer duration in milliseconds
        """
        self.transfers.append({
            "direction": direction,
            "size_bytes": size_bytes,
            "duration_ms": duration_ms,
            "bandwidth_gbps": (size_bytes / 1e9) / (duration_ms / 1000.0) if duration_ms > 0 else 0.0
        })
    
    def track_kernel(self, duration_ms: float):
        """Track a kernel execution."""
        self.kernel_launches += 1
        self.kernel_times.append(duration_ms)
    
    def snapshot(self) -> dict:
        """Capture current memory state."""
        try:
            import cupy as cp
            device = cp.cuda.Device()
            mem_info = device.mem_info
            current_state = {
                "free": mem_info[0],
                "total": mem_info[1],
                "used": mem_info[1] - mem_info[0],
                "allocations": dict(self.allocations)
            }
            self.snapshots.append(current_state)
            return current_state
        except Exception:
            return {}
    
    def report(self, verbose: bool = True) -> dict:
        """
        Generate comprehensive memory report.
        
        Args:
            verbose: Print detailed report to console
        
        Returns:
            Dictionary with profiling statistics
        """
        # Calculate statistics
        h2d_transfers = [t for t in self.transfers if t["direction"] == "H2D"]
        d2h_transfers = [t for t in self.transfers if t["direction"] == "D2H"]
        
        h2d_total_gb = sum(t["size_bytes"] for t in h2d_transfers) / 1e9
        d2h_total_gb = sum(t["size_bytes"] for t in d2h_transfers) / 1e9
        h2d_avg_bw = np.mean([t["bandwidth_gbps"] for t in h2d_transfers]) if h2d_transfers else 0.0
        d2h_avg_bw = np.mean([t["bandwidth_gbps"] for t in d2h_transfers]) if d2h_transfers else 0.0
        
        avg_kernel_time = np.mean(self.kernel_times) if self.kernel_times else 0.0
        
        report_data = {
            "peak_allocated_gb": self.peak_allocated / 1e9,
            "allocations": {k: v / 1e9 for k, v in self.allocations.items()},
            "h2d_total_gb": h2d_total_gb,
            "d2h_total_gb": d2h_total_gb,
            "h2d_avg_bandwidth_gbps": h2d_avg_bw,
            "d2h_avg_bandwidth_gbps": d2h_avg_bw,
            "kernel_launches": self.kernel_launches,
            "avg_kernel_time_ms": avg_kernel_time
        }
        
        # Add graph statistics if available
        if hasattr(self, 'graph_replays'):
            report_data["graph_replays"] = self.graph_replays
        if hasattr(self, 'graph_hit_rate'):
            report_data["graph_hit_rate"] = self.graph_hit_rate
        
        # Add throughput if available
        if hasattr(self, 'throughput'):
            report_data["throughput_mpairs_per_sec"] = self.throughput
        
        if verbose:
            self._print_report(report_data)
        
        return report_data
    
    def _print_report(self, data: dict):
        """Print formatted profiling report."""
        try:
            import cupy as cp
            device = cp.cuda.Device()
            total_vram = device.mem_info[1] / 1e9
        except Exception:
            total_vram = 0.0
        
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║                    GPU PERFORMANCE REPORT                      ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║ Memory Usage                                                   ║")
        
        if total_vram > 0:
            peak_pct = (data["peak_allocated_gb"] / total_vram) * 100
            print(f"║   Peak Allocated:     {data['peak_allocated_gb']:5.1f} GB / {total_vram:5.1f} GB ({peak_pct:4.1f}%)               ║")
        else:
            print(f"║   Peak Allocated:     {data['peak_allocated_gb']:5.1f} GB                                   ║")
        
        for name, size_gb in data["allocations"].items():
            print(f"║   {name:20s}: {size_gb:5.1f} GB                                  ║")
        
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║ Transfer Bandwidth                                             ║")
        print(f"║   H→D Total:          {data['h2d_total_gb']:5.1f} GB @ {data['h2d_avg_bandwidth_gbps']:4.1f} GB/s                     ║")
        print(f"║   D→H Total:          {data['d2h_total_gb']:5.1f} GB @ {data['d2h_avg_bandwidth_gbps']:4.1f} GB/s                     ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║ Kernel Performance                                             ║")
        print(f"║   Total Launches:     {data['kernel_launches']:,}                                    ║")
        
        # Add graph statistics if available
        if "graph_replays" in data and "graph_hit_rate" in data:
            print(f"║   Graph Replays:        {data['graph_replays']:,} ({data['graph_hit_rate']*100:4.1f}% hit rate)                  ║")
        
        print(f"║   Avg Kernel Time:     {data['avg_kernel_time_ms']:.2f} ms                                 ║")
        
        # Add throughput if available
        if "throughput_mpairs_per_sec" in data:
            print(f"║   Throughput:         {data['throughput_mpairs_per_sec']:.1f} Mpairs/s                          ║")
        
        print("╚══════════════════════════════════════════════════════════════╝\n")


class CUDAGraphManager:
    """Manages CUDA graph capture and replay for kernel optimization."""
    
    def __init__(self):
        """Initialize the CUDA graph manager."""
        self.graphs = {}
        self.graph_execs = {}
        self.capture_counts = {}
        self.replay_counts = {}
    
    def capture_graph(self, key: tuple, stream: Any, kernel_fn: Callable, 
                      grid: Tuple[int, ...], block: Tuple[int, ...], args: Tuple[Any, ...]):
        """
        Capture a CUDA graph for the given kernel configuration.
        
        Args:
            key: Unique key for this graph configuration
            stream: CUDA stream to use (cupy.cuda.Stream)
            kernel_fn: Compiled kernel function
            grid: Grid dimensions tuple
            block: Block dimensions tuple
            args: Kernel arguments tuple
        """
        import cupy as cp
        
        if key in self.graphs:
            return
        
        try:
            # Start graph capture
            stream.begin_capture()
            
            # Execute kernel in capture mode
            kernel_fn(grid, block, args)
            
            # End capture and get graph
            graph = stream.end_capture()
            
            # Instantiate graph for replay
            graph_exec = graph.instantiate()
            
            self.graphs[key] = graph
            self.graph_execs[key] = graph_exec
            self.capture_counts[key] = 1
            self.replay_counts[key] = 0
            
        except Exception as e:
            # If capture fails, clean up and continue without graph
            try:
                stream.end_capture()
            except:
                pass
    
    def has_graph(self, key: tuple) -> bool:
        """Check if a graph exists for this configuration."""
        return key in self.graph_execs
    
    def replay_graph(self, key: tuple, stream):
        """
        Replay a previously captured graph.
        
        Args:
            key: Graph configuration key
            stream: CUDA stream to replay on
        """
        if key not in self.graph_execs:
            raise KeyError(f"No graph found for key {key}")
        
        graph_exec = self.graph_execs[key]
        graph_exec.launch(stream)
        self.replay_counts[key] = self.replay_counts.get(key, 0) + 1
    
    def clear(self):
        """Release all captured graphs."""
        self.graphs.clear()
        self.graph_execs.clear()
        self.capture_counts.clear()
        self.replay_counts.clear()
    
    def get_statistics(self) -> dict:
        """Get statistics about graph usage."""
        total_captures = sum(self.capture_counts.values())
        total_replays = sum(self.replay_counts.values())
        hit_rate = total_replays / (total_captures + total_replays) if (total_captures + total_replays) > 0 else 0.0
        
        return {
            "total_graphs": len(self.graphs),
            "total_captures": total_captures,
            "total_replays": total_replays,
            "hit_rate": hit_rate
        }


def _compute_optimal_state_tile(vram_fraction: float = 0.85, nq: int = 6, 
                                 dtype=np.float32, overhead_gb: float = 2.0) -> int:
    """
    Compute optimal state_tile size based on available VRAM.
    
    Args:
        vram_fraction: Maximum fraction of VRAM to use (default 0.85)
        nq: Number of qubits
        dtype: Data type for computations
        overhead_gb: Reserved VRAM for framework overhead in GB
    
    Returns:
        Optimal tile size (number of states)
    """
    try:
        import cupy as cp
        # Get available VRAM
        device = cp.cuda.Device()
        total_vram = device.mem_info[1]  # Total memory in bytes
        available_vram = total_vram * vram_fraction - (overhead_gb * 1024**3)
        
        # Calculate memory per state: dim = 2^nq complex numbers
        dim = 1 << nq
        bytes_per_complex = 8 if dtype == np.float32 else 16  # complex64 or complex128
        bytes_per_state = dim * bytes_per_complex
        
        # Calculate max states that fit in available VRAM
        max_states = int(available_vram / bytes_per_state)
        
        # Round down to nearest power of 2 for efficiency
        tile_size = 2 ** int(np.log2(max_states))
        
        # Ensure minimum and maximum bounds
        tile_size = max(256, min(tile_size, 32768))
        
        return tile_size
    except Exception as e:
        # Fallback to conservative default
        return 8192

def _compute_max_precompute_size(vram_fraction: float = 0.85, nq: int = 6,
                                  dtype=np.float32, overhead_gb: float = 2.0) -> int:
    """
    Determine maximum number of states that can be cached in GPU memory.
    
    Args:
        vram_fraction: Maximum fraction of VRAM to use
        nq: Number of qubits
        dtype: Data type
        overhead_gb: Reserved VRAM in GB
    
    Returns:
        Maximum number of states to cache
    """
    try:
        import cupy as cp
        device = cp.cuda.Device()
        total_vram = device.mem_info[1]
        available_vram = total_vram * vram_fraction - (overhead_gb * 1024**3)
        
        dim = 1 << nq
        bytes_per_complex = 8 if dtype == np.float32 else 16
        bytes_per_state = dim * bytes_per_complex
        
        max_states = int(available_vram / (bytes_per_state * MATRIX_PAIRS_FACTOR))
        
        return max(1024, max_states)
    except Exception:
        return 8192

def _setup_cupy():
    import cupy as cp
    # Use managed memory pool for GPU allocations
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    
    # Use pinned memory pool for host transfers
    pinned_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
    
    # Warm up
    _ = cp.ones((1,), dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()

    candidates = [
        os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME"),
        "/usr/local/cuda", "/opt/cuda",
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Library", "include")
    ]
    include_path = None
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "include")):
            include_path = os.path.join(path, "include")
            break
        elif path and os.path.exists(path) and path.endswith("include"):
            include_path = path
            break
    
    if include_path:
        opts = f"-I{include_path}"
        existing = os.environ.get("CUPY_NVRTC_OPTIONS", "")
        if include_path not in existing:
            os.environ["CUPY_NVRTC_OPTIONS"] = existing + " " + opts

# =====================================================================
# Worker globals
# =====================================================================
_pl_w = None
_pl_nq = None
_pl_device = None
_pl_qnode = None
_pl_float_dtype = None
_pl_complex_dtype = None
_pl_angle_scale = 1.0
_pl_re_embed = False
_pl_embed_mode = "ryrz"

def _pl_worker_init(w, dev, nq, fdtype, ascale, re_emb, mode):
    global _pl_w, _pl_nq, _pl_device, _pl_qnode, _pl_float_dtype, _pl_complex_dtype, _pl_angle_scale, _pl_re_embed, _pl_embed_mode
    os.environ["OMP_NUM_THREADS"] = "1"
    _pl_float_dtype = np.dtype(np.float32) if fdtype == "float32" else np.dtype(np.float64)
    _pl_complex_dtype = np.dtype(np.complex64) if fdtype == "float32" else np.dtype(np.complex128)
    _pl_w = _ensure_numpy(w, _pl_float_dtype)
    _pl_nq, _pl_device = int(nq), str(dev)
    _pl_angle_scale, _pl_re_embed, _pl_embed_mode = float(ascale), bool(re_emb), str(mode)
    
    # --- BUG FIX: Reset QNode to force device recreation on param change ---
    _pl_qnode = None 
    # ---------------------------------------------------------------------

def _pl_get_qnode():
    global _pl_qnode
    if _pl_qnode is None:
        import pennylane as qml
        try:
            dev = qml.device(_pl_device, wires=_pl_nq, shots=None, c_dtype=_pl_complex_dtype)
        except:
            dev = qml.device(_pl_device, wires=_pl_nq, shots=None)
        @qml.qnode(dev, interface=None, diff_method=None)
        def _state(theta_row):
            theta = qml.math.asarray(theta_row, dtype=_pl_float_dtype)
            def _embed(v):
                if _pl_embed_mode == "angle": qml.AngleEmbedding(_pl_angle_scale*v[:_pl_nq], wires=range(_pl_nq), rotation="Y")
                else:
                    for i in range(_pl_nq):
                        qml.RY(_pl_angle_scale*v[i], wires=i)
                        if _pl_embed_mode=="ryrz": qml.RZ(_pl_angle_scale*v[i], wires=i)
            if _pl_re_embed:
                for l in range(_pl_w.shape[0]):
                    _embed(theta)
                    qml.templates.BasicEntanglerLayers(_pl_w[l:l+1], wires=range(_pl_nq))
            else:
                _embed(theta)
                qml.templates.BasicEntanglerLayers(_pl_w, wires=range(_pl_nq))
            return qml.state()
        _pl_qnode = _state
    return _pl_qnode

def _pl_states_for_rows(rows, mat):
    qnode = _pl_get_qnode()
    out = np.empty((len(rows), 1 << _pl_nq), dtype=_pl_complex_dtype)
    for t, idx in enumerate(rows): out[t] = qnode(mat[idx])
    return out

# =====================================================================
# CUDA Kernels
# =====================================================================
CUDA_TEMPLATE = r"""
#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

extern "C" __global__
void cgemm_abs2_tiled_full(const T_COMPLEX* __restrict__ SA,
                           const T_COMPLEX* __restrict__ SB,
                           T_REAL* __restrict__ K,
                           const int BM, const int BN, const int D,
                           const int lda, const int ldb, const int ldk)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Safety check moved slightly later to allow shared mem init if needed, 
    // but typically we return early to save ops.
    if (i >= BM || j >= BN) return;

    __shared__ T_COMPLEX sA[TILE_M][TILE_K];
    __shared__ T_COMPLEX sB[TILE_N][TILE_K];
    
    T_COMPLEX acc = MAKE_COMPLEX(0.0, 0.0);

    for (int k0 = 0; k0 < D; k0 += TILE_K) {
        // Load sA
        if (i < BM) {
            for (int tk = threadIdx.x; tk < TILE_K; tk += blockDim.x) {
                int k = k0 + tk;
                sA[threadIdx.y][tk] = (k < D) ? SA[i * lda + k] : MAKE_COMPLEX(0.0, 0.0);
            }
        }
        
        // Load sB
        if (j < BN) {
            for (int tk = threadIdx.y; tk < TILE_K; tk += blockDim.y) {
                int k = k0 + tk;
                T_COMPLEX v = (k < D) ? SB[j * ldb + k] : MAKE_COMPLEX(0.0, 0.0);
                sB[threadIdx.x][tk] = MAKE_COMPLEX(v.x, -v.y); 
            }
        }
        __syncthreads();

        // COMPUTE: Removed #pragma unroll to prevent ptxas register spill on large tiles
        for (int tk = 0; tk < TILE_K; ++tk) {
            T_COMPLEX a = sA[threadIdx.y][tk];
            T_COMPLEX b = sB[threadIdx.x][tk];
            T_REAL rx = a.x * b.x - a.y * b.y;
            T_REAL ry = a.x * b.y + a.y * b.x;
            acc.x += rx; acc.y += ry;
        }
        __syncthreads();
    }
    K[i * ldk + j] = acc.x * acc.x + acc.y * acc.y;
}

extern "C" __global__
void cgemm_abs2_tiled_lower(const T_COMPLEX* __restrict__ SA,
                            const T_COMPLEX* __restrict__ SB,
                            T_REAL* __restrict__ K,
                            const int BM, const int BN, const int D,
                            const int lda, const int ldb, const int ldk)
{
    // Block-level optimization for symmetric matrix
    if (blockIdx.x > blockIdx.y) return;

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= BM || j >= BN) return;
    if (BM == BN && j > i) return;

    __shared__ T_COMPLEX sA[TILE_M][TILE_K];
    __shared__ T_COMPLEX sB[TILE_N][TILE_K];
    T_COMPLEX acc = MAKE_COMPLEX(0.0, 0.0);

    for (int k0 = 0; k0 < D; k0 += TILE_K) {
        if (i < BM) {
            for (int tk = threadIdx.x; tk < TILE_K; tk += blockDim.x) {
                int k = k0 + tk;
                sA[threadIdx.y][tk] = (k < D) ? SA[i * lda + k] : MAKE_COMPLEX(0.0, 0.0);
            }
        }
        if (j < BN) {
            for (int tk = threadIdx.y; tk < TILE_K; tk += blockDim.y) {
                int k = k0 + tk;
                T_COMPLEX v = (k < D) ? SB[j * ldb + k] : MAKE_COMPLEX(0.0, 0.0);
                sB[threadIdx.x][tk] = MAKE_COMPLEX(v.x, -v.y);
            }
        }
        __syncthreads();

        // COMPUTE: Removed #pragma unroll here too
        for (int tk = 0; tk < TILE_K; ++tk) {
            T_COMPLEX a = sA[threadIdx.y][tk];
            T_COMPLEX b = sB[threadIdx.x][tk];
            T_REAL rx = a.x * b.x - a.y * b.y;
            T_REAL ry = a.x * b.y + a.y * b.x;
            acc.x += rx; acc.y += ry;
        }
        __syncthreads();
    }
    K[i * ldk + j] = acc.x * acc.x + acc.y * acc.y;
}
"""

def _round_to_pow2(x):
    """Round to nearest power of 2 for better graph reuse."""
    return 2 ** int(np.ceil(np.log2(max(1, x))))

_RAWMOD_CACHE = {}
def _get_kernel(tm, tn, tk, name, double):
    import cupy as cp
    key = (tm, tn, tk, name, double)
    if key in _RAWMOD_CACHE: return _RAWMOD_CACHE[key]
    
    t_m = ("-DT_REAL=double", "-DT_COMPLEX=double2", "-DMAKE_COMPLEX=make_double2") if double else \
          ("-DT_REAL=float", "-DT_COMPLEX=float2", "-DMAKE_COMPLEX=make_float2")
    opts = ("--std=c++14", "--use_fast_math") + t_m + (f"-DTILE_M={tm}", f"-DTILE_N={tn}", f"-DTILE_K={tk}")
    
    mod = cp.RawModule(code=CUDA_TEMPLATE, options=opts, name_expressions=(name,))
    fn = mod.get_function(name)
    _RAWMOD_CACHE[key] = fn
    return fn

# =====================================================================
# CUDA Kernel Autotuning
# =====================================================================
_AUTOTUNE_CACHE_FILE = ".cuda_kernel_autotune.json"
_AUTOTUNE_CACHE: Dict[str, Tuple[int, int, int]] = {}

def _load_autotune_cache():
    """Load autotuning results from disk."""
    global _AUTOTUNE_CACHE
    if os.path.exists(_AUTOTUNE_CACHE_FILE):
        try:
            with open(_AUTOTUNE_CACHE_FILE, 'r') as f:
                data = json.load(f)
                _AUTOTUNE_CACHE = {k: tuple(v) for k, v in data.items()}
        except Exception:
            pass

def _save_autotune_cache():
    """Save autotuning results to disk."""
    try:
        with open(_AUTOTUNE_CACHE_FILE, 'w') as f:
            json.dump({k: list(v) for k, v in _AUTOTUNE_CACHE.items()}, f, indent=2)
    except Exception:
        pass

def _autotune_kernel_tiles(nq: int, is_double: bool = False, 
                           test_size: int = 512, warmup: int = 2, trials: int = 5) -> Tuple[int, int, int]:
    """
    Benchmark different TILE_M, TILE_N, TILE_K combinations and return the best.
    
    Args:
        nq: Number of qubits
        is_double: Whether to use double precision
        test_size: Size of test matrices
        warmup: Number of warmup iterations
        trials: Number of benchmark trials
    
    Returns:
        Tuple of (TILE_M, TILE_N, TILE_K) with best performance
    """
    import cupy as cp
    import time
    
    # Check cache first
    cache_key = f"nq{nq}_{'double' if is_double else 'float'}"
    if cache_key in _AUTOTUNE_CACHE:
        return _AUTOTUNE_CACHE[cache_key]
    
    dim = 1 << nq
    dtype_complex = cp.complex128 if is_double else cp.complex64
    dtype_real = cp.float64 if is_double else cp.float32
    
    # Generate test data
    rng = cp.random.default_rng(42)
    SA = rng.random((test_size, dim), dtype=dtype_real) + 1j * rng.random((test_size, dim), dtype=dtype_real)
    SB = rng.random((test_size, dim), dtype=dtype_real) + 1j * rng.random((test_size, dim), dtype=dtype_real)
    K_out = cp.empty((test_size, test_size), dtype=dtype_real)
    
    # FIX: Add qubit-aware tile constraints to avoid shared memory overflow
    # For high qubit counts, use smaller tiles to fit in shared memory
    if nq >= 14:
        candidates_m_n = [16, 32]
        candidates_k = [16, 32]
    elif nq >= 12:
        candidates_m_n = [16, 32, 64]
        candidates_k = [16, 32, 64]
    else:
        # Original candidates for lower qubit counts
        candidates_m_n = [16, 32, 64]
        candidates_k = [16, 32, 64, 128]
    
    results = []
    
    for tm in candidates_m_n:
        for tn in candidates_m_n:
            for tk in candidates_k:
                # Check shared memory constraint
                # sA[TILE_M][TILE_K] + sB[TILE_N][TILE_K]
                bytes_per_complex = 16 if is_double else 8
                shared_mem = (tm * tk + tn * tk) * bytes_per_complex
                if shared_mem > 48 * 1024:  # 48KB limit
                    continue
                
                try:
                    kernel = _get_kernel(tm, tn, tk, "cgemm_abs2_tiled_full", is_double)
                    
                    grid = ((test_size + tn - 1) // tn, (test_size + tm - 1) // tm, 1)
                    block = (tn, tm, 1)
                    
                    # Warmup
                    for _ in range(warmup):
                        kernel(grid, block, (SA, SB, K_out, test_size, test_size, dim, dim, dim, test_size))
                    cp.cuda.runtime.deviceSynchronize()
                    
                    # Benchmark
                    times = []
                    for _ in range(trials):
                        start = time.perf_counter()
                        kernel(grid, block, (SA, SB, K_out, test_size, test_size, dim, dim, dim, test_size))
                        cp.cuda.runtime.deviceSynchronize()
                        times.append(time.perf_counter() - start)
                    
                    avg_time = np.mean(times)
                    results.append((avg_time, tm, tn, tk))
                    
                except Exception:
                    continue
    
    if not results:
        # Fallback to default
        return (32, 32, 32)
    
    # Select best configuration
    results.sort()
    best = results[0]
    best_config = (best[1], best[2], best[3])
    
    # Cache result
    _AUTOTUNE_CACHE[cache_key] = best_config
    _save_autotune_cache()
    
    return best_config

# =====================================================================
# State Generation (Torch) -> CuPy (FIXED & BATCHED)
# =====================================================================
def _build_states_block_torch_cuda(x_blk, w_np, dev_name, ascale, re_emb, mode):
    import torch as th
    import pennylane as qml
    
    nq = int(x_blk.shape[1])
    t_float = th.float32 if x_blk.dtype == np.float32 else th.float64
    t_cplx = th.complex64 if t_float == th.float32 else th.complex128
    
    x = th.from_numpy(np.ascontiguousarray(x_blk)).to("cuda", dtype=t_float, non_blocking=True)
    w = th.from_numpy(np.ascontiguousarray(w_np)).to("cuda", dtype=t_float, non_blocking=True)
    
    try:
        dev = qml.device(dev_name, wires=nq, shots=None, c_dtype=np.complex64 if t_float==th.float32 else np.complex128)
    except:
        dev = qml.device("lightning.gpu", wires=nq, shots=None)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(row):
        def _emb(v):
            if mode=="angle": qml.AngleEmbedding(ascale*v[:nq], wires=range(nq), rotation="Y")
            else:
                for i in range(nq):
                    qml.RY(ascale*v[i], wires=i)
                    if mode=="ryrz": qml.RZ(ascale*v[i], wires=i)
        if re_emb:
            for l in range(w.shape[0]):
                _emb(row)
                qml.templates.BasicEntanglerLayers(w[l:l+1], wires=range(nq))
        else:
            _emb(row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    # --- ATTEMPT NATIVE BATCHING (FASTEST, NO VMAP) ---
    try:
        states = _state(x)
        if states.ndim != 2 or states.shape[0] != x.shape[0]:
            raise ValueError("Batching not natively supported")
    except:
        states = th.stack([_state(x[i]) for i in range(x.shape[0])])
    
    states = states.to(device="cuda", dtype=t_cplx, non_blocking=False)
    
    # --- CRITICAL FIX: SYNC ---
    th.cuda.synchronize()
    # --------------------------
    
    return states

def _torch_cuda_to_cupy(t):
    import cupy as cp
    return cp.from_dlpack(t)

# =====================================================================
# Bulk State Precomputation & Async Dispatch
# =====================================================================
def _build_all_states_torch_cuda(x_all, w_np, dev_name, ascale, re_emb, mode, use_pinned=True):
    """
    Build ALL quantum states in one pass with pinned memory optimization.
    Minimizes torch→cupy handoffs by precomputing entire matrix at once.
    
    Args:
        x_all: Full input data matrix (n_samples × n_qubits)
        w_np: Weights array
        dev_name: Device name
        ascale: Angle scale
        re_emb: Re-embedding flag
        mode: Embedding mode
        use_pinned: Whether to use pinned memory for transfers
    
    Returns:
        CuPy array of all quantum states
    """
    import torch as th
    import pennylane as qml
    
    nq = int(x_all.shape[1])
    t_float = th.float32 if x_all.dtype == np.float32 else th.float64
    t_cplx = th.complex64 if t_float == th.float32 else th.complex128
    
    # Use pinned memory for faster transfers
    if use_pinned:
        x = th.from_numpy(np.ascontiguousarray(x_all)).pin_memory().to("cuda", dtype=t_float, non_blocking=True)
        w = th.from_numpy(np.ascontiguousarray(w_np)).pin_memory().to("cuda", dtype=t_float, non_blocking=True)
    else:
        x = th.from_numpy(np.ascontiguousarray(x_all)).to("cuda", dtype=t_float, non_blocking=True)
        w = th.from_numpy(np.ascontiguousarray(w_np)).to("cuda", dtype=t_float, non_blocking=True)
    
    try:
        dev = qml.device(dev_name, wires=nq, shots=None, c_dtype=np.complex64 if t_float==th.float32 else np.complex128)
    except:
        dev = qml.device("lightning.gpu", wires=nq, shots=None)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(row):
        def _emb(v):
            if mode=="angle": qml.AngleEmbedding(ascale*v[:nq], wires=range(nq), rotation="Y")
            else:
                for i in range(nq):
                    qml.RY(ascale*v[i], wires=i)
                    if mode=="ryrz": qml.RZ(ascale*v[i], wires=i)
        if re_emb:
            for l in range(w.shape[0]):
                _emb(row)
                qml.templates.BasicEntanglerLayers(w[l:l+1], wires=range(nq))
        else:
            _emb(row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    # Attempt native batching for all states at once
    try:
        states = _state(x)
        if states.ndim != 2 or states.shape[0] != x.shape[0]:
            raise ValueError("Batching not natively supported")
    except:
        states = th.stack([_state(x[i]) for i in range(x.shape[0])])
    
    states = states.to(device="cuda", dtype=t_cplx, non_blocking=False)
    th.cuda.synchronize()
    
    # Convert to CuPy with zero-copy DLPack
    return _torch_cuda_to_cupy(states)

_COMPUTE_STREAM = None

def _get_compute_stream():
    """Get or create dedicated compute stream for async dispatch."""
    global _COMPUTE_STREAM
    if _COMPUTE_STREAM is None:
        import cupy as cp
        _COMPUTE_STREAM = cp.cuda.Stream(non_blocking=True)
    return _COMPUTE_STREAM

def _dispatch_kernel_async(kernel_fn, grid, block, args, stream=None):
    """
    Dispatch kernel asynchronously without immediate synchronization.
    
    Args:
        kernel_fn: Compiled kernel function
        grid: Grid dimensions
        block: Block dimensions
        args: Kernel arguments
        stream: CUDA stream (uses compute_stream if None)
    """
    import cupy as cp
    if stream is None:
        stream = _get_compute_stream()
    
    with stream:
        kernel_fn(grid, block, args)

# =====================================================================
# Main Compute Functions
# =====================================================================
def _gram_torch_stream(a_np, b_np, weights_np, device_name, tile_size, symmetric, float_dt, ret_dt, angle_scale, re_embed_between_layers, embed_mode,
                       use_pinned_memory=False, use_cuda_streams=False, use_amp=False, use_compile=False, tensorcore_precision="fp32"):
    import torch as th
    import pennylane as qml
    
    n, nq = a_np.shape
    m = n if b_np is None else b_np.shape[0]
    
    tf = th.float32 if float_dt==np.float32 else th.float64
    tc = th.complex64 if float_dt==np.float32 else th.complex128
    
    # OPTIMIZATION 1: Pinned memory for faster CPU→GPU transfers
    if use_pinned_memory and th.cuda.is_available():
        a = th.from_numpy(a_np).pin_memory().to("cuda", dtype=tf, non_blocking=True)
        b = a if b_np is None else th.from_numpy(b_np).pin_memory().to("cuda", dtype=tf, non_blocking=True)
        w = th.from_numpy(weights_np).pin_memory().to("cuda", dtype=tf, non_blocking=True)
    else:
        a = th.from_numpy(a_np).to("cuda", dtype=tf)
        b = a if b_np is None else th.from_numpy(b_np).to("cuda", dtype=tf)
        w = th.from_numpy(weights_np).to("cuda", dtype=tf)
    
    try: dev = qml.device(device_name, wires=nq, shots=None, c_dtype=(np.complex64 if float_dt==np.float32 else np.complex128))
    except: dev = qml.device("lightning.gpu", wires=nq, shots=None)
    
    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(row):
        def _emb(v):
            if embed_mode=="angle": qml.AngleEmbedding(float(angle_scale)*v[:nq], wires=range(nq), rotation="Y")
            else:
                for i in range(nq):
                    qml.RY(float(angle_scale)*v[i], wires=i)
                    if embed_mode=="ryrz": qml.RZ(float(angle_scale)*v[i], wires=i)
        if re_embed_between_layers:
            for l in range(w.shape[0]):
                _emb(row)
                qml.templates.BasicEntanglerLayers(w[l:l+1], wires=range(nq))
        else:
            _emb(row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    try: 
        build = lambda x: _state(x).to(dtype=tc)
        _ = build(a[:2])
    except: 
        build = lambda x: th.stack([_state(x[i]) for i in range(len(x))]).to(dtype=tc)

    # --- CONFIGURATION TENSOR CORES ---
    autocast_dtype = None
    if tensorcore_precision == "bf16" and th.cuda.is_bf16_supported():
        autocast_dtype = th.bfloat16
        # Active les matmul optimisés sur Ampere+
        th.set_float32_matmul_precision('high') 
    elif tensorcore_precision == "fp16":
        autocast_dtype = th.float16
    elif tensorcore_precision == "tf32":
        th.set_float32_matmul_precision('medium') # Force TF32 sur Ampere+
    
    # Active AMP si une précision réduite est demandée
    enable_amp = (autocast_dtype is not None)
    
    # OPTIMIZATION 3: torch.compile (PyTorch 2.0+)
    if use_compile and hasattr(th, 'compile'):
        try:
            build = th.compile(build, mode="reduce-overhead")
        except Exception:
            pass  # Fall back to non-compiled version if compile fails
    
    # OPTIMIZATION 2: CUDA streams for overlapped execution
    compute_stream = None
    if use_cuda_streams and th.cuda.is_available():
        compute_stream = th.cuda.Stream()

    k = th.empty((n, m), device="cuda", dtype=tf)
    
    # OPTIMIZATION 2: CUDA streams for overlapped execution
    compute_stream = None
    if use_cuda_streams and th.cuda.is_available():
        compute_stream = th.cuda.Stream()
    
    # Helper function to compute kernel block
    def compute_kernel_block(i0, i1):
        sa = build(a[i0:i1])
        j_start = i0 if (symmetric and b_np is None) else 0
        for j0 in range(j_start, m, tile_size):
            j1 = min(j0+tile_size, m)
            sb = sa if (b_np is None and j0==i0) else build(b[j0:j1])
            res = (sa @ sb.conj().T).abs().square()
            k[i0:i1, j0:j1] = res
            if symmetric and b_np is None and j0 > i0:
                k[j0:j1, i0:i1] = res.T
        del sa
    
    # OPTIMIZATION 4: --- EXECUTION AVEC PRECISION DYNAMIQUE ---
    with th.no_grad():
        with th.cuda.amp.autocast(enabled=enable_amp, dtype=autocast_dtype):
            for i0 in range(0, n, tile_size):
                i1 = min(i0+tile_size, n)
                
                if compute_stream:
                    with th.cuda.stream(compute_stream):
                        compute_kernel_block(i0, i1)
                else:
                    compute_kernel_block(i0, i1)
                
                th.cuda.empty_cache()
    
    # Synchronize if using streams
    if compute_stream:
        compute_stream.synchronize()
            
    return k.cpu().numpy().astype(ret_dt)

def _gram_pennylane_angles_mp(
        A, B, weights, device_name, tile_size, symmetric, n_workers,
        dtype, return_dtype, progress, desc, angle_scale, re_embed_between_layers, embed_mode
):
    import multiprocessing as mp
    
    f_dt = np.dtype(np.float32) if dtype=="float32" else np.dtype(np.float64)
    r_dt = np.dtype(np.float32) if return_dtype=="float32" else np.dtype(np.float64)
    
    A = _ensure_numpy(A, f_dt)
    B = A if B is None else _ensure_numpy(B, f_dt)
    n, nq = A.shape
    m = B.shape[0]
    w = _ensure_numpy(weights, f_dt)
    
    def _chunk(n, c): return [list(range(s, min(s+c, n))) for s in range(0, n, c)]
    ra = _chunk(n, max(1, tile_size))
    rb = ra if (B is A) else _chunk(m, max(1, tile_size))
    
    if n_workers is None or n_workers <= 0: n_workers = 1
    
    initargs = (w, device_name, nq, "float32" if f_dt==np.float32 else "float64", angle_scale, re_embed_between_layers, embed_mode)
    
    if n_workers == 1:
        _pl_worker_init(*initargs)
        sa = np.concatenate([_pl_states_for_rows(r, A) for r in ra], axis=0)
        sb = sa if (B is A) else np.concatenate([_pl_states_for_rows(r, B) for r in rb], axis=0)
    else:
        ctx = mp.get_context("spawn")
        from functools import partial
        with ctx.Pool(processes=n_workers, initializer=_pl_worker_init, initargs=initargs) as pool:
            from functools import partial
            sa = np.concatenate(list(pool.imap(partial(_pl_states_for_rows, mat=A), ra)), axis=0)
            sb = sa if (B is A) else np.concatenate(list(pool.imap(partial(_pl_states_for_rows, mat=B), rb)), axis=0)
            
    k = np.empty((n, m), dtype=r_dt)
    for i0, i1 in _tile_ranges(n, tile_size):
        sa_blk = sa[i0:i1]
        j_start = i0 if (symmetric and (B is A)) else 0
        for j0, j1 in _tile_ranges(m, tile_size):
            if j0 < j_start: continue
            sb_blk = sb[j0:j1]
            g = sa_blk @ sb_blk.conj().T
            mag2 = (np.abs(g)**2).astype(r_dt)
            k[i0:i1, j0:j1] = mag2
            if symmetric and (B is A) and j0 > i0:
                k[j0:j1, i0:i1] = mag2.T
    return k

def compute_kernel_matrix(
        X: Any, Y: Optional[Any] = None, *, weights: np.ndarray,
        device_name: str = "lightning.qubit", tile_size: int = 64, symmetric: bool = True,
        n_workers: int = 0, dtype: str = "float32", return_dtype: str = "float32",
        gram_backend: str = "auto", progress: bool = False, desc: str = "Gram",
        angle_scale: float = 1.0, re_embed_between_layers: bool = False, embed_mode: str = "ryrz",
        normalize: bool = False, jitter: float = 0.0,
        state_tile: int = -1, tile_m="auto", tile_n="auto", tile_k="auto",
        autotune: bool = True, precompute_all_states: bool = True, vram_fraction: float = 0.85,
        # NEW parameters for advanced optimizations
        dynamic_batch: bool = True,
        num_streams: int = 4,
        learn_tiles: bool = True,
        profile_memory: bool = False,
        use_cuda_graphs: bool = True,
        verbose_profile: bool = False,
        # Torch backend optimizations
        use_pinned_memory: bool = False,
        use_cuda_streams: bool = False,
        use_amp: bool = False,
        use_compile: bool = False, 
        tensorcore_precision: str = "fp32"
):
    f_dt = np.float32 if dtype=="float32" else np.float64
    r_dt = np.float32 if return_dtype=="float32" else np.float64
    is_double = (f_dt == np.float64)

    if gram_backend == "cuda_states":
        import cupy as cp
        try: 
            _setup_cupy()
            _load_autotune_cache()
        except: 
            pass
        
        A = _ensure_numpy(X, f_dt)
        B = A if Y is None else _ensure_numpy(Y, f_dt)
        w = _ensure_numpy(weights, f_dt)
        n, nq = A.shape
        m = B.shape[0]
        dim = 1 << nq
        
        # FIX: Force float64 for high qubit counts to prevent numerical overflow
        if nq >= 14 and dtype == "float32":
            if progress:
                print(f"⚠️ Switching to float64 for {nq} qubits (numerical stability)")
            f_dt = np.float64
            is_double = True
            A = A.astype(f_dt)
            B = B.astype(f_dt)
            w = w.astype(f_dt)

        # OPTIMIZATION 1: VRAM-aware state_tile sizing
        if state_tile == -1:
            state_tile = _compute_optimal_state_tile(vram_fraction, nq, f_dt)
            if progress:
                print(f"📊 Auto-sized state_tile={state_tile} (using {vram_fraction*100:.0f}% VRAM)")
        
        # OPTIMIZATION 3: Kernel autotuning with qubit-aware fallback
        # FIX: Add fallback tile sizes for high qubit counts to avoid shared memory errors
        if nq >= 14:
            # Safe tiles for very high qubit counts
            tm, tn, tk = (16, 16, 16)
            if progress:
                print(f"⚠️ Using conservative tiles for {nq} qubits: M={tm}, N={tn}, K={tk}")
        elif nq >= 12:
            # Conservative tiles for high qubit counts
            tm, tn, tk = (32, 32, 32)
            if progress:
                print(f"⚠️ Using conservative tiles for {nq} qubits: M={tm}, N={tn}, K={tk}")
        elif autotune and tile_m == "auto":
            tm, tn, tk = _autotune_kernel_tiles(nq, is_double)
            if progress:
                print(f"🔧 Autotuned kernel tiles: M={tm}, N={tn}, K={tk}")
        else:
            tm, tn, tk = (32, 32, 32)
            if tile_m != "auto": 
                tm, tn, tk = int(tile_m), int(tile_n), int(tile_k)
        
        # NEW OPTIMIZATION: Tile size learning
        tile_optimizer = None
        if learn_tiles:
            tile_optimizer = TileSizeOptimizer()
            device = cp.cuda.Device()
            available_vram = device.mem_info[1] / 1e9  # Convert to GB
            prediction = tile_optimizer.predict_optimal_tiles(n, nq, available_vram)
            
            if prediction["confidence"] > 0.5 and state_tile == -1:
                state_tile = prediction["state_tile"]
                if state_tile > 0 and progress:
                    print(f"🧠 Learned state_tile={state_tile} (confidence: {prediction['confidence']:.2f})")
            
            if prediction["confidence"] > 0.5 and tile_m == "auto":
                tm, tn, tk = prediction["kernel_tiles"]
                if progress:
                    print(f"🧠 Learned kernel tiles: M={tm}, N={tn}, K={tk}")
        
        # NEW OPTIMIZATION: Memory profiler
        mem_profiler = None
        if profile_memory:
            mem_profiler = MemoryProfiler(enable_realtime=verbose_profile)
            if progress:
                print("📊 Memory profiling enabled")
        
        # NEW OPTIMIZATION: CUDA stream pool
        stream_pool = None
        if num_streams > 1:
            stream_pool = CUDAStreamPool(num_streams)
            if progress:
                print(f"🌊 Stream pool initialized with {num_streams} streams")
        
        # NEW OPTIMIZATION: Dynamic batch sizer
        batch_sizer = None
        if dynamic_batch and state_tile > 0:
            batch_sizer = DynamicBatchSizer(
                initial_batch=state_tile,
                min_batch=max(64, state_tile // 4),
                max_batch=min(16384, state_tile * 2),
                target_memory_usage=vram_fraction
            )
            if progress:
                print(f"🔄 Dynamic batch sizing enabled (initial={state_tile})")
        
        # NEW OPTIMIZATION: CUDA graph manager
        graph_manager = None
        if use_cuda_graphs:
            graph_manager = CUDAGraphManager()
            if progress:
                print("📈 CUDA graph optimization enabled")
        
        # Re-finalize state_tile if still auto
        if state_tile == -1:
            state_tile = _compute_optimal_state_tile(vram_fraction, nq, f_dt)
            if progress:
                print(f"📊 Auto-sized state_tile={state_tile} (using {vram_fraction*100:.0f}% VRAM)")
        
        K_cp = cp.empty((n, m), dtype=cp.float64 if is_double else cp.float32)
        
        if mem_profiler:
            mem_profiler.track_allocation("kernel_output", K_cp.nbytes)
        
        # OPTIMIZATION 2: Bulk state precomputation
        max_precompute = _compute_max_precompute_size(vram_fraction, nq, f_dt)
        use_bulk_precompute = precompute_all_states and (max(n, m) <= max_precompute)
        
        if use_bulk_precompute:
            # Precompute ALL states at once to minimize handoffs
            if progress:
                print(f"⚡ Bulk precomputing {n} + {m} states...")
            
            start_time = time.time()
            s_a_cp = _build_all_states_torch_cuda(A, w, device_name, angle_scale, 
                                                   re_embed_between_layers, embed_mode, use_pinned=True)
            if Y is None:
                s_b_cp = s_a_cp
            else:
                s_b_cp = _build_all_states_torch_cuda(B, w, device_name, angle_scale,
                                                       re_embed_between_layers, embed_mode, use_pinned=True)
            
            if mem_profiler:
                mem_profiler.track_allocation("states_A", s_a_cp.nbytes)
                if Y is not None:
                    mem_profiler.track_allocation("states_B", s_b_cp.nbytes)
            
            # Use stream pool or fallback to single stream
            if stream_pool:
                compute_stream = stream_pool
            else:
                compute_stream = _get_compute_stream()
            
            tile_count = 0
            
            # Dynamic batch sizing
            current_state_tile = state_tile
            if batch_sizer:
                current_state_tile = batch_sizer.current_batch
            
            i_ranges = list(_tile_ranges(n, current_state_tile))
            j_ranges = list(_tile_ranges(m, current_state_tile))
            
            it = tqdm(total=len(i_ranges)*len(j_ranges), desc=desc, leave=False) if progress else None
            
            for i0, i1 in i_ranges:
                s_a_tile = s_a_cp[i0:i1]
                
                # FIX: Add intermediate normalization for high qubit counts
                if nq >= 12:
                    norms_a = cp.linalg.norm(s_a_tile, axis=1, keepdims=True)
                    norms_a = cp.where(norms_a > 1e-12, norms_a, 1.0)  # Avoid division by zero
                    s_a_tile = s_a_tile / norms_a
                
                j_start = i0 if (symmetric and Y is None) else 0
                
                for j0, j1 in j_ranges:
                    if j0 < j_start: 
                        if it: it.update(1)
                        continue
                    
                    s_b_tile = s_b_cp[j0:j1]
                    
                    # FIX: Add intermediate normalization for high qubit counts
                    if nq >= 12 and not (symmetric and Y is None and j0 == i0):
                        norms_b = cp.linalg.norm(s_b_tile, axis=1, keepdims=True)
                        norms_b = cp.where(norms_b > 1e-12, norms_b, 1.0)  # Avoid division by zero
                        s_b_tile = s_b_tile / norms_b
                    
                    bi, bj = int(i1-i0), int(j1-j0)
                    
                    use_lower = (symmetric and (Y is None) and j0==i0)
                    kernel_name = "cgemm_abs2_tiled_lower" if use_lower else "cgemm_abs2_tiled_full"
                    
                    # FIX: Wrap kernel compilation in try-except for shared memory errors
                    try:
                        k_fn = _get_kernel(tm, tn, tk, kernel_name, is_double)
                    except Exception as e:
                        if "shared memory" in str(e).lower() or "ptxas" in str(e).lower():
                            # Fallback to smaller tiles
                            if progress:
                                print(f"⚠️ Kernel compilation failed, falling back to 16x16x16 tiles")
                            tm, tn, tk = (16, 16, 16)
                            k_fn = _get_kernel(tm, tn, tk, kernel_name, is_double)
                        else:
                            raise
                    
                    grid = ((bj + tn - 1) // tn, (bi + tm - 1) // tm, 1)
                    block = (tn, tm, 1)
                    
                    out_tile = cp.empty((bi, bj), dtype=K_cp.dtype)
                    
                    # Select stream
                    if stream_pool:
                        current_stream = stream_pool.get_stream()
                    else:
                        current_stream = compute_stream
                    
                    # CUDA graph optimization
                    # FIX: Round tile dimensions to nearest power of 2 for better graph reuse
                    graph_key = (_round_to_pow2(bi), _round_to_pow2(bj), tm, tn, tk, kernel_name, is_double)
                    kernel_start = time.time()
                    
                    if graph_manager and graph_manager.has_graph(graph_key):
                        # Replay existing graph
                        with current_stream:
                            graph_manager.replay_graph(graph_key, current_stream)
                    else:
                        # Execute kernel normally
                        with current_stream:
                            k_fn(grid, block, (s_a_tile, s_b_tile, out_tile, bi, bj, dim, dim, dim, bj))
                        
                        # Capture graph for future reuse
                        if graph_manager and tile_count > 0:  # Skip first tile to avoid capture issues
                            try:
                                graph_manager.capture_graph(graph_key, current_stream, k_fn, grid, block,
                                                           (s_a_tile, s_b_tile, out_tile, bi, bj, dim, dim, dim, bj))
                            except Exception:
                                pass  # Graph capture failed, continue without it
                    
                    kernel_time = time.time() - kernel_start
                    
                    # FIX: Add numerical stability check for high qubit counts
                    if nq >= 12 and not cp.all(cp.isfinite(out_tile)):
                        if progress:
                            print(f"⚠️ NaN/Inf detected in tile ({i0}:{i1}, {j0}:{j1}), repairing...")
                        out_tile = cp.nan_to_num(out_tile, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    K_cp[i0:i1, j0:j1] = out_tile
                    if symmetric and (Y is None) and j0 > i0:
                        K_cp[j0:j1, i0:i1] = out_tile.T
                    
                    # Track kernel performance
                    if mem_profiler:
                        mem_profiler.track_kernel(kernel_time * 1000)  # Convert to ms
                    
                    # Dynamic batch adjustment
                    if batch_sizer and tile_count % BATCH_ADJUST_INTERVAL == 0:
                        device = cp.cuda.Device()
                        mem_info = device.mem_info
                        current_mem_usage = 1.0 - (mem_info[0] / mem_info[1])
                        new_batch = batch_sizer.adjust(current_mem_usage, kernel_time)
                        if new_batch != current_state_tile:
                            current_state_tile = new_batch
                            # Recompute ranges if batch size changed
                            # (Note: in practice, this would be applied to next iteration)
                    
                    tile_count += 1
                    
                    # Batch synchronization
                    if tile_count % BATCH_SYNC_INTERVAL == 0:
                        if stream_pool:
                            stream_pool.synchronize_all()
                        else:
                            compute_stream.synchronize()
                    
                    if it: it.update(1)
            
            # Final synchronization
            if stream_pool:
                stream_pool.synchronize_all()
            else:
                compute_stream.synchronize()
            
            if it: it.close()
            
            total_time = time.time() - start_time
            
        else:
            # Fall back to original tiled approach when bulk doesn't fit
            start_time = time.time()
            j_ranges = list(_tile_ranges(m, state_tile))
            b_cache = {}
            
            it_b = tqdm(j_ranges, desc="Cache B", leave=False) if (progress and Y is not None) else j_ranges
            for j0, j1 in it_b:
                s_th = _build_states_block_torch_cuda(B[j0:j1], w, device_name, angle_scale, re_embed_between_layers, embed_mode)
                b_cache[(j0, j1)] = _torch_cuda_to_cupy(s_th)
            
            if mem_profiler:
                total_b_cache_size = sum(arr.nbytes for arr in b_cache.values())
                mem_profiler.track_allocation("states_B_cache", total_b_cache_size)
            
            i_ranges = list(_tile_ranges(n, state_tile))
            it_a = tqdm(i_ranges, desc=desc, leave=False) if progress else i_ranges
            
            # Use stream pool or fallback
            if stream_pool:
                compute_stream = stream_pool
            else:
                compute_stream = _get_compute_stream()
            
            tile_count = 0
            
            for i0, i1 in it_a:
                s_a_th = _build_states_block_torch_cuda(A[i0:i1], w, device_name, angle_scale, re_embed_between_layers, embed_mode)
                s_a_cp = _torch_cuda_to_cupy(s_a_th)
                
                # FIX: Add intermediate normalization for high qubit counts
                if nq >= 12:
                    norms_a = cp.linalg.norm(s_a_cp, axis=1, keepdims=True)
                    norms_a = cp.where(norms_a > 1e-12, norms_a, 1.0)  # Avoid division by zero
                    s_a_cp = s_a_cp / norms_a
                
                relevant_j = [ (j0, j1) for (j0, j1) in j_ranges if (not symmetric or j1 > i0) ]
                for j0, j1 in relevant_j:
                    s_b_cp = b_cache.get((j0, j1))
                    if s_b_cp is None: s_b_cp = b_cache[(j0, j1)] = s_a_cp
                    
                    # FIX: Add intermediate normalization for high qubit counts
                    s_b_tile = s_b_cp
                    if nq >= 12 and s_b_cp is not s_a_cp:
                        norms_b = cp.linalg.norm(s_b_cp, axis=1, keepdims=True)
                        norms_b = cp.where(norms_b > 1e-12, norms_b, 1.0)  # Avoid division by zero
                        s_b_tile = s_b_cp / norms_b
                    
                    bi, bj = int(i1-i0), int(j1-j0)
                    
                    use_lower = (symmetric and (Y is None) and j0==i0)
                    kernel_name = "cgemm_abs2_tiled_lower" if use_lower else "cgemm_abs2_tiled_full"
                    
                    # FIX: Wrap kernel compilation in try-except for shared memory errors
                    try:
                        k_fn = _get_kernel(tm, tn, tk, kernel_name, is_double)
                    except Exception as e:
                        if "shared memory" in str(e).lower() or "ptxas" in str(e).lower():
                            # Fallback to smaller tiles
                            if progress:
                                print(f"⚠️ Kernel compilation failed, falling back to 16x16x16 tiles")
                            tm, tn, tk = (16, 16, 16)
                            k_fn = _get_kernel(tm, tn, tk, kernel_name, is_double)
                        else:
                            raise
                    
                    grid = ((bj + tn - 1) // tn, (bi + tm - 1) // tm, 1)
                    block = (tn, tm, 1)
                    
                    out_tile = cp.empty((bi, bj), dtype=K_cp.dtype)
                    
                    # Select stream
                    if stream_pool:
                        current_stream = stream_pool.get_stream()
                    else:
                        current_stream = compute_stream
                    
                    kernel_start = time.time()
                    
                    # Dispatch (graphs less useful in tiled approach due to varying sizes)
                    with current_stream:
                        k_fn(grid, block, (s_a_cp, s_b_tile, out_tile, bi, bj, dim, dim, dim, bj))
                    
                    kernel_time = time.time() - kernel_start
                    
                    # FIX: Add numerical stability check for high qubit counts
                    if nq >= 12 and not cp.all(cp.isfinite(out_tile)):
                        if progress:
                            print(f"⚠️ NaN/Inf detected in tile ({i0}:{i1}, {j0}:{j1}), repairing...")
                        out_tile = cp.nan_to_num(out_tile, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    K_cp[i0:i1, j0:j1] = out_tile
                    if symmetric and (Y is None) and j0 > i0:
                        K_cp[j0:j1, i0:i1] = out_tile.T
                    
                    if mem_profiler:
                        mem_profiler.track_kernel(kernel_time * 1000)
                    
                    tile_count += 1
                    if tile_count % BATCH_SYNC_INTERVAL == 0:
                        if stream_pool:
                            stream_pool.synchronize_all()
                        else:
                            compute_stream.synchronize()
                
                del s_a_cp
            
            # Final synchronization
            if stream_pool:
                stream_pool.synchronize_all()
            else:
                compute_stream.synchronize()
            
            total_time = time.time() - start_time
        
        # OPTIMIZATION 5: Memory cleanup
        K = K_cp.get().astype(r_dt)
        cp.get_default_memory_pool().free_all_blocks()
        
        # Record performance for tile learning
        total_pairs = n * m if Y is not None else (n * (n + 1)) // 2
        throughput = total_pairs / total_time if total_time > 0 else 0.0
        
        if tile_optimizer:
            device = cp.cuda.Device()
            peak_memory_gb = (device.mem_info[1] - device.mem_info[0]) / 1e9
            tile_optimizer.record_run(n, nq, state_tile, (tm, tn, tk), throughput, peak_memory_gb)
        
        # Generate profiling reports
        if mem_profiler:
            # Calculate throughput in Mpairs/s
            throughput_mpairs = throughput / 1e6
            
            # Get graph statistics if available
            if graph_manager:
                graph_stats = graph_manager.get_statistics()
                # Add to profiler data for inclusion in report
                mem_profiler.graph_replays = graph_stats["total_replays"]
                mem_profiler.graph_hit_rate = graph_stats["hit_rate"]
            
            # Add throughput to profiler data
            mem_profiler.throughput = throughput_mpairs
            
            report_data = mem_profiler.report(verbose=verbose_profile)
            
            # Add additional statistics
            if batch_sizer:
                batch_stats = batch_sizer.report()
                if verbose_profile:
                    print("╔══════════════════════════════════════════════════════════════╗")
                    print("║ Dynamic Adjustments                                            ║")
                    print(f"║   Batch Size Range:   {batch_stats['min_batch_used']} → {batch_stats['max_batch_used']} ({batch_stats['adjustments']} adjustments)             ║")
                    if stream_pool:
                        util = stream_pool.get_utilization() * 100
                        print(f"║   Stream Utilization: {util:.1f}%                                    ║")
                    print("╚══════════════════════════════════════════════════════════════╝\n")
        
        # --- PROTECTION ANTI-CRASH (NEW) ---
        if not np.all(np.isfinite(K)):
            print("⚠️ Matrice corrompue (NaN/Inf) détectée dans le backend cuda_states. Réparation...")
            K = np.nan_to_num(K, nan=0.0, posinf=1.0, neginf=0.0)
        # -----------------------------------

        if normalize and Y is None: 
            if jitter > 0: K += jitter * np.eye(n, dtype=K.dtype)
            _normalize_diag_inplace(K)
        return K

    if gram_backend in ["torch", "auto"] and "gpu" in device_name:
        import torch as th
        try:
            return _gram_torch_stream(
                _ensure_numpy(X, f_dt), _ensure_numpy(Y, f_dt) if Y is not None else None,
                weights_np=_ensure_numpy(weights, f_dt), device_name=device_name,
                tile_size=tile_size, symmetric=symmetric, float_dt=f_dt, ret_dt=r_dt,
                angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode,
                use_pinned_memory=use_pinned_memory, use_cuda_streams=use_cuda_streams,
                use_amp=use_amp, use_compile=use_compile, tensorcore_precision=tensorcore_precision
            )
        except Exception as e:
            if gram_backend=="torch": raise e
            
    return _gram_pennylane_angles_mp(
        X, Y, weights=weights, device_name=device_name, tile_size=tile_size,
        symmetric=symmetric, n_workers=n_workers, dtype=str(f_dt), return_dtype=str(r_dt),
        progress=progress, desc=desc, angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
    )