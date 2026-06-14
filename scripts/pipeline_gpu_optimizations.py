"""GPU memory and optimization helpers for pipeline backends."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import json
import os
import sys

import numpy as np

MATRIX_PAIRS_FACTOR = 2
BATCH_SYNC_INTERVAL = 32
BATCH_ADJUST_INTERVAL = 10


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
        with cp.cuda.using_allocator(pinned_pool.malloc):
            _PINNED_BUFFERS[key] = cp.zeros(shape, dtype=dtype)
    return _PINNED_BUFFERS[key]


class DynamicBatchSizer:
    """Adjusts batch sizes at runtime based on GPU memory pressure and throughput."""

    def __init__(self, initial_batch: int, min_batch: int = 64, max_batch: int = 16384,
                 target_memory_usage: float = 0.85):
        self.current_batch = initial_batch
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_memory = target_memory_usage
        self.history = []
        self.adjustments = 0
        self.total_kernel_time = 0.0
        self.kernel_count = 0

    def adjust(self, current_memory_used: float, last_kernel_time: float) -> int:
        self.total_kernel_time += last_kernel_time
        self.kernel_count += 1

        if current_memory_used > self.target_memory + 0.05:
            self.current_batch = max(self.min_batch, int(self.current_batch * 0.75))
            self.adjustments += 1
        elif current_memory_used < self.target_memory - 0.10:
            if len(self.history) >= 5:
                recent_times = [h[1] for h in self.history[-5:]]
                variance = np.var(recent_times)
                mean_time = np.mean(recent_times)
                if variance < (mean_time * 0.1) ** 2:
                    self.current_batch = min(self.max_batch, int(self.current_batch * 1.25))
                    self.adjustments += 1

        self.history.append((current_memory_used, last_kernel_time, self.current_batch))
        return self.current_batch

    def report(self) -> dict:
        batch_sizes = [h[2] for h in self.history]
        return {
            "adjustments": self.adjustments,
            "current_batch": self.current_batch,
            "min_batch_used": min(batch_sizes) if batch_sizes else self.current_batch,
            "max_batch_used": max(batch_sizes) if batch_sizes else self.current_batch,
            "avg_kernel_time": self.total_kernel_time / self.kernel_count if self.kernel_count > 0 else 0.0,
            "total_kernels": self.kernel_count,
        }


class CUDAStreamPool:
    """Manages multiple CUDA streams for concurrent operations."""

    def __init__(self, num_streams: int = 4):
        import cupy as cp

        self.device_id = int(cp.cuda.runtime.getDevice())
        self.num_streams = num_streams
        with cp.cuda.Device(self.device_id):
            self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
        self.current_idx = 0
        self.usage_count = [0] * num_streams
        self.total_operations = 0

    def get_stream(self):
        import cupy as cp

        cp.cuda.Device(self.device_id).use()
        stream = self.streams[self.current_idx]
        self.usage_count[self.current_idx] += 1
        self.current_idx = (self.current_idx + 1) % self.num_streams
        self.total_operations += 1
        return stream

    def synchronize_all(self):
        import cupy as cp

        cp.cuda.Device(self.device_id).use()
        for stream in self.streams:
            stream.synchronize()

    def synchronize(self):
        self.synchronize_all()

    def get_utilization(self) -> float:
        if self.total_operations == 0:
            return 0.0
        expected_per_stream = self.total_operations / self.num_streams
        if expected_per_stream == 0:
            return 0.0
        variance = np.var(self.usage_count)
        return 1.0 - min(1.0, variance / (expected_per_stream ** 2))

    def __enter__(self):
        import cupy as cp

        cp.cuda.Device(self.device_id).use()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.synchronize_all()
        return False


class TileSizeOptimizer:
    """Learns optimal tile sizes from historical runs."""

    def __init__(self, history_file: str = ".tile_optimizer_history.json"):
        self.history_file = history_file
        self.history = []
        self.load_history()

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def save_history(self):
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

    def record_run(self, n_samples: int, n_qubits: int, state_tile: int,
                   kernel_tiles: tuple, throughput: float, peak_memory: float):
        entry = {
            "n_samples": n_samples,
            "n_qubits": n_qubits,
            "state_tile": state_tile,
            "kernel_tiles": list(kernel_tiles),
            "throughput": throughput,
            "peak_memory": peak_memory,
        }
        self.history.append(entry)
        self.save_history()

    def predict_optimal_tiles(self, n_samples: int, n_qubits: int,
                               available_vram: float) -> dict:
        if not self.history:
            return {
                "state_tile": -1,
                "kernel_tiles": (32, 32, 32),
                "confidence": 0.0,
                "source": "default",
            }

        similar = [h for h in self.history if h["n_qubits"] == n_qubits]
        if not similar:
            similar = self.history

        valid = [h for h in similar if h["peak_memory"] <= available_vram * 0.9]
        if not valid:
            return {
                "state_tile": -1,
                "kernel_tiles": (32, 32, 32),
                "confidence": 0.0,
                "source": "no_valid_history",
            }

        valid.sort(key=lambda x: x["throughput"], reverse=True)
        best = valid[0]
        confidence = min(1.0, len(valid) / 10.0)

        return {
            "state_tile": best["state_tile"],
            "kernel_tiles": tuple(best["kernel_tiles"]),
            "confidence": confidence,
            "source": "learned",
        }

    def get_statistics(self) -> dict:
        if not self.history:
            return {
                "total_runs": 0,
                "unique_configs": 0,
                "avg_throughput": 0.0,
            }

        unique_configs = len(set(
            (h["n_qubits"], h["state_tile"], tuple(h["kernel_tiles"]))
            for h in self.history
        ))

        return {
            "total_runs": len(self.history),
            "unique_configs": unique_configs,
            "avg_throughput": np.mean([h["throughput"] for h in self.history]),
        }


class MemoryProfiler:
    """Detailed GPU memory analysis and profiling."""

    def __init__(self, enable_realtime: bool = False):
        self.enable_realtime = enable_realtime
        self.allocations = {}
        self.transfers = []
        self.snapshots = []
        self.peak_allocated = 0.0
        self.kernel_launches = 0
        self.kernel_times = []
        self.stream_operations = 0

    def track_allocation(self, name: str, size_bytes: int):
        if name not in self.allocations:
            self.allocations[name] = 0
        self.allocations[name] += size_bytes

        total = sum(self.allocations.values())
        self.peak_allocated = max(self.peak_allocated, total)

    def track_transfer(self, direction: str, size_bytes: int, duration_ms: float):
        self.transfers.append({
            "direction": direction,
            "size_bytes": size_bytes,
            "duration_ms": duration_ms,
            "bandwidth_gbps": (size_bytes / 1e9) / (duration_ms / 1000.0) if duration_ms > 0 else 0.0,
        })

    def track_kernel(self, duration_ms: float):
        self.kernel_launches += 1
        self.kernel_times.append(duration_ms)

    def record_stream_usage(self, stream_count: int):
        self.stream_operations = stream_count

    def snapshot(self) -> dict:
        try:
            import cupy as cp

            device = cp.cuda.Device()
            mem_info = device.mem_info
            current_state = {
                "free": mem_info[0],
                "total": mem_info[1],
                "used": mem_info[1] - mem_info[0],
                "allocations": dict(self.allocations),
            }
            self.snapshots.append(current_state)
            return current_state
        except Exception:
            return {}

    def report(self, verbose: bool = True) -> dict:
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
            "avg_kernel_time_ms": avg_kernel_time,
        }

        if hasattr(self, "graph_replays"):
            report_data["graph_replays"] = self.graph_replays
        if hasattr(self, "graph_hit_rate"):
            report_data["graph_hit_rate"] = self.graph_hit_rate
        if hasattr(self, "throughput"):
            report_data["throughput_mpairs_per_sec"] = self.throughput

        if verbose:
            self._print_report(report_data)

        return report_data

    def _print_report(self, data: dict):
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

        if "graph_replays" in data and "graph_hit_rate" in data:
            print(f"║   Graph Replays:        {data['graph_replays']:,} ({data['graph_hit_rate']*100:4.1f}% hit rate)                  ║")

        print(f"║   Avg Kernel Time:     {data['avg_kernel_time_ms']:.2f} ms                                 ║")

        if "throughput_mpairs_per_sec" in data:
            print(f"║   Throughput:         {data['throughput_mpairs_per_sec']:.1f} Mpairs/s                          ║")

        print("╚══════════════════════════════════════════════════════════════╝\n")


class CUDAGraphManager:
    """Manages CUDA graph capture and replay for kernel optimization."""

    def __init__(self):
        self.graphs = {}
        self.graph_execs = {}
        self.capture_counts = {}
        self.replay_counts = {}

    def capture_graph(self, key: tuple, stream: Any, kernel_fn: Callable,
                      grid: Tuple[int, ...], block: Tuple[int, ...], args: Tuple[Any, ...]):
        if key in self.graphs:
            return

        try:
            stream.begin_capture()
            kernel_fn(grid, block, args)
            graph = stream.end_capture()
            graph_exec = graph.instantiate()

            self.graphs[key] = graph
            self.graph_execs[key] = graph_exec
            self.capture_counts[key] = 1
            self.replay_counts[key] = 0
        except Exception:
            try:
                stream.end_capture()
            except Exception:
                pass

    def has_graph(self, key: tuple) -> bool:
        return key in self.graph_execs

    def replay_graph(self, key: tuple, stream):
        if key not in self.graph_execs:
            raise KeyError(f"No graph found for key {key}")

        graph_exec = self.graph_execs[key]
        graph_exec.launch(stream)
        self.replay_counts[key] = self.replay_counts.get(key, 0) + 1

    def clear(self):
        self.graphs.clear()
        self.graph_execs.clear()
        self.capture_counts.clear()
        self.replay_counts.clear()

    def get_statistics(self) -> dict:
        total_captures = sum(self.capture_counts.values())
        total_replays = sum(self.replay_counts.values())
        hit_rate = total_replays / (total_captures + total_replays) if (total_captures + total_replays) > 0 else 0.0

        return {
            "total_graphs": len(self.graphs),
            "total_captures": total_captures,
            "total_replays": total_replays,
            "hit_rate": hit_rate,
        }


def _compute_optimal_state_tile(vram_fraction: float = 0.85, nq: int = 6,
                                dtype=np.float32, overhead_gb: float = 2.0) -> int:
    try:
        import cupy as cp

        device = cp.cuda.Device()
        total_vram = device.mem_info[1]
        available_vram = total_vram * vram_fraction - (overhead_gb * 1024**3)

        dim = 1 << nq
        bytes_per_complex = 8 if dtype == np.float32 else 16
        bytes_per_state = dim * bytes_per_complex

        max_states = int(available_vram / bytes_per_state)
        tile_size = 2 ** int(np.log2(max_states))

        qubit_cap = 32768
        if nq >= 16:
            qubit_cap = 256
        elif nq >= 14:
            qubit_cap = 512
        elif nq >= 12:
            qubit_cap = 1024

        tile_size = max(256, min(tile_size, qubit_cap))
        return tile_size
    except Exception:
        return 8192


def _compute_max_precompute_size(vram_fraction: float = 0.85, nq: int = 6,
                                 dtype=np.float32, overhead_gb: float = 2.0) -> int:
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


def _can_precompute_all(n_samples_a: int, n_samples_b: int, n_qubits: int, dtype,
                        vram_fraction: float = 0.85, overhead_gb: float = 2.0) -> bool:
    try:
        import cupy as cp

        device = cp.cuda.Device()
        available_vram = device.mem_info[0]
        total_vram = device.mem_info[1]

        dim = 1 << n_qubits
        bytes_per_complex = 16 if dtype == np.float64 else 8

        b_samples = 0 if n_samples_b is None else n_samples_b
        state_count = n_samples_a + b_samples

        states_mem = state_count * dim * bytes_per_complex
        kernel_cols = n_samples_a if b_samples == 0 else b_samples
        kernel_mem = n_samples_a * kernel_cols * 8
        workspace_mem = max(states_mem, kernel_mem)
        usable_vram = min(available_vram, total_vram * vram_fraction) - int(overhead_gb * 1024**3)
        if usable_vram <= 0:
            return False

        total_needed = states_mem + kernel_mem + workspace_mem
        return total_needed < usable_vram
    except Exception:
        return False


def _setup_cupy():
    import cupy as cp

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    pinned_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)

    _ = cp.ones((1,), dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()

    candidates = [
        os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME"),
        "/usr/local/cuda", "/opt/cuda",
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Library", "include"),
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
