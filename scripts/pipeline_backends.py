# pipeline_backends.py — Optimized, Synchronized, Vmap-Free & Numerically Safe
from typing import Optional, Any
import time
import numpy as np
from tqdm import tqdm

# =====================================================================
# Helpers
# =====================================================================
from scripts.pipeline_helpers import (
    _compute_self_kernel_diag,
    _ensure_numpy,
    _normalize_cross_inplace,
    _normalize_diag_inplace,
    _tile_ranges,
)

from scripts.pipeline_gpu_optimizations import (
    BATCH_ADJUST_INTERVAL,
    BATCH_SYNC_INTERVAL,
    CUDAStreamPool,
    CUDAGraphManager,
    DynamicBatchSizer,
    MemoryProfiler,
    TileSizeOptimizer,
    _can_precompute_all,
    _compute_max_precompute_size,
    _compute_optimal_state_tile,
    _setup_cupy,
)

from scripts.pipeline_compute import (
    _AUTOTUNE_CACHE,
    _autotune_kernel_tiles,
    _build_all_states_torch_cuda,
    _build_states_block_torch_cuda,
    _get_kernel,
    _gram_pennylane_angles_mp,
    _gram_torch_stream,
    _launch_output_stationary_kernel,
    _load_autotune_cache,
    _normalize_state_tile_cp,
    _round_to_pow2,
    _torch_cuda_to_cupy,
)

# =====================================================================
# Main Compute Functions
# =====================================================================

def compute_kernel_matrix(
        X: Any, Y: Optional[Any] = None, *, weights: np.ndarray,
        device_name: str = "lightning.qubit", tile_size: int = 128, symmetric: bool = True,
        n_workers: int = 16, dtype: str = "float64", return_dtype: str = "float64",
        gram_backend: str = "auto", progress: bool = False, desc: str = "Gram",
        angle_scale: float = 1.0, re_embed_between_layers: bool = False, embed_mode: str = "ryrz",
        normalize: bool = False, jitter: float = 0.0,
        state_tile: int = -1, tile_m="auto", tile_n="auto", tile_k="auto",
        autotune: bool = True, precompute_all_states: bool = True, vram_fraction: float = 0.95,
        # NEW parameters for advanced optimizations
        dynamic_batch: bool = False,
        num_streams: int = 2,
        learn_tiles: bool = True,
        profile_memory: bool = False,
        use_cuda_graphs: bool = False,
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
    raw_result = None

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
                print(f"Warning: Switching to float64 for {nq} qubits (numerical stability)")
            f_dt = np.float64
            is_double = True
            A = A.astype(f_dt)
            B = B.astype(f_dt)
            w = w.astype(f_dt)

        # OPTIMIZATION 1: VRAM-aware state_tile sizing
        if state_tile == -1:
            state_tile = _compute_optimal_state_tile(vram_fraction, nq, f_dt)
            if progress:
                print(f"Auto-sized state_tile={state_tile} (using {vram_fraction*100:.0f}% VRAM)")
        
        # OPTIMIZATION 3: Kernel autotuning with qubit-aware fallback
        # Reuse cached autotune results first when available.
        cache_key = f"nq{nq}_{'double' if is_double else 'float'}"
        cached_tiles = _AUTOTUNE_CACHE.get(cache_key) if autotune and tile_m == "auto" else None
        kernel_tiles_locked = False
        if cached_tiles is not None:
            tm, tn, tk = cached_tiles
            kernel_tiles_locked = True
            if progress:
                print(f"Loaded cached kernel tiles: M={tm}, N={tn}, K={tk}")
        # FIX: Add fallback tile sizes for high qubit counts to avoid shared memory errors
        elif nq >= 14:
            # Safe tiles for very high qubit counts
            tm, tn, tk = (16, 16, 16)
            kernel_tiles_locked = True
            if progress:
                print(f"Warning: Using conservative tiles for {nq} qubits: M={tm}, N={tn}, K={tk}")
        elif nq >= 12:
            # Conservative tiles for high qubit counts
            tm, tn, tk = (32, 32, 32)
            kernel_tiles_locked = True
            if progress:
                print(f"Warning: Using conservative tiles for {nq} qubits: M={tm}, N={tn}, K={tk}")
        elif autotune and tile_m == "auto":
            tm, tn, tk = _autotune_kernel_tiles(nq, is_double)
            if progress:
                print(f"Autotuned kernel tiles: M={tm}, N={tn}, K={tk}")
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
                    print(f"Learned state_tile={state_tile} (confidence: {prediction['confidence']:.2f})")
            
            if prediction["confidence"] > 0.5 and tile_m == "auto" and not kernel_tiles_locked:
                tm, tn, tk = prediction["kernel_tiles"]
                if progress:
                    print(f"Learned kernel tiles: M={tm}, N={tn}, K={tk}")
        
        # NEW OPTIMIZATION: Memory profiler
        mem_profiler = None
        if profile_memory:
            mem_profiler = MemoryProfiler(enable_realtime=verbose_profile)
            if progress:
                print("Memory profiling enabled")
        
        # NEW OPTIMIZATION: CUDA stream pool
        stream_pool = None
        if num_streams > 1:
            stream_pool = CUDAStreamPool(num_streams)
            if progress:
                print(f"Stream pool initialized with {num_streams} streams")
        
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
                print(f"Dynamic batch sizing enabled (initial={state_tile})")
        
        # NEW OPTIMIZATION: CUDA graph manager
        graph_manager = None
        if use_cuda_graphs:
            graph_manager = CUDAGraphManager()
            if progress:
                print("CUDA graph optimization enabled")
        
        # Re-finalize state_tile if still auto
        if state_tile == -1:
            state_tile = _compute_optimal_state_tile(vram_fraction, nq, f_dt)
            if progress:
                print(f"Auto-sized state_tile={state_tile} (using {vram_fraction*100:.0f}% VRAM)")
        
        K_cp = cp.empty((n, m), dtype=cp.float64)
        
        if mem_profiler:
            mem_profiler.track_allocation("kernel_output", K_cp.nbytes)
        
        # Add VRAM estimation to progress output
        if progress:
            dim = 1 << nq
            states_gb = n * dim * (16 if is_double else 8) / 1e9
            kernel_gb = n * m * 8 / 1e9
            print(f"Estimated VRAM: states={states_gb:.1f}GB, kernel={kernel_gb:.1f}GB, "
                  f"total={states_gb + kernel_gb:.1f}GB")
        
        # OPTIMIZATION 2: Bulk state precomputation with VRAM-aware check
        max_precompute = _compute_max_precompute_size(vram_fraction, nq, f_dt)
        
        # Check if bulk precomputation is feasible
        if precompute_all_states:
            can_precompute = _can_precompute_all(
                n, 0 if Y is None else m, nq, f_dt, vram_fraction
            )
            if not can_precompute:
                if progress:
                    print(f"Warning: VRAM insufficient for bulk precompute ({n} samples × {nq} qubits). "
                          f"Falling back to tiled approach.")
                precompute_all_states = False
                
                # Also reduce state_tile to fit
                max_states = _compute_max_precompute_size(vram_fraction, nq, f_dt)
                if state_tile > max_states:
                    state_tile = max(256, max_states // 2)
                    if progress:
                        print(f"   Reduced state_tile to {state_tile}")
        
        use_bulk_precompute = precompute_all_states and (max(n, m) <= max_precompute)
        
        if use_bulk_precompute:
            # Precompute ALL states at once to minimize handoffs
            if progress:
                    print(f"Bulk precomputing {n} + {m} states...")
            
            start_time = time.time()
            transfer_start = time.time()
            s_a_cp = _build_all_states_torch_cuda(
                A, w, device_name, angle_scale, re_embed_between_layers, embed_mode,
                use_pinned=True, progress=progress, desc="States A"
            )
            transfer_time_a = (time.time() - transfer_start) * 1000  # Convert to ms
            
            if Y is None:
                s_b_cp = s_a_cp
                transfer_time_b = 0
            else:
                transfer_start = time.time()
                s_b_cp = _build_all_states_torch_cuda(
                    B, w, device_name, angle_scale, re_embed_between_layers, embed_mode,
                    use_pinned=True, progress=progress, desc="States B"
                )
                transfer_time_b = (time.time() - transfer_start) * 1000  # Convert to ms
            
            if mem_profiler:
                mem_profiler.track_allocation("states_A", s_a_cp.nbytes)
                mem_profiler.track_transfer("H2D", s_a_cp.nbytes, transfer_time_a)
                if Y is not None:
                    mem_profiler.track_allocation("states_B", s_b_cp.nbytes)
                    mem_profiler.track_transfer("H2D", s_b_cp.nbytes, transfer_time_b)
            
            # Use stream pool or fallback to single stream
            if stream_pool:
                compute_stream = stream_pool
            else:
                compute_stream = cp.cuda.Stream.null
            
            tile_count = 0
            
            # Dynamic batch sizing
            current_state_tile = state_tile
            if batch_sizer:
                current_state_tile = batch_sizer.current_batch
            
            i_ranges = list(_tile_ranges(n, current_state_tile))
            j_ranges = list(_tile_ranges(m, current_state_tile))
            
            it = tqdm(total=len(i_ranges)*len(j_ranges), desc=desc, leave=False) if progress else None
            
            for i0, i1 in i_ranges:
                j_start = i0 if (symmetric and Y is None) else 0
                
                for j0, j1 in j_ranges:
                    if j0 < j_start: 
                        if it: it.update(1)
                        continue

                    bi, bj = int(i1-i0), int(j1-j0)
                    
                    kernel_name = "cgemm_abs2_os_full"
                    
                    try:
                        k_fn = _get_kernel(tm, tn, tk, kernel_name, is_double)
                    except Exception as e:
                        raise
                    
                    # Select stream
                    if stream_pool:
                        current_stream = stream_pool.get_stream()
                    else:
                        current_stream = compute_stream
                    
                    # CUDA graph optimization
                    # FIX: Round tile dimensions to nearest power of 2 for better graph reuse
                    graph_key = (_round_to_pow2(bi), _round_to_pow2(bj), tm, tn, tk, kernel_name, is_double)
                    kernel_start = time.time()
                    with current_stream:
                        s_a_tile = s_a_cp[i0:i1]
                        if nq >= 12:
                            s_a_tile = _normalize_state_tile_cp(s_a_tile)

                        if symmetric and Y is None and j0 == i0:
                            s_b_tile = s_a_tile
                        else:
                            s_b_tile = s_b_cp[j0:j1]
                            if nq >= 12:
                                s_b_tile = _normalize_state_tile_cp(s_b_tile)

                        out_tile = cp.empty((bi, bj), dtype=cp.float64)

                        if graph_manager and graph_manager.has_graph(graph_key):
                            # Graph replay is skipped here: tiles carry new data buffers on each iteration.
                            out_tile, _, _, _, _, _ = _launch_output_stationary_kernel(
                                k_fn, s_a_tile, s_b_tile, tn, tm, out_tile=out_tile
                            )
                        else:
                            # Execute kernel normally
                            out_tile, s_a_contig, s_b_contig, grid, block, args = _launch_output_stationary_kernel(
                                k_fn, s_a_tile, s_b_tile, tn, tm, out_tile=out_tile
                            )

                        # FIX: Add numerical stability check for high qubit counts
                        # Replacement values: NaN→0 (no overlap), +Inf→1 (perfect overlap), -Inf→0 (invalid)
                        if nq >= 12 and not cp.all(cp.isfinite(out_tile)):
                            if progress:
                                print(f"Warning: NaN/Inf detected in tile ({i0}:{i1}, {j0}:{j1}), repairing...")
                            out_tile = cp.nan_to_num(out_tile, nan=0.0, posinf=1.0, neginf=0.0)

                        K_cp[i0:i1, j0:j1] = out_tile
                        if symmetric and (Y is None) and j0 > i0:
                            K_cp[j0:j1, i0:i1] = out_tile.T

                    # Capture graph for future reuse
                    if (
                        graph_manager
                        and not graph_manager.has_graph(graph_key)
                        and tile_count > 0
                    ):
                        try:
                            graph_manager.capture_graph(graph_key, current_stream, k_fn, grid, block, args)
                        except Exception:
                            pass  # Graph capture failed, continue without it
                    
                    kernel_time = time.time() - kernel_start
                    
                    # Track kernel performance
                    if mem_profiler:
                        mem_profiler.track_kernel(kernel_time * 1000)  # Convert to ms
                        # Record stream usage
                        if stream_pool:
                            mem_profiler.record_stream_usage(tile_count)
                    
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
                s_th = _build_states_block_torch_cuda(
                    B[j0:j1], w, device_name, angle_scale, re_embed_between_layers, embed_mode,
                    progress=progress, desc="States B"
                )
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
                compute_stream = cp.cuda.Stream.null
            
            tile_count = 0
            
            for i0, i1 in it_a:
                s_a_th = _build_states_block_torch_cuda(
                    A[i0:i1], w, device_name, angle_scale, re_embed_between_layers, embed_mode,
                    progress=progress, desc="States A"
                )
                s_a_cp = _torch_cuda_to_cupy(s_a_th)
                
                relevant_j = [ (j0, j1) for (j0, j1) in j_ranges if (not symmetric or j1 > i0) ]
                for j0, j1 in relevant_j:
                    s_b_cp = b_cache.get((j0, j1))
                    if s_b_cp is None: s_b_cp = b_cache[(j0, j1)] = s_a_cp

                    bi, bj = int(i1-i0), int(j1-j0)
                    
                    kernel_name = "cgemm_abs2_os_full"
                    
                    try:
                        k_fn = _get_kernel(tm, tn, tk, kernel_name, is_double)
                    except Exception as e:
                        raise
                    
                    # Select stream
                    if stream_pool:
                        current_stream = stream_pool.get_stream()
                    else:
                        current_stream = compute_stream
                    
                    kernel_start = time.time()
                    
                    # Dispatch (graphs less useful in tiled approach due to varying sizes)
                    with current_stream:
                        s_a_tile = s_a_cp
                        if nq >= 12:
                            s_a_tile = _normalize_state_tile_cp(s_a_tile)

                        if s_b_cp is s_a_cp:
                            s_b_tile = s_a_tile
                        else:
                            s_b_tile = s_b_cp
                            if nq >= 12:
                                s_b_tile = _normalize_state_tile_cp(s_b_tile)

                        out_tile, _, _, _, _, _ = _launch_output_stationary_kernel(
                            k_fn, s_a_tile, s_b_tile, tn, tm
                        )

                        # FIX: Add numerical stability check for high qubit counts
                        # Replacement values: NaN→0 (no overlap), +Inf→1 (perfect overlap), -Inf→0 (invalid)
                        if nq >= 12 and not cp.all(cp.isfinite(out_tile)):
                            if progress:
                                print(f"Warning: NaN/Inf detected in tile ({i0}:{i1}, {j0}:{j1}), repairing...")
                            out_tile = cp.nan_to_num(out_tile, nan=0.0, posinf=1.0, neginf=0.0)

                        K_cp[i0:i1, j0:j1] = out_tile
                        if symmetric and (Y is None) and j0 > i0:
                            K_cp[j0:j1, i0:i1] = out_tile.T
                    
                    kernel_time = time.time() - kernel_start
                    
                    if mem_profiler:
                        mem_profiler.track_kernel(kernel_time * 1000)
                        # Record stream usage
                        if stream_pool:
                            mem_profiler.record_stream_usage(tile_count)
                    
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
        transfer_start = time.time()
        K = K_cp.get().astype(r_dt)
        transfer_time_d2h = (time.time() - transfer_start) * 1000  # Convert to ms
        
        if mem_profiler:
            mem_profiler.track_transfer("D2H", K.nbytes, transfer_time_d2h)
        
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
                    print("+--------------------------------------------------------------+")
                    print("| Dynamic adjustments                                          |")
                    print(f"|   Batch size range: {batch_stats['min_batch_used']} -> {batch_stats['max_batch_used']} ({batch_stats['adjustments']} adjustments) |")
                    if stream_pool:
                        util = stream_pool.get_utilization() * 100
                        print(f"|   Stream utilization: {util:.1f}%                                   |")
                    print("+--------------------------------------------------------------+\n")
        
        # --- PROTECTION ANTI-CRASH (NEW) ---
        if not np.all(np.isfinite(K)):
            print("Warning: corrupted matrix (NaN/Inf) detected in the cuda_states backend. Repairing...")
            K = np.nan_to_num(K, nan=0.0, posinf=1.0, neginf=0.0)
        # -----------------------------------
        raw_result = K

    if raw_result is None and gram_backend in ["torch", "auto"] and "gpu" in device_name:
        import torch as th
        try:
            raw_result = _gram_torch_stream(
                _ensure_numpy(X, f_dt), _ensure_numpy(Y, f_dt) if Y is not None else None,
                weights_np=_ensure_numpy(weights, f_dt), device_name=device_name,
                tile_size=tile_size, symmetric=symmetric, float_dt=f_dt, ret_dt=r_dt,
                angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode,
                use_pinned_memory=use_pinned_memory, use_cuda_streams=use_cuda_streams,
                use_amp=use_amp, use_compile=use_compile, tensorcore_precision=tensorcore_precision
            )
        except Exception as e:
            if gram_backend=="torch": raise e

    if raw_result is None:
        raw_result = _gram_pennylane_angles_mp(
            X, Y, weights=weights, device_name=device_name, tile_size=tile_size,
            symmetric=symmetric, n_workers=n_workers, dtype=str(f_dt), return_dtype=str(r_dt),
            progress=progress, desc=desc, angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
        )

    K = np.asarray(raw_result, order="C")
    if not normalize:
        return K

    if Y is None:
        if jitter > 0:
            K = np.array(K, copy=True, order="C")
            K += jitter * np.eye(K.shape[0], dtype=K.dtype)
        _normalize_diag_inplace(K)
        return K

    if progress:
        print("Computing self-kernel diagonals for strict cross-kernel normalization.")

    diag_x = _compute_self_kernel_diag(
        X, weights=weights,
        device_name=device_name, tile_size=tile_size, symmetric=True,
        n_workers=n_workers, dtype=dtype, return_dtype=return_dtype,
        gram_backend=gram_backend, angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers,
        embed_mode=embed_mode, jitter=jitter, state_tile=state_tile,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, autotune=autotune,
        precompute_all_states=precompute_all_states, vram_fraction=vram_fraction,
        dynamic_batch=dynamic_batch, num_streams=num_streams, learn_tiles=learn_tiles,
        profile_memory=profile_memory, use_cuda_graphs=use_cuda_graphs, verbose_profile=verbose_profile,
        use_pinned_memory=use_pinned_memory, use_cuda_streams=use_cuda_streams,
        use_amp=use_amp, use_compile=use_compile, tensorcore_precision=tensorcore_precision,
    )

    if Y is X:
        diag_y = diag_x
    else:
        diag_y = _compute_self_kernel_diag(
            Y, weights=weights,
            device_name=device_name, tile_size=tile_size, symmetric=True,
            n_workers=n_workers, dtype=dtype, return_dtype=return_dtype,
            gram_backend=gram_backend, angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers,
            embed_mode=embed_mode, jitter=jitter, state_tile=state_tile,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, autotune=autotune,
            precompute_all_states=precompute_all_states, vram_fraction=vram_fraction,
            dynamic_batch=dynamic_batch, num_streams=num_streams, learn_tiles=learn_tiles,
            profile_memory=profile_memory, use_cuda_graphs=use_cuda_graphs, verbose_profile=verbose_profile,
            use_pinned_memory=use_pinned_memory, use_cuda_streams=use_cuda_streams,
            use_amp=use_amp, use_compile=use_compile, tensorcore_precision=tensorcore_precision,
        )

    K = np.array(K, copy=True, order="C")
    _normalize_cross_inplace(K, diag_x, diag_y)
    return K
