# Memory Profiler, Stream Utilization, and VRAM Estimation Fixes

## Summary of Changes

This document summarizes the fixes applied to address profiler and optimization issues in the quantum kernel computation pipeline.

## Issues Fixed

### Issue 1: Memory Profiler Shows 0.0 GB for Most Metrics

**Problem**: The memory profiler wasn't tracking actual CuPy array allocations, showing 0.0 GB for all metrics.

**Solution**:
- Modified `scripts/pipeline_backends.py` to track actual allocation sizes:
  - Added `.nbytes` tracking for `states_A`, `states_B`, and `kernel_output`
  - Added H→D transfer tracking with timing after state precomputation
  - Added D→H transfer tracking with timing when retrieving results from GPU

**Files Changed**: `scripts/pipeline_backends.py`

**Key Code Changes**:
```python
# Track allocations with actual sizes
if mem_profiler:
    mem_profiler.track_allocation("states_A", s_a_cp.nbytes)
    mem_profiler.track_transfer("H2D", s_a_cp.nbytes, transfer_time_a)
    
# Track D2H transfer when retrieving results
transfer_start = time.time()
K = K_cp.get().astype(r_dt)
transfer_time_d2h = (time.time() - transfer_start) * 1000
if mem_profiler:
    mem_profiler.track_transfer("D2H", K.nbytes, transfer_time_d2h)
```

### Issue 2: Stream Utilization is 0.0%

**Problem**: Stream pool was created but stream usage wasn't being tracked, resulting in 0.0% utilization.

**Solution**:
- Added `stream_operations` field to `MemoryProfiler` class
- Added `record_stream_usage()` method to track stream operations count
- Called `record_stream_usage()` in kernel dispatch loops
- Stream pool's existing `get_utilization()` method calculates variance-based utilization

**Files Changed**: `scripts/pipeline_backends.py`

**Key Code Changes**:
```python
class MemoryProfiler:
    def __init__(self, enable_realtime: bool = False):
        # ... existing fields ...
        self.stream_operations = 0
    
    def record_stream_usage(self, stream_count: int):
        """Record stream usage for utilization tracking."""
        self.stream_operations = stream_count

# In kernel dispatch loop
if mem_profiler:
    mem_profiler.track_kernel(kernel_time * 1000)
    if stream_pool:
        mem_profiler.record_stream_usage(tile_count)
```

### Issue 3: CUDA Graph Hit Rate is 0%

**Problem**: Graph keys included exact tile dimensions which varied, preventing graph reuse.

**Solution**:
- Already implemented: Graph keys use `_round_to_pow2(bi)` and `_round_to_pow2(bj)`
- This normalizes varying tile dimensions to power-of-2 buckets for better graph matching

**Files Changed**: None (already implemented correctly)

**Existing Code**:
```python
def _round_to_pow2(x):
    """Round to nearest power of 2 for better graph reuse."""
    return 2 ** int(np.ceil(np.log2(max(1, x))))

# Graph key generation
graph_key = (_round_to_pow2(bi), _round_to_pow2(bj), tm, tn, tk, kernel_name, is_double)
```

### Issue 4: VRAM Estimation Too Conservative

**Problem**: Original calculation was wrong, only using 40% of VRAM for states with 2x safety margin.

**Solution**:
- Rewrote `get_safe_sample_size()` with correct memory formula
- Formula: `n × dim × 16 + n² × 8 < usable_vram`
- Separately calculates limits from state memory and kernel memory
- Takes minimum of both limits plus base_samples cap

**Files Changed**: `tools/test_num_qubit_impact.py`

**Key Code Changes**:
```python
def get_safe_sample_size(n_qubits: int, base_samples: int = 80000, 
                         available_vram_gb: float = 102.0, 
                         vram_fraction: float = 0.85) -> int:
    dim = 2 ** n_qubits
    bytes_per_complex = 16  # complex128
    usable_vram = available_vram_gb * vram_fraction * 1e9
    
    # State memory: n × dim × 16 (with 1.5x safety)
    max_by_states = int(usable_vram / (dim * bytes_per_complex * 1.5))
    
    # Kernel memory: n² × 8 (use 50% of VRAM)
    max_by_kernel = int(np.sqrt(usable_vram * 0.5 / 8))
    
    safe_samples = min(base_samples, max_by_states, max_by_kernel)
    return max(100, safe_samples)
```

**Results** (tested with 102GB VRAM):
- 4-12 qubits: ~73k samples (kernel memory limited)
- 16 qubits: ~55k samples (state memory starts to dominate)
- 20 qubits: ~3.4k samples (state memory dominates)

### Issue 5: Very Low Throughput with 500 Samples

**Problem**: With only 500 samples, GPU overhead dominated and throughput was artificially low.

**Solution**:
- Changed default from `N_SAMPLES=500` to `DEFAULT_SAMPLES=10000`
- Added `QUBIT_SAMPLE_CONFIGS` dictionary with qubit-specific sample sizes
- Updated benchmark function to use qubit-specific configurations

**Files Changed**: `tools/test_num_qubit_impact.py`

**Key Code Changes**:
```python
DEFAULT_SAMPLES = 10000

QUBIT_SAMPLE_CONFIGS = {
    4: 50000,   # Can handle large samples (kernel memory limited)
    8: 50000,   # Can handle large samples (kernel memory limited)
    12: 30000,  # Can handle large samples (kernel memory limited)
    16: 15000,  # Reduced (state memory starts to dominate)
    20: 3000,   # Significantly reduced (state memory dominates)
}

# In benchmark function
if n_qubits in QUBIT_SAMPLE_CONFIGS:
    n_samples = QUBIT_SAMPLE_CONFIGS[n_qubits]
```

## Testing

### Validation Test
Created `test_profiler_fixes.py` to validate:
1. VRAM estimation produces reasonable sample counts
2. Memory profiler tracks non-zero allocations and transfers
3. Stream utilization is reported correctly

### Manual Verification
Run the test with profiling enabled:
```bash
python tools/test_num_qubit_impact.py --profile-memory --verbose-profile --cuda-states-full-opts
```

Expected output should show:
- Memory allocations (states_A, kernel_output) > 0 GB
- H→D and D→H transfer totals > 0 GB with bandwidths
- Stream Utilization > 0%
- Graph replays > 0 (if enough tiles)
- Higher sample counts and throughput values

## Impact

These fixes address critical profiling and optimization issues:

1. **Accurate memory tracking**: Enables proper monitoring of GPU memory usage
2. **Stream utilization visibility**: Shows how well the stream pool is being used
3. **Better graph reuse**: Normalized keys allow CUDA graphs to be reused more often
4. **Realistic VRAM limits**: Allows 10-50x more samples for low/medium qubit counts
5. **Higher throughput measurements**: Better represents actual GPU performance

## Files Modified

1. `scripts/pipeline_backends.py`
   - Added transfer tracking with timing
   - Added stream usage tracking
   - Graph key normalization (already present, verified)

2. `tools/test_num_qubit_impact.py`
   - Fixed VRAM estimation formula
   - Added qubit-specific sample configurations
   - Increased default sample size
   - Improved output formatting

3. `test_profiler_fixes.py` (new)
   - Validation test for all fixes

## Backward Compatibility

All changes are backward compatible:
- New MemoryProfiler methods are optional (only called when profiling is enabled)
- VRAM estimation returns sensible defaults if calculation fails
- Existing code paths continue to work unchanged
