# CUDA_STATES Optimization Usage Guide

This guide documents how to leverage all the new cuda_states backend optimization parameters across the test files and training script.

## Overview

The `pipeline_backends.py` now provides comprehensive GPU optimization features:
- **VRAM-aware tiling**: Auto-compute optimal state_tile from available VRAM
- **Kernel autotuning**: Benchmark tile sizes (TILE_M, TILE_N, TILE_K) on first run
- **Bulk precomputation**: Build all states in one pass to minimize handoffs
- **Dynamic batch sizing**: Adjust batch size based on memory pressure
- **CUDA stream pool**: Parallel execution with multiple streams
- **CUDA graph optimization**: Capture and replay kernels for efficiency
- **Tile learning**: Learn optimal tiles from historical runs
- **Memory profiling**: Detailed GPU memory usage analysis

## Training Script: `train_svm_qkernel.py`

### Basic Usage

```bash
# Train with default optimizations (all enabled)
python train_svm_qkernel.py \
    --config configs/cifar10.yaml \
    --gram-backend cuda_states
```

### Full Optimization Control

```bash
# Train with custom optimization settings
python train_svm_qkernel.py \
    --config configs/cifar10.yaml \
    --gram-backend cuda_states \
    --state-tile -1 \              # Auto VRAM-aware sizing
    --vram-fraction 0.9 \           # Use 90% of VRAM
    --autotune \                    # Enable kernel autotuning
    --precompute-all-states \       # Bulk precompute all states
    --dynamic-batch \               # Enable dynamic batch sizing
    --num-streams 4 \               # Use 4 CUDA streams
    --use-cuda-graphs \             # Enable CUDA graph optimization
    --profile-memory \              # Enable memory profiling
    --verbose-profile               # Show detailed profiling output
```

### Disable Specific Optimizations

```bash
# Disable selected optimizations
python train_svm_qkernel.py \
    --config configs/cifar10.yaml \
    --gram-backend cuda_states \
    --no-autotune \                 # Disable kernel autotuning
    --no-precompute \               # Disable bulk precomputation
    --no-dynamic-batch \            # Disable dynamic batch sizing
    --no-cuda-graphs                # Disable CUDA graph optimization
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--state-tile` | int | -1 | State tile size (-1 for auto VRAM-aware) |
| `--vram-fraction` | float | 0.85 | Maximum VRAM fraction to use (0-1) |
| `--autotune` | flag | True | Enable kernel tile autotuning |
| `--no-autotune` | flag | - | Disable kernel autotuning |
| `--precompute-all-states` | flag | True | Bulk precompute all quantum states |
| `--no-precompute` | flag | - | Disable bulk precomputation |
| `--dynamic-batch` | flag | True | Enable dynamic batch sizing |
| `--no-dynamic-batch` | flag | - | Disable dynamic batch sizing |
| `--num-streams` | int | 4 | Number of CUDA streams for parallelism |
| `--learn-tiles` | flag | True | Learn optimal tiles from run history |
| `--no-learn-tiles` | flag | - | Disable tile learning |
| `--use-cuda-graphs` | flag | True | Enable CUDA graph optimization |
| `--no-cuda-graphs` | flag | - | Disable CUDA graph optimization |
| `--profile-memory` | flag | False | Enable GPU memory profiling |
| `--verbose-profile` | flag | False | Show detailed profiling output |

## Test Files

### 1. `tools/test_num_qubit_impact.py`

Tests performance scaling with qubit count.

**Basic Usage:**
```bash
# Run with default optimizations
python tools/test_num_qubit_impact.py
```

**With Custom Optimizations:**
```bash
# Test with all optimizations enabled
python tools/test_num_qubit_impact.py --cuda-states-full-opts

# Test with memory profiling
python tools/test_num_qubit_impact.py \
    --profile-memory \
    --verbose-profile

# Test with custom configuration
python tools/test_num_qubit_impact.py \
    --state-tile 2048 \
    --vram-fraction 0.9 \
    --num-streams 8 \
    --no-autotune
```

### 2. `tools/test_tile_samples_impact.py`

Tests tile size and sample count impact, including comprehensive optimization studies.

**Features:**
- Original tests: `test_cuda_states_tile_impact()`, `test_numpy_tile_workers_impact()`, `test_sample_scaling()`
- **New**: `test_state_tile_optimization()` - Test auto vs fixed state_tile values
- **New**: `test_num_streams_impact()` - Test stream count impact (1, 2, 4, 8)
- **New**: `test_vram_fraction_impact()` - Test memory pressure at different targets
- **New**: `test_optimization_ablation()` - Compare each optimization's contribution
- **New**: `test_sample_scaling_with_optimizations()` - Verify O(N²) with optimizations

**Usage:**
```bash
# Run all tests including new optimization tests
python tools/test_tile_samples_impact.py
```

**Output:** Results saved to `benchmark_results/tile_samples_impact_results.csv`

### 3. `benchmark_production.py`

Comprehensive production benchmark suite.

**Basic Usage:**
```bash
# Run all benchmarks
python benchmark_production.py

# Run specific tests
python benchmark_production.py --tests qubit sample tile

# Run with optimization ablation study
python benchmark_production.py --run-ablation

# Run with full profiling
python benchmark_production.py --profile-all
```

**New Test Functions:**

1. **`benchmark_optimization_ablation()`**
   - Compares performance with each optimization individually disabled
   - Configurations tested:
     - All Optimizations (baseline)
     - No Autotune
     - No Precompute
     - No Dynamic Batch
     - No CUDA Graphs
     - Baseline (No Opts)

2. **`benchmark_with_profiling()`**
   - Runs with full memory profiling enabled
   - Shows detailed GPU memory breakdown
   - Reports CUDA graph statistics
   - Stream utilization metrics

**CLI Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `--tests` | list | Tests to run: qubit, sample, tile, ablation, profile, all |
| `--backends` | list | Backends to test: cuda_states, torch, numpy, all |
| `--run-ablation` | flag | Run optimization ablation study |
| `--profile-all` | flag | Enable memory profiling for all tests |

**Examples:**
```bash
# Run optimization ablation study only
python benchmark_production.py --tests ablation

# Run profiling test with cuda_states backend
python benchmark_production.py --tests profile --backends cuda_states

# Run all tests with ablation and profiling
python benchmark_production.py --run-ablation --profile-all
```

## Expected Console Output

When running with profiling enabled, you should see:

```
📊 Auto-sized state_tile=8192 (using 85% VRAM)
🔧 Autotuned kernel tiles: M=32, N=32, K=64
🧠 Learned state_tile=8192 (confidence: 0.75)
📊 Memory profiling enabled
🌊 Stream pool initialized with 4 streams
🔄 Dynamic batch sizing enabled (initial=8192)
📈 CUDA graph optimization enabled

╔══════════════════════════════════════════════════════════════╗
║                    GPU PERFORMANCE REPORT                      ║
╠══════════════════════════════════════════════════════════════╣
║ Memory Usage                                                   ║
║   Peak Allocated:      12.5 GB /  96.0 GB ( 13.0%)           ║
║   states_A          :   8.0 GB                                ║
║   states_B          :   0.0 GB                                ║
║   kernel_output     :   4.5 GB                                ║
╠══════════════════════════════════════════════════════════════╣
║ Transfer Bandwidth                                             ║
║   H→D Total:           8.5 GB @ 12.3 GB/s                     ║
║   D→H Total:           4.5 GB @ 15.2 GB/s                     ║
╠══════════════════════════════════════════════════════════════╣
║ Kernel Performance                                             ║
║   Total Launches:     256                                      ║
║   Graph Replays:      230 (89.8% hit rate)                    ║
║   Avg Kernel Time:    2.35 ms                                 ║
║   Throughput:         458.2 Mpairs/s                          ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Dynamic Adjustments                                            ║
║   Batch Size Range:   4096 → 8192 (3 adjustments)             ║
║   Stream Utilization: 94.2%                                    ║
╚══════════════════════════════════════════════════════════════╝
```

## Hardware Requirements

Target Hardware: NVIDIA RTX 6000 Ada Generation
- VRAM: 96GB per GPU
- CUDA: 13.0+
- Compute Capability: 8.9

The optimizations are designed to maximize utilization of this high-end hardware while maintaining stability and efficiency.

## Optimization Recommendations

### For Maximum Throughput
```bash
--state-tile -1 \           # Auto-size based on VRAM
--vram-fraction 0.95 \      # Use 95% of available VRAM
--autotune \                # Let system find optimal kernel tiles
--precompute-all-states \   # Minimize handoffs
--dynamic-batch \           # Adjust to memory pressure
--num-streams 8 \           # Maximum parallelism
--use-cuda-graphs           # Maximum kernel efficiency
```

### For Maximum Stability
```bash
--state-tile 4096 \         # Fixed, conservative tile size
--vram-fraction 0.75 \      # Leave headroom for other processes
--autotune \                # Still benefit from tuning
--precompute-all-states \   # Keep bulk optimization
--num-streams 4 \           # Moderate parallelism
--use-cuda-graphs           # Kernel optimization
```

### For Debugging/Development
```bash
--state-tile 2048 \         # Small, manageable size
--vram-fraction 0.5 \       # Conservative memory use
--no-autotune \             # Disable tuning for predictability
--no-dynamic-batch \        # Fixed batch sizes
--num-streams 1 \           # Serial execution
--no-cuda-graphs \          # Disable graph optimization
--profile-memory \          # Enable profiling
--verbose-profile           # Show all details
```

## Troubleshooting

### Out of Memory Errors
- Reduce `--vram-fraction` (try 0.7 or 0.5)
- Set explicit `--state-tile` to a smaller value (e.g., 2048 or 1024)
- Disable `--no-precompute` to avoid large bulk allocations

### Slow Performance
- Ensure `--autotune` is enabled
- Increase `--num-streams` (try 8 or 16)
- Increase `--vram-fraction` if memory allows
- Ensure `--use-cuda-graphs` is enabled

### Unexpected Results
- Enable `--profile-memory --verbose-profile` to see what's happening
- Disable optimizations one by one to identify issues
- Check CUDA/PyTorch versions compatibility

## Output Files

All benchmarks save results to `benchmark_results/`:
- `qubit_impact_results.csv` - Qubit scaling test results
- `tile_samples_impact_results.csv` - Tile and sample impact results
- `production_benchmark.csv` - Comprehensive benchmark results
- `production_benchmark.png` - Performance plots
- `production_benchmark_summary.json` - Summary statistics

## WandB Integration

When using `train_svm_qkernel.py`, all optimization settings are logged to WandB:
- `fold_X/config/state_tile`
- `fold_X/config/vram_fraction`
- `fold_X/config/autotune`
- `fold_X/config/num_streams`
- `fold_X/config/dynamic_batch`
- `fold_X/config/use_cuda_graphs`
- `fold_X/config/precompute_all_states`
- `fold_X/config/learn_tiles`

This allows tracking which optimizations were used for each experiment.

## Implementation Notes

All files maintain backward compatibility:
- Default values match recommended settings for RTX 6000 Ada
- Parameters are passed through to `compute_kernel_matrix()` in `pipeline_backends.py`
- Non-cuda_states backends ignore the cuda-specific parameters
- All optimizations can be individually disabled for ablation studies

## Future Enhancements

Potential additions mentioned in the problem statement but not yet implemented:
- Heatmap generation for state_tile × num_streams
- Additional plots comparing optimization impacts
- More detailed optimization reports with per-optimization speedup calculations
