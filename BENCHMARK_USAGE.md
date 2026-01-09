# Comprehensive Benchmark Usage Guide

This document describes how to use `benchmark_production.py` to comprehensively benchmark ALL optimizations available in both the `cuda_states` and `torch` backends.

## Overview

The benchmark script provides 10 comprehensive benchmark sections covering:

1. **Backend Comparison** - Compare all backends with default optimizations
2. **cuda_states Optimization Ablation** - Test each optimization individually
3. **cuda_states State Tile Optimization** - Find optimal state_tile values
4. **cuda_states VRAM Fraction Impact** - Test impact of vram_fraction
5. **cuda_states Stream Pool Impact** - Test num_streams impact
6. **torch Optimization Ablation** - Test each torch optimization
7. **torch Tile Size Optimization** - Find optimal tile_size
8. **Memory Profiling** - Run with detailed memory profiling
9. **Qubit Scaling Analysis** - Measure exponential scaling
10. **Sample Scaling Analysis** - Verify O(N²) scaling

## Quick Start

### Run All Benchmarks

```bash
python benchmark_production.py --all
```

### Run Specific Benchmark Sections

```bash
# Backend comparison only
python benchmark_production.py --backend-comparison

# cuda_states optimization study
python benchmark_production.py --cuda-states-ablation --cuda-states-state-tile \
    --cuda-states-vram --cuda-states-streams

# torch optimization study
python benchmark_production.py --torch-ablation --torch-tiles

# Scaling analysis
python benchmark_production.py --qubit-scaling --sample-scaling
```

## CLI Arguments

### Test Selection

| Argument | Description |
|----------|-------------|
| `--all` | Run all benchmarks |
| `--backend-comparison` | Compare cuda_states, torch, and numpy backends |
| `--cuda-states-ablation` | Test each cuda_states optimization individually |
| `--cuda-states-state-tile` | Test state_tile values [-1, 512, 1024, 2048, 4096, 8192] |
| `--cuda-states-vram` | Test vram_fraction values [0.5, 0.7, 0.85, 0.95] |
| `--cuda-states-streams` | Test num_streams values [1, 2, 4, 8] |
| `--torch-ablation` | Test each torch optimization individually |
| `--torch-tiles` | Test tile_size values [64, 128, 256, 512, 1024, 2048] |
| `--memory-profiling` | Run with detailed memory profiling |
| `--qubit-scaling` | Test scaling with qubit counts [4-20] |
| `--sample-scaling` | Test scaling with sample counts [1000-16000] |

### Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-samples` | 8000 | Number of samples for benchmarks |
| `--n-qubits` | 10 | Number of qubits for benchmarks |
| `--output-dir` | benchmark_results | Output directory for results |
| `--warmup-runs` | 1 | Number of warmup runs before timing |
| `--benchmark-runs` | 3 | Number of timed benchmark runs |
| `--verbose` | False | Print detailed information |

## Usage Examples

### Example 1: Quick Backend Comparison

Test the three backends with a smaller workload:

```bash
python benchmark_production.py --backend-comparison --n-samples 4000 --n-qubits 8
```

### Example 2: Full cuda_states Optimization Study

Comprehensive study of all cuda_states optimizations:

```bash
python benchmark_production.py \
    --cuda-states-ablation \
    --cuda-states-state-tile \
    --cuda-states-vram \
    --cuda-states-streams \
    --n-samples 8000 \
    --n-qubits 10 \
    --verbose
```

### Example 3: torch Backend Optimization

Find optimal torch settings:

```bash
python benchmark_production.py \
    --torch-ablation \
    --torch-tiles \
    --n-samples 8000 \
    --verbose
```

### Example 4: Memory Profiling

Run with detailed memory profiling:

```bash
python benchmark_production.py \
    --memory-profiling \
    --backend-comparison \
    --n-samples 4000 \
    --verbose
```

### Example 5: Scaling Analysis

Analyze how performance scales:

```bash
python benchmark_production.py \
    --qubit-scaling \
    --sample-scaling \
    --verbose
```

### Example 6: Production-Ready Full Suite

Run the complete benchmark suite with production settings:

```bash
python benchmark_production.py \
    --all \
    --n-samples 8000 \
    --n-qubits 10 \
    --warmup-runs 2 \
    --benchmark-runs 5 \
    --output-dir production_benchmark_results \
    --verbose
```

## Output Files

The benchmark generates comprehensive output in the specified output directory:

### CSV Files

- `backend_comparison.csv` - Backend comparison results
- `cuda_states_ablation.csv` - cuda_states ablation study results
- `cuda_states_state_tile.csv` - State tile optimization results
- `cuda_states_vram_fraction.csv` - VRAM fraction impact results
- `cuda_states_streams.csv` - Stream pool impact results
- `torch_ablation.csv` - torch ablation study results
- `torch_tile_sizes.csv` - torch tile optimization results
- `memory_profiling.csv` - Memory profiling results
- `qubit_scaling.csv` - Qubit scaling results
- `sample_scaling.csv` - Sample scaling results
- `production_benchmark.csv` - Combined results from all tests
- `production_benchmark_summary.json` - Summary statistics

### Plots

- `production_benchmark.png` - Comprehensive visualization with 6 subplots:
  - Throughput vs Qubits
  - Time vs Qubits (Log Scale)
  - GPU Memory Usage
  - Sample Scaling (O(N²) verification)
  - Tile Size Optimization
  - GPU Speedup vs CPU

## Interpreting Results

### Backend Comparison

Look for:
- **Best throughput** (Mpairs/s): Higher is better
- **GPU memory usage**: Ensure it fits in available VRAM
- **Speedup vs CPU**: How much faster GPU backends are

### Optimization Ablation

- **Baseline**: Performance with all optimizations OFF
- **Individual optimizations**: Contribution of each optimization
- **Full optimizations**: Performance with all optimizations ON
- **Speedup factor**: Full optimizations vs Baseline

### Tile Size Optimization

- **Optimal tile size**: Balance between memory usage and performance
- **Performance curve**: How throughput varies with tile size
- **Memory impact**: Larger tiles may use more memory

### VRAM Fraction Impact

- **Sweet spot**: Best vram_fraction for your workload
- **Stability**: Higher fractions may be less stable
- **Throughput**: Impact on performance

### Stream Pool Impact

- **Optimal stream count**: More isn't always better
- **Diminishing returns**: Look for performance plateau
- **Overhead**: Too many streams can add overhead

### Scaling Analysis

- **Qubit scaling**: Should show exponential growth (2^n)
- **Sample scaling**: Should show O(N²) behavior
- **Backend comparison**: How different backends scale

## Performance Tips

1. **Start small**: Test with smaller workloads first (e.g., --n-samples 2000)
2. **Use warmup**: Always use at least 1 warmup run to prime the GPU
3. **Multiple runs**: Use 3-5 benchmark runs for accurate timing
4. **Monitor memory**: Watch GPU memory usage, especially for large qubit counts
5. **Optimize incrementally**: Test one optimization at a time before combining

## Troubleshooting

### Out of Memory Errors

- Reduce `--n-samples` or `--n-qubits`
- Lower `vram_fraction` (e.g., 0.7 instead of 0.95)
- Use smaller state_tile values
- Close other GPU applications

### Slow Performance

- Ensure GPU isn't being used by other processes
- Check that CUDA is properly installed
- Verify that optimizations are enabled
- Try different tile sizes

### Import Errors

Ensure all dependencies are installed:
```bash
pip install numpy pandas matplotlib seaborn torch cupy
```

## Advanced Usage

### Custom Configuration

You can modify `BACKEND_CONFIGS` in the script to test custom configurations:

```python
BACKEND_CONFIGS = {
    "cuda_states": {
        "state_tile": 4096,  # Custom value
        "vram_fraction": 0.9,  # Custom value
        "num_streams": 8,  # Custom value
        # ... other parameters
    },
}
```

### Batch Testing

Create a shell script to run multiple configurations:

```bash
#!/bin/bash
for samples in 2000 4000 8000 16000; do
    for qubits in 8 10 12 14; do
        python benchmark_production.py \
            --backend-comparison \
            --n-samples $samples \
            --n-qubits $qubits \
            --output-dir results_${samples}_${qubits}
    done
done
```

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM, CUDA 11.0+
- **Recommended**: NVIDIA GPU with 16GB+ VRAM, CUDA 12.0+
- **Optimal**: NVIDIA RTX 6000 Ada (96GB VRAM), CUDA 13.0

## References

- [GPU Optimizations Documentation](GPU_OPTIMIZATIONS.md)
- [CUDA Optimizations Usage](CUDA_OPTIMIZATIONS_USAGE.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
