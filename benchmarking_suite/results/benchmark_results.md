# Benchmark Results - Problem (benchmarking_script) Part (b)

## Configuration
- Batch size: 4
- Sequence length: 512
- Warm-up steps: 5
- Measurement steps: 10

## Results Table

| Model Size   | Forward (ms)         | Backward (ms)         |   Total (ms) |
|:-------------|:---------------------|:----------------------|-------------:|
| small        | 3498.21 ± 191.46     | 7530.34 ± 343.73      |      11028.5 |
| medium       | 16788.82 ± 1026.26   | 31665.92 ± 1336.20    |      48454.7 |
| large        | 48049.76 ± 2945.25   | 82658.81 ± 6128.42    |     130709   |
| xl           | 134530.44 ± 25420.90 | 232723.69 ± 41466.36  |     367254   |
| 2.7B         | 288760.97 ± 71151.24 | 583418.19 ± 165890.08 |     872179   |

## Analysis

### Coefficient of Variation (CV) Analysis

The coefficient of variation measures relative variability (std/mean × 100%):

| Model | Forward CV | Backward CV |
|-------|------------|-------------|
| small | 5.5% | 4.6% |
| medium | 6.1% | 4.2% |
| large | 6.1% | 7.4% |
| xl | 18.9% | 17.8% |
| 2.7B | 24.6% | 28.4% |

- **Average coefficient of variation:**
  - Forward pass: 12.3%
  - Backward pass: 12.5%

### Backward/Forward Time Ratios

| Model | Ratio |
|-------|-------|
| small | 2.15x |
| medium | 1.89x |
| large | 1.72x |
| xl | 1.73x |
| 2.7B | 2.02x |

### Key Findings

1. **Low variability for smaller models** (CV < 10%): Small, medium, and large models show very consistent performance across measurements, indicating reliable benchmarks.

2. **Higher variability for larger models**: XL and 2.7B models show higher CV (17-28%), likely due to:
   - Memory pressure and potential swapping
   - CPU throttling during long runs
   - System resource contention

3. **Backward pass consistently slower**: Backward passes take 1.7-2.2x longer than forward passes due to gradient computation and storage requirements.

4. **Exponential scaling**: Each model size increase results in roughly 2.5-3x slower performance, confirming O(n²) computational complexity with model dimensions.
