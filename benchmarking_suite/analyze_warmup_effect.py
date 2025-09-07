#!/usr/bin/env python3
"""
Script to analyze the effect of warm-up steps on benchmark results
For Problem (benchmarking_script) part (c)
"""

import subprocess
import sys
import numpy as np
import pandas as pd


def run_benchmark_with_warmup(model_size, n_warmup, n_steps=10):
    """Run benchmark with specified number of warm-up steps."""
    
    cmd = [
        sys.executable,
        "benchmark_model.py",
        "--model-size", model_size,
        "--n-warmup", str(n_warmup),
        "--n-steps", str(n_steps),
        "--batch-size", "4",
        "--sequence-length", "512",
        "--backward"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse timing from output
        lines = output.split('\n')
        
        for line in lines:
            if "Total:" in line and "±" in line:
                parts = line.split(":")[-1].strip().split("±")
                total_mean = float(parts[0].strip().replace(" ms", ""))
                total_std = float(parts[1].strip().replace(" ms", ""))
                return total_mean, total_std
        
        return None, None
    
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None, None


def main():
    print("=" * 60)
    print("ANALYZING WARM-UP EFFECT")
    print("=" * 60)
    
    # Test with medium model for reasonable runtime
    model_size = "medium"
    warmup_configs = [0, 1, 2, 5, 10]
    
    results = []
    
    print(f"\nTesting {model_size} model with different warm-up configurations")
    print("-" * 60)
    
    for n_warmup in warmup_configs:
        print(f"Running with {n_warmup} warm-up steps...")
        
        # Run multiple trials to see variability
        trials = []
        for trial in range(3):
            mean_time, std_time = run_benchmark_with_warmup(model_size, n_warmup, n_steps=5)
            if mean_time is not None:
                trials.append(mean_time)
        
        if trials:
            avg_time = np.mean(trials)
            trial_std = np.std(trials)
            
            results.append({
                "warmup_steps": n_warmup,
                "avg_time_ms": avg_time,
                "trial_std_ms": trial_std,
                "num_trials": len(trials)
            })
            
            print(f"  Average: {avg_time:.2f} ms (std across trials: {trial_std:.2f} ms)")
    
    # Create DataFrame and analyze
    df = pd.DataFrame(results)
    
    if not df.empty:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        print(df.to_string(index=False))
        
        # Analysis
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)
        
        # Compare no warm-up vs with warm-up
        no_warmup = df[df['warmup_steps'] == 0]['avg_time_ms'].values[0]
        with_warmup_5 = df[df['warmup_steps'] == 5]['avg_time_ms'].values[0]
        
        speedup = (no_warmup - with_warmup_5) / with_warmup_5 * 100
        
        print(f"\nEffect of warm-up steps:")
        print(f"  No warm-up:     {no_warmup:.2f} ms")
        print(f"  5 warm-up steps: {with_warmup_5:.2f} ms")
        print(f"  Difference:      {no_warmup - with_warmup_5:.2f} ms ({speedup:.1f}% slower without warm-up)")
        
        # Variability analysis
        no_warmup_std = df[df['warmup_steps'] == 0]['trial_std_ms'].values[0]
        with_warmup_std = df[df['warmup_steps'] == 5]['trial_std_ms'].values[0]
        
        print(f"\nVariability (std across trials):")
        print(f"  No warm-up:     {no_warmup_std:.2f} ms")
        print(f"  5 warm-up steps: {with_warmup_std:.2f} ms")
        
        print("\n" + "=" * 60)
        print("EXPLANATION")
        print("=" * 60)
        
        print("""
The warm-up effect occurs due to several factors:

1. **CUDA Kernel Compilation**: On first execution, CUDA kernels need to be
   compiled and optimized for the specific GPU architecture. This is a 
   one-time cost that affects the first few iterations.

2. **GPU State**: The GPU may be in a low-power state initially and needs
   to ramp up to full performance clocks. Warm-up steps ensure the GPU
   is running at optimal frequencies.

3. **Memory Caching**: Initial runs may experience cache misses as data
   is loaded into various cache levels. Subsequent runs benefit from
   warmed caches.

4. **PyTorch JIT Compilation**: PyTorch may perform just-in-time optimization
   of the computation graph during the first few executions.

5. **Memory Allocation**: The first iterations may trigger memory pool
   allocations that are reused in subsequent iterations.

Even 1-2 warm-up steps show significant improvement, but 5 steps is generally
sufficient to reach steady-state performance for most models.
        """)


if __name__ == "__main__":
    main()