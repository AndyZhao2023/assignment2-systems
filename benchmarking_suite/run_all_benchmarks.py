#!/usr/bin/env python3
"""
Script to run benchmarks for all model sizes and generate results table
For Problem (benchmarking_script) part (b)
"""

import subprocess
import json
import pandas as pd
import sys
import os
import logging
import datetime
import time


def run_benchmark(model_size, backward=True, n_warmup=5, n_steps=10):
    """Run benchmark for a specific model size and parse results."""
    
    start_time = time.time()
    cmd = [
        sys.executable, 
        "benchmark_model.py",
        "--model-size", model_size,
        "--n-warmup", str(n_warmup),
        "--n-steps", str(n_steps),
        "--batch-size", "4",
        "--sequence-length", "512",
    ]
    
    if backward:
        cmd.append("--backward")
    
    logging.info(f"Starting benchmark for {model_size} model...")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        elapsed_time = time.time() - start_time
        logging.info(f"Benchmark for {model_size} completed in {elapsed_time:.1f}s")
        
        # Parse the output to extract timing information
        lines = output.split('\n')
        
        forward_time = None
        backward_time = None
        
        for line in lines:
            if "Forward pass:" in line and "±" in line:
                # Extract mean and std
                parts = line.split(":")[-1].strip().split("±")
                forward_mean = float(parts[0].strip().replace(" ms", ""))
                forward_std = float(parts[1].strip().replace(" ms", ""))
                forward_time = (forward_mean, forward_std)
                logging.info(f"{model_size} - Forward: {forward_mean:.2f}±{forward_std:.2f}ms")
            
            if "Backward pass:" in line and "±" in line:
                # Extract mean and std  
                parts = line.split(":")[-1].strip().split("±")
                backward_mean = float(parts[0].strip().replace(" ms", ""))
                backward_std = float(parts[1].strip().replace(" ms", ""))
                backward_time = (backward_mean, backward_std)
                logging.info(f"{model_size} - Backward: {backward_mean:.2f}±{backward_std:.2f}ms")
        
        result_data = {
            "model_size": model_size,
            "forward_mean": forward_time[0] if forward_time else None,
            "forward_std": forward_time[1] if forward_time else None,
            "backward_mean": backward_time[0] if backward_time else None,
            "backward_std": backward_time[1] if backward_time else None,
        }
        
        logging.info(f"Successfully parsed results for {model_size}")
        return result_data
    
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Error running benchmark for {model_size} after {elapsed_time:.1f}s: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Unexpected error for {model_size} after {elapsed_time:.1f}s: {e}")
        return None


def main():
    # Setup logging
    log_filename = f"benchmark_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Keep console output too
        ]
    )
    
    model_sizes = ["small", "medium", "large", "xl", "2.7B"]
    results = []
    overall_start_time = time.time()
    
    logging.info("=" * 60)
    logging.info("STARTING BENCHMARK SUITE")
    logging.info("=" * 60)
    logging.info("Configuration: batch_size=4, sequence_length=512")
    logging.info("Warm-up steps: 5, Measurement steps: 10")
    logging.info(f"Model sizes to test: {', '.join(model_sizes)}")
    logging.info(f"Log file: {log_filename}")
    logging.info("=" * 60)
    
    successful_models = []
    failed_models = []
    
    for i, model_size in enumerate(model_sizes, 1):
        logging.info(f"[{i}/{len(model_sizes)}] Processing {model_size} model...")
        result = run_benchmark(model_size, backward=True)
        if result:
            results.append(result)
            successful_models.append(model_size)
            logging.info(f"✓ {model_size} benchmark completed successfully")
        else:
            failed_models.append(model_size)
            logging.warning(f"✗ {model_size} benchmark failed")
        
        logging.info("-" * 40)
    
    # Summary of execution
    overall_elapsed = time.time() - overall_start_time
    logging.info("=" * 60)
    logging.info("BENCHMARK SUITE COMPLETED")
    logging.info("=" * 60)
    logging.info(f"Total execution time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    logging.info(f"Successful models: {successful_models}")
    if failed_models:
        logging.warning(f"Failed models: {failed_models}")
    
    # Create DataFrame and display results
    df = pd.DataFrame(results)
    
    if not df.empty:
        logging.info("\n" + "=" * 60)
        logging.info("BENCHMARK RESULTS (all times in milliseconds)")
        logging.info("=" * 60)
        
        # Format the output nicely
        formatted_df = pd.DataFrame({
            "Model Size": df["model_size"],
            "Forward (ms)": df.apply(lambda x: f"{x['forward_mean']:.2f} ± {x['forward_std']:.2f}" 
                                    if pd.notna(x['forward_mean']) else "N/A", axis=1),
            "Backward (ms)": df.apply(lambda x: f"{x['backward_mean']:.2f} ± {x['backward_std']:.2f}" 
                                     if pd.notna(x['backward_mean']) else "N/A", axis=1),
            "Total (ms)": df.apply(lambda x: f"{x['forward_mean'] + x['backward_mean']:.2f}" 
                                  if pd.notna(x['forward_mean']) and pd.notna(x['backward_mean']) 
                                  else "N/A", axis=1)
        })
        
        # Log the results table
        for line in formatted_df.to_string(index=False).split('\n'):
            logging.info(line)
        
        # Save to CSV for later analysis
        df.to_csv("benchmark_results.csv", index=False)
        logging.info("Results saved to benchmark_results.csv")
        
        # Save to Markdown for writeup
        with open("benchmark_results.md", "w") as f:
            f.write("# Benchmark Results - Problem (benchmarking_script) Part (b)\n\n")
            f.write("## Configuration\n")
            f.write("- Batch size: 4\n")
            f.write("- Sequence length: 512\n") 
            f.write("- Warm-up steps: 5\n")
            f.write("- Measurement steps: 10\n\n")
            f.write("## Results Table\n\n")
            f.write(formatted_df.to_markdown(index=False))
            f.write("\n\n## Analysis\n\n")
            f.write(f"- Average coefficient of variation:\n")
            f.write(f"  - Forward pass: {avg_forward_cv:.1f}%\n")
            f.write(f"  - Backward pass: {avg_backward_cv:.1f}%\n\n")
            
            # Add backward/forward ratios to markdown
            f.write("- Backward/Forward time ratios:\n")
            for _, row in df.iterrows():
                if pd.notna(row['forward_mean']) and pd.notna(row['backward_mean']):
                    ratio = row['backward_mean'] / row['forward_mean']
                    f.write(f"  - {row['model_size']:8s}: {ratio:.2f}x\n")
            
            f.write("\n- Note: Backward pass typically takes ~2-3x longer than forward pass ")
            f.write("due to gradient computation and storage requirements.\n")
        
        logging.info("Results saved to benchmark_results.md")
        
        # Analysis for part (b)
        logging.info("\n" + "=" * 60)
        logging.info("ANALYSIS")
        logging.info("=" * 60)
        
        # Check if standard deviations are small
        avg_forward_cv = (df["forward_std"] / df["forward_mean"]).mean() * 100
        avg_backward_cv = (df["backward_std"] / df["backward_mean"]).mean() * 100
        
        logging.info(f"Average coefficient of variation (CV):")
        logging.info(f"  Forward pass:  {avg_forward_cv:.1f}%")
        logging.info(f"  Backward pass: {avg_backward_cv:.1f}%")
        
        if avg_forward_cv < 10 and avg_backward_cv < 10:
            logging.info("\n✓ Low variability observed - standard deviations are small relative to means")
            logging.info("  This indicates consistent performance across measurements.")
        else:
            logging.warning("\n⚠ Higher variability observed - consider increasing warm-up steps or")
            logging.warning("  measurement steps for more stable results.")
        
        # Backward vs Forward ratio
        logging.info(f"\nBackward/Forward time ratio:")
        for _, row in df.iterrows():
            if pd.notna(row['forward_mean']) and pd.notna(row['backward_mean']):
                ratio = row['backward_mean'] / row['forward_mean']
                logging.info(f"  {row['model_size']:8s}: {ratio:.2f}x")
        
        logging.info("\nNote: Backward pass typically takes ~2-3x longer than forward pass")
        logging.info("due to gradient computation and storage requirements.")
    
    else:
        logging.warning("No results collected. Please check for errors above.")
        
    logging.info(f"\nComplete log saved to: {log_filename}")


if __name__ == "__main__":
    main()