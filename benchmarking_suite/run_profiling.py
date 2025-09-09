#!/usr/bin/env python3
"""
Runner script to execute profiling for all model/context length combinations
CS336 Assignment 2 - Nsight Systems Profiling
"""

import subprocess
import os
import json
from datetime import datetime
import argparse
from typing import List, Dict

from logging_config import create_profiling_logger

# Set up logging
logger = create_profiling_logger(__name__)


# Model sizes to test
MODEL_SIZES = ["small", "medium", "large", "xl", "2.7B"]

# Context lengths to test
CONTEXT_LENGTHS = [128, 256, 512, 1024]

# Default batch size
DEFAULT_BATCH_SIZE = 4


def create_output_dir(base_dir: str = "profiling_results") -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_profiling_command(
    model_size: str,
    context_length: int,
    output_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_nsys: bool = True
) -> Dict:
    """
    Run a single profiling command.

    Args:
        model_size: Model configuration name
        context_length: Sequence/context length
        output_dir: Directory for output files
        batch_size: Batch size for testing
        use_nsys: Whether to use nsys profiling

    Returns:
        Dictionary with results or error status
    """

    # Base output filename
    base_name = f"{model_size}_ctx{context_length}_bs{batch_size}"
    json_output = os.path.join(output_dir, f"{base_name}.json")

    # Build command
    if use_nsys:
        # Nsight Systems profiling command
        profile_output = os.path.join(output_dir, f"{base_name}")
        cmd = [
            "uv", "run",
            "nsys", "profile",
            "--pytorch=autograd-nvtx",  # Automatic PyTorch annotation with NVTX
            f"--output={profile_output}",
            "--force-overwrite=true",
            "--python-backtrace=cuda",  # Python backtraces for CUDA calls
            "python", "benchmark_profiling.py",
            "--model-size", model_size,
            "--sequence-length", str(context_length),
            "--context-length", str(context_length),
            "--batch-size", str(batch_size),
            "--n-warmup", "5",
            "--n-steps", "10",
            "--with-optimizer",
            "--annotate-attention",
            "--output-json", json_output
        ]
    else:
        # Direct Python execution (for testing without nsys)
        cmd = [
            "python", "benchmark_profiling.py",
            "--model-size", model_size,
            "--sequence-length", str(context_length),
            "--context-length", str(context_length),
            "--batch-size", str(batch_size),
            "--n-warmup", "5",
            "--n-steps", "10",
            "--with-optimizer",
            "--annotate-attention",
            "--output-json", json_output
        ]

    logger.info("Running: %s with context length %s", model_size, context_length)
    logger.info("Command: %s", ' '.join(cmd))

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            check=False
        )

        # Check if successful
        if result.returncode == 0:
            logger.info("✓ Success: %s", base_name)

            # Load and return JSON results if available
            if os.path.exists(json_output):
                with open(json_output, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"status": "success", "note": "No JSON output"}
        
        # Detailed error analysis
        stderr_lower = result.stderr.lower()
        stdout_lower = result.stdout.lower()
        
        # Check for actual OOM errors (not just exit code 1)
        oom_indicators = [
            "outofmemoryerror", 
            "out of memory", 
            "cuda out of memory",
            "cuda error: out of memory",
            "runtime error: cuda out of memory"
        ]
        
        if any(indicator in stderr_lower or indicator in stdout_lower for indicator in oom_indicators):
            logger.warning("✗ OOM: %s", base_name)
            logger.warning("OOM details: %s", result.stderr[:200])
            return {"status": "OOM", "model_size": model_size, "context_length": context_length}
        
        # Check for import/module errors
        import_indicators = ["importerror", "modulenotfounderror", "no module named"]
        if any(indicator in stderr_lower for indicator in import_indicators):
            logger.error("✗ Import Error: %s", base_name)
            logger.error("Import error details: %s", result.stderr[:300])
            return {"status": "import_error", "error": result.stderr}
        
        # Check for CUDA errors
        cuda_indicators = ["cuda error", "cuda runtime error", "cuda driver", "no cuda device"]
        if any(indicator in stderr_lower for indicator in cuda_indicators):
            logger.error("✗ CUDA Error: %s", base_name)
            logger.error("CUDA error details: %s", result.stderr[:300])
            return {"status": "cuda_error", "error": result.stderr}
        
        # Check for argument/configuration errors
        if result.returncode == 2 or "argument" in stderr_lower or "usage:" in stderr_lower:
            logger.error("✗ Argument Error: %s", base_name)
            logger.error("Argument error details: %s", result.stderr[:300])
            return {"status": "argument_error", "error": result.stderr}
        
        # Generic error with more details
        logger.error("✗ Error: %s (exit code %d)", base_name, result.returncode)
        logger.error("Command: %s", ' '.join(cmd))
        logger.error("STDERR: %s", result.stderr[:500])
        if result.stdout:
            logger.error("STDOUT: %s", result.stdout[:200])
        
        return {
            "status": "error", 
            "error": result.stderr, 
            "exit_code": result.returncode,
            "stdout": result.stdout[:200] if result.stdout else None
        }

    except subprocess.TimeoutExpired:
        logger.error("✗ Timeout: %s", base_name)
        return {"status": "timeout", "model_size": model_size, "context_length": context_length}
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("✗ Exception: %s - %s", base_name, str(e))
        return {"status": "exception", "error": str(e)}


def generate_summary_report(results: List[Dict], output_dir: str):
    """Generate a summary report of all profiling runs."""

    summary_file = os.path.join(output_dir, "summary_report.md")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Profiling Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Create results table
        f.write("## Results Table\n\n")
        f.write("| Model | Context | Status | Forward (ms) | Backward (ms) | Total (ms) |\n")
        f.write("|-------|---------|--------|--------------|---------------|------------|\n")

        for result in results:
            model = result.get("model_size", "?")
            ctx = result.get("context_length", result.get("sequence_length", "?"))
            status = result.get("status", "unknown")

            if status == "OOM":
                f.write(f"| {model} | {ctx} | OOM | - | - | - |\n")
            elif status == "timeout":
                f.write(f"| {model} | {ctx} | Timeout | - | - | - |\n")
            elif "stats" in result:
                stats = result["stats"]
                forward = f"{stats.get('forward_times_mean', 0)*1000:.1f}"
                backward = f"{stats.get('backward_times_mean', 0)*1000:.1f}"
                total = f"{stats.get('total_times_mean', 0)*1000:.1f}"
                f.write(f"| {model} | {ctx} | ✓ | {forward} | {backward} | {total} |\n")
            else:
                f.write(f"| {model} | {ctx} | {status} | - | - | - |\n")

        # Add comparison with previous benchmarks if available
        f.write("\n## Comparison with Python Timer Benchmarks\n\n")
        f.write("Compare the forward pass times from nsys profiling with the previous ")
        f.write("Python standard library timing results to verify consistency.\n\n")

        # Add notes
        f.write("## Notes\n\n")
        f.write("- OOM: Out of Memory error occurred\n")
        f.write("- Times are averaged over 10 measurement steps (after 5 warmup steps)\n")
        f.write("- Profiling includes NVTX annotations for detailed kernel analysis\n")
        f.write("- Use `nsys-ui <profile>.nsys-rep` to view detailed timeline\n")

    logger.info("Summary report saved to: %s", summary_file)


def main():
    """Main function to run profiling for all model/context combinations."""
    parser = argparse.ArgumentParser(description="Run profiling for all model/context combinations")
    parser.add_argument("--output-dir", type=str, default="profiling_results",
                        help="Base directory for output files")
    parser.add_argument("--models", nargs="+", default=MODEL_SIZES,
                        help="Model sizes to test")
    parser.add_argument("--contexts", nargs="+", type=int, default=CONTEXT_LENGTHS,
                        help="Context lengths to test")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for testing")
    parser.add_argument("--no-nsys", action="store_true",
                        help="Run without nsys (for testing)")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip xl and 2.7B models")

    args = parser.parse_args()

    # Filter models if skipping large ones
    models = args.models
    if args.skip_large:
        models = [m for m in models if m not in ["xl", "2.7B"]]

    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    logger.info("Output directory: %s", output_dir)

    # Change to benchmarking_suite directory
    original_dir = os.getcwd()
    benchmarking_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(benchmarking_dir)

    # Collect all results
    all_results = []
    total_configs = len(models) * len(args.contexts)
    current = 0

    logger.info("Testing %s configurations:", total_configs)
    logger.info("Models: %s", models)
    logger.info("Context lengths: %s", args.contexts)
    logger.info("Batch size: %s", args.batch_size)
    logger.info("=" * 60)

    try:
        for model_size in models:
            for context_length in args.contexts:
                current += 1
                logger.info("[%s/%s] Starting configuration", current, total_configs)

                result = run_profiling_command(
                    model_size=model_size,
                    context_length=context_length,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    use_nsys=not args.no_nsys
                )

                all_results.append(result)

                # Save intermediate results
                intermediate_file = os.path.join(output_dir, "all_results.json")
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2)

        # Generate summary report
        generate_summary_report(all_results, output_dir)

        logger.info("=" * 60)
        logger.info("Profiling complete!")
        logger.info("Results saved in: %s", output_dir)

    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
