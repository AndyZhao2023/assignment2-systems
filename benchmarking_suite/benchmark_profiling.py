#!/usr/bin/env python3
"""
NVTX-annotated benchmarking script for CS336 Assignment 2
Profiling with Nsight Systems to analyze Transformer model performance
"""

import argparse
import contextlib
import json
import sys
import timeit
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from logging_config import create_profiling_logger

# Set up logging
logger = create_profiling_logger(__name__)

# Check if NVTX is available
try:
    from torch.cuda import nvtx

    # Check if NVTX actually works by trying to use it
    if torch.cuda.is_available():
        try:
            with nvtx.range("test"):
                pass
            NVTX_AVAILABLE = True
        except RuntimeError:
            NVTX_AVAILABLE = False
    else:
        NVTX_AVAILABLE = False
except ImportError:
    NVTX_AVAILABLE = False

# If NVTX not available, create dummy implementation
if not NVTX_AVAILABLE:

    class DummyNVTX:
        """Dummy NVTX class for when CUDA/NVTX is not available."""

        @staticmethod
        def range(_):
            """Dummy range method that returns a null context."""
            return contextlib.nullcontext()

    nvtx = DummyNVTX()


# Model configurations from Table 1 in Section 1.1.2
MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32
    },
}


def create_model(model_size: str, **model_kwargs) -> BasicsTransformerLM:
    """Create a Transformer model with the specified configuration.

    Args:
        model_size: Size configuration name from MODEL_CONFIGS
        **model_kwargs: Additional model parameters (vocab_size, context_length, device)
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Invalid model size: {model_size}. "
            f"Choose from {list(MODEL_CONFIGS.keys())}"
        )

    # Set defaults
    vocab_size = model_kwargs.get("vocab_size", 10000)
    context_length = model_kwargs.get("context_length", 512)
    device = model_kwargs.get("device", "cpu")

    config = MODEL_CONFIGS[model_size]

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,  # Default RoPE theta
    )

    return model.to(device)


def generate_random_batch(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    device: str = "cpu"
) -> torch.LongTensor:
    """Generate random input data for benchmarking."""
    return torch.randint(
        0, vocab_size, (batch_size, sequence_length), device=device
    )


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NVTX-annotated version of scaled dot product attention.
    This replaces the original cs336_basics implementation for profiling.
    Maintains exact mathematical equivalence with the original.
    """
    import math
    from einops import einsum
    from cs336_basics.nn_utils import softmax

    # Use NVTX only if available
    nvtx_available = NVTX_AVAILABLE

    if nvtx_available:
        with nvtx.range("scaled_dot_product_attention"):
            with nvtx.range("computing_attention_scores"):
                # Compute attention scores using einsum (matches original)
                d_k = K.shape[-1]
                attention_scores = einsum(
                    Q, K, "... query d_k, ... key d_k -> ... query key"
                ) / math.sqrt(d_k)

            with nvtx.range("applying_mask"):
                # Apply mask if provided (matches original logic)
                if mask is not None:
                    attention_scores = torch.where(
                        mask, attention_scores, float("-inf")
                    )

            with nvtx.range("computing_softmax"):
                # Apply softmax (using cs336_basics softmax function)
                attention_weights = softmax(attention_scores, dim=-1)

            with nvtx.range("final_matmul"):
                # Compute output using einsum (matches original)
                output = einsum(
                    attention_weights,
                    V,
                    "... query key, ... key d_v -> ... query d_v"
                )
    else:
        # CPU/non-CUDA version without NVTX
        d_k = K.shape[-1]
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

        attention_weights = softmax(attention_scores, dim=-1)
        output = einsum(
            attention_weights, V, "... query key, ... key d_v -> ... query d_v"
        )

    return output


def benchmark_with_profiling(model: nn.Module, data: torch.Tensor, **config) -> Dict:
    """
    Benchmark model with NVTX annotations for profiling.
    Returns detailed timing information.

    Args:
        model: The model to benchmark
        data: Input data tensor
        **config: Configuration dict containing:
            n_warmup (int): Number of warmup steps (default: 5)
            n_steps (int): Number of measurement steps (default: 10)
            use_optimizer (bool): Include optimizer step (default: False)
            use_cuda_sync (bool): Use CUDA synchronization (default: True)
    """
    # Extract config parameters with defaults
    n_warmup = config.get("n_warmup", 5)
    n_steps = config.get("n_steps", 10)
    use_optimizer = config.get("use_optimizer", False)
    use_cuda_sync = config.get("use_cuda_sync", True)

    results = {
        "forward_times": [],
        "backward_times": [],
        "optimizer_times": [],
        "total_times": [],
    }

    # Create loss function
    loss_fn = nn.CrossEntropyLoss()

    # Create target data (shifted input for language modeling)
    target = torch.cat([data[:, 1:], data[:, :1]], dim=1)

    # Create optimizer if requested
    optimizer = None
    if use_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warm-up steps (marked for filtering in profiler)
    with nvtx.range("warmup_phase"):
        for i in range(n_warmup):
            with nvtx.range(f"warmup_step_{i}"):
                # Forward pass
                with nvtx.range("forward_pass"):
                    output = model(data)
                    loss = loss_fn(
                        output.reshape(-1, output.size(-1)), target.reshape(-1)
                    )

                # Backward pass
                with nvtx.range("backward_pass"):
                    loss.backward()

                # Optimizer step
                if optimizer:
                    with nvtx.range("optimizer_step"):
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    model.zero_grad()

                if use_cuda_sync and data.is_cuda:
                    torch.cuda.synchronize()

    # Measurement steps
    with nvtx.range("measurement_phase"):
        for i in range(n_steps):
            with nvtx.range(f"measurement_step_{i}"):
                total_start = timeit.default_timer()

                # Time forward pass
                with nvtx.range("forward_pass"):
                    forward_start = timeit.default_timer()
                    output = model(data)
                    loss = loss_fn(
                        output.reshape(-1, output.size(-1)), target.reshape(-1)
                    )
                    if use_cuda_sync and data.is_cuda:
                        torch.cuda.synchronize()
                    forward_end = timeit.default_timer()
                    results["forward_times"].append(forward_end - forward_start)

                # Time backward pass
                with nvtx.range("backward_pass"):
                    backward_start = timeit.default_timer()
                    loss.backward()
                    if use_cuda_sync and data.is_cuda:
                        torch.cuda.synchronize()
                    backward_end = timeit.default_timer()
                    results["backward_times"].append(backward_end - backward_start)

                # Time optimizer step
                if optimizer:
                    with nvtx.range("optimizer_step"):
                        optimizer_start = timeit.default_timer()
                        optimizer.step()
                        optimizer.zero_grad()
                        if use_cuda_sync and data.is_cuda:
                            torch.cuda.synchronize()
                        optimizer_end = timeit.default_timer()
                        results["optimizer_times"].append(
                            optimizer_end - optimizer_start
                        )
                else:
                    model.zero_grad()
                    results["optimizer_times"].append(0.0)

                total_end = timeit.default_timer()
                results["total_times"].append(total_end - total_start)

    # Calculate statistics
    stats = {}
    for key, value in results.items():
        if value:
            stats[f"{key}_mean"] = np.mean(value)
            stats[f"{key}_std"] = np.std(value)

    return stats


def create_markdown_report(
    args,
    stats: Dict,
    total_params: int,
    device: str,
    status: str = "success"
) -> str:
    """
    Create a formatted markdown report of the benchmarking results.
    
    Args:
        args: Command-line arguments
        stats: Dictionary of timing statistics
        total_params: Total number of model parameters
        device: Device used (cuda/cpu)
        status: Status of the run ("success" or "OOM")
    
    Returns:
        Formatted markdown string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the markdown report
    md_lines = []
    md_lines.append(f"# Benchmarking Results: {args.model_size.upper()} Model\n")
    md_lines.append(f"*Generated: {timestamp}*\n")
    
    # Configuration section
    md_lines.append("## Configuration\n")
    md_lines.append(f"- **Model Size**: {args.model_size}")
    md_lines.append(f"- **Device**: {device}")
    md_lines.append(f"- **Total Parameters**: {total_params / 1e6:.2f}M")
    md_lines.append(f"- **Vocabulary Size**: {args.vocab_size:,}")
    md_lines.append("")
    
    # Test parameters section
    md_lines.append("## Test Parameters\n")
    md_lines.append(f"- **Batch Size**: {args.batch_size}")
    md_lines.append(f"- **Sequence Length**: {args.sequence_length}")
    md_lines.append(f"- **Context Length**: {args.context_length}")
    md_lines.append(f"- **Warmup Steps**: {args.n_warmup}")
    md_lines.append(f"- **Measurement Steps**: {args.n_steps}")
    md_lines.append("")
    
    if status == "OOM":
        # Handle OOM case
        md_lines.append("## Status\n")
        md_lines.append("⚠️ **Out of Memory Error**")
        md_lines.append("")
        md_lines.append(f"The model configuration ({args.model_size}) with sequence length "
                       f"{args.sequence_length} exceeded available GPU memory.")
        md_lines.append("")
    else:
        # Performance results table
        md_lines.append("## Performance Results\n")
        md_lines.append("| Metric | Mean (ms) | Std Dev (ms) |")
        md_lines.append("|--------|-----------|--------------|")
        
        # Forward pass
        forward_mean = stats.get('forward_times_mean', 0) * 1000
        forward_std = stats.get('forward_times_std', 0) * 1000
        md_lines.append(f"| Forward Pass | {forward_mean:.2f} | ±{forward_std:.2f} |")
        
        # Backward pass
        backward_mean = stats.get('backward_times_mean', 0) * 1000
        backward_std = stats.get('backward_times_std', 0) * 1000
        md_lines.append(f"| Backward Pass | {backward_mean:.2f} | ±{backward_std:.2f} |")
        
        # Optimizer step (if applicable)
        if args.with_optimizer:
            opt_mean = stats.get('optimizer_times_mean', 0) * 1000
            opt_std = stats.get('optimizer_times_std', 0) * 1000
            if opt_mean > 0:
                md_lines.append(f"| Optimizer Step | {opt_mean:.2f} | ±{opt_std:.2f} |")
        
        # Total time
        total_mean = stats.get('total_times_mean', 0) * 1000
        total_std = stats.get('total_times_std', 0) * 1000
        md_lines.append(f"| **Total per Step** | **{total_mean:.2f}** | **±{total_std:.2f}** |")
        md_lines.append("")
        
        # Throughput calculations
        md_lines.append("## Throughput Metrics\n")
        if total_mean > 0:
            steps_per_second = 1000 / total_mean
            tokens_per_step = args.batch_size * args.sequence_length
            tokens_per_second = tokens_per_step * steps_per_second
            md_lines.append(f"- **Steps/second**: {steps_per_second:.2f}")
            md_lines.append(f"- **Tokens/step**: {tokens_per_step:,}")
            md_lines.append(f"- **Tokens/second**: {tokens_per_second:,.0f}")
        md_lines.append("")
    
    # Additional information
    md_lines.append("## Additional Information\n")
    md_lines.append(f"- **Optimizer**: {'AdamW' if args.with_optimizer else 'None'}")
    md_lines.append(f"- **CUDA Synchronization**: {'Enabled' if device == 'cuda' and not args.no_cuda_sync else 'Disabled'}")
    md_lines.append(f"- **NVTX Annotations**: {'Enhanced' if args.enhanced_annotations else 'Attention Only' if args.annotate_attention else 'Disabled'}")
    md_lines.append("")
    
    # Notes section
    md_lines.append("## Notes\n")
    if NVTX_AVAILABLE and device == "cuda":
        md_lines.append("- Profile generated with NVTX annotations for detailed kernel analysis")
        md_lines.append("- Use `nsys-ui` to visualize the generated `.nsys-rep` file")
    else:
        md_lines.append("- NVTX annotations not available (CPU mode or CUDA not available)")
    md_lines.append("")
    
    return "\n".join(md_lines)


def main() -> int:
    """Main function to run profiling benchmark."""
    parser = argparse.ArgumentParser(
        description="Profiling-enabled Transformer benchmark"
    )

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size configuration",
    )
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--context-length", type=int, default=512, help="Maximum sequence length"
    )

    # Data configuration
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for benchmarking"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length for input data",
    )

    # Benchmarking configuration
    parser.add_argument(
        "--n-warmup", type=int, default=5, help="Number of warm-up steps"
    )
    parser.add_argument(
        "--n-steps", type=int, default=10, help="Number of measurement steps"
    )
    parser.add_argument(
        "--with-optimizer",
        action="store_true",
        help="Include optimizer step in benchmarking",
    )
    parser.add_argument(
        "--annotate-attention",
        action="store_true",
        help="Replace attention with NVTX-annotated version",
    )
    parser.add_argument(
        "--enhanced-annotations",
        action="store_true",
        help="Apply enhanced NVTX annotations to all model components",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--no-cuda-sync", action="store_true", help="Disable CUDA synchronization"
    )

    # Output configuration
    parser.add_argument(
        "--output-json", type=str, default=None, help="Save results to JSON file"
    )
    parser.add_argument(
        "--output-md", type=str, default=None, help="Save results to Markdown file"
    )

    args = parser.parse_args()

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    use_cuda_sync = not args.no_cuda_sync and device == "cuda"

    # Apply annotations if requested
    if args.enhanced_annotations:
        from enhanced_profiling_annotations import apply_enhanced_annotations

        logger.info("Applying enhanced NVTX annotations to all model components")
    elif args.annotate_attention:
        cs336_basics.model.scaled_dot_product_attention = (
            annotated_scaled_dot_product_attention
        )
        logger.info("Using NVTX-annotated attention implementation")

    logger.info("Profiling %s model on %s", args.model_size, device)
    logger.info("Batch size: %s, Sequence length: %s", args.batch_size, args.sequence_length)
    logger.info("Context length: %s", args.context_length)
    logger.info("Warm-up steps: %s, Measurement steps: %s", args.n_warmup, args.n_steps)
    logger.info("-" * 60)

    try:
        # Create model
        with nvtx.range("model_creation"):
            model = create_model(
                args.model_size,
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                device=device,
            )

            # Apply enhanced annotations after model creation
            if args.enhanced_annotations:
                model = apply_enhanced_annotations(model)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total parameters: %.2fM", total_params / 1e6)

        # Generate random data
        with nvtx.range("data_generation"):
            data = generate_random_batch(
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                vocab_size=args.vocab_size,
                device=device,
            )

        # Run benchmarking with profiling
        logger.info("Running profiling benchmark...")
        stats = benchmark_with_profiling(
            model,
            data,
            n_warmup=args.n_warmup,
            n_steps=args.n_steps,
            use_optimizer=args.with_optimizer,
            use_cuda_sync=use_cuda_sync,
        )

        # Log results
        logger.info("Results (averaged over %s steps):", args.n_steps)
        logger.info(
            "Forward pass:  %.2f ± %.2f ms", stats['forward_times_mean']*1000, stats['forward_times_std']*1000
        )
        logger.info(
            "Backward pass: %.2f ± %.2f ms", stats['backward_times_mean']*1000, stats['backward_times_std']*1000
        )
        if args.with_optimizer and stats["optimizer_times_mean"] > 0:
            logger.info(
                "Optimizer step: %.2f ± %.2f ms", stats['optimizer_times_mean']*1000, stats['optimizer_times_std']*1000
            )
        logger.info(
            "Total per step: %.2f ± %.2f ms", stats['total_times_mean']*1000, stats['total_times_std']*1000
        )

        # Save results if requested
        if args.output_json:
            results = {
                "model_size": args.model_size,
                "context_length": args.context_length,
                "sequence_length": args.sequence_length,
                "batch_size": args.batch_size,
                "total_params": total_params,
                "device": device,
                "timestamp": datetime.now().isoformat(),
                "stats": stats,
            }
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info("Results saved to %s", args.output_json)

        # Save markdown report if requested
        if args.output_md:
            md_content = create_markdown_report(args, stats, total_params, device)
            with open(args.output_md, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info("Markdown report saved to %s", args.output_md)

        return 0

    except torch.cuda.OutOfMemoryError:
        logger.error(
            "Out of memory for %s model with sequence length %s", args.model_size, args.sequence_length
        )

        # Save OOM status if output file specified
        if args.output_json:
            results = {
                "model_size": args.model_size,
                "context_length": args.context_length,
                "sequence_length": args.sequence_length,
                "batch_size": args.batch_size,
                "status": "OOM",
                "timestamp": datetime.now().isoformat(),
            }
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        # Save OOM markdown report if requested
        if args.output_md:
            md_content = create_markdown_report(args, {}, 0, "cuda", status="OOM")
            with open(args.output_md, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info("OOM markdown report saved to %s", args.output_md)

        return 1

    except (RuntimeError, ValueError, ImportError) as e:
        logger.error("Error: %s", str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())

