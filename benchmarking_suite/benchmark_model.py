#!/usr/bin/env python3
"""
Benchmarking script for CS336 Assignment 2
Problem (benchmarking_script): End-to-end benchmarking of Transformer models
"""

import argparse
import timeit
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import sys

# Import the Transformer model from cs336_basics
from cs336_basics.model import BasicsTransformerLM


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
    }
}


def create_model(
    model_size: str,
    vocab_size: int = 10000,
    context_length: int = 512,
    device: str = "cpu"
) -> BasicsTransformerLM:
    """Create a Transformer model with the specified configuration."""
    
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_size]
    
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0  # Default RoPE theta
    )
    
    return model.to(device)


def generate_random_batch(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    device: str = "cpu"
) -> torch.LongTensor:
    """Generate random input data for benchmarking."""
    return torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)


def benchmark_forward_pass(
    model: nn.Module,
    data: torch.Tensor,
    n_warmup: int = 5,
    n_steps: int = 10,
    use_cuda_sync: bool = True
) -> Tuple[float, float]:
    """Benchmark the forward pass of the model."""
    
    times = []
    
    # Warm-up steps
    for _ in range(n_warmup):
        _ = model(data)
        if use_cuda_sync and data.is_cuda:
            torch.cuda.synchronize()
    
    # Measurement steps
    for _ in range(n_steps):
        start_time = timeit.default_timer()
        _ = model(data)
        if use_cuda_sync and data.is_cuda:
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time


def benchmark_forward_backward_pass(
    model: nn.Module,
    data: torch.Tensor,
    n_warmup: int = 5,
    n_steps: int = 10,
    use_cuda_sync: bool = True
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Benchmark both forward and backward passes of the model."""
    
    forward_times = []
    backward_times = []
    
    # Create loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Create target data (shifted input for language modeling)
    target = torch.cat([data[:, 1:], data[:, :1]], dim=1)
    
    # Warm-up steps
    for _ in range(n_warmup):
        # Forward pass
        output = model(data)
        loss = loss_fn(output.reshape(-1, output.size(-1)), target.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Clear gradients for next iteration
        model.zero_grad()
        
        if use_cuda_sync and data.is_cuda:
            torch.cuda.synchronize()
    
    # Measurement steps
    for _ in range(n_steps):
        # Time forward pass
        forward_start = timeit.default_timer()
        output = model(data)
        loss = loss_fn(output.reshape(-1, output.size(-1)), target.reshape(-1))
        if use_cuda_sync and data.is_cuda:
            torch.cuda.synchronize()
        forward_end = timeit.default_timer()
        forward_times.append(forward_end - forward_start)
        
        # Time backward pass
        backward_start = timeit.default_timer()
        loss.backward()
        if use_cuda_sync and data.is_cuda:
            torch.cuda.synchronize()
        backward_end = timeit.default_timer()
        backward_times.append(backward_end - backward_start)
        
        # Clear gradients for next iteration
        model.zero_grad()
    
    forward_mean = np.mean(forward_times)
    forward_std = np.std(forward_times)
    backward_mean = np.mean(backward_times)
    backward_std = np.std(backward_times)
    
    return (forward_mean, forward_std), (backward_mean, backward_std)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models")
    
    # Model configuration
    parser.add_argument("--model-size", type=str, default="small",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size configuration")
    parser.add_argument("--vocab-size", type=int, default=10000,
                        help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Data configuration
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for benchmarking")
    parser.add_argument("--sequence-length", type=int, default=512,
                        help="Sequence length for input data")
    
    # Benchmarking configuration
    parser.add_argument("--n-warmup", type=int, default=5,
                        help="Number of warm-up steps")
    parser.add_argument("--n-steps", type=int, default=10,
                        help="Number of measurement steps")
    parser.add_argument("--backward", action="store_true",
                        help="Benchmark both forward and backward passes")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--no-cuda-sync", action="store_true",
                        help="Disable CUDA synchronization (not recommended for accurate timing)")
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    use_cuda_sync = not args.no_cuda_sync and device == "cuda"
    
    print(f"Benchmarking {args.model_size} model on {device}")
    print(f"Batch size: {args.batch_size}, Sequence length: {args.sequence_length}")
    print(f"Warm-up steps: {args.n_warmup}, Measurement steps: {args.n_steps}")
    print("-" * 60)
    
    # Create model
    model = create_model(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Generate random data
    data = generate_random_batch(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        device=device
    )
    
    if args.backward:
        # Benchmark forward and backward passes
        print("\nBenchmarking forward and backward passes...")
        (forward_mean, forward_std), (backward_mean, backward_std) = benchmark_forward_backward_pass(
            model, data, args.n_warmup, args.n_steps, use_cuda_sync
        )
        
        print(f"\nForward pass:  {forward_mean*1000:.2f} ± {forward_std*1000:.2f} ms")
        print(f"Backward pass: {backward_mean*1000:.2f} ± {backward_std*1000:.2f} ms")
        print(f"Total:         {(forward_mean + backward_mean)*1000:.2f} ± {(forward_std + backward_std)*1000:.2f} ms")
    else:
        # Benchmark only forward pass
        print("\nBenchmarking forward pass only...")
        mean_time, std_time = benchmark_forward_pass(
            model, data, args.n_warmup, args.n_steps, use_cuda_sync
        )
        
        print(f"\nForward pass: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")


if __name__ == "__main__":
    main()