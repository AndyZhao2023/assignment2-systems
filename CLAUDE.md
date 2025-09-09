# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is CS336 Spring 2025 Assignment 2: Systems. The assignment focuses on implementing optimized Transformer language models with distributed training and optimization.

## Project Structure

- `cs336_basics/`: Module containing the staff implementation from assignment 1 (language model basics)
- `cs336_systems/`: Main module where assignment 2 implementations go (currently empty, needs implementation)
- `tests/`: Test suite for the assignment implementations
  - `test_attention.py`: Tests for FlashAttention2 implementations (PyTorch and Triton)
  - `test_ddp.py`: Tests for bucketed distributed data parallel
  - `test_ddp_individual_parameters.py`: Tests for individual parameter DDP
  - `test_sharded_optimizer.py`: Tests for optimizer state sharding
- `adapters.py`: Interface definitions that need to be implemented

## Key Commands

### Running Tests
```bash
# Run all tests with detailed output
uv run pytest -v ./tests

# Run specific test files
uv run pytest tests/test_attention.py
uv run pytest tests/test_ddp.py
uv run pytest tests/test_ddp_individual_parameters.py
uv run pytest tests/test_sharded_optimizer.py

# Run tests and create submission
./test_and_make_submission.sh
```

### Development
```bash
# Install dependencies and verify setup
uv run python

# Run with specific Python module
uv run python -m cs336_systems
```

## Implementation Requirements

The assignment requires implementing several key components in the `cs336_systems` module:

1. **FlashAttention2 Implementations**:
   - PyTorch-only implementation (`get_flashattention_autograd_function_pytorch`)
   - Triton kernel implementation (`get_flashattention_autograd_function_triton`)
   - Both need forward and backward passes with proper gradient computation

2. **Distributed Data Parallel (DDP)**:
   - Individual parameter gradient synchronization (`get_ddp_individual_parameters`)
   - Bucketed gradient synchronization (`get_ddp_bucketed`)
   - Associated hooks for after backward pass and training batch start

3. **Optimizer State Sharding**:
   - Implementation of sharded optimizer (`get_sharded_optimizer`)

## Testing Approach

- Tests use PyTorch's `torch.testing.assert_close` with tolerances (rtol=1e-2, atol=1e-2)
- GPU tests are automatically skipped if CUDA is not available
- Tests check for proper saved tensors in autograd functions
- DDP tests use multiprocessing for distributed testing

## Dependencies

Key dependencies (managed via `uv`):
- PyTorch ~2.6.0 (or ~2.2.2 for Intel Macs)
- Triton (for GPU kernels)
- einops (for tensor operations)
- pytest for testing
- wandb for experiment tracking