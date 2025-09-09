# Vast.ai GPU Profiling Commands

## Setup Commands (Run once after connecting to vast.ai instance)

```bash
# 1. Clone your repository
git clone https://github.com/AndyZhao2023/assignment2-systems.git
cd assignment2-systems

# 2. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 3. Install dependencies
uv sync

# 4. Verify CUDA and nsys are available
nvidia-smi
nsys --version

# 5. Make scripts executable
chmod +x benchmarking_suite/profile_all.sh
chmod +x benchmarking_suite/*.py
```

## Quick Test (Verify everything works)

```bash
# Test with small model and short context
cd benchmarking_suite

# Test without nsys first
uv run python benchmark_profiling.py \
    --model-size small \
    --sequence-length 128 \
    --context-length 128 \
    --batch-size 4 \
    --n-warmup 2 \
    --n-steps 3 \
    --with-optimizer \
    --annotate-attention

# Test with nsys (basic)
uv run nsys profile \
    --pytorch \
    --output=test_profile \
    --force-overwrite=true \
    python benchmark_profiling.py \
    --model-size small \
    --sequence-length 128 \
    --n-warmup 2 \
    --n-steps 3 \
    --annotate-attention

# Test with enhanced annotations (detailed profiling)
uv run nsys profile \
    --pytorch \
    --output=test_profile_detailed \
    --force-overwrite=true \
    python benchmark_profiling.py \
    --model-size small \
    --sequence-length 128 \
    --n-warmup 2 \
    --n-steps 3 \
    --enhanced-annotations
```

## Option 1: Run All Profiles Automatically (Recommended)

```bash
cd benchmarking_suite

# Run the automated profiling script
uv run bash profile_all.sh

# Or use Python runner for more control
uv run python run_profiling.py \
    --output-dir profiling_results \
    --batch-size 4

# For testing, skip large models
uv run python run_profiling.py \
    --skip-large \
    --output-dir profiling_results_test
```

## Option 2: Run Individual Profiles Manually

Run these commands for each model/context combination:

### Small Model
```bash
cd benchmarking_suite

# Context 128
uv run nsys profile --pytorch --output=profiles/small_ctx128 --force-overwrite=true \
    python benchmark_profiling.py --model-size small --sequence-length 128 --context-length 128 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/small_ctx128.json

# Context 256
uv run nsys profile --pytorch --output=profiles/small_ctx256 --force-overwrite=true \
    python benchmark_profiling.py --model-size small --sequence-length 256 --context-length 256 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/small_ctx256.json

# Context 512
uv run nsys profile --pytorch --output=profiles/small_ctx512 --force-overwrite=true \
    python benchmark_profiling.py --model-size small --sequence-length 512 --context-length 512 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/small_ctx512.json

# Context 1024
uv run nsys profile --pytorch --output=profiles/small_ctx1024 --force-overwrite=true \
    python benchmark_profiling.py --model-size small --sequence-length 1024 --context-length 1024 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/small_ctx1024.json
```

### Medium Model
```bash
# Context 128
uv run nsys profile --pytorch --output=profiles/medium_ctx128 --force-overwrite=true \
    python benchmark_profiling.py --model-size medium --sequence-length 128 --context-length 128 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/medium_ctx128.json

# Context 256
uv run nsys profile --pytorch --output=profiles/medium_ctx256 --force-overwrite=true \
    python benchmark_profiling.py --model-size medium --sequence-length 256 --context-length 256 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/medium_ctx256.json

# Context 512
uv run nsys profile --pytorch --output=profiles/medium_ctx512 --force-overwrite=true \
    python benchmark_profiling.py --model-size medium --sequence-length 512 --context-length 512 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/medium_ctx512.json

# Context 1024
uv run nsys profile --pytorch --output=profiles/medium_ctx1024 --force-overwrite=true \
    python benchmark_profiling.py --model-size medium --sequence-length 1024 --context-length 1024 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/medium_ctx1024.json
```

### Large Model
```bash
# Context 128
uv run nsys profile --pytorch --output=profiles/large_ctx128 --force-overwrite=true \
    python benchmark_profiling.py --model-size large --sequence-length 128 --context-length 128 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/large_ctx128.json

# Context 256
uv run nsys profile --pytorch --output=profiles/large_ctx256 --force-overwrite=true \
    python benchmark_profiling.py --model-size large --sequence-length 256 --context-length 256 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/large_ctx256.json

# Context 512
uv run nsys profile --pytorch --output=profiles/large_ctx512 --force-overwrite=true \
    python benchmark_profiling.py --model-size large --sequence-length 512 --context-length 512 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/large_ctx512.json

# Context 1024 (may OOM)
uv run nsys profile --pytorch --output=profiles/large_ctx1024 --force-overwrite=true \
    python benchmark_profiling.py --model-size large --sequence-length 1024 --context-length 1024 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/large_ctx1024.json
```

### XL Model (likely to OOM on longer contexts)
```bash
# Context 128
uv run nsys profile --pytorch --output=profiles/xl_ctx128 --force-overwrite=true \
    python benchmark_profiling.py --model-size xl --sequence-length 128 --context-length 128 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/xl_ctx128.json

# Context 256
uv run nsys profile --pytorch --output=profiles/xl_ctx256 --force-overwrite=true \
    python benchmark_profiling.py --model-size xl --sequence-length 256 --context-length 256 \
    --batch-size 4 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/xl_ctx256.json

# Context 512 (may OOM)
uv run nsys profile --pytorch --output=profiles/xl_ctx512 --force-overwrite=true \
    python benchmark_profiling.py --model-size xl --sequence-length 512 --context-length 512 \
    --batch-size 2 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/xl_ctx512.json
```

### 2.7B Model (likely to OOM on most configs)
```bash
# Context 128 with reduced batch size
uv run nsys profile --pytorch --output=profiles/2.7B_ctx128 --force-overwrite=true \
    python benchmark_profiling.py --model-size 2.7B --sequence-length 128 --context-length 128 \
    --batch-size 2 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/2.7B_ctx128.json

# Context 256 with batch size 1
uv run nsys profile --pytorch --output=profiles/2.7B_ctx256 --force-overwrite=true \
    python benchmark_profiling.py --model-size 2.7B --sequence-length 256 --context-length 256 \
    --batch-size 1 --n-warmup 5 --n-steps 10 --with-optimizer --annotate-attention \
    --output-json profiles/2.7B_ctx256.json
```

## Analyze Results

After profiling, analyze the results:

```bash
cd benchmarking_suite

# If you used automatic profiling
uv run python analyze_profiles.py profiling_results/batch_*/

# If you have previous Python benchmark results
uv run python analyze_profiles.py profiling_results/batch_*/ \
    --python-benchmark ../benchmarking_suite/results/benchmark_results.md \
    --output profiling_analysis.md

# View the analysis
cat profiling_analysis.md
```

## Download Results

After profiling, download the results to your local machine:

```bash
# Create a tar archive of results
cd assignment2-systems
tar -czf profiling_results.tar.gz benchmarking_suite/profiling_results/

# Download using scp or vast.ai's file transfer
# The exact command depends on your vast.ai instance setup
```

## Viewing Profiles Locally

After downloading `.nsys-rep` files to your local machine:

1. Install NVIDIA Nsight Systems on your local machine
2. Open the GUI: `nsys-ui`
3. Open the `.nsys-rep` files
4. Look for:
   - NVTX ranges showing warmup vs measurement phases
   - Forward/backward pass timings
   - Attention component breakdown
   - CUDA kernel execution patterns

## Troubleshooting

### If you get CUDA out of memory errors:
```bash
# Reduce batch size
--batch-size 2  # or even 1 for large models

# Reduce sequence length
--sequence-length 256 --context-length 256

# Skip optimizer to save memory
# Remove --with-optimizer flag
```

### If nsys is not found:
```bash
# Check if it's installed
which nsys

# It might be in a different location
/usr/local/cuda/bin/nsys --version

# Add to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
```

### If profiling takes too long:
```bash
# Reduce measurement steps
--n-warmup 2 --n-steps 5

# Profile fewer configurations
--skip-large  # Skip xl and 2.7B models
```

## Expected Output

After successful profiling, you should have:
1. `.nsys-rep` files for each configuration (for GUI viewing)
2. `.json` files with timing statistics
3. Summary report with all results
4. Analysis comparing with Python timing

The answer to part (a) should show that Nsight Systems forward pass timing matches Python standard library timing within ~5-10% variance.