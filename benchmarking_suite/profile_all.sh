#!/bin/bash
#
# Batch profiling script for CS336 Assignment 2
# Run all model/context length combinations with Nsight Systems
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODELS=("small" "medium" "large" "xl" "2.7B")
CONTEXT_LENGTHS=(128 256 512 1024)
BATCH_SIZE=4
WARMUP_STEPS=5
MEASURE_STEPS=10

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="profiling_results/batch_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "CS336 Assignment 2 - Nsight Systems Profiling"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Log file for this run
LOG_FILE="${OUTPUT_DIR}/profiling_log.txt"
SUMMARY_FILE="${OUTPUT_DIR}/summary.csv"

# Initialize summary CSV
echo "model,context_length,status,forward_ms,backward_ms,optimizer_ms,total_ms" > "${SUMMARY_FILE}"

# Function to run a single profiling session
run_profile() {
    local model=$1
    local context_len=$2
    local output_name="${model}_ctx${context_len}_bs${BATCH_SIZE}"
    
    echo -e "${YELLOW}[$(date +%H:%M:%S)] Profiling: ${model} with context=${context_len}${NC}"
    echo "[$(date +%H:%M:%S)] Profiling: ${model} with context=${context_len}" >> "${LOG_FILE}"
    
    # Create nsys command
    nsys_cmd="nsys profile \
        --pytorch=autograd-nvtx \
        --output=${OUTPUT_DIR}/${output_name} \
        --force-overwrite=true \
        --stats=true \
        --python-backtrace=true \
        python benchmark_profiling.py \
        --model-size ${model} \
        --sequence-length ${context_len} \
        --context-length ${context_len} \
        --batch-size ${BATCH_SIZE} \
        --n-warmup ${WARMUP_STEPS} \
        --n-steps ${MEASURE_STEPS} \
        --with-optimizer \
        --annotate-attention \
        --output-json ${OUTPUT_DIR}/${output_name}.json"
    
    # Run the command
    if eval ${nsys_cmd} >> "${LOG_FILE}" 2>&1; then
        echo -e "${GREEN}  ✓ Success${NC}"
        
        # Extract timing from JSON if available
        if [ -f "${OUTPUT_DIR}/${output_name}.json" ]; then
            python3 -c "
import json
with open('${OUTPUT_DIR}/${output_name}.json', 'r') as f:
    data = json.load(f)
    if 'stats' in data:
        stats = data['stats']
        forward = stats.get('forward_times_mean', 0) * 1000
        backward = stats.get('backward_times_mean', 0) * 1000
        optimizer = stats.get('optimizer_times_mean', 0) * 1000
        total = stats.get('total_times_mean', 0) * 1000
        print(f'${model},${context_len},success,{forward:.2f},{backward:.2f},{optimizer:.2f},{total:.2f}')
    else:
        print(f'${model},${context_len},{data.get(\"status\", \"unknown\")},,,,')
" >> "${SUMMARY_FILE}"
        fi
    else
        # Check if it was OOM
        if grep -q "OutOfMemoryError\|out of memory" "${LOG_FILE}"; then
            echo -e "${RED}  ✗ Out of Memory${NC}"
            echo "${model},${context_len},OOM,,,," >> "${SUMMARY_FILE}"
        else
            echo -e "${RED}  ✗ Failed${NC}"
            echo "${model},${context_len},failed,,,," >> "${SUMMARY_FILE}"
        fi
    fi
    
    echo ""
}

# Main profiling loop
total=$((${#MODELS[@]} * ${#CONTEXT_LENGTHS[@]}))
current=0

for model in "${MODELS[@]}"; do
    for context_len in "${CONTEXT_LENGTHS[@]}"; do
        current=$((current + 1))
        echo "[$current/$total]"
        run_profile "$model" "$context_len"
        
        # Add small delay between runs to avoid thermal throttling
        sleep 2
    done
done

echo "=========================================="
echo "Profiling Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "To view profiles in GUI:"
echo "  nsys-ui ${OUTPUT_DIR}/<model>_ctx<length>_bs${BATCH_SIZE}.nsys-rep"
echo "=========================================="

# Generate final markdown report
python3 - << EOF
import csv
import os
from datetime import datetime

output_dir = "${OUTPUT_DIR}"
summary_file = "${SUMMARY_FILE}"

# Read CSV results
results = []
with open(summary_file, 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

# Generate markdown report
report_file = os.path.join(output_dir, "profiling_report.md")
with open(report_file, 'w') as f:
    f.write("# Nsight Systems Profiling Report\\n\\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
    
    f.write("## Configuration\\n")
    f.write(f"- Batch Size: ${BATCH_SIZE}\\n")
    f.write(f"- Warmup Steps: ${WARMUP_STEPS}\\n")
    f.write(f"- Measurement Steps: ${MEASURE_STEPS}\\n\\n")
    
    f.write("## Results\\n\\n")
    f.write("| Model | Context | Status | Forward (ms) | Backward (ms) | Optimizer (ms) | Total (ms) |\\n")
    f.write("|-------|---------|--------|--------------|---------------|----------------|------------|\\n")
    
    for r in results:
        model = r['model']
        ctx = r['context_length']
        status = r['status']
        
        if status == 'success':
            forward = f"{float(r['forward_ms']):.2f}" if r['forward_ms'] else '-'
            backward = f"{float(r['backward_ms']):.2f}" if r['backward_ms'] else '-'
            optimizer = f"{float(r['optimizer_ms']):.2f}" if r['optimizer_ms'] else '-'
            total = f"{float(r['total_ms']):.2f}" if r['total_ms'] else '-'
            status_icon = '✓'
        elif status == 'OOM':
            forward = backward = optimizer = total = '-'
            status_icon = '✗ OOM'
        else:
            forward = backward = optimizer = total = '-'
            status_icon = '✗'
        
        f.write(f"| {model} | {ctx} | {status_icon} | {forward} | {backward} | {optimizer} | {total} |\\n")
    
    f.write("\\n## Notes\\n\\n")
    f.write("- All times are in milliseconds\\n")
    f.write("- OOM = Out of Memory\\n")
    f.write("- Times are averaged over ${MEASURE_STEPS} measurement steps\\n")
    f.write("- Warmup steps (${WARMUP_STEPS}) are excluded from measurements\\n")

print(f"Report generated: {report_file}")
EOF