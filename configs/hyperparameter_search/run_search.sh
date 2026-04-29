#!/usr/bin/env bash
# Parallel hyperparameter search driver.
#
# Runs every YAML config in this directory through the training runner,
# fanning out across available GPUs (or CPU cores if no GPU is present).
# None of these configs need a warm-start checkpoint — they train from
# scratch so that learning-rate / gamma / network / method effects are
# measured in isolation.
#
# Usage:
#   ./configs/hyperparameter_search/run_search.sh               # run every *.yaml here
#   ./configs/hyperparameter_search/run_search.sh lr-*.yaml     # run a subset (glob from repo root)
#
# Environment overrides:
#   MAX_PARALLEL=N      max concurrent runs (default: one per GPU, or 1 on CPU)
#   NUM_GPUS=N          override detected GPU count
#   OUTPUT_DIR=path     override output directory (default: experiments)
#   REPORT=1            after each run, render report.html + episode.gif

set -euo pipefail

# Resolve repo root so the script works regardless of cwd.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

# GPU memory: disable JAX preallocation and cap each process at a fraction of
# GPU memory so multiple sweeps can share the same device without OOM.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.08}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}"
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" = "0" ]; then
    NUM_GPUS=1
fi
MAX_PARALLEL="${MAX_PARALLEL:-$NUM_GPUS}"
echo "Detected $NUM_GPUS device(s). Max $MAX_PARALLEL concurrent run(s)."

OUTPUT_DIR="${OUTPUT_DIR:-experiments}"
RUNNER="python -m red_within_blue.training.runner"
REPORTER="python -m red_within_blue.analysis.experiment_report"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Collect configs (from args, or all *.yaml in this directory)
# ---------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=()
    for cfg in "$SCRIPT_DIR"/*.yaml; do
        CONFIGS+=("$cfg")
    done
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configs to run."
    exit 0
fi

echo "Running ${#CONFIGS[@]} hyperparameter-search config(s):"
for c in "${CONFIGS[@]}"; do
    echo "  $(basename "$c" .yaml)"
done
echo ""

# ---------------------------------------------------------------------------
# Parallel launcher
# ---------------------------------------------------------------------------
gpu_idx=0
failed=0
running=0

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    gpu=$((gpu_idx % NUM_GPUS))

    while [ "$running" -ge "$MAX_PARALLEL" ]; do
        if wait -n 2>/dev/null; then
            :
        else
            failed=$((failed + 1))
        fi
        running=$((running - 1))
    done

    echo "  Starting: $name (device $gpu)"
    CUDA_VISIBLE_DEVICES=$gpu $RUNNER --config "$cfg" --output-dir "$OUTPUT_DIR" \
        > "${OUTPUT_DIR}/${name}.log" 2>&1 &
    gpu_idx=$((gpu_idx + 1))
    running=$((running + 1))
done

echo ""
echo "Waiting for remaining jobs..."

while [ "$running" -gt 0 ]; do
    if wait -n 2>/dev/null; then
        :
    else
        failed=$((failed + 1))
    fi
    running=$((running - 1))
done

echo ""
echo "=== Training: $((${#CONFIGS[@]} - failed))/${#CONFIGS[@]} succeeded ==="

# ---------------------------------------------------------------------------
# Optional: generate HTML report + episode GIF for each completed run
# ---------------------------------------------------------------------------
if [ "${REPORT:-0}" = "1" ]; then
    echo ""
    echo "=== Generating reports ==="
    for cfg in "${CONFIGS[@]}"; do
        name=$(basename "$cfg" .yaml)
        run_dir="${OUTPUT_DIR}/${name}"
        if [ -f "${run_dir}/checkpoint.npz" ]; then
            echo "  Reporting: $name"
            $REPORTER --config "$cfg" --experiment-dir "$run_dir" \
                >> "${OUTPUT_DIR}/${name}.log" 2>&1 || echo "    report failed for $name"
        fi
    done
fi

exit $failed
