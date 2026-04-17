#!/usr/bin/env bash
# Run all sweep experiments in parallel, respecting warm-start dependencies.
#
# Usage:
#   ./run_all_sweeps.sh                          # run all sweeps
#   ./run_all_sweeps.sh configs/sweeps/lr-*.yaml  # run a subset (no staging)
#
# Two-stage execution:
#   Stage 1: Solo-agent sweeps (no warm-start dependency) — run in parallel
#   Stage 2: Multi-agent + grid scaling (warm-start from solo-explore) — run in parallel
#
# Results land in experiments/<experiment_name>/

set -euo pipefail

# GPU memory: disable JAX preallocation and cap each process at a fraction of
# GPU memory so many sweeps can share the same GPU(s) without OOM.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.08}"

# Detect available GPUs (comma-separated list of IDs) and round-robin across them.
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}"
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" = "0" ]; then
    NUM_GPUS=1
fi

# Max concurrent experiments. Default: one per GPU (safe for tight memory).
# Override with MAX_PARALLEL=N to run N at a time across all GPUs.
MAX_PARALLEL="${MAX_PARALLEL:-$NUM_GPUS}"
echo "Detected $NUM_GPUS GPU(s). Max $MAX_PARALLEL concurrent experiment(s)."

OUTPUT_DIR="${OUTPUT_DIR:-experiments}"
RUNNER="python -m red_within_blue.training.runner"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Helper: launch a batch of configs in parallel, wait for all, report
# ---------------------------------------------------------------------------
run_batch() {
    local label="$1"
    shift
    local configs=("$@")

    if [ ${#configs[@]} -eq 0 ]; then
        return 0
    fi

    echo "=== $label: launching ${#configs[@]} experiments ==="
    echo ""

    # Launch with bounded concurrency (MAX_PARALLEL). Each job maps to a GPU
    # via round-robin. When MAX_PARALLEL is reached, wait for any to finish
    # before starting the next.
    local gpu_idx=0
    local failed=0
    declare -A pid_name
    local running=0

    for cfg in "${configs[@]}"; do
        local name
        name=$(basename "$cfg" .yaml)
        local gpu=$((gpu_idx % NUM_GPUS))

        # Throttle: wait for a slot if we hit MAX_PARALLEL
        while [ "$running" -ge "$MAX_PARALLEL" ]; do
            if wait -n 2>/dev/null; then
                :
            else
                failed=$((failed + 1))
            fi
            running=$((running - 1))
        done

        echo "  Starting: $name (GPU $gpu)"
        CUDA_VISIBLE_DEVICES=$gpu $RUNNER --config "$cfg" --output-dir "$OUTPUT_DIR" \
            > "${OUTPUT_DIR}/${name}.log" 2>&1 &
        pid_name[$!]="$name"
        gpu_idx=$((gpu_idx + 1))
        running=$((running + 1))
    done

    echo ""
    echo "Waiting for remaining $label jobs to complete..."
    echo ""

    # Drain remaining jobs
    while [ "$running" -gt 0 ]; do
        if wait -n 2>/dev/null; then
            :
        else
            failed=$((failed + 1))
        fi
        running=$((running - 1))
    done

    echo ""
    echo "=== $label: $((${#configs[@]} - failed))/${#configs[@]} succeeded ==="
    echo ""

    return $failed
}

# ---------------------------------------------------------------------------
# If args provided, run them directly (no staging)
# ---------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    run_batch "Custom" "$@"
    exit $?
fi

# ---------------------------------------------------------------------------
# Stage 1: solo-agent sweeps (no warm-start needed)
# ---------------------------------------------------------------------------
SOLO_CONFIGS=()
WARM_CONFIGS=()

for cfg in configs/sweeps/*.yaml; do
    if grep -q "warm_start:" "$cfg"; then
        WARM_CONFIGS+=("$cfg")
    else
        SOLO_CONFIGS+=("$cfg")
    fi
done

echo "Found ${#SOLO_CONFIGS[@]} solo configs, ${#WARM_CONFIGS[@]} warm-start configs"
echo ""

run_batch "Stage 1 (solo)" "${SOLO_CONFIGS[@]}"
STAGE1_FAILED=$?

# Also run the main solo-explore config to produce the checkpoint for stage 2
if [ ! -f "${OUTPUT_DIR}/solo-explore/checkpoint.npz" ]; then
    echo "=== Training solo-explore baseline for warm-start ==="
    $RUNNER --config configs/solo-explore.yaml --output-dir "$OUTPUT_DIR" \
        > "${OUTPUT_DIR}/solo-explore-baseline.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "  DONE: solo-explore baseline"
    else
        echo "  FAIL: solo-explore baseline — stage 2 will fail"
        echo "  See ${OUTPUT_DIR}/solo-explore-baseline.log"
        exit 1
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Stage 2: warm-start experiments (multi-agent, grid scaling, connectivity)
# ---------------------------------------------------------------------------
if [ ${#WARM_CONFIGS[@]} -gt 0 ]; then
    run_batch "Stage 2 (warm-start)" "${WARM_CONFIGS[@]}"
    STAGE2_FAILED=$?
else
    STAGE2_FAILED=0
fi

TOTAL=$((${#SOLO_CONFIGS[@]} + ${#WARM_CONFIGS[@]}))
TOTAL_FAILED=$((STAGE1_FAILED + STAGE2_FAILED))
echo "=== All done: $((TOTAL - TOTAL_FAILED))/${TOTAL} succeeded ==="

if [ $TOTAL_FAILED -gt 0 ]; then
    exit 1
fi
