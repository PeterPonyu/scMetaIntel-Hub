#!/bin/bash
# Parallel Ollama model downloader — tiered by size, retry-safe.
#
# Strategy:
#   Tier S (≤2GB):   up to 4 concurrent pulls — tiny, finish fast
#   Tier M (2-6GB):  up to 3 concurrent pulls — moderate size
#   Tier L (6-15GB): up to 2 concurrent pulls — large, avoid disk thrash
#
# ollama pull is natively resumable: partial downloads survive interruption.
# Re-running this script skips completed models and resumes partial ones.
#
# Usage:
#   bash scripts/ollama_parallel_pull.sh          # full run
#   bash scripts/ollama_parallel_pull.sh --dry-run # show plan only

set -o pipefail

MAX_RETRIES=3
RETRY_DELAY=15

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

# ── Helper: pull one model with retries ──────────────────────────
pull_model() {
    local MODEL="$1"
    local TAG="$2"  # display tag like [S3/7]

    # Skip if already pulled
    if ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qxF "$MODEL"; then
        echo "$TAG SKIP $MODEL (already pulled)"
        return 0
    fi

    local ATTEMPT=0
    while [ $ATTEMPT -lt $MAX_RETRIES ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "$TAG pulling $MODEL (attempt $ATTEMPT/$MAX_RETRIES) ..."
        if ollama pull "$MODEL" 2>&1 | tail -1; then
            # Verify
            if ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qxF "$MODEL"; then
                echo "$TAG OK $MODEL"
                return 0
            fi
        fi
        echo "$TAG RETRY $MODEL in ${RETRY_DELAY}s ..."
        sleep $RETRY_DELAY
    done

    echo "$TAG FAILED $MODEL after $MAX_RETRIES attempts"
    return 1
}
export -f pull_model
export MAX_RETRIES RETRY_DELAY

# ── Build tier lists ─────────────────────────────────────────────
TIER_S=(
    "granite3.3:2b"
    "falcon3:3b"
    "qwen2.5:3b"
    "llama3.2:3b"
)

TIER_M=(
    "phi4-mini:latest"
    "qwen3:4b"
    "gemma3:4b-it-q8_0"
    "falcon3:7b"
    "deepseek-r1:7b"
    "aya-expanse:8b"
    "deepseek-r1:8b"
    "granite3.3:8b"
    "glm4:9b"
)

TIER_L=(
    "falcon3:10b"
    "mistral:7b-instruct-v0.3-q8_0"
    "deepseek-r1:14b"
    "gemma2:9b-instruct-q8_0"
    "qwen2.5:14b-instruct-q8_0"
)

# Count what actually needs downloading
count_needed() {
    local NEEDED=0
    local ARR=("$@")
    for m in "${ARR[@]}"; do
        if ! ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qxF "$m"; then
            NEEDED=$((NEEDED + 1))
        fi
    done
    echo $NEEDED
}

S_NEED=$(count_needed "${TIER_S[@]}")
M_NEED=$(count_needed "${TIER_M[@]}")
L_NEED=$(count_needed "${TIER_L[@]}")
TOTAL=$((S_NEED + M_NEED + L_NEED))

echo "================================================================"
echo "  Parallel Ollama Download Plan"
echo "  $(date)"
echo "================================================================"
echo "  Tier S (≤2GB,  4 parallel): ${#TIER_S[@]} models, $S_NEED need download"
echo "  Tier M (2-6GB, 3 parallel): ${#TIER_M[@]} models, $M_NEED need download"
echo "  Tier L (6-15GB,2 parallel): ${#TIER_L[@]} models, $L_NEED need download"
echo "  Total to download: $TOTAL"
echo "================================================================"

if $DRY_RUN; then
    echo ""
    echo "DRY RUN — no downloads. Remove --dry-run to execute."
    exit 0
fi

if [ $TOTAL -eq 0 ]; then
    echo "All models already pulled. Nothing to do."
    exit 0
fi

FAILED_MODELS=()

# ── Tier S: 4 parallel ──────────────────────────────────────────
if [ $S_NEED -gt 0 ]; then
    echo ""
    echo "── Tier S: Small models (≤2GB), 4 parallel ──"
    PIDS=()
    IDX=0
    for m in "${TIER_S[@]}"; do
        IDX=$((IDX + 1))
        pull_model "$m" "[S$IDX/${#TIER_S[@]}]" &
        PIDS+=($!)
    done
    # Wait for all, collect exit codes
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILED_MODELS+=("tier-S:pid-$pid")
        fi
    done
    echo "── Tier S complete ──"
fi

# ── Tier M: 3 parallel (batch through) ──────────────────────────
if [ $M_NEED -gt 0 ]; then
    echo ""
    echo "── Tier M: Medium models (2-6GB), 3 parallel ──"
    IDX=0
    BATCH_PIDS=()
    for m in "${TIER_M[@]}"; do
        IDX=$((IDX + 1))
        pull_model "$m" "[M$IDX/${#TIER_M[@]}]" &
        BATCH_PIDS+=($!)

        # When we hit 3 concurrent, wait for any one to finish before launching next
        if [ ${#BATCH_PIDS[@]} -ge 3 ]; then
            wait -n 2>/dev/null || true
            # Clean finished PIDs
            NEW_PIDS=()
            for pid in "${BATCH_PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                fi
            done
            BATCH_PIDS=("${NEW_PIDS[@]}")
        fi
    done
    # Wait for remaining
    for pid in "${BATCH_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "── Tier M complete ──"
fi

# ── Tier L: 2 parallel ──────────────────────────────────────────
if [ $L_NEED -gt 0 ]; then
    echo ""
    echo "── Tier L: Large models (6-15GB), 2 parallel ──"
    IDX=0
    BATCH_PIDS=()
    for m in "${TIER_L[@]}"; do
        IDX=$((IDX + 1))
        pull_model "$m" "[L$IDX/${#TIER_L[@]}]" &
        BATCH_PIDS+=($!)

        if [ ${#BATCH_PIDS[@]} -ge 2 ]; then
            wait -n 2>/dev/null || true
            NEW_PIDS=()
            for pid in "${BATCH_PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                fi
            done
            BATCH_PIDS=("${NEW_PIDS[@]}")
        fi
    done
    for pid in "${BATCH_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "── Tier L complete ──"
fi

# ── Final verification ───────────────────────────────────────────
echo ""
echo "================================================================"
echo "  FINAL VERIFICATION"
echo "================================================================"
ALL_MODELS=("${TIER_S[@]}" "${TIER_M[@]}" "${TIER_L[@]}")
OK=0
FAIL=0
for m in "${ALL_MODELS[@]}"; do
    if ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qxF "$m"; then
        OK=$((OK + 1))
    else
        FAIL=$((FAIL + 1))
        echo "  MISSING: $m"
    fi
done
echo "  Verified: $OK / ${#ALL_MODELS[@]}"
if [ $FAIL -gt 0 ]; then
    echo "  $FAIL model(s) still missing. Re-run this script to retry."
else
    echo "  ALL MODELS DOWNLOADED SUCCESSFULLY."
fi
echo "  $(date)"
echo "================================================================"
