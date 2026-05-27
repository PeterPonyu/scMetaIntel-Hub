#!/usr/bin/env bash
# Auto-pipeline: wait for any running 05_bench_public, then run the
# surgical Task A+D re-run for the top-10 models, then refresh the
# claim audit and all article figures.
#
# Designed to be launched in the background once and run unattended.
#
# Usage:
#   nohup bash scripts/run_audit_pipeline.sh > logs/audit_pipeline.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "Pipeline start"

log_matching_bench_processes() {
    pids="$(pgrep -f "05_bench_public" || true)"
    if [ -z "$pids" ]; then
        return 0
    fi
    for pid in $pids; do
        ps -p "$pid" -o pid=,etime=,comm= 2>/dev/null | sed 's/^/  /' >&2 || true
    done
}

# 1. Wait for any running 05_bench_public to finish (bounded)
#
# Bound the wait so unrelated/stale processes whose command line happens to
# contain "05_bench_public" (e.g. an editor buffer, a tmux pane, a previous
# failed run) cannot block the audit pipeline forever. Override the timeout
# with BENCH_WAIT_TIMEOUT_SEC (default: 600s). Set BENCH_WAIT_TIMEOUT_SEC=0
# to skip the wait entirely.
BENCH_WAIT_TIMEOUT_SEC="${BENCH_WAIT_TIMEOUT_SEC:-600}"
BENCH_WAIT_POLL_SEC="${BENCH_WAIT_POLL_SEC:-30}"

case "$BENCH_WAIT_TIMEOUT_SEC" in
    ''|*[!0-9]*)
        log "ERROR: BENCH_WAIT_TIMEOUT_SEC must be a non-negative integer (got: $BENCH_WAIT_TIMEOUT_SEC)."
        exit 2
        ;;
esac
case "$BENCH_WAIT_POLL_SEC" in
    ''|*[!0-9]*)
        log "ERROR: BENCH_WAIT_POLL_SEC must be a positive integer (got: $BENCH_WAIT_POLL_SEC)."
        exit 2
        ;;
esac
if [ "$BENCH_WAIT_POLL_SEC" -le 0 ]; then
    log "ERROR: BENCH_WAIT_POLL_SEC must be greater than 0 to avoid a tight polling loop."
    exit 2
fi

if [ "$BENCH_WAIT_TIMEOUT_SEC" -le 0 ]; then
    log "BENCH_WAIT_TIMEOUT_SEC=0; skipping wait for 05_bench_public."
    if pgrep -f "05_bench_public" > /dev/null; then
        log "WARNING: 05_bench_public processes still match (continuing anyway):"
        log_matching_bench_processes
    fi
else
    log "Waiting for 05_bench_public processes to drain (timeout=${BENCH_WAIT_TIMEOUT_SEC}s)..."
    waited=0
    while pgrep -f "05_bench_public" > /dev/null; do
        if [ "$waited" -ge "$BENCH_WAIT_TIMEOUT_SEC" ]; then
            log "ERROR: timed out after ${waited}s waiting for 05_bench_public to drain."
            log "Matching process summaries (pid elapsed command):"
            log_matching_bench_processes
            log "If these are stale/unrelated, kill them or rerun with BENCH_WAIT_TIMEOUT_SEC=0 to skip the wait."
            exit 2
        fi
        remaining=$((BENCH_WAIT_TIMEOUT_SEC - waited))
        if [ "$BENCH_WAIT_POLL_SEC" -lt "$remaining" ]; then
            sleep_for="$BENCH_WAIT_POLL_SEC"
        else
            sleep_for="$remaining"
        fi
        sleep "$sleep_for"
        waited=$((waited + sleep_for))
    done
    log "Public bench is no longer running (waited ${waited}s)"
fi

# This wrapper is intentionally public so maintainers can version the pipeline
# entrypoint, but several downstream scripts remain private/local tooling. Fail
# before starting services if a clean checkout does not contain those scripts.
required_private_scripts=(
    scripts/surgical_rerun_a_d.py
    scripts/analysis_claim_support.py
    scripts/analysis_scaling_and_ci.py
    scripts/generate_article_figures.py
)
missing_private_scripts=()
for script in "${required_private_scripts[@]}"; do
    if [ ! -f "$script" ]; then
        missing_private_scripts+=("$script")
    fi
done
if [ "${#missing_private_scripts[@]}" -gt 0 ]; then
    log "ERROR: audit pipeline requires private/local scripts that are not present in this checkout:"
    printf '  %s\n' "${missing_private_scripts[@]}" >&2
    log "Restore the private audit tooling or keep this wrapper disabled in public clones."
    exit 2
fi

# 2. Verify Ollama is up; restart if needed
if ! curl -sf http://localhost:11434/api/tags > /dev/null; then
    log "Starting Ollama"
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    disown
    sleep 5
fi
log "Ollama is up with $(curl -s http://localhost:11434/api/tags | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("models",[])))') models"

# 3. Surgical re-run: Tasks A and D for top-10 models
log "Stripping cached Task A/D entries and launching bench for top-10"
conda run -n dl python scripts/surgical_rerun_a_d.py --top-k 10
log "Surgical re-run complete"

# 4. Re-run claim audit (will pick up per-query units automatically)
log "Running claim audit"
conda run -n dl python scripts/analysis_claim_support.py

# 5. Refresh CIs and scaling
log "Refreshing scaling/CI plots"
conda run -n dl python scripts/analysis_scaling_and_ci.py

# 6. Regenerate all article figures
log "Regenerating article figures"
conda run -n dl python scripts/generate_article_figures.py

log "Pipeline complete"
