#!/usr/bin/env bash
# Pull all 17 LLM models for scMetaIntel-Hub benchmarking.
# All models are served via Ollama.
# Usage: OLLAMA_HOME=/home/zeyufu/.ollama bash scripts/pull_models.sh
set -e

OLLAMA_BIN="${OLLAMA_BIN:-/home/zeyufu/miniconda3/envs/dl/bin/ollama}"

if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "Starting Ollama service..."
  OLLAMA_HOME=/home/zeyufu/.ollama OLLAMA_FLASH_ATTENTION=1 "$OLLAMA_BIN" serve &
  sleep 5
fi

MODELS=(
  # Qwen family
  "qwen3.5:27b"
  "qwen3.5:9b"
  "qwen3.5:9b-q8_0"
  "qwen3:32b"
  "qwen3:14b"
  "qwen3:14b-q8_0"
  "qwen3:8b"
  "qwen2.5:7b-instruct-q8_0"
  "qwen2.5:1.5b"
  "qwen2.5:0.5b"
  # Cross-architecture
  "gemma3:27b-it-qat"
  "gemma3:12b-it-q8_0"
  "mistral-small:24b-instruct-2501-q4_K_M"
  "mistral-nemo:12b-instruct-2407-q8_0"
  "phi4:14b-q8_0"
  "llama3.1:8b-instruct-q8_0"
  "command-r:35b-08-2024-q4_K_M"
)

for model in "${MODELS[@]}"; do
  echo "Pulling $model..."
  OLLAMA_HOME=/home/zeyufu/.ollama "$OLLAMA_BIN" pull "$model" || echo "  WARN: Failed to pull $model"
done

echo "Done. $(OLLAMA_HOME=/home/zeyufu/.ollama "$OLLAMA_BIN" list | wc -l) models available."

# ============================================================
# HuggingFace models (embeddings + rerankers + finetune bases)
# Delegates to the unified Python download manager.
# Usage:
#   bash scripts/pull_models.sh --hf          # embeddings + rerankers
#   bash scripts/pull_models.sh --hf-all      # embeddings + rerankers + finetune bases
# For finer control, use the Python script directly:
#   python scripts/download_models.py --check
#   python scripts/download_models.py --phase all
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$1" = "--hf" ]; then
  echo ""
  echo "=== Pulling HuggingFace embedding + reranker models ==="
  python3 "$SCRIPT_DIR/download_models.py" --category hf-embedding || true
  python3 "$SCRIPT_DIR/download_models.py" --category hf-reranker || true
  echo "HuggingFace model downloads complete."
fi

if [ "$1" = "--hf-all" ]; then
  echo ""
  echo "=== Pulling ALL HuggingFace models (embeds + rerankers + finetune bases) ==="
  python3 "$SCRIPT_DIR/download_models.py" --category hf-embedding || true
  python3 "$SCRIPT_DIR/download_models.py" --category hf-reranker || true
  python3 "$SCRIPT_DIR/download_models.py" --category hf-finetune || true
  echo "HuggingFace model downloads complete."
fi
