"""
Shared utilities for benchmark scripts.

Centralizes Ollama management functions that were duplicated across
04_bench_llm.py, 05_bench_public.py, 08_bench_ablation.py, 09_bench_e2e.py.
"""

import logging
import time

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import (
    OLLAMA_API_TAGS,
    OLLAMA_API_PS,
    OLLAMA_API_GENERATE,
    OLLAMA_KEEP_ALIVE_UNLOAD,
    TIMEOUT_OLLAMA_CHECK,
    TIMEOUT_OLLAMA_MGMT,
)

logger = logging.getLogger(__name__)


def check_ollama() -> list[str]:
    """Check if Ollama server is running and return available model names."""
    try:
        resp = requests.get(OLLAMA_API_TAGS, timeout=TIMEOUT_OLLAMA_CHECK)
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info(f"Ollama models available: {models}")
        return models
    except Exception:
        logger.error("Ollama server not running. Start with: ollama serve")
        return []


def unload_ollama_models():
    """Unload all models from Ollama VRAM to free GPU memory between runs."""
    try:
        resp = requests.get(OLLAMA_API_PS, timeout=TIMEOUT_OLLAMA_CHECK)
        loaded = resp.json().get("models", [])
        for m in loaded:
            name = m["name"]
            logger.info(f"  Unloading {name} from VRAM ...")
            requests.post(
                OLLAMA_API_GENERATE,
                json={"model": name, "keep_alive": OLLAMA_KEEP_ALIVE_UNLOAD},
                timeout=TIMEOUT_OLLAMA_MGMT,
            )
        if loaded:
            time.sleep(2)  # brief pause for VRAM release
            logger.info(f"  Unloaded {len(loaded)} model(s)")
    except Exception as e:
        logger.warning(f"  Failed to unload models: {e}")
