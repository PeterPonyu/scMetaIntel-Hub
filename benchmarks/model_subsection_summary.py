#!/usr/bin/env python3
"""Build subsection-wise model summary with local availability status."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scmetaintel.config import (
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    LLM_MODELS,
    DEFAULT_LLM,
    DEFAULT_LLM_FAST,
    DEFAULT_EMBEDDING,
    DEFAULT_EMBED_HYBRID,
    DEFAULT_EMBED_BIO,
    DEFAULT_RERANKER,
)


def read_json(path: Path, fallback: dict | None = None) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return fallback or {}


def get_ollama_models() -> list[str]:
    """Return pulled Ollama model tags, e.g. ['qwen2.5:1.5b', ...]."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=False
        )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return []
        # Skip header line
        models = []
        for ln in lines[1:]:
            parts = ln.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []


def llm_available(ollama_name: str, pulled: list[str]) -> bool:
    """Exact tag match; fallback to base-name family prefix match."""
    if ollama_name in pulled:
        return True
    base = ollama_name.split(":", 1)[0]
    return any(m.startswith(base + ":") for m in pulled)


def main():
    cache_report_path = ROOT / "benchmarks" / "results" / "model_cache_report.json"
    cache_report = read_json(cache_report_path, fallback={"embedding": {}, "reranker": {}})

    pulled = get_ollama_models()

    embedding_status = {}
    for key, cfg in EMBEDDING_MODELS.items():
        st = cache_report.get("embedding", {}).get(key, {})
        embedding_status[key] = {
            "repo": cfg["name"],
            "status": st.get("status", "unknown"),
            "dim": cfg.get("dim"),
            "max_tokens": cfg.get("max_tokens"),
            "type": cfg.get("type"),
            "tier": cfg.get("tier"),
        }

    reranker_status = {}
    for key, cfg in RERANKER_MODELS.items():
        st = cache_report.get("reranker", {}).get(key, {})
        reranker_status[key] = {
            "repo": cfg["name"],
            "status": st.get("status", "unknown"),
        }

    llm_status = {}
    for key, cfg in LLM_MODELS.items():
        ollama_name = cfg["ollama_name"]
        llm_status[key] = {
            "ollama_name": ollama_name,
            "pulled": llm_available(ollama_name, pulled),
            "ctx": cfg.get("ctx"),
            "quant": cfg.get("quant"),
            "size_b": cfg.get("size_b"),
        }

    summary = {
        "section_0_infrastructure": {
            "ollama_pulled_models": pulled,
            "notes": "Large Ollama model pulls may be network-limited; smaller Qwen2.5 models are currently pulled.",
        },
        "section_2_embedding_benchmark": {
            "configured_models": list(EMBEDDING_MODELS.keys()),
            "default_dense": DEFAULT_EMBEDDING,
            "default_hybrid": DEFAULT_EMBED_HYBRID,
            "default_bio": DEFAULT_EMBED_BIO,
            "availability": embedding_status,
        },
        "section_3_retrieval_policy": {
            "primary_dense": DEFAULT_EMBED_HYBRID,
            "primary_reranker": DEFAULT_RERANKER,
            "embedding_availability": {k: v["status"] for k, v in embedding_status.items()},
            "reranker_availability": {k: v["status"] for k, v in reranker_status.items()},
        },
        "section_4_llm_benchmark": {
            "configured_models": list(LLM_MODELS.keys()),
            "default_llm": DEFAULT_LLM,
            "default_llm_fast": DEFAULT_LLM_FAST,
            "availability": llm_status,
        },
        "section_5_context_efficiency": {
            "model_used_by_script_default": DEFAULT_LLM,
            "is_pulled": llm_status.get(DEFAULT_LLM, {}).get("pulled", False),
            "fallback_pulled_models": [k for k, v in llm_status.items() if v.get("pulled")],
        },
        "section_6_finetune": {
            "embedding_base_default": "biolord-2023",
            "embedding_base_available": embedding_status.get("biolord-2023", {}).get("status") == "cached",
            "reranker_base_default": "biomedbert-base (CrossEncoder)",
            "reranker_base_available": embedding_status.get("biomedbert-base", {}).get("status") == "cached",
            "llm_finetune_base_recommended": "qwen3-14b",
            "llm_finetune_base_pulled": llm_status.get("qwen3-14b", {}).get("pulled", False),
        },
        "section_7_end_to_end": {
            "baseline": {
                "embedding": "bge-m3",
                "embedding_ready": embedding_status.get("bge-m3", {}).get("status") == "cached",
                "llm": "qwen2.5-1.5b",
                "llm_ready": llm_status.get("qwen2.5-1.5b", {}).get("pulled", False),
                "reranker": None,
                "ready": embedding_status.get("bge-m3", {}).get("status") == "cached"
                         and llm_status.get("qwen2.5-1.5b", {}).get("pulled", False),
            },
            "optimized": {
                "embedding": "bge-m3",
                "embedding_ready": embedding_status.get("bge-m3", {}).get("status") == "cached",
                "reranker": "bge-reranker-v2-m3",
                "reranker_ready": reranker_status.get("bge-reranker-v2-m3", {}).get("status") == "cached",
                "llm": "qwen3.5-27b",
                "llm_ready": llm_status.get("qwen3.5-27b", {}).get("pulled", False),
                "ready": (
                    embedding_status.get("bge-m3", {}).get("status") == "cached"
                    and reranker_status.get("bge-reranker-v2-m3", {}).get("status") == "cached"
                    and llm_status.get("qwen3.5-27b", {}).get("pulled", False)
                ),
            },
            "finetuned": {
                "embedding": "bge-m3-ft (placeholder)",
                "reranker": "bge-reranker-v2-m3-ft (placeholder)",
                "llm": "qwen3-14b-ft (placeholder)",
                "ready": False,
                "notes": "Fine-tuned artifacts are not yet produced in benchmarks/finetuned/.",
            },
        },
    }

    out = ROOT / "benchmarks" / "results" / "model_subsection_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
