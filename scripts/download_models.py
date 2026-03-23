#!/usr/bin/env python3
"""
Unified model download manager for scMetaIntel-Hub.

Downloads all models required by the pipeline across 4 categories:
  - Ollama LLMs (17 models via ``ollama pull``)
  - HuggingFace embeddings (18 models via snapshot_download)
  - HuggingFace rerankers (7 models via snapshot_download)
  - HuggingFace fine-tuning base models (8 models via snapshot_download)

Usage:
    python scripts/download_models.py --phase core
    python scripts/download_models.py --phase benchmark
    python scripts/download_models.py --phase finetune
    python scripts/download_models.py --phase all
    python scripts/download_models.py --category hf-embedding
    python scripts/download_models.py --check
    python scripts/download_models.py --dry-run --phase all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import (
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    LLM_MODELS,
    HF_ENDPOINT,
    DEFAULT_LLM,
    DEFAULT_LLM_FAST,
    DEFAULT_EMBEDDING,
    DEFAULT_EMBED_BIO,
    DEFAULT_RERANKER,
    OLLAMA_HOST,
    PROJECT_ROOT,
)

logger = logging.getLogger("download_models")


# ---------------------------------------------------------------------------
# Enums and data types
# ---------------------------------------------------------------------------

class Category(Enum):
    OLLAMA = "ollama"
    HF_EMBEDDING = "hf-embedding"
    HF_RERANKER = "hf-reranker"
    HF_FINETUNE = "hf-finetune"


class Status(Enum):
    CACHED = "cached"
    DOWNLOADED = "downloaded"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"


@dataclass
class ModelSpec:
    key: str
    display_name: str
    category: Category
    download_id: str
    disabled: bool = False
    estimated_size_gb: float = 0.0
    note: str = ""


@dataclass
class DownloadResult:
    model: ModelSpec
    status: Status
    elapsed_sec: float = 0.0
    path: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# FINETUNE_CANDIDATES loader (avoids importing full benchmark module)
# ---------------------------------------------------------------------------

def _load_finetune_candidates() -> Dict[str, dict]:
    """Import FINETUNE_CANDIDATES from benchmarks/06_bench_finetune.py."""
    bench_path = PROJECT_ROOT / "benchmarks" / "06_bench_finetune.py"
    if not bench_path.exists():
        logger.warning(f"Fine-tune benchmark not found at {bench_path}")
        return {}
    spec = importlib.util.spec_from_file_location("bench_finetune", bench_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        logger.warning(f"Could not load finetune candidates: {e}")
        return {}
    return getattr(mod, "FINETUNE_CANDIDATES", {})


# ---------------------------------------------------------------------------
# Registry — builds categorized model lists from project config
# ---------------------------------------------------------------------------

class Registry:

    def __init__(self, include_disabled: bool = False):
        self.include_disabled = include_disabled
        self._finetune_candidates: Dict[str, dict] | None = None

    @property
    def finetune_candidates(self) -> Dict[str, dict]:
        if self._finetune_candidates is None:
            self._finetune_candidates = _load_finetune_candidates()
        return self._finetune_candidates

    def get_ollama_models(self) -> List[ModelSpec]:
        models = []
        for key, cfg in LLM_MODELS.items():
            models.append(ModelSpec(
                key=key,
                display_name=cfg["ollama_name"],
                category=Category.OLLAMA,
                download_id=cfg["ollama_name"],
                estimated_size_gb=cfg.get("vram_gb", 0),
                note=cfg.get("note", ""),
            ))
        return models

    def get_hf_embedding_models(self) -> List[ModelSpec]:
        models = []
        for key, cfg in EMBEDDING_MODELS.items():
            disabled = cfg.get("disabled", False)
            if disabled and not self.include_disabled:
                continue
            models.append(ModelSpec(
                key=key,
                display_name=cfg["name"],
                category=Category.HF_EMBEDDING,
                download_id=cfg["name"],
                disabled=disabled,
                estimated_size_gb=cfg.get("vram_gb", 0.5),
                note=cfg.get("note", ""),
            ))
        return models

    def get_hf_reranker_models(self) -> List[ModelSpec]:
        models = []
        for key, cfg in RERANKER_MODELS.items():
            disabled = cfg.get("disabled", False)
            if disabled and not self.include_disabled:
                continue
            models.append(ModelSpec(
                key=key,
                display_name=cfg["name"],
                category=Category.HF_RERANKER,
                download_id=cfg["name"],
                disabled=disabled,
                estimated_size_gb=cfg.get("vram_gb", 0.5),
                note=cfg.get("note", ""),
            ))
        return models

    def get_hf_finetune_models(self) -> List[ModelSpec]:
        models = []
        for key, cfg in self.finetune_candidates.items():
            models.append(ModelSpec(
                key=key,
                display_name=cfg["hf_name"],
                category=Category.HF_FINETUNE,
                download_id=cfg["hf_name"],
                estimated_size_gb=cfg.get("vram_4bit_gb", 5) * 4,
                note=cfg.get("note", ""),
            ))
        return models

    def get_core_models(self) -> List[ModelSpec]:
        """The 4 default runtime models needed for basic pipeline operation."""
        core = []
        # Default LLM (Ollama)
        llm_cfg = LLM_MODELS[DEFAULT_LLM]
        core.append(ModelSpec(
            key=DEFAULT_LLM,
            display_name=llm_cfg["ollama_name"],
            category=Category.OLLAMA,
            download_id=llm_cfg["ollama_name"],
            estimated_size_gb=llm_cfg.get("vram_gb", 0),
        ))
        # Fast fallback LLM if different
        if DEFAULT_LLM_FAST != DEFAULT_LLM and DEFAULT_LLM_FAST in LLM_MODELS:
            fast_cfg = LLM_MODELS[DEFAULT_LLM_FAST]
            core.append(ModelSpec(
                key=DEFAULT_LLM_FAST,
                display_name=fast_cfg["ollama_name"],
                category=Category.OLLAMA,
                download_id=fast_cfg["ollama_name"],
                estimated_size_gb=fast_cfg.get("vram_gb", 0),
            ))
        # Default dense embedding (HF)
        emb_cfg = EMBEDDING_MODELS[DEFAULT_EMBEDDING]
        core.append(ModelSpec(
            key=DEFAULT_EMBEDDING,
            display_name=emb_cfg["name"],
            category=Category.HF_EMBEDDING,
            download_id=emb_cfg["name"],
            estimated_size_gb=emb_cfg.get("vram_gb", 0),
        ))
        # Bio embedding (HF)
        bio_cfg = EMBEDDING_MODELS[DEFAULT_EMBED_BIO]
        core.append(ModelSpec(
            key=DEFAULT_EMBED_BIO,
            display_name=bio_cfg["name"],
            category=Category.HF_EMBEDDING,
            download_id=bio_cfg["name"],
            estimated_size_gb=bio_cfg.get("vram_gb", 0),
        ))
        # Default reranker (HF)
        rer_cfg = RERANKER_MODELS[DEFAULT_RERANKER]
        core.append(ModelSpec(
            key=DEFAULT_RERANKER,
            display_name=rer_cfg["name"],
            category=Category.HF_RERANKER,
            download_id=rer_cfg["name"],
            estimated_size_gb=rer_cfg.get("vram_gb", 0),
        ))
        return core

    def get_router_models(self) -> List[ModelSpec]:
        """Ollama models used by the task-specific router (TASK_MODEL_MAP)."""
        from scmetaintel.router import TASK_MODEL_MAP
        seen = set()
        models = []
        for task, model_key in TASK_MODEL_MAP.items():
            if model_key in seen or model_key not in LLM_MODELS:
                continue
            seen.add(model_key)
            cfg = LLM_MODELS[model_key]
            models.append(ModelSpec(
                key=model_key,
                display_name=cfg["ollama_name"],
                category=Category.OLLAMA,
                download_id=cfg["ollama_name"],
                estimated_size_gb=cfg.get("vram_gb", 0),
            ))
        return models

    def resolve_phase(self, phase: str) -> List[ModelSpec]:
        if phase == "core":
            return self.get_core_models()
        elif phase == "router":
            return self.get_router_models()
        elif phase == "benchmark":
            return (
                self.get_ollama_models()
                + self.get_hf_embedding_models()
                + self.get_hf_reranker_models()
            )
        elif phase == "finetune":
            return self.get_hf_finetune_models()
        elif phase == "all":
            return (
                self.get_ollama_models()
                + self.get_hf_embedding_models()
                + self.get_hf_reranker_models()
                + self.get_hf_finetune_models()
            )
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def resolve_category(self, category_str: str) -> List[ModelSpec]:
        cat = Category(category_str)
        dispatch = {
            Category.OLLAMA: self.get_ollama_models,
            Category.HF_EMBEDDING: self.get_hf_embedding_models,
            Category.HF_RERANKER: self.get_hf_reranker_models,
            Category.HF_FINETUNE: self.get_hf_finetune_models,
        }
        return dispatch[cat]()


# ---------------------------------------------------------------------------
# OllamaDownloader
# ---------------------------------------------------------------------------

class OllamaDownloader:

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.ollama_bin = os.environ.get(
            "OLLAMA_BIN",
            shutil.which("ollama") or "/home/zeyufu/miniconda3/envs/dl/bin/ollama",
        )
        self._installed_cache: List[str] | None = None

    def is_serving(self) -> bool:
        import requests
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def start_service(self) -> bool:
        if self.is_serving():
            return True
        logger.info("Starting Ollama service...")
        env = os.environ.copy()
        env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
        try:
            subprocess.Popen(
                [self.ollama_bin, "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.error(f"Ollama binary not found at {self.ollama_bin}")
            return False
        for _ in range(30):
            time.sleep(0.5)
            if self.is_serving():
                logger.info("Ollama service started.")
                return True
        logger.error("Ollama service did not start within 15 seconds.")
        return False

    def list_installed(self) -> List[str]:
        if self._installed_cache is not None:
            return self._installed_cache
        import requests
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            self._installed_cache = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            self._installed_cache = []
        return self._installed_cache

    def invalidate_cache(self):
        self._installed_cache = None

    def is_cached(self, ollama_name: str) -> bool:
        installed = self.list_installed()
        if ollama_name in installed:
            return True
        # Match by base name (e.g. "qwen2.5:1.5b" matches "qwen2.5:1.5b-...")
        base = ollama_name.split(":")[0] if ":" in ollama_name else ollama_name
        tag = ollama_name.split(":", 1)[1] if ":" in ollama_name else ""
        for m in installed:
            if m == ollama_name or m.startswith(ollama_name):
                return True
            # match "model:tag" against "model:tag-suffix"
            if ":" in m:
                m_base, m_tag = m.split(":", 1)
                if m_base == base and m_tag.startswith(tag):
                    return True
        return False

    def pull(self, model: ModelSpec) -> DownloadResult:
        t0 = time.time()
        try:
            result = subprocess.run(
                [self.ollama_bin, "pull", model.download_id],
                capture_output=True,
                text=True,
                timeout=7200,
            )
            elapsed = time.time() - t0
            self.invalidate_cache()
            if result.returncode == 0:
                return DownloadResult(
                    model=model, status=Status.DOWNLOADED,
                    elapsed_sec=elapsed,
                )
            else:
                return DownloadResult(
                    model=model, status=Status.FAILED,
                    elapsed_sec=elapsed,
                    error=result.stderr.strip()[:300],
                )
        except subprocess.TimeoutExpired:
            return DownloadResult(
                model=model, status=Status.FAILED,
                elapsed_sec=time.time() - t0,
                error="Timed out after 7200s",
            )
        except Exception as e:
            return DownloadResult(
                model=model, status=Status.FAILED,
                elapsed_sec=time.time() - t0,
                error=str(e)[:300],
            )


# ---------------------------------------------------------------------------
# HuggingFaceDownloader
# ---------------------------------------------------------------------------

class HuggingFaceDownloader:

    # Default: sequential single-thread downloads for stability.
    # huggingface_hub uses 8 concurrent threads by default which saturates
    # bandwidth and causes partial-download failures on large models.
    DEFAULT_MAX_WORKERS = 1
    DEFAULT_RETRIES = 3
    DEFAULT_RETRY_DELAY = 10  # seconds between retries

    def __init__(self, endpoint: str = HF_ENDPOINT, token: str | None = None,
                 max_workers: int = DEFAULT_MAX_WORKERS,
                 retries: int = DEFAULT_RETRIES):
        self.endpoint = endpoint
        self.token = token or os.environ.get("HF_TOKEN")
        self.max_workers = max_workers
        self.retries = retries

    @staticmethod
    def _cache_root() -> Path:
        return Path(
            os.environ.get(
                "HF_HUB_CACHE",
                Path.home() / ".cache" / "huggingface" / "hub",
            )
        )

    def _repo_dir(self, repo_id: str) -> Path:
        return self._cache_root() / f"models--{repo_id.replace('/', '--')}"

    def has_incomplete_blobs(self, repo_id: str) -> Tuple[bool, int]:
        """Check if a cached model has .incomplete blob files (partial downloads)."""
        blobs_dir = self._repo_dir(repo_id) / "blobs"
        if not blobs_dir.exists():
            return False, 0
        incomplete = list(blobs_dir.glob("*.incomplete"))
        return len(incomplete) > 0, len(incomplete)

    def clean_incomplete_blobs(self, repo_id: str) -> int:
        """Remove .incomplete blob files so snapshot_download can resume cleanly."""
        blobs_dir = self._repo_dir(repo_id) / "blobs"
        if not blobs_dir.exists():
            return 0
        removed = 0
        for f in blobs_dir.glob("*.incomplete"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def is_cached(self, repo_id: str) -> Tuple[bool, str]:
        # If there are incomplete blobs, the model is NOT fully cached
        has_partial, n_partial = self.has_incomplete_blobs(repo_id)
        if has_partial:
            return False, ""
        # Method 1: huggingface_hub API
        path = ""
        try:
            from huggingface_hub import snapshot_download
            p = snapshot_download(repo_id=repo_id, local_files_only=True)
            path = str(p)
        except Exception:
            # Method 2: manual snapshot directory check
            cache_root = self._cache_root()
            repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
            snapshots_dir = repo_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = sorted(
                    [p for p in snapshots_dir.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if snapshots:
                    path = str(snapshots[0])
        if not path:
            return False, ""
        # Guard against gated-model stubs: if the snapshot only has tiny
        # metadata files (LICENSE, README) but no actual model weights,
        # we should NOT consider it cached.  Real embedding models are >=5 MB
        # and LLMs are >> 1 GB.
        total_blob_bytes = sum(
            f.stat().st_size
            for f in self._repo_dir(repo_id).joinpath("blobs").rglob("*")
            if f.is_file()
        ) if self._repo_dir(repo_id).joinpath("blobs").exists() else 0
        if total_blob_bytes < 1_000_000:  # < 1 MB → stub / gated rejection
            return False, ""
        return True, path

    def cached_size_mb(self, repo_id: str) -> float:
        repo_dir = self._repo_dir(repo_id)
        if not repo_dir.exists():
            return 0.0
        total = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
        return round(total / 1e6, 1)

    def _do_download(self, repo_id: str) -> str:
        """Single download attempt. Returns snapshot path on success, raises on failure."""
        from huggingface_hub import snapshot_download

        os.environ["HF_ENDPOINT"] = self.endpoint
        # Force sequential downloads: max_workers=1 prevents concurrent file
        # downloads that saturate the network and cause partial-download failures.
        path = snapshot_download(
            repo_id=repo_id,
            token=self.token,
            endpoint=self.endpoint,
            max_workers=self.max_workers,
        )
        return str(path)

    def download(self, model: ModelSpec) -> DownloadResult:
        t0 = time.time()
        last_err = ""

        for attempt in range(1, self.retries + 1):
            # Clean up incomplete blobs from prior failed attempts so
            # snapshot_download starts fresh on the failed files
            n_cleaned = self.clean_incomplete_blobs(model.download_id)
            if n_cleaned:
                logger.info(f"    Cleaned {n_cleaned} incomplete blob(s) before attempt {attempt}")

            try:
                path = self._do_download(model.download_id)
                # Verify no incomplete blobs remain after download
                has_partial, n_partial = self.has_incomplete_blobs(model.download_id)
                if has_partial:
                    raise RuntimeError(
                        f"Download finished but {n_partial} .incomplete blob(s) remain"
                    )
                elapsed = time.time() - t0
                return DownloadResult(
                    model=model, status=Status.DOWNLOADED,
                    elapsed_sec=elapsed, path=path,
                )
            except Exception as e:
                last_err = str(e)[:300]
                # Don't retry gated/auth errors — they won't succeed
                if any(s in last_err.lower() for s in ["gated", "401", "403", "access"]):
                    last_err += " (hint: set --hf-token or $HF_TOKEN for gated models)"
                    break
                if attempt < self.retries:
                    delay = self.DEFAULT_RETRY_DELAY * attempt
                    logger.warning(
                        f"    Attempt {attempt}/{self.retries} failed: {last_err[:120]}"
                        f" — retrying in {delay}s..."
                    )
                    time.sleep(delay)

        return DownloadResult(
            model=model, status=Status.FAILED,
            elapsed_sec=time.time() - t0, error=last_err,
        )


# ---------------------------------------------------------------------------
# CacheChecker
# ---------------------------------------------------------------------------

class CacheChecker:

    def __init__(self, registry: Registry):
        self.registry = registry
        self.ollama = OllamaDownloader()
        self.hf = HuggingFaceDownloader()

    def check_all(self) -> Dict[str, List[dict]]:
        report: Dict[str, List[dict]] = {
            "ollama": [],
            "hf-embedding": [],
            "hf-reranker": [],
            "hf-finetune": [],
        }

        # Ollama
        for model in self.registry.get_ollama_models():
            is_avail = self.ollama.is_cached(model.download_id)
            report["ollama"].append({
                "key": model.key,
                "name": model.download_id,
                "status": "cached" if is_avail else "missing",
                "vram_gb": model.estimated_size_gb,
            })

        # HF embeddings — always show all (including disabled) for status
        reg_all = Registry(include_disabled=True)
        for model in reg_all.get_hf_embedding_models():
            is_cached, path = self.hf.is_cached(model.download_id)
            has_partial, n_partial = self.hf.has_incomplete_blobs(model.download_id)
            size_mb = self.hf.cached_size_mb(model.download_id)
            if has_partial:
                status = "partial"
            elif is_cached:
                status = "cached"
            elif model.disabled:
                status = "disabled"
            else:
                status = "missing"
            report["hf-embedding"].append({
                "key": model.key,
                "repo": model.download_id,
                "status": status,
                "path": path,
                "size_mb": size_mb,
                "disabled": model.disabled,
                "incomplete_blobs": n_partial,
            })

        for model in reg_all.get_hf_reranker_models():
            is_cached, path = self.hf.is_cached(model.download_id)
            has_partial, n_partial = self.hf.has_incomplete_blobs(model.download_id)
            size_mb = self.hf.cached_size_mb(model.download_id)
            if has_partial:
                status = "partial"
            elif is_cached:
                status = "cached"
            elif model.disabled:
                status = "disabled"
            else:
                status = "missing"
            report["hf-reranker"].append({
                "key": model.key,
                "repo": model.download_id,
                "status": status,
                "path": path,
                "size_mb": size_mb,
                "disabled": model.disabled,
                "incomplete_blobs": n_partial,
            })

        for model in self.registry.get_hf_finetune_models():
            is_cached, path = self.hf.is_cached(model.download_id)
            has_partial, n_partial = self.hf.has_incomplete_blobs(model.download_id)
            size_mb = self.hf.cached_size_mb(model.download_id)
            if has_partial:
                status = "partial"
            elif is_cached:
                status = "cached"
            else:
                status = "missing"
            report["hf-finetune"].append({
                "key": model.key,
                "repo": model.download_id,
                "status": status,
                "path": path,
                "size_mb": size_mb,
                "incomplete_blobs": n_partial,
            })

        return report

    def print_report(self, report: Dict[str, List[dict]]):
        labels = {
            "ollama": "Ollama LLMs",
            "hf-embedding": "HuggingFace Embeddings",
            "hf-reranker": "HuggingFace Rerankers",
            "hf-finetune": "HuggingFace Fine-tune Base Models",
        }
        for cat_key, entries in report.items():
            cached = sum(1 for e in entries if e["status"] == "cached")
            partial = sum(1 for e in entries if e["status"] == "partial")
            total = len(entries)
            disabled = sum(1 for e in entries if e.get("disabled"))
            header = f"{labels[cat_key]} ({cached}/{total} cached"
            if partial:
                header += f", {partial} partial"
            if disabled:
                header += f", {disabled} disabled"
            header += ")"
            print(f"\n{'=' * 64}")
            print(f"  {header}")
            print(f"{'=' * 64}")

            for e in entries:
                tag = e["status"].upper()
                name = e.get("name") or e.get("repo", "")
                disp = name if len(name) <= 50 else name[:47] + "..."
                if cat_key == "ollama":
                    size_info = f"~{e['vram_gb']:.1f} GB VRAM"
                else:
                    size_info = f"{e.get('size_mb', 0):.0f} MB" if e.get("size_mb") else ""
                    if e.get("incomplete_blobs"):
                        size_info += f" ({e['incomplete_blobs']} incomplete)"
                print(f"  [{tag:>8s}]  {e['key']:<24s} {disp:<52s} {size_info}")

        # Summary line
        total_all = sum(len(v) for v in report.values())
        cached_all = sum(
            1 for v in report.values() for e in v if e["status"] == "cached"
        )
        partial_all = sum(
            1 for v in report.values() for e in v if e["status"] == "partial"
        )
        print(f"\n  Total: {cached_all}/{total_all} models cached", end="")
        if partial_all:
            print(f", {partial_all} partial (re-run to resume)")
        else:
            print()


# ---------------------------------------------------------------------------
# DownloadOrchestrator
# ---------------------------------------------------------------------------

class DownloadOrchestrator:

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.ollama = OllamaDownloader()
        self.hf = HuggingFaceDownloader()
        self.results: List[DownloadResult] = []

    def set_hf_options(self, endpoint: str | None = None, token: str | None = None,
                       max_workers: int | None = None, retries: int | None = None):
        if endpoint:
            self.hf.endpoint = endpoint
        if token:
            self.hf.token = token
        if max_workers is not None:
            self.hf.max_workers = max_workers
        if retries is not None:
            self.hf.retries = retries

    def download_models(self, models: List[ModelSpec]) -> List[DownloadResult]:
        ollama_models = [m for m in models if m.category == Category.OLLAMA]
        hf_models = [m for m in models if m.category != Category.OLLAMA]

        if ollama_models:
            self._download_ollama(ollama_models)
        if hf_models:
            self._download_hf(hf_models)

        return self.results

    def _download_ollama(self, models: List[ModelSpec]):
        print(f"\n--- Ollama LLMs ({len(models)} models) ---")

        if self.dry_run:
            for m in models:
                is_cached = self.ollama.is_cached(m.download_id)
                tag = "CACHED" if is_cached else "WOULD DOWNLOAD"
                print(f"  [DRY-RUN {tag}] {m.key} ({m.download_id}) ~{m.estimated_size_gb:.1f} GB")
                self.results.append(DownloadResult(
                    model=m,
                    status=Status.CACHED if is_cached else Status.SKIPPED,
                ))
            return

        if not self.ollama.start_service():
            logger.error("Cannot start Ollama. Skipping all Ollama downloads.")
            for m in models:
                self.results.append(DownloadResult(
                    model=m, status=Status.FAILED,
                    error="Ollama service unavailable",
                ))
            return

        for i, m in enumerate(models, 1):
            if self.ollama.is_cached(m.download_id):
                print(f"  [{i}/{len(models)}] CACHED: {m.key} ({m.download_id})")
                self.results.append(DownloadResult(model=m, status=Status.CACHED))
                continue

            print(f"  [{i}/{len(models)}] Pulling: {m.key} ({m.download_id})...")
            result = self.ollama.pull(m)
            self.results.append(result)
            tag = "OK" if result.status == Status.DOWNLOADED else "FAILED"
            msg = f"  -> {tag} ({result.elapsed_sec:.1f}s)"
            if result.error:
                msg += f" Error: {result.error}"
            print(msg)

    def _download_hf(self, models: List[ModelSpec]):
        enabled = [m for m in models if not m.disabled]
        disabled = [m for m in models if m.disabled]

        print(f"\n--- HuggingFace Models ({len(enabled)} enabled, {len(disabled)} disabled) ---")

        for m in disabled:
            print(f"  [DISABLED] {m.key} ({m.download_id}) — skipped")
            self.results.append(DownloadResult(model=m, status=Status.DISABLED))

        for i, m in enumerate(enabled, 1):
            is_cached, path = self.hf.is_cached(m.download_id)

            if self.dry_run:
                tag = "CACHED" if is_cached else "WOULD DOWNLOAD"
                print(f"  [DRY-RUN {tag}] {m.key} ({m.download_id}) ~{m.estimated_size_gb:.1f} GB")
                self.results.append(DownloadResult(
                    model=m,
                    status=Status.CACHED if is_cached else Status.SKIPPED,
                    path=path,
                ))
                continue

            if is_cached:
                print(f"  [{i}/{len(enabled)}] CACHED: {m.key} ({m.download_id})")
                self.results.append(DownloadResult(model=m, status=Status.CACHED, path=path))
                continue

            print(f"  [{i}/{len(enabled)}] Downloading: {m.key} ({m.download_id})"
                  f" [retries={self.hf.retries}, workers={self.hf.max_workers}]...")
            result = self.hf.download(m)
            self.results.append(result)
            tag = "OK" if result.status == Status.DOWNLOADED else "FAILED"
            msg = f"  -> {tag} ({result.elapsed_sec:.1f}s)"
            if result.error:
                msg += f"\n     Error: {result.error}"
            print(msg)

    def print_summary(self):
        cached = [r for r in self.results if r.status == Status.CACHED]
        downloaded = [r for r in self.results if r.status == Status.DOWNLOADED]
        failed = [r for r in self.results if r.status == Status.FAILED]
        skipped = [r for r in self.results if r.status == Status.SKIPPED]
        disabled = [r for r in self.results if r.status == Status.DISABLED]

        print(f"\n{'=' * 60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Already cached: {len(cached)}")
        print(f"  Downloaded:     {len(downloaded)}")
        print(f"  Failed:         {len(failed)}")
        print(f"  Skipped:        {len(skipped)}")
        print(f"  Disabled:       {len(disabled)}")
        print(f"  Total:          {len(self.results)}")

        if failed:
            print(f"\nFailed models ({len(failed)}):")
            for r in failed:
                print(f"  - {r.model.key} ({r.model.download_id}): {r.error}")

        total_time = sum(r.elapsed_sec for r in self.results)
        if total_time > 0:
            print(f"\nTotal time: {total_time:.0f}s ({total_time / 60:.1f}m)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _deduplicate(models: List[ModelSpec]) -> List[ModelSpec]:
    seen: set = set()
    unique: List[ModelSpec] = []
    for m in models:
        if m.download_id not in seen:
            seen.add(m.download_id)
            unique.append(m)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Download all models for scMetaIntel-Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --phase core                    # 4 default runtime models
  %(prog)s --phase router                  # 4 task-specific router models
  %(prog)s --phase benchmark               # All 39 benchmark models
  %(prog)s --phase finetune                # 8 HF fine-tuning base models
  %(prog)s --phase all                     # Everything (~47 models)
  %(prog)s --category hf-embedding         # Only HF embeddings
  %(prog)s --category ollama               # Only Ollama LLMs
  %(prog)s --check                         # Report cache status
  %(prog)s --dry-run --phase all           # Preview what would download
  %(prog)s --phase benchmark --include-disabled   # Include broken models
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--phase",
        choices=["core", "router", "benchmark", "finetune", "all"],
        help="Download phase preset",
    )
    group.add_argument(
        "--category",
        choices=["ollama", "hf-embedding", "hf-reranker", "hf-finetune"],
        help="Download a specific model category",
    )
    group.add_argument(
        "--check", action="store_true",
        help="Report cache status for all models (no downloads)",
    )

    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    parser.add_argument("--include-disabled", action="store_true",
                        help="Include disabled models")
    parser.add_argument("--hf-endpoint", default=None,
                        help=f"HuggingFace endpoint URL (default: {HF_ENDPOINT})")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token for gated models")
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of retry attempts per HF model (default: 3)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max concurrent file downloads per HF model (default: 1 for stability)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    registry = Registry(include_disabled=args.include_disabled)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # --check mode
    if args.check:
        checker = CacheChecker(registry)
        report = checker.check_all()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            checker.print_report(report)
        return

    # Resolve model list
    if args.phase:
        models = registry.resolve_phase(args.phase)
    else:
        models = registry.resolve_category(args.category)

    models = _deduplicate(models)
    logger.info(f"Resolved {len(models)} models to download (phase={args.phase}, category={args.category})")

    orchestrator = DownloadOrchestrator(dry_run=args.dry_run)
    orchestrator.set_hf_options(
        endpoint=args.hf_endpoint,
        token=args.hf_token,
        max_workers=args.max_workers,
        retries=args.retries,
    )

    orchestrator.download_models(models)

    if args.json:
        results_data = [
            {
                "key": r.model.key,
                "category": r.model.category.value,
                "download_id": r.model.download_id,
                "status": r.status.value,
                "elapsed_sec": round(r.elapsed_sec, 1),
                "path": r.path,
                "error": r.error,
            }
            for r in orchestrator.results
        ]
        print(json.dumps(results_data, indent=2))
    else:
        orchestrator.print_summary()


if __name__ == "__main__":
    main()
