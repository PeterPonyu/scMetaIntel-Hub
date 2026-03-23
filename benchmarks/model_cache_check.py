#!/usr/bin/env python3
"""Check local HuggingFace cache availability for configured models."""

from pathlib import Path
import json
import os
from huggingface_hub import snapshot_download

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import EMBEDDING_MODELS, RERANKER_MODELS


def check(repo_id: str) -> dict:
    try:
        p = snapshot_download(repo_id=repo_id, local_files_only=True)
        return {"status": "cached", "path": p}
    except Exception as e:
        cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if snapshots:
                snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return {"status": "cached", "path": str(snapshots[0])}
        return {"status": "missing", "error": str(e).splitlines()[0][:180]}


def main():
    report = {"embedding": {}, "reranker": {}}
    for key, cfg in EMBEDDING_MODELS.items():
        report["embedding"][key] = {"repo": cfg["name"], **check(cfg["name"])}
    for key, cfg in RERANKER_MODELS.items():
        report["reranker"][key] = {"repo": cfg["name"], **check(cfg["name"])}

    out = Path(__file__).resolve().parent / "results" / "model_cache_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    emb_cached = sum(1 for x in report["embedding"].values() if x["status"] == "cached")
    rer_cached = sum(1 for x in report["reranker"].values() if x["status"] == "cached")

    print(f"embedding_cached {emb_cached}/{len(report['embedding'])}")
    print(f"reranker_cached {rer_cached}/{len(report['reranker'])}")
    print(f"report_path {out}")


if __name__ == "__main__":
    main()
