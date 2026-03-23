"""
Embedding generation and Qdrant indexing for scMetaIntel-Hub.

Provides both:
- generic benchmark-friendly helpers (`Embedder`, `build_index_from_ground_truth`)
- class-based project API (`StudyEmbedder`)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, ScoredPoint, VectorParams

from .config import get_config
from .models import CharacteristicsSummary, EnrichedStudy, SampleMeta, PubMedInfo

logger = logging.getLogger(__name__)


def resolve_local_snapshot(repo_id: str) -> str | None:
    """Return the newest local HF snapshot path for a repo, if cached."""
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return None
    snapshots = [p for p in repo_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def resolve_load_name(repo_id: str) -> str:
    """Use local snapshot paths only for model families that require it here."""
    if repo_id.startswith("Qwen/"):
        return resolve_local_snapshot(repo_id) or repo_id
    return repo_id


class Embedder:
    """Unified embedding interface for all supported benchmark/runtime models."""

    def __init__(self, model_key: str | None = None, device: str | None = None):
        cfg = get_config()
        self.model_key = model_key or cfg.embedding.dense_model_key
        self.model_cfg = __import__("scmetaintel.config", fromlist=["EMBEDDING_MODELS"]).EMBEDDING_MODELS[self.model_key]
        # Respect per-model device override (e.g. "cpu" for Blackwell-incompatible BERT models)
        if device is not None:
            self.device = device
        else:
            self.device = self.model_cfg.get("device", "cuda")
        self._model = None
        self._is_flag = self.model_key == "bge-m3"

    def _load(self):
        if self._model is not None:
            return
        name = self.model_cfg["name"]
        load_name = resolve_load_name(name)
        if self._is_flag:
            try:
                from FlagEmbedding import BGEM3FlagModel
                self._model = BGEM3FlagModel(load_name, use_fp16=True)
            except Exception as e:
                logger.warning(
                    "FlagEmbedding BGEM3 load failed for %s (%s); falling back to SentenceTransformer dense-only mode.",
                    name,
                    e,
                )
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    load_name,
                    device=self.device,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self._is_flag = False
        else:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                load_name,
                device=self.device,
                trust_remote_code=True,
                local_files_only=True,
            )
        logger.info(f"Loaded embedding model: {name}")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        self._load()
        if self._is_flag:
            out = self._model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return np.array(out["dense_vecs"], dtype=np.float32)
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embs, dtype=np.float32)

    def encode_sparse(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        if not self._is_flag:
            raise NotImplementedError("Sparse encoding only supported for BGE-M3")
        self._load()
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return out["lexical_weights"]

    @property
    def dim(self) -> int:
        return int(self.model_cfg["dim"])


class StudyEmbedder:
    """Project-level study embedder that indexes enriched studies into Qdrant."""

    def __init__(self):
        self.cfg = get_config()
        self.embedder = Embedder(model_key=self.cfg.embedding.dense_model_key, device=self.cfg.embedding.device)
        self.client = get_qdrant_client(self.cfg.paths.qdrant_dir)

    def get_collection_info(self) -> dict:
        collection = self.client.get_collection(self.cfg.retrieval.collection_name)
        info = {
            "collection": self.cfg.retrieval.collection_name,
            "points_count": collection.points_count,
            "status": getattr(collection.status, "value", str(collection.status)),
        }
        if hasattr(collection, "vectors_count"):
            info["vectors_count"] = collection.vectors_count
        return info

    def index_studies(self, studies: List[EnrichedStudy], force_reindex: bool = False):
        create_collection(self.client, self.cfg.retrieval.collection_name, self.embedder.dim, recreate=force_reindex)
        index_studies(self.client, self.cfg.retrieval.collection_name, studies, self.embedder)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


def get_qdrant_client(path: Optional[Path] = None) -> QdrantClient:
    cfg = get_config()
    return QdrantClient(path=str(path or cfg.paths.qdrant_dir))


def create_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False):
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field in ["organism", "series_type", "domain"]:
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        logger.info(f"Created Qdrant collection '{name}' (dim={dim})")


def _study_text(study: EnrichedStudy | Dict) -> str:
    if isinstance(study, EnrichedStudy):
        return study.to_search_text()
    return study.get("document_text") or "\n".join(
        p for p in [study.get("title", ""), study.get("summary", ""), study.get("overall_design", "")] if p
    )


def _study_payload(study: EnrichedStudy | Dict) -> Dict:
    if isinstance(study, EnrichedStudy):
        return {
            "gse_id": study.gse_id,
            "title": study.title,
            "summary": study.summary[:2000],
            "document_text": study.to_search_text()[:4000],
            "organism": study.organism,
            "platform": study.platform,
            "series_type": study.series_type,
            "domain": study.domain,
            "modalities": study.modalities,
            "n_samples": study.n_samples,
            "submission_date": study.submission_date,
            "tissues": study.characteristics_summary.tissues,
            "diseases": study.characteristics_summary.diseases,
            "cell_types": study.characteristics_summary.cell_types,
            "treatments": study.characteristics_summary.treatments,
            "search_text": study.to_search_text()[:4000],
        }
    tissues = study.get("tissues") or study.get("characteristics_summary", {}).get("tissues", [])
    diseases = study.get("diseases") or study.get("characteristics_summary", {}).get("diseases", [])
    cell_types = study.get("cell_types") or study.get("characteristics_summary", {}).get("cell_types", [])
    return {
        "gse_id": study.get("gse_id", ""),
        "title": study.get("title", ""),
        "summary": study.get("summary", "")[:2000],
        "document_text": _study_text(study)[:4000],
        "organism": study.get("organism", ""),
        "platform": study.get("platform", ""),
        "series_type": study.get("series_type", ""),
        "domain": study.get("domain", ""),
        "modalities": study.get("modalities", []),
        "n_samples": study.get("n_samples", study.get("sample_count", 0)),
        "submission_date": study.get("submission_date", ""),
        "tissues": tissues,
        "diseases": diseases,
        "cell_types": cell_types,
        "treatments": study.get("treatments", []),
        "search_text": _study_text(study)[:4000],
    }


def index_studies(
    client: QdrantClient,
    collection_name: str,
    docs: List[EnrichedStudy | Dict],
    embedder: Embedder,
    batch_size: int = 64,
):
    texts = [_study_text(d) for d in docs]
    logger.info(f"Encoding {len(texts)} documents ...")
    vectors = embedder.encode(texts, batch_size=batch_size, show_progress=True)

    points = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        payload = _study_payload(doc)
        points.append(PointStruct(id=i, vector=vec.tolist(), payload=payload))

    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
    logger.info(f"Indexed {len(points)} studies into '{collection_name}'")


def search_dense(
    client: QdrantClient,
    collection_name: str,
    query_vec: np.ndarray,
    top_k: int = 50,
    filters: Optional[Dict] = None,
) -> List[ScoredPoint]:
    qdrant_filter = None
    if filters:
        conditions = []
        for field, value in filters.items():
            conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        qdrant_filter = Filter(must=conditions)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist(),
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )
    return response.points


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------


def _dict_to_study(data: Dict) -> EnrichedStudy:
    if "samples" in data or "characteristics_summary" in data:
        return EnrichedStudy.from_dict(data)
    pubmed = None
    pubs = data.get("pubmed", [])
    if isinstance(pubs, list) and pubs:
        first = pubs[0]
        pubmed = PubMedInfo(
            pmid=first.get("pmid", ""),
            title=first.get("title", ""),
            abstract=first.get("abstract", ""),
            mesh_terms=first.get("mesh_terms", []),
        )
    return EnrichedStudy(
        gse_id=data.get("gse_id", ""),
        title=data.get("title", ""),
        summary=data.get("summary", ""),
        overall_design=data.get("overall_design", ""),
        organism=data.get("organism", ""),
        platform=data.get("platform", ""),
        series_type=data.get("series_type", ""),
        n_samples=data.get("n_samples", data.get("sample_count", 0)),
        submission_date=data.get("submission_date", ""),
        domain=data.get("domain", ""),
        modalities=data.get("modalities", []),
        characteristics_summary=CharacteristicsSummary(
            tissues=data.get("tissues", []),
            diseases=data.get("diseases", []),
            cell_types=data.get("cell_types", []),
            treatments=data.get("treatments", []),
        ),
        pubmed=pubmed,
    )


def load_enriched_studies(enriched_dir: Path) -> List[EnrichedStudy]:
    studies = []
    for p in sorted(enriched_dir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            studies.append(_dict_to_study(data))
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
    return studies


def build_index_from_ground_truth(
    model_key: Optional[str] = None,
    ground_truth_dir: Optional[Path] = None,
    qdrant_path: Optional[Path] = None,
    collection_name: Optional[str] = None,
) -> Tuple[QdrantClient, Embedder]:
    cfg = get_config()
    gt_dir = ground_truth_dir or cfg.paths.ground_truth_dir
    docs = []
    for p in sorted(gt_dir.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))
    embedder = Embedder(model_key=model_key or cfg.embedding.dense_model_key, device=cfg.embedding.device)
    client = get_qdrant_client(qdrant_path)
    cname = collection_name or cfg.retrieval.collection_name
    create_collection(client, cname, embedder.dim, recreate=True)
    index_studies(client, cname, docs, embedder)
    return client, embedder


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build vector index for scMetaIntel-Hub")
    parser.add_argument("--input", type=Path, default=None, help="Enriched metadata directory")
    parser.add_argument("--rebuild", action="store_true", help="Recreate collection")
    parser.add_argument("--info", action="store_true", help="Show collection info")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = get_config()
    embedder = StudyEmbedder()

    if args.info:
        print(embedder.get_collection_info())
        return

    input_dir = args.input or cfg.paths.enriched_dir
    studies = load_enriched_studies(input_dir)
    logger.info(f"Loaded {len(studies)} studies from {input_dir}")
    if not studies:
        logger.warning("No studies found to index.")
        return
    embedder.index_studies(studies, force_reindex=args.rebuild)
    print(embedder.get_collection_info())


if __name__ == "__main__":
    main()
