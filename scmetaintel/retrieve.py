"""
Hybrid retrieval for scMetaIntel-Hub.

This module merges:
- `HybridRetriever` from the standalone scMetaIntel project
- `Reranker` and `RetrievalPipeline` from GEO-DataHub/scmetaintel/retrieve.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .answer import parse_query as llm_parse_query
from .config import QDRANT_COLLECTION, RERANKER_MODELS, get_config
from .embed import Embedder, get_qdrant_client, resolve_load_name, search_dense
from .models import CharacteristicsSummary, EnrichedStudy, ParsedQuery, SearchResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Reranker:
    def __init__(self, model_key: str | None = None, device: str | None = None):
        cfg = get_config()
        self.model_key = model_key or cfg.embedding.reranker_model_key
        model_cfg = RERANKER_MODELS.get(self.model_key, {})
        # Respect per-model device override (e.g. "cpu" for Blackwell-incompatible models)
        if device is not None:
            self.device = device
        else:
            self.device = model_cfg.get("device", "cuda")
        self._model = None
        self._is_cross_encoder = False

    def _load(self):
        if self._model is not None:
            return
        cfg = get_config()
        model_cfg = RERANKER_MODELS[self.model_key]
        model_name = model_cfg["name"]
        load_name = resolve_load_name(model_name)
        # Always use sentence-transformers CrossEncoder (FlagReranker's compute_score
        # calls prepare_for_model which is removed in transformers>=5.3)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(
            load_name, local_files_only=True, trust_remote_code=True)
        tokenizer = getattr(self._model, "tokenizer", None)
        model = getattr(self._model, "model", None)
        if tokenizer is not None and tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
        if tokenizer is not None and model is not None and getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        self._is_cross_encoder = True
        logger.info(f"Loaded reranker via CrossEncoder: {model_name}")

    def score(self, query: str, passages: List[str]) -> List[float]:
        self._load()
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs)
        return list(scores)


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder,
        qdrant_client=None,
        collection_name: str = QDRANT_COLLECTION,
        reranker: Optional[Reranker] = None,
        strategy: str = "hybrid+filter+rerank",
        top_k: int = 50,
        rerank_k: int = 10,
    ):
        self.embedder = embedder
        self.client = qdrant_client or get_qdrant_client()
        self.collection = collection_name
        self.reranker = reranker
        self.strategy = strategy
        self.top_k = top_k
        self.rerank_k = rerank_k
        self._all_docs: Optional[List[Dict]] = None

    def _get_all_docs(self) -> List[Dict]:
        if self._all_docs is not None:
            return self._all_docs
        docs = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                docs.append({"id": p.id, **p.payload})
            if offset is None:
                break
        self._all_docs = docs
        return docs

    def retrieve(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        use_dense = "dense" in self.strategy or "hybrid" in self.strategy
        use_sparse = "sparse" in self.strategy or "hybrid" in self.strategy
        use_filter = "filter" in self.strategy
        use_rerank = "rerank" in self.strategy

        results_by_id: Dict[str, Dict] = {}
        dense_ranking, sparse_ranking = [], []

        if use_dense:
            qvec = self.embedder.encode([query])[0]
            f = filters if use_filter else None
            hits = search_dense(self.client, self.collection, qvec, top_k=self.top_k, filters=f)
            for h in hits:
                gse = h.payload.get("gse_id", str(h.id))
                dense_ranking.append(gse)
                results_by_id[gse] = {**h.payload, "dense_score": h.score}

        if use_sparse:
            query_tokens = set(query.lower().split())
            all_docs = self._get_all_docs()
            scored = []
            for doc in all_docs:
                text = doc.get("document_text", doc.get("search_text", doc.get("title", ""))).lower()
                doc_tokens = Counter(text.split())
                overlap = sum(doc_tokens[t] for t in query_tokens if t in doc_tokens)
                if overlap > 0:
                    scored.append((doc["gse_id"], overlap, doc))
            scored.sort(key=lambda x: x[1], reverse=True)
            for gse, sc, doc in scored[: self.top_k]:
                sparse_ranking.append(gse)
                if gse not in results_by_id:
                    results_by_id[gse] = doc
                results_by_id[gse]["sparse_score"] = sc

        if use_dense and use_sparse:
            ranking = [g for g, _ in reciprocal_rank_fusion([dense_ranking, sparse_ranking])[: self.top_k]]
        elif use_dense:
            ranking = dense_ranking
        else:
            ranking = sparse_ranking

        if use_rerank and self.reranker and ranking:
            passages = [results_by_id[g].get("document_text", results_by_id[g].get("search_text", results_by_id[g].get("title", ""))) for g in ranking[: self.top_k]]
            scores = self.reranker.score(query, passages)
            if not isinstance(scores, list):
                scores = [scores]
            scored_pairs = list(zip(ranking[: self.top_k], scores))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            ranking = [g for g, _ in scored_pairs[: self.rerank_k]]
            for g, s in scored_pairs:
                results_by_id[g]["rerank_score"] = s

        out = []
        for rank, gse in enumerate(ranking[: self.rerank_k], start=1):
            entry = results_by_id[gse]
            entry["rank"] = rank
            entry["gse_id"] = gse
            out.append(entry)
        return out


# ---------------------------------------------------------------------------
# Context Management Utilities (for 05b_bench_context_management.py)
# ---------------------------------------------------------------------------


def reorder_mmr(
    studies: List[Dict],
    query_embedding: np.ndarray,
    embedder: Embedder,
    lambda_param: float = 0.5,
) -> List[Dict]:
    """Maximal Marginal Relevance reordering for diversity.

    Iteratively selects the study most similar to query but dissimilar
    to already-selected studies.
    """
    if len(studies) <= 1:
        return studies

    # Encode all study texts
    texts = [
        s.get("document_text", s.get("search_text", s.get("title", "")))
        for s in studies
    ]
    doc_embeddings = embedder.encode(texts)

    # Normalize
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    d_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10)

    # Query similarities
    query_sims = d_norms @ q_norm

    selected_indices = []
    remaining = list(range(len(studies)))

    while remaining:
        if not selected_indices:
            # First pick: highest query similarity
            best_idx = max(remaining, key=lambda i: query_sims[i])
        else:
            # MMR: balance relevance vs diversity
            best_score = -float("inf")
            best_idx = remaining[0]
            for i in remaining:
                rel = query_sims[i]
                # Max similarity to any selected doc
                max_sim = max(float(d_norms[i] @ d_norms[j]) for j in selected_indices)
                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [studies[i] for i in selected_indices]


def reorder_recency(studies: List[Dict]) -> List[Dict]:
    """Reorder studies by submission_date (most recent first)."""
    def _date_key(s):
        d = s.get("submission_date", "")
        return d if isinstance(d, str) else ""
    return sorted(studies, key=_date_key, reverse=True)


def expand_query_ontology(
    query: str,
    parsed: Dict,
    ontology_indices: Dict,
    max_synonyms: int = 3,
) -> str:
    """Expand query with synonyms from CL/UBERON/MONDO ontologies.

    Args:
        query: Original query string.
        parsed: Parsed query dict with fields like tissue, disease, cell_type.
        ontology_indices: Dict of {prefix: OntologyIndex} with lookup tables built.
        max_synonyms: Max synonyms to add per matched term.
    """
    expansions = []
    field_to_ontology = {
        "tissue": ["UBERON"],
        "disease": ["MONDO"],
        "cell_type": ["CL"],
    }
    for field, prefixes in field_to_ontology.items():
        value = parsed.get(field, "")
        if not value:
            continue
        key = value.lower().strip()
        for prefix in prefixes:
            idx = ontology_indices.get(prefix)
            if idx is None:
                continue
            term_id = idx.lookup.get(key)
            if term_id:
                syns = idx.synonyms.get(term_id, [])
                expansions.extend(syns[:max_synonyms])

    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


def multi_step_retrieve(
    pipeline: "RetrievalPipeline",
    query: str,
    model_key: str | None = None,
    steps: int = 2,
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """Multi-step retrieval: retrieve → LLM refines query → retrieve again → merge.

    Step 1: Initial retrieval with original query.
    Step 2: Use LLM to generate a refined/expanded query based on initial results.
    Step 3: Retrieve again with refined query, merge via RRF.
    """
    from .answer import llm_call as _llm_call

    # Step 1: initial retrieval
    results_1 = pipeline.retrieve(query, filters=filters)

    if steps <= 1 or not results_1:
        return results_1

    # Step 2: LLM-based query refinement
    top_titles = [r.get("title", "") for r in results_1[:5]]
    refine_prompt = (
        f"Original query: {query}\n\n"
        f"Top results found:\n" +
        "\n".join(f"- {t}" for t in top_titles) +
        f"\n\nGenerate a more specific search query that would find "
        f"additional relevant datasets. Return ONLY the refined query, nothing else."
    )
    try:
        refined = _llm_call(
            refine_prompt,
            model_key=model_key or "qwen2.5-0.5b",
            system="You refine scientific search queries. Return only the query.",
            temperature=0.0,
            max_tokens=100,
            timeout=30,
        ).strip()
    except Exception:
        return results_1

    if not refined or len(refined) < 5:
        return results_1

    # Step 3: retrieve with refined query
    results_2 = pipeline.retrieve(refined, filters=filters)

    # Merge via RRF
    ranking_1 = [r.get("gse_id", "") for r in results_1]
    ranking_2 = [r.get("gse_id", "") for r in results_2]
    merged_ranking = [g for g, _ in reciprocal_rank_fusion([ranking_1, ranking_2])]

    # Build unified results dict
    all_results = {}
    for r in results_1 + results_2:
        gse = r.get("gse_id", "")
        if gse not in all_results:
            all_results[gse] = r

    return [all_results[g] for g in merged_ranking if g in all_results][: pipeline.rerank_k]


class HybridRetriever:
    def __init__(self):
        self.cfg = get_config()
        self.embedder = Embedder(model_key=self.cfg.embedding.dense_model_key, device=self.cfg.embedding.device)
        self.client = get_qdrant_client(self.cfg.paths.qdrant_dir)
        try:
            self.reranker = Reranker(device=self.cfg.embedding.device)
        except Exception as e:
            logger.warning(f"Reranker unavailable: {e}")
            self.reranker = None

    def parse_query(self, query: str) -> ParsedQuery:
        try:
            data = llm_parse_query(query, model_key=self.cfg.llm.model_key)
        except Exception as e:
            logger.warning(f"LLM parse failed: {e}")
            data = self._fallback_parse(query)
        if isinstance(data, ParsedQuery):
            return data
        return ParsedQuery(
            raw_query=query,
            organism=data.get("organism", "") or "",
            tissue=data.get("tissue", "") or "",
            disease=data.get("disease", "") or "",
            cell_type=data.get("cell_type", "") or "",
            assay=data.get("assay", "") or "",
            treatment=data.get("treatment", "") or "",
            free_text=data.get("free_text", query) or query,
        )

    def _fallback_parse(self, query: str) -> Dict:
        q = query.lower()
        data = {"organism": "", "tissue": "", "disease": "", "cell_type": "", "assay": "", "treatment": "", "free_text": query}
        if "human" in q or "homo sapiens" in q:
            data["organism"] = "Homo sapiens"
        elif "mouse" in q or "mus musculus" in q:
            data["organism"] = "Mus musculus"
        assay_map = {
            "scatac": "scATAC-seq",
            "scatac-seq": "scATAC-seq",
            "cite-seq": "CITE-seq",
            "multiome": "Multiome",
            "spatial": "Spatial",
            "scrna": "scRNA-seq",
            "scRNA-seq": "scRNA-seq",
        }
        for k, v in assay_map.items():
            if k.lower() in q:
                data["assay"] = v
                break
        return data

    def _build_filter_dict(self, parsed: ParsedQuery) -> Optional[Dict]:
        filters = {}
        if parsed.organism:
            filters["organism"] = parsed.organism
        return filters or None

    def _payload_to_study(self, payload: Dict) -> EnrichedStudy:
        tissues = payload.get("tissues", [])
        diseases = payload.get("diseases", [])
        cell_types = payload.get("cell_types", [])
        treatments = payload.get("treatments", [])
        return EnrichedStudy(
            gse_id=payload.get("gse_id", ""),
            title=payload.get("title", ""),
            summary=payload.get("summary", payload.get("document_text", "")),
            overall_design=payload.get("overall_design", ""),
            organism=payload.get("organism", ""),
            platform=payload.get("platform", ""),
            series_type=payload.get("series_type", ""),
            n_samples=payload.get("n_samples", payload.get("sample_count", 0)),
            submission_date=payload.get("submission_date", ""),
            domain=payload.get("domain", ""),
            modalities=payload.get("modalities", []),
            characteristics_summary=CharacteristicsSummary(
                tissues=tissues,
                diseases=diseases,
                cell_types=cell_types,
                treatments=treatments,
            ),
        )

    def search(self, query: str, top_k: int | None = None, use_reranker: bool = True) -> tuple[ParsedQuery, list[SearchResult]]:
        parsed = self.parse_query(query)
        strategy = "hybrid+filter+rerank" if use_reranker and self.reranker else "hybrid+filter"
        pipeline = RetrievalPipeline(
            embedder=self.embedder,
            qdrant_client=self.client,
            collection_name=self.cfg.retrieval.collection_name,
            reranker=self.reranker if use_reranker else None,
            strategy=strategy,
            top_k=top_k or self.cfg.retrieval.top_k_retrieve,
            rerank_k=self.cfg.retrieval.top_k_rerank,
        )
        filter_dict = self._build_filter_dict(parsed)
        results = pipeline.retrieve(parsed.free_text or query, filters=filter_dict)
        out = []
        for row in results:
            out.append(
                SearchResult(
                    study=self._payload_to_study(row),
                    score=float(row.get("dense_score", row.get("score", 0.0) or 0.0)),
                    rerank_score=float(row.get("rerank_score", 0.0) or 0.0),
                )
            )
        return parsed, out

    def load_full_study(self, gse_id: str) -> EnrichedStudy | None:
        candidates = [
            self.cfg.paths.enriched_dir / f"{gse_id}.json",
            self.cfg.paths.ground_truth_dir / f"{gse_id}.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                try:
                    return EnrichedStudy.from_dict(data)
                except Exception:
                    return self._payload_to_study(data)
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid retrieval for scMetaIntel-Hub")
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    retriever = HybridRetriever()
    parsed, results = retriever.search(args.query, top_k=args.top_k, use_reranker=not args.no_rerank)
    print(f"\nQuery: {args.query}")
    print(f"Parsed: organism={parsed.organism}, tissue={parsed.tissue}, disease={parsed.disease}, assay={parsed.assay}")
    print(f"\nResults ({len(results)}):\n")
    for i, r in enumerate(results, 1):
        s = r.study
        print(f"  {i}. {s.gse_id} — {s.title}")
        print(f"     Score: {r.score:.3f} | Rerank: {r.rerank_score:.3f}")
        print(f"     Organism: {s.organism} | Tissues: {s.characteristics_summary.tissues}")
        print(f"     Diseases: {s.characteristics_summary.diseases}\n")


if __name__ == "__main__":
    main()
