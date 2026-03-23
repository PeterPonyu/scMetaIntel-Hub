"""
Ontology normalization pipeline for scMetaIntel-Hub.

Normalizes raw GEO metadata terms to:
- Cell Ontology (CL)
- UBERON
- MONDO
- optionally EFO later
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import get_config
from .embed import resolve_load_name
from .models import EnrichedStudy, OntologyMapping

logger = logging.getLogger(__name__)


class OntologyIndex:
    def __init__(self, name: str, prefix: str):
        self.name = name
        self.prefix = prefix
        self.terms: dict[str, str] = {}
        self.synonyms: dict[str, list[str]] = {}
        self.lookup: dict[str, str] = {}
        self.embeddings: np.ndarray | None = None
        self.embedding_ids: list[str] = []

    def build_lookup(self):
        self.lookup = {}
        for term_id, name in self.terms.items():
            self.lookup[name.lower().strip()] = term_id
        for term_id, syns in self.synonyms.items():
            for syn in syns:
                key = syn.lower().strip()
                if key not in self.lookup:
                    self.lookup[key] = term_id

    def exact_match(self, text: str) -> OntologyMapping | None:
        key = text.lower().strip()
        if key in self.lookup:
            term_id = self.lookup[key]
            return OntologyMapping(
                raw_term=text,
                ontology_id=term_id,
                ontology_name=self.terms.get(term_id, ""),
                ontology_source=self.name,
                confidence=1.0,
                method="exact",
            )
        return None

    def embedding_match(self, text_embedding: np.ndarray, top_k: int = 3, threshold: float = 0.7) -> list[OntologyMapping]:
        if self.embeddings is None or not self.embedding_ids:
            return []
        text_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-10)
        emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = emb_norms @ text_norm
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < threshold:
                continue
            term_id = self.embedding_ids[idx]
            results.append(
                OntologyMapping(
                    raw_term="",
                    ontology_id=term_id,
                    ontology_name=self.terms.get(term_id, ""),
                    ontology_source=self.name,
                    confidence=score,
                    method="embedding",
                )
            )
        return results


def _parse_obo_simple(obo_path: Path, prefix: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    terms: dict[str, str] = {}
    synonyms: dict[str, list[str]] = {}
    current_id = None
    current_name = None
    current_syns: list[str] = []
    in_term = False
    is_obsolete = False
    prefix_bare = prefix.rstrip(":")

    with open(obo_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                if current_id and current_name and not is_obsolete:
                    terms[current_id] = current_name
                    synonyms[current_id] = current_syns
                current_id = None
                current_name = None
                current_syns = []
                in_term = True
                is_obsolete = False
                continue
            if line.startswith("[") and line.endswith("]"):
                if in_term and current_id and current_name and not is_obsolete:
                    terms[current_id] = current_name
                    synonyms[current_id] = current_syns
                in_term = False
                current_id = None
                current_name = None
                current_syns = []
                is_obsolete = False
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line[4:].strip()
                if not current_id.startswith(prefix_bare):
                    current_id = None
            elif line.startswith("name: "):
                current_name = line[6:].strip()
            elif line.startswith("synonym: "):
                start = line.find('"')
                end = line.find('"', start + 1)
                if start >= 0 and end > start:
                    current_syns.append(line[start + 1 : end])
            elif line.startswith("is_obsolete: true"):
                is_obsolete = True

    if in_term and current_id and current_name and not is_obsolete:
        terms[current_id] = current_name
        synonyms[current_id] = current_syns
    return terms, synonyms


def load_obo(obo_path: Path) -> OntologyIndex:
    name_map = {
        "cl": ("CL", "CL:"),
        "uberon": ("UBERON", "UBERON:"),
        "mondo": ("MONDO", "MONDO:"),
        "efo": ("EFO", "EFO:"),
    }
    stem = obo_path.stem.lower().split("-")[0]
    name, prefix = name_map.get(stem, (stem.upper(), f"{stem.upper()}:"))
    index = OntologyIndex(name=name, prefix=prefix)
    try:
        import pronto
        ont = pronto.Ontology(obo_path)
        for term in ont.terms():
            term_id = str(term.id)
            if not term_id.startswith(prefix.rstrip(":")):
                continue
            term_name = str(term.name) if term.name else ""
            if not term_name:
                continue
            index.terms[term_id] = term_name
            syns = []
            for syn in term.synonyms:
                if syn.description:
                    syns.append(str(syn.description))
            index.synonyms[term_id] = syns
    except Exception as e:
        logger.warning(f"pronto failed for {obo_path}: {e}. Using simple parser.")
        terms, synonyms = _parse_obo_simple(obo_path, prefix)
        index.terms = terms
        index.synonyms = synonyms
    index.build_lookup()
    logger.info(f"Loaded {index.name}: {len(index.terms)} terms")
    return index


class OntologyNormalizer:
    def __init__(self, ontology_dir: Path | None = None):
        cfg = get_config()
        self.ontology_dir = ontology_dir or cfg.paths.ontologies_dir
        self.indices: dict[str, OntologyIndex] = {}
        self.bio_encoder = None
        self._cache: dict[str, OntologyMapping] = {}

    def load_ontologies(self):
        obo_files = {
            "CL": "cl.obo",
            "UBERON": "uberon-basic.obo",
            "MONDO": "mondo.obo",
        }
        for name, filename in obo_files.items():
            path = self.ontology_dir / filename
            if path.exists():
                try:
                    self.indices[name] = load_obo(path)
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
            else:
                logger.warning(f"Ontology file not found: {path}")

    def _load_bio_encoder(self):
        if self.bio_encoder is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            cfg = get_config()
            logger.info(f"Loading ontology encoder: {cfg.embedding.bio_model}")
            load_name = resolve_load_name(cfg.embedding.bio_model)
            self.bio_encoder = SentenceTransformer(
                load_name,
                device=cfg.embedding.device,
                trust_remote_code=True,
                local_files_only=True,
            )

    def build_embedding_indices(self, cache_path: Path | None = None):
        self._load_bio_encoder()
        cache_path = cache_path or (self.ontology_dir / "ontology_embeddings.pkl")
        if cache_path.exists():
            logger.info(f"Loading cached ontology embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            for name, data in cached.items():
                if name in self.indices:
                    self.indices[name].embeddings = data["embeddings"]
                    self.indices[name].embedding_ids = data["ids"]
            return
        for name, idx in self.indices.items():
            texts, ids = [], []
            for term_id, term_name in idx.terms.items():
                syns = idx.synonyms.get(term_id, [])
                texts.append(f"{term_name}: {', '.join(syns[:5])}" if syns else term_name)
                ids.append(term_id)
            if texts:
                embeddings = self.bio_encoder.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
                idx.embeddings = embeddings
                idx.embedding_ids = ids
        cache_data = {}
        for name, idx in self.indices.items():
            if idx.embeddings is not None:
                cache_data[name] = {"embeddings": idx.embeddings, "ids": idx.embedding_ids}
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Cached ontology embeddings to {cache_path}")

    def normalize(self, text: str, category: str, use_embedding: bool = True) -> OntologyMapping:
        cache_key = f"{category}:{text.lower().strip()}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        ontology_names = {"cell_type": ["CL"], "tissue": ["UBERON"], "disease": ["MONDO"]}.get(category, list(self.indices.keys()))
        for onto_name in ontology_names:
            if onto_name not in self.indices:
                continue
            result = self.indices[onto_name].exact_match(text)
            if result:
                self._cache[cache_key] = result
                return result
        if use_embedding:
            self._load_bio_encoder()
            text_emb = self.bio_encoder.encode(text, normalize_embeddings=True)
            for onto_name in ontology_names:
                if onto_name not in self.indices:
                    continue
                matches = self.indices[onto_name].embedding_match(text_emb, top_k=1, threshold=0.65)
                if matches:
                    result = matches[0]
                    result.raw_term = text
                    self._cache[cache_key] = result
                    return result
        result = OntologyMapping(raw_term=text, confidence=0.0, method="none")
        self._cache[cache_key] = result
        return result

    def normalize_study(self, study: EnrichedStudy) -> dict[str, list[OntologyMapping]]:
        mappings: dict[str, list[OntologyMapping]] = defaultdict(list)
        cs = study.characteristics_summary
        for tissue in cs.tissues:
            mappings["tissue"].append(self.normalize(tissue, "tissue"))
        for disease in cs.diseases:
            mappings["disease"].append(self.normalize(disease, "disease"))
        for cell_type in cs.cell_types:
            mappings["cell_type"].append(self.normalize(cell_type, "cell_type"))
        study.ontology_mappings = dict(mappings)
        return study.ontology_mappings


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ontology normalization for scMetaIntel-Hub")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--normalize", type=Path)
    parser.add_argument("--test", type=str, nargs="+")
    parser.add_argument("--category", default="tissue", choices=["tissue", "disease", "cell_type"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    normalizer = OntologyNormalizer()
    normalizer.load_ontologies()

    if args.build_index or args.test or args.normalize:
        normalizer.build_embedding_indices()

    if args.test:
        for term in args.test:
            result = normalizer.normalize(term, args.category)
            print(f"'{term}' -> {result.ontology_id} ({result.ontology_name}) [{result.method}, conf={result.confidence:.2f}]")

    if args.normalize:
        for json_path in sorted(args.normalize.glob("*.json")):
            with open(json_path) as f:
                data = json.load(f)
            study = EnrichedStudy.from_dict(data) if ("samples" in data or "characteristics_summary" in data) else EnrichedStudy(
                gse_id=data.get("gse_id", ""),
                title=data.get("title", ""),
                organism=data.get("organism", ""),
            )
            normalizer.normalize_study(study)
            with open(json_path, "w") as f:
                json.dump(study.to_dict(), f, indent=2, default=str)
        logger.info("Done normalizing studies.")


if __name__ == "__main__":
    main()
