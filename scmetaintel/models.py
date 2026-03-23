"""Data models and schemas for scMetaIntel."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleMeta:
    gsm_id: str
    title: str = ""
    source: str = ""
    characteristics: dict[str, str] = field(default_factory=dict)
    organism: str = ""
    platform: str = ""


@dataclass
class PubMedInfo:
    pmid: str
    title: str = ""
    abstract: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class OntologyMapping:
    raw_term: str
    ontology_id: str = ""
    ontology_name: str = ""
    ontology_source: str = ""  # CL, UBERON, MONDO, EFO
    confidence: float = 0.0
    method: str = ""  # exact, synonym, embedding, llm


@dataclass
class CharacteristicsSummary:
    tissues: list[str] = field(default_factory=list)
    diseases: list[str] = field(default_factory=list)
    cell_types: list[str] = field(default_factory=list)
    treatments: list[str] = field(default_factory=list)
    organisms: list[str] = field(default_factory=list)
    donor_count: int = 0
    sex: list[str] = field(default_factory=list)
    age_range: str = ""


@dataclass
class EnrichedStudy:
    gse_id: str
    title: str = ""
    summary: str = ""
    overall_design: str = ""
    organism: str = ""
    platform: str = ""
    series_type: str = ""
    n_samples: int = 0
    submission_date: str = ""
    domain: str = ""
    modalities: list[str] = field(default_factory=list)
    samples: dict[str, SampleMeta] = field(default_factory=dict)
    characteristics_summary: CharacteristicsSummary = field(default_factory=CharacteristicsSummary)
    pubmed: PubMedInfo | None = None
    ontology_mappings: dict[str, list[OntologyMapping]] = field(default_factory=dict)
    files_summary: dict[str, Any] = field(default_factory=dict)

    def to_search_text(self) -> str:
        """Generate a combined text for embedding / search."""
        parts = [
            f"Study: {self.title}",
            f"Organism: {self.organism}",
        ]
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.overall_design:
            parts.append(f"Design: {self.overall_design}")
        if self.modalities:
            parts.append(f"Modalities: {', '.join(self.modalities)}")
        if self.domain:
            parts.append(f"Domain: {self.domain}")

        cs = self.characteristics_summary
        if cs.tissues:
            parts.append(f"Tissues: {', '.join(cs.tissues)}")
        if cs.diseases:
            parts.append(f"Diseases: {', '.join(cs.diseases)}")
        if cs.cell_types:
            parts.append(f"Cell types: {', '.join(cs.cell_types)}")
        if cs.treatments:
            parts.append(f"Treatments: {', '.join(cs.treatments)}")
        if cs.donor_count > 0:
            parts.append(f"Donors: {cs.donor_count}")

        if self.pubmed:
            if self.pubmed.abstract:
                parts.append(f"Abstract: {self.pubmed.abstract}")
            if self.pubmed.mesh_terms:
                parts.append(f"MeSH: {', '.join(self.pubmed.mesh_terms)}")
            if self.pubmed.keywords:
                parts.append(f"Keywords: {', '.join(self.pubmed.keywords)}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EnrichedStudy:
        """Deserialize from dict."""
        samples = {}
        for gsm_id, s in d.get("samples", {}).items():
            if isinstance(s, dict):
                # Only pass keys that SampleMeta accepts
                samples[gsm_id] = SampleMeta(
                    gsm_id=s.get("gsm_id", gsm_id),
                    title=s.get("title", ""),
                    source=s.get("source", ""),
                    characteristics=s.get("characteristics", {}),
                    organism=s.get("organism", ""),
                    platform=s.get("platform", ""),
                )

        pubmed = None
        pm = d.get("pubmed")
        if isinstance(pm, list) and pm:
            first = pm[0]
            pubmed = PubMedInfo(
                pmid=first.get("pmid", ""),
                title=first.get("title", ""),
                abstract=first.get("abstract", ""),
                mesh_terms=first.get("mesh_terms", []),
                keywords=first.get("keywords", []),
            )
        elif isinstance(pm, dict) and pm:
            pubmed = PubMedInfo(**pm)

        cs_data = d.get("characteristics_summary", {})
        # Filter to only CharacteristicsSummary fields
        cs_fields = {"tissues", "diseases", "cell_types", "treatments",
                      "organisms", "donor_count", "sex", "age_range"}
        cs = CharacteristicsSummary(**{k: v for k, v in cs_data.items() if k in cs_fields})

        ontology_mappings = {}
        for key, mappings in d.get("ontology_mappings", {}).items():
            ontology_mappings[key] = [OntologyMapping(**m) for m in mappings]

        return cls(
            gse_id=d["gse_id"],
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            overall_design=d.get("overall_design", ""),
            organism=d.get("organism", ""),
            platform=d.get("platform", ""),
            series_type=d.get("series_type", ""),
            n_samples=d.get("n_samples", 0),
            submission_date=d.get("submission_date", ""),
            domain=d.get("domain", ""),
            modalities=d.get("modalities", []),
            samples=samples,
            characteristics_summary=cs,
            pubmed=pubmed,
            ontology_mappings=ontology_mappings,
            files_summary=d.get("files_summary", {}),
        )


@dataclass
class ParsedQuery:
    raw_query: str
    organism: str = ""
    tissue: str = ""
    disease: str = ""
    cell_type: str = ""
    assay: str = ""
    treatment: str = ""
    free_text: str = ""
    expanded_terms: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SearchResult:
    study: EnrichedStudy
    score: float = 0.0
    rerank_score: float = 0.0
    match_reason: str = ""


@dataclass
class AnswerResponse:
    query: str
    parsed: ParsedQuery
    results: list[SearchResult]
    answer_text: str = ""
    retrieval_stats: dict[str, Any] = field(default_factory=dict)
