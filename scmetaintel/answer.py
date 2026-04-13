"""
Grounded answer generation and structured LLM helper tasks.

This module merges:
- the class-based `AnswerGenerator` API from standalone scMetaIntel
- the benchmark-friendly helper functions from GEO-DataHub/scmetaintel/answer.py
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, Iterable, List

import requests

# Regex to extract embedded <think>...</think> blocks from DeepSeek-R1 responses.
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)

from .config import (
    LLM_MODELS,
    family_always_thinks,
    family_json_hint,
    family_supports_think_api,
    family_think_method,
    get_family_config,
    resolve_model_family,
    resolve_model_key,
    think_token_budget,
    get_config,
    BENCH_TEMPERATURE,
    TIMEOUT_LLM_LONG,
)
from .models import AnswerResponse, ParsedQuery, SearchResult

logger = logging.getLogger(__name__)


# Import centralized prompts from config — single source of truth
from .config import PROMPTS, DEFAULT_NUM_CTX

SYSTEM_PROMPT = PROMPTS["chat"]
ANSWER_SYSTEM = PROMPTS["answer"]
PARSE_SYSTEM = PROMPTS["parse"]
EXTRACT_SYSTEM = PROMPTS["extract"]

# Regex for stripping markdown code fences around JSON
_MD_CODE_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
)


def extract_json(text: str) -> dict | None:
    """Robustly extract a JSON object from LLM output.

    Handles:
    - Raw JSON: ``{"key": "value"}``
    - Markdown-wrapped: ````json\\n{...}\\n````  (Mistral, Falcon, GLM common)
    - Preamble text before JSON  (Phi, Aya common)
    - Trailing text after JSON
    Returns None if no valid JSON object found.
    """
    # 1. Try markdown code block extraction first
    md = _MD_CODE_BLOCK_RE.search(text)
    if md:
        try:
            return json.loads(md.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. Find the outermost { ... } via brace counting (more robust than greedy regex)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break  # Malformed — fall through

    # 3. Last resort: greedy regex (original behavior)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None

ANSWER_PROMPT = """Based on the user's query and the retrieved dataset metadata below, provide a comprehensive answer.

## User Query
{query}

## Parsed Constraints
- Organism: {organism}
- Tissue: {tissue}
- Disease: {disease}
- Cell type: {cell_type}
- Assay: {assay}

## Retrieved Datasets ({n_results} found)

{context}

## Instructions
1. Summarize the relevant datasets found
2. For each dataset, mention: GSE ID, key metadata, and relevance to the query
3. If datasets can be integrated or compared, mention that
4. Cite every claim with [GSE_ID]
5. End with a brief summary of coverage and any gaps

Answer:"""


def _resolve_model(model_key_or_name: str | None) -> str:
    cfg = get_config()
    if not model_key_or_name:
        return cfg.llm.model
    if model_key_or_name in LLM_MODELS:
        return LLM_MODELS[model_key_or_name]["ollama_name"]
    return model_key_or_name


def ollama_generate(
    prompt: str,
    model: str | None = None,
    system: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int = 120,
    think: bool | None = None,
) -> Dict:
    cfg = get_config()
    model_name = _resolve_model(model)
    temperature = cfg.llm.temperature if temperature is None else temperature
    max_tokens = cfg.llm.max_tokens if max_tokens is None else max_tokens

    # Resolve model key via O(1) index (was: O(n) scan per call)
    model_key = (model if model in LLM_MODELS else None) or resolve_model_key(model_name)
    model_info = LLM_MODELS.get(model_key, {}) if model_key else {}
    model_family = model_info.get("family", "")

    # Resolve thinking mode: explicit param > per-model config > False
    if think is None:
        think = model_info.get("think", False)

    # Get full family capabilities (think dispatch, output quirks, etc.)
    fam = get_family_config(model_family)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": DEFAULT_NUM_CTX,
        },
    }
    # Send Ollama "think" API parameter for families that support it
    if fam["think_api"]:
        payload["think"] = think

    resp = requests.post(
        f"{cfg.llm.base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message", {})
    response_text = msg.get("content", "")
    thinking_text = msg.get("thinking", "") or ""

    # Embedded think (DeepSeek-R1): extract <think>...</think> from response.
    if fam["think_method"] == "embedded" and not thinking_text:
        think_match = _THINK_TAG_RE.search(response_text)
        if think_match:
            thinking_text = think_match.group(1).strip()
            response_text = (response_text[:think_match.start()]
                             + response_text[think_match.end():]).strip()

    # Fallback: some models (Gemma3, Granite) in think mode put the answer
    # in the thinking field, leaving content empty.
    if think and not response_text.strip() and thinking_text.strip():
        logger.warning(
            f"Think-mode fallback: {model_name} returned empty content "
            f"with non-empty thinking ({len(thinking_text)} chars). "
            f"Extracting answer from thinking field."
        )
        response_text = thinking_text

    return {
        "response": response_text,
        "thinking": thinking_text,
        "total_duration_ns": data.get("total_duration", 0),
        "eval_count": data.get("eval_count", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "model": model_name,
        "think_enabled": think or fam["always_thinks"],
    }


def ollama_generate_stream(
    prompt: str,
    model: str | None = None,
    system: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int = 120,
) -> Iterable[str]:
    cfg = get_config()
    model_name = _resolve_model(model)
    temperature = cfg.llm.temperature if temperature is None else temperature
    max_tokens = cfg.llm.max_tokens if max_tokens is None else max_tokens

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    with requests.post(
        f"{cfg.llm.base_url}/api/chat",
        json={
            "model": model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": DEFAULT_NUM_CTX,
            },
        },
        timeout=timeout,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield text


def llm_call(prompt: str, model_key: str | None = None, system: str = "", **kwargs) -> str:
    result = ollama_generate(prompt, model=model_key, system=system, **kwargs)
    return result["response"]


def parse_query(query: str, model_key: str | None = None, think: bool | None = None) -> Dict:
    family = resolve_model_family(model_key or "")
    max_tok = think_token_budget(512, think or False, family)
    system = PARSE_SYSTEM + "\n" + family_json_hint(family)
    raw = llm_call(query, model_key=model_key, system=system, temperature=BENCH_TEMPERATURE,
                   max_tokens=max_tok, think=think, timeout=TIMEOUT_LLM_LONG)
    data = extract_json(raw)
    if data:
        if not data.get("free_text"):
            data["free_text"] = query
        return data
    logger.warning(f"Failed to parse query response: {raw[:200]}")
    return {"raw_query": query, "free_text": query}


def extract_metadata(title: str, summary: str, model_key: str | None = None, think: bool | None = None) -> Dict:
    prompt = f"Title: {title}\n\nSummary: {summary}"
    family = resolve_model_family(model_key or "")
    max_tok = think_token_budget(768, think or False, family)
    system = EXTRACT_SYSTEM + "\n" + family_json_hint(family)
    raw = llm_call(prompt, model_key=model_key, system=system, temperature=BENCH_TEMPERATURE,
                   max_tokens=max_tok, think=think, timeout=TIMEOUT_LLM_LONG)
    data = extract_json(raw)
    if data:
        return data
    return {"tissues": [], "diseases": [], "cell_types": [], "modalities": [], "organism": ""}


def format_study_context(result: SearchResult) -> str:
    s = result.study
    cs = s.characteristics_summary
    lines = [
        f"### {s.gse_id}: {s.title}",
        f"- Organism: {s.organism}",
        f"- Domain: {s.domain}",
        f"- Samples: {s.n_samples}",
        f"- Modalities: {', '.join(s.modalities) if s.modalities else 'N/A'}",
        f"- Tissues: {', '.join(cs.tissues) if cs.tissues else 'N/A'}",
        f"- Diseases: {', '.join(cs.diseases) if cs.diseases else 'N/A'}",
        f"- Cell types: {', '.join(cs.cell_types) if cs.cell_types else 'N/A'}",
        f"- Treatments: {', '.join(cs.treatments) if cs.treatments else 'N/A'}",
        f"- Submission: {s.submission_date}",
    ]
    if s.summary:
        summary = s.summary[:500] + "..." if len(s.summary) > 500 else s.summary
        lines.append(f"- Summary: {summary}")
    if s.overall_design:
        design = s.overall_design[:300] + "..." if len(s.overall_design) > 300 else s.overall_design
        lines.append(f"- Design: {design}")
    if s.pubmed:
        lines.append(f"- PubMed: {s.pubmed.pmid}")
        if s.pubmed.mesh_terms:
            lines.append(f"- MeSH: {', '.join(s.pubmed.mesh_terms[:10])}")
    lines.append(f"- Retrieval score: {result.score:.3f} / Rerank: {result.rerank_score:.3f}")
    return "\n".join(lines)


def _study_dict_to_text(study: Dict, fmt: str = "structured", idx: int = 1) -> str:
    gse = study.get("gse_id", "???")
    tissues = study.get("tissues") or study.get("characteristics_summary", {}).get("tissues", [])
    diseases = study.get("diseases") or study.get("characteristics_summary", {}).get("diseases", [])
    cell_types = study.get("cell_types") or study.get("characteristics_summary", {}).get("cell_types", [])
    summary = study.get("summary", "")
    if fmt == "full":
        return (
            f"[{idx}] {gse}: {study.get('title', '')}\n"
            f"  Summary: {summary}\n"
            f"  Organism: {study.get('organism', '')}\n"
            f"  Tissues: {', '.join(tissues)}\n"
            f"  Diseases: {', '.join(diseases)}\n"
            f"  Cell types: {', '.join(cell_types)}\n"
        )
    if fmt == "minimal":
        return f"[{idx}] {gse}: {study.get('title', '')} | {', '.join(tissues) or 'N/A'} | {', '.join(diseases) or 'N/A'}"
    return (
        f"[{idx}] {gse}: {study.get('title', '')}\n"
        f"  Organism: {study.get('organism', '')} | "
        f"Tissues: {', '.join(tissues)} | Diseases: {', '.join(diseases)}\n"
    )


def format_context(studies: List[Dict | SearchResult], fmt: str = "structured") -> str:
    parts: List[str] = []
    for i, item in enumerate(studies, 1):
        if isinstance(item, SearchResult):
            if fmt == "structured":
                s = item.study
                cs = s.characteristics_summary
                parts.append(
                    f"[{i}] {s.gse_id}: {s.title}\n"
                    f"  Organism: {s.organism} | Tissues: {', '.join(cs.tissues)} | Diseases: {', '.join(cs.diseases)}\n"
                )
            else:
                parts.append(format_study_context(item))
        else:
            parts.append(_study_dict_to_text(item, fmt=fmt, idx=i))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Context Management Utilities (for 05b_bench_context_management.py)
# ---------------------------------------------------------------------------


def chunk_study_fields(study: Dict) -> List[Dict]:
    """Split a study dict into semantic field-level chunks.

    Returns list of {"gse_id": str, "chunk_type": str, "text": str}
    """
    gse = study.get("gse_id", "???")
    chunks = []

    # Header chunk: title + organism + modalities
    title = study.get("title", "")
    organism = study.get("organism", "")
    modalities = study.get("modalities", [])
    header = f"{gse}: {title}"
    if organism:
        header += f" | Organism: {organism}"
    if modalities:
        header += f" | Modalities: {', '.join(modalities)}"
    chunks.append({"gse_id": gse, "chunk_type": "header", "text": header})

    # Summary chunk
    summary = study.get("summary", "")
    if summary and len(summary) > 20:
        chunks.append({"gse_id": gse, "chunk_type": "summary", "text": f"{gse} summary: {summary}"})

    # Metadata chunk: tissues, diseases, cell_types
    cs = study.get("characteristics_summary", {})
    tissues = study.get("tissues") or cs.get("tissues", [])
    diseases = study.get("diseases") or cs.get("diseases", [])
    cell_types = study.get("cell_types") or cs.get("cell_types", [])
    meta_parts = []
    if tissues:
        meta_parts.append(f"Tissues: {', '.join(tissues)}")
    if diseases:
        meta_parts.append(f"Diseases: {', '.join(diseases)}")
    if cell_types:
        meta_parts.append(f"Cell types: {', '.join(cell_types)}")
    if meta_parts:
        chunks.append({"gse_id": gse, "chunk_type": "metadata", "text": f"{gse} metadata: {' | '.join(meta_parts)}"})

    # Abstract chunk (from PubMed if available)
    pubmed = study.get("pubmed", {}) or {}
    abstract = pubmed.get("abstract", "")
    if abstract and len(abstract) > 20:
        chunks.append({"gse_id": gse, "chunk_type": "abstract", "text": f"{gse} abstract: {abstract}"})

    return chunks


def format_context_chunked(
    studies: List[Dict],
    chunk_types: List[str] | None = None,
) -> str:
    """Format context using field-level chunks instead of whole documents.

    Args:
        studies: List of study dicts.
        chunk_types: If specified, only include these chunk types
                     (e.g. ["header", "metadata"]). Default: all.
    """
    parts: List[str] = []
    for study in studies:
        chunks = chunk_study_fields(study)
        for chunk in chunks:
            if chunk_types and chunk["chunk_type"] not in chunk_types:
                continue
            parts.append(chunk["text"])
    return "\n".join(parts)


def compress_context(
    studies: List[Dict],
    query: str,
    model_key: str | None = None,
    target_tokens: int = 500,
) -> str:
    """Use a fast LLM to summarize retrieved docs relative to the query.

    Args:
        studies: Retrieved study dicts.
        query: User query (for relevance-aware compression).
        model_key: LLM to use for summarization (default: fast model).
        target_tokens: Approximate target length in words.
    """
    # Build raw context for the LLM to compress
    raw = format_context(studies, fmt="structured")
    compress_prompt = (
        f"Summarize the following dataset descriptions in {target_tokens} words, "
        f"focusing on information relevant to this query: \"{query}\"\n\n"
        f"Preserve all GSE IDs and key metadata (tissues, diseases, organisms).\n\n"
        f"Datasets:\n{raw}"
    )
    compressed = llm_call(
        compress_prompt,
        model_key=model_key or "qwen2.5-0.5b",
        system="You are a concise scientific summarizer. Preserve GSE accession IDs.",
        temperature=0.0,
        max_tokens=target_tokens * 2,
        timeout=60,
    )
    return compressed


def allocate_token_budget(
    studies: List[Dict],
    budget: int = 2000,
    fmt: str = "structured",
) -> List[Dict]:
    """Adaptively trim studies list to fit within a token budget.

    Estimates ~4 chars per token. Keeps as many full studies as fit.
    """
    kept = []
    total_chars = 0
    char_budget = budget * 4  # rough estimate: 4 chars ≈ 1 token
    for study in studies:
        text = _study_dict_to_text(study, fmt=fmt, idx=len(kept) + 1)
        if total_chars + len(text) > char_budget and kept:
            break
        kept.append(study)
        total_chars += len(text)
    return kept
    def __init__(self):
        self.cfg = get_config()

    def generate(self, query: str, parsed: ParsedQuery, results: List[SearchResult]) -> AnswerResponse:
        context = "\n\n".join(format_study_context(r) for r in results)
        prompt = ANSWER_PROMPT.format(
            query=query,
            organism=parsed.organism or "any",
            tissue=parsed.tissue or "any",
            disease=parsed.disease or "any",
            cell_type=parsed.cell_type or "any",
            assay=parsed.assay or "any",
            n_results=len(results),
            context=context,
        )
        answer_text = self._generate_with_llm(prompt)
        if answer_text is None:
            answer_text = self._generate_fallback(query, parsed, results)
        return AnswerResponse(
            query=query,
            parsed=parsed,
            results=results,
            answer_text=answer_text,
            retrieval_stats={
                "total_results": len(results),
                "top_score": results[0].score if results else 0.0,
                "top_rerank_score": results[0].rerank_score if results else 0.0,
            },
        )

    def _generate_with_llm(self, prompt: str) -> str | None:
        try:
            result = ollama_generate(
                prompt,
                model=self.cfg.llm.model_key,
                system=SYSTEM_PROMPT,
                temperature=self.cfg.llm.temperature,
                max_tokens=self.cfg.llm.max_tokens,
            )
            return result["response"].strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            try:
                result = ollama_generate(
                    prompt,
                    model=self.cfg.llm.fallback_model_key,
                    system=SYSTEM_PROMPT,
                    temperature=self.cfg.llm.temperature,
                    max_tokens=self.cfg.llm.max_tokens,
                )
                return result["response"].strip()
            except Exception as e2:
                logger.warning(f"Fallback LLM generation failed: {e2}")
                return None

    def _generate_fallback(self, query: str, parsed: ParsedQuery, results: List[SearchResult]) -> str:
        if not results:
            return f"No datasets found matching: {query}"
        lines = [f"## Results for: \"{query}\"", "", f"Found **{len(results)} datasets** matching your query.\n"]
        for i, r in enumerate(results, 1):
            s = r.study
            cs = s.characteristics_summary
            lines.append(f"### {i}. {s.gse_id} — {s.title}")
            lines.append(f"- **Organism**: {s.organism}")
            if cs.tissues:
                lines.append(f"- **Tissues**: {', '.join(cs.tissues)}")
            if cs.diseases:
                lines.append(f"- **Diseases**: {', '.join(cs.diseases)}")
            if cs.cell_types:
                lines.append(f"- **Cell types**: {', '.join(cs.cell_types)}")
            if s.n_samples:
                lines.append(f"- **Samples**: {s.n_samples}")
            lines.append(f"- **Relevance score**: {r.rerank_score or r.score:.3f}")
            lines.append("")
        return "\n".join(lines)

    def generate_streaming(self, query: str, parsed: ParsedQuery, results: List[SearchResult]):
        context = "\n\n".join(format_study_context(r) for r in results)
        prompt = ANSWER_PROMPT.format(
            query=query,
            organism=parsed.organism or "any",
            tissue=parsed.tissue or "any",
            disease=parsed.disease or "any",
            cell_type=parsed.cell_type or "any",
            assay=parsed.assay or "any",
            n_results=len(results),
            context=context,
        )
        try:
            yield from ollama_generate_stream(
                prompt,
                model=self.cfg.llm.model_key,
                system=SYSTEM_PROMPT,
                temperature=self.cfg.llm.temperature,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception as e:
            logger.warning(f"Streaming generation failed: {e}")
            yield self._generate_fallback(query, parsed, results)


def generate_answer(
    query: str,
    studies: List[Dict | SearchResult],
    model_key: str | None = None,
    context_format: str = "structured",
    system_prompt: str = ANSWER_SYSTEM,
) -> Dict:
    context = format_context(studies, fmt=context_format)
    prompt = (
        f"Retrieved studies:\n{context}\n\n"
        f"User query: {query}\n\n"
        f"Provide a comprehensive answer citing relevant GSE accessions."
    )
    t0 = time.time()
    result = ollama_generate(prompt, model=model_key, system=system_prompt)
    elapsed_ms = (time.time() - t0) * 1000
    answer = result["response"]
    cited = list(sorted(set(re.findall(r"GSE\d+", answer))))
    return {
        "answer": answer,
        "cited_gse": cited,
        "model": _resolve_model(model_key),
        "duration_ms": round(elapsed_ms, 1),
        "eval_tokens": result.get("eval_count", 0),
        "prompt_tokens": result.get("prompt_eval_count", 0),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a grounded answer from raw study JSON context")
    parser.add_argument("query", help="User query")
    args = parser.parse_args()
    result = generate_answer(args.query, [])
    print(result["answer"])


if __name__ == "__main__":
    main()
