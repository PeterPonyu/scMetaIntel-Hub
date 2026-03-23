#!/usr/bin/env python3
"""
Quick smoke tests for all 5 benchmark tasks from the brainstorm blueprint.

Uses a fast model (qwen3:8b) for speed. Tests:
  Task 1: Information Extraction (IE)
  Task 2: Cell Type Normalization
  Task 3: Bio-QA (experimental design)
  Task 4: RAG Retrieval (embedding + search)
  Task 5: Cross-Dataset Reasoning

Usage:
    conda run -n dl python benchmarks/quick_smoke_test.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

# Bypass proxy for local Ollama
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1"

OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GT_DIR = Path(__file__).resolve().parent / "ground_truth"
# Use a fast model for smoke test — avoid qwen3 thinking mode
TEST_MODEL = "qwen2.5:7b-instruct-q8_0"


def ollama_chat(prompt: str, system: str = "", model: str = TEST_MODEL,
                temperature: float = 0.1, max_tokens: int = 1024) -> dict:
    """Simple Ollama chat call."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.time() - t0
    return {
        "response": data.get("message", {}).get("content", ""),
        "elapsed_s": round(elapsed, 2),
        "eval_count": data.get("eval_count", 0),
    }


def load_sample_gse(n=3):
    """Load n diverse GSE ground truth files."""
    files = sorted(GT_DIR.glob("GSE*.json"))
    # Pick spread: first, middle, near-end
    indices = [0, len(files) // 3, 2 * len(files) // 3]
    samples = []
    for idx in indices[:n]:
        with open(files[idx]) as f:
            samples.append(json.load(f))
    return samples


# ============================================================
# Task 1: Information Extraction
# ============================================================
def test_task1_ie(gse: dict) -> dict:
    """Extract structured metadata from GSE text."""
    print(f"\n{'='*60}")
    print(f"TASK 1: Information Extraction — {gse['gse_id']}")
    print(f"{'='*60}")

    system = (
        "You are a biomedical metadata extractor. Given a GEO dataset's title and "
        "summary, extract structured metadata. Return ONLY valid JSON with:\n"
        '{"organism": "", "tissues": [], "diseases": [], "cell_types": [], '
        '"sequencing_technology": "", "estimated_cell_count": ""}'
    )
    prompt = f"Title: {gse['title']}\n\nSummary: {gse.get('summary', '')}\n\nOverall design: {gse.get('overall_design', '')}"

    result = ollama_chat(prompt, system=system, temperature=0.0, max_tokens=512)
    raw = result["response"]

    # Try to parse JSON from response
    extracted = {}
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            extracted = json.loads(m.group())
    except json.JSONDecodeError:
        pass

    # Compare with ground truth
    gt_organism = gse.get("organism", "")
    gt_tissues = gse.get("tissues", [])
    gt_diseases = gse.get("diseases", [])

    print(f"  Title: {gse['title'][:80]}...")
    print(f"  GT organism: {gt_organism}")
    print(f"  Extracted:   {extracted.get('organism', '???')}")
    print(f"  GT tissues:  {gt_tissues[:3]}")
    print(f"  Extracted:   {extracted.get('tissues', [])[:3]}")
    print(f"  GT diseases: {gt_diseases[:3]}")
    print(f"  Extracted:   {extracted.get('diseases', [])[:3]}")
    print(f"  Cell types:  {extracted.get('cell_types', [])[:5]}")
    print(f"  Time: {result['elapsed_s']}s | Tokens: {result['eval_count']}")

    # Score: organism exact match
    org_match = gt_organism.lower() in extracted.get("organism", "").lower() if gt_organism else True
    print(f"  Organism match: {'PASS' if org_match else 'FAIL'}")

    return {"task": "IE", "gse": gse["gse_id"], "org_match": org_match,
            "extracted": extracted, "elapsed_s": result["elapsed_s"]}


# ============================================================
# Task 2: Cell Type Normalization
# ============================================================
def test_task2_cell_normalization() -> dict:
    """Normalize free-text cell type names to Cell Ontology."""
    print(f"\n{'='*60}")
    print("TASK 2: Cell Type Normalization")
    print(f"{'='*60}")

    test_cases = [
        ("activated T helper cells",       "CL:0000492"),  # CD4+ alpha-beta T cell
        ("monocyte-derived macrophages",    "CL:0000235"),  # macrophage
        ("Treg cells",                      "CL:0000815"),  # regulatory T cell
        ("cancer-associated fibroblasts",   "CL:0000057"),  # fibroblast
        ("alveolar type II cells",          "CL:0002063"),  # type II pneumocyte
    ]

    system = (
        "You are a cell biology ontology expert. Map the given cell type name to "
        "the most specific Cell Ontology (CL) term.\n"
        "Return ONLY valid JSON: {\"input\": \"\", \"cl_id\": \"CL:XXXXXXX\", "
        "\"cl_name\": \"\", \"confidence\": 0.0-1.0, \"reasoning\": \"\"}"
    )

    results = []
    for cell_name, expected_cl in test_cases:
        result = ollama_chat(f"Cell type: {cell_name}", system=system,
                             temperature=0.0, max_tokens=256)
        raw = result["response"]
        parsed = {}
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
        except json.JSONDecodeError:
            pass

        got_cl = parsed.get("cl_id", "???")
        cl_name = parsed.get("cl_name", "???")
        conf = parsed.get("confidence", 0)
        match = got_cl == expected_cl
        print(f"  '{cell_name}' → {got_cl} ({cl_name}) conf={conf} | expected={expected_cl} | {'PASS' if match else 'MISS'}")
        results.append({"input": cell_name, "expected": expected_cl, "got": got_cl,
                         "cl_name": cl_name, "match": match, "elapsed_s": result["elapsed_s"]})

    exact_acc = sum(1 for r in results if r["match"]) / len(results)
    print(f"\n  Exact match accuracy: {exact_acc:.0%} ({sum(1 for r in results if r['match'])}/{len(results)})")
    return {"task": "cell_normalization", "accuracy": exact_acc, "details": results}


# ============================================================
# Task 3: Bio-QA (experimental design)
# ============================================================
def test_task3_bioqa(gse: dict) -> dict:
    """Ask questions about experimental design and verify answers."""
    print(f"\n{'='*60}")
    print(f"TASK 3: Bio-QA — {gse['gse_id']}")
    print(f"{'='*60}")

    context = (
        f"Title: {gse['title']}\n\n"
        f"Summary: {gse.get('summary', '')}\n\n"
        f"Overall design: {gse.get('overall_design', '')}\n\n"
        f"Organism: {gse.get('organism', '')}\n"
        f"Series type: {gse.get('series_type', '')}"
    )

    questions = [
        ("What organism was studied?", gse.get("organism", "")),
        ("What tissue or organ was the focus?", ", ".join(gse.get("tissues", [])[:2]) or "unknown"),
        ("What sequencing technology was used?", gse.get("series_type", "")),
    ]

    system = (
        "You are a single-cell biology expert. Answer the question based ONLY on "
        "the provided dataset description. Be concise (1-2 sentences max)."
    )

    results = []
    for q, expected in questions:
        prompt = f"Dataset:\n{context}\n\nQuestion: {q}"
        result = ollama_chat(prompt, system=system, temperature=0.0, max_tokens=200)
        answer = result["response"].strip()

        # Check if expected answer is mentioned
        if expected:
            found = any(word.lower() in answer.lower() for word in expected.split()[:3] if len(word) > 2)
        else:
            found = True  # No ground truth to compare

        print(f"  Q: {q}")
        print(f"  A: {answer[:120]}")
        print(f"  Expected contains: '{expected[:60]}' → {'FOUND' if found else 'MISSING'}")
        print()
        results.append({"question": q, "answer": answer[:200], "expected": expected,
                         "found": found, "elapsed_s": result["elapsed_s"]})

    return {"task": "bio_qa", "gse": gse["gse_id"], "results": results}


# ============================================================
# Task 4: RAG Retrieval (embedding similarity)
# ============================================================
def test_task4_retrieval() -> dict:
    """Test embedding-based retrieval using BGE-M3."""
    print(f"\n{'='*60}")
    print("TASK 4: RAG Retrieval (Embedding Similarity)")
    print(f"{'='*60}")

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  SKIP: sentence_transformers not available")
        return {"task": "retrieval", "status": "skipped"}

    # Load embedding model
    print("  Loading BGE-M3...")
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-m3")
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Load a sample of ground truth docs
    files = sorted(GT_DIR.glob("GSE*.json"))[:100]  # first 100 for speed
    docs = []
    gse_ids = []
    for f in files:
        d = json.load(open(f))
        text = d.get("document_text", "") or f"{d.get('title','')} {d.get('summary','')}"
        docs.append(text[:512])  # truncate for speed
        gse_ids.append(d["gse_id"])

    # Encode documents
    print(f"  Encoding {len(docs)} documents...")
    t0 = time.time()
    doc_embeddings = model.encode(docs, batch_size=32, show_progress_bar=False)
    encode_time = time.time() - t0
    print(f"  Encoded in {encode_time:.1f}s ({len(docs)/encode_time:.0f} docs/s)")

    # Test queries
    queries = [
        ("single cell RNA-seq lung cancer human", ["lung", "cancer"]),
        ("mouse brain development scRNA-seq", ["mouse", "brain"]),
        ("T cell immune checkpoint tumor", ["T cell", "tumor", "immune"]),
    ]

    results = []
    for query, keywords in queries:
        q_emb = model.encode([query])
        scores = np.dot(doc_embeddings, q_emb.T).flatten()
        top_k = np.argsort(scores)[::-1][:5]

        print(f"\n  Query: '{query}'")
        hits = []
        for rank, idx in enumerate(top_k):
            gse = gse_ids[idx]
            title = json.load(open(GT_DIR / f"{gse}.json")).get("title", "")[:80]
            score = scores[idx]
            # Check if any keyword appears in the doc
            relevant = any(kw.lower() in docs[idx].lower() for kw in keywords)
            hits.append({"rank": rank+1, "gse": gse, "score": float(score), "relevant": relevant})
            print(f"    #{rank+1} {gse} (score={score:.3f}) {'✓' if relevant else '✗'} {title[:60]}")

        # Precision@5
        p5 = sum(1 for h in hits if h["relevant"]) / len(hits)
        results.append({"query": query, "p_at_5": p5, "hits": hits})

    avg_p5 = sum(r["p_at_5"] for r in results) / len(results)
    print(f"\n  Average P@5: {avg_p5:.2f}")
    return {"task": "retrieval", "avg_p5": avg_p5, "encode_speed": len(docs)/encode_time,
            "details": results}


# ============================================================
# Task 5: Cross-Dataset Reasoning
# ============================================================
def test_task5_cross_dataset(samples: list) -> dict:
    """Judge whether two datasets study related topics."""
    print(f"\n{'='*60}")
    print("TASK 5: Cross-Dataset Reasoning")
    print(f"{'='*60}")

    system = (
        "You are a bioinformatics expert. Given two single-cell dataset descriptions, "
        "assess their relationship.\n"
        "Return ONLY valid JSON:\n"
        '{"related": true/false, "relationship_type": "same_topic|complementary|unrelated", '
        '"shared_aspects": [], "differences": [], "integrable": true/false, "reasoning": ""}'
    )

    # Create pairs: (similar, different)
    pairs = []
    if len(samples) >= 2:
        pairs.append((samples[0], samples[1], "test_pair_1"))
    if len(samples) >= 3:
        pairs.append((samples[0], samples[2], "test_pair_2"))

    results = []
    for gse_a, gse_b, label in pairs:
        prompt = (
            f"Dataset A ({gse_a['gse_id']}):\n"
            f"  Title: {gse_a['title']}\n"
            f"  Summary: {gse_a.get('summary', '')[:300]}\n"
            f"  Organism: {gse_a.get('organism', '')}\n\n"
            f"Dataset B ({gse_b['gse_id']}):\n"
            f"  Title: {gse_b['title']}\n"
            f"  Summary: {gse_b.get('summary', '')[:300]}\n"
            f"  Organism: {gse_b.get('organism', '')}\n"
        )

        result = ollama_chat(prompt, system=system, temperature=0.0, max_tokens=512)
        raw = result["response"]
        parsed = {}
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
        except json.JSONDecodeError:
            pass

        same_org = gse_a.get("organism", "") == gse_b.get("organism", "")
        print(f"  {gse_a['gse_id']} vs {gse_b['gse_id']}")
        print(f"    Same organism: {same_org}")
        print(f"    Model says related: {parsed.get('related', '?')}")
        print(f"    Relationship: {parsed.get('relationship_type', '?')}")
        print(f"    Integrable: {parsed.get('integrable', '?')}")
        print(f"    Shared: {parsed.get('shared_aspects', [])[:3]}")
        print(f"    Reasoning: {parsed.get('reasoning', '')[:120]}")
        print(f"    Time: {result['elapsed_s']}s")
        print()

        results.append({
            "pair": f"{gse_a['gse_id']}_vs_{gse_b['gse_id']}",
            "same_organism": same_org,
            "model_output": parsed,
            "elapsed_s": result["elapsed_s"],
        })

    return {"task": "cross_dataset", "results": results}


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  scMetaIntel Quick Smoke Test — All 5 Benchmark Tasks")
    print(f"  Model: {TEST_MODEL}")
    print(f"  Ground truth: {sum(1 for _ in GT_DIR.glob('GSE*.json'))} GSE files")
    print("=" * 60)

    # Verify Ollama is up
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if TEST_MODEL not in models:
            # Try to find a close match
            for m in models:
                if "8b" in m or "1.5b" in m:
                    print(f"  Note: {TEST_MODEL} not found, using {m}")
                    break
        print(f"  Ollama OK, {len(models)} models available")
    except Exception as e:
        print(f"  ERROR: Cannot connect to Ollama at {OLLAMA_BASE}: {e}")
        sys.exit(1)

    # Load sample data
    samples = load_sample_gse(3)
    print(f"  Loaded {len(samples)} sample GSE: {[s['gse_id'] for s in samples]}")

    all_results = {}
    t_total = time.time()

    # Task 1: IE
    for gse in samples[:2]:
        r = test_task1_ie(gse)
        all_results[f"task1_{r['gse']}"] = r

    # Task 2: Cell Type Normalization
    all_results["task2"] = test_task2_cell_normalization()

    # Task 3: Bio-QA
    for gse in samples[:2]:
        r = test_task3_bioqa(gse)
        all_results[f"task3_{r['gse']}"] = r

    # Task 4: Retrieval
    all_results["task4"] = test_task4_retrieval()

    # Task 5: Cross-Dataset Reasoning
    all_results["task5"] = test_task5_cross_dataset(samples)

    total_time = time.time() - t_total

    # Summary
    print(f"\n{'='*60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Model: {TEST_MODEL}")
    print()

    # Task 1 summary
    ie_results = [v for k, v in all_results.items() if k.startswith("task1_")]
    if ie_results:
        ie_pass = sum(1 for r in ie_results if r.get("org_match"))
        print(f"  Task 1 (IE):           {ie_pass}/{len(ie_results)} organism match")

    # Task 2 summary
    t2 = all_results.get("task2", {})
    if t2:
        print(f"  Task 2 (Cell Norm):    {t2.get('accuracy', 0):.0%} exact match")

    # Task 3 summary
    qa_results = [v for k, v in all_results.items() if k.startswith("task3_")]
    if qa_results:
        total_q = sum(len(r.get("results", [])) for r in qa_results)
        found_q = sum(sum(1 for q in r.get("results", []) if q.get("found")) for r in qa_results)
        print(f"  Task 3 (Bio-QA):       {found_q}/{total_q} answers contain expected info")

    # Task 4 summary
    t4 = all_results.get("task4", {})
    if t4 and t4.get("avg_p5") is not None:
        print(f"  Task 4 (Retrieval):    P@5={t4['avg_p5']:.2f}, {t4.get('encode_speed', 0):.0f} docs/s")

    # Task 5 summary
    t5 = all_results.get("task5", {})
    if t5:
        n_pairs = len(t5.get("results", []))
        valid = sum(1 for r in t5.get("results", []) if r.get("model_output", {}).get("relationship_type"))
        print(f"  Task 5 (Cross-Dataset): {valid}/{n_pairs} pairs analyzed")

    print()

    # Save results
    out_path = Path(__file__).parent / "results" / "smoke_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
