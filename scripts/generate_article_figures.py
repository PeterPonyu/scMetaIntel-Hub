#!/usr/bin/env python3
"""Generate article figures and tables from benchmark results.

Produces:
  - Table 1: Corpus statistics
  - Table 2: Query category distribution
  - Table 3: Full LLM comparison matrix
  - Table 4: Retrieval strategy comparison
  - Table 5: Top-5 model comparison
  - Figure 2: Radar chart — general vs biomedical embedders
  - Figure 3: Composite score vs model size scatter plot
  - Figure 4: Per-task heatmap (models × tasks)
  - Figure 5: Context k vs quality curve
  - Figure 6: E2E pipeline comparison bar chart

Output: article_figures/ directory with PNG/PDF + CSV tables.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
GT_DIR = ROOT / "benchmarks" / "ground_truth"
QUERIES_PATH = ROOT / "benchmarks" / "eval_queries.json"
OUT_DIR = ROOT / "article_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Color palette ──
FAMILY_COLORS = {
    "qwen": "#2196F3",
    "gemma": "#FF9800",
    "mistral": "#9C27B0",
    "phi": "#4CAF50",
    "llama": "#F44336",
    "command-r": "#795548",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Table 1: Corpus statistics
# ═══════════════════════════════════════════════════════════════════════════
def table_1_corpus_stats():
    """Compute corpus statistics from ground truth files."""
    files = sorted(GT_DIR.glob("GSE*.json"))
    n = len(files)
    organisms = Counter()
    has_tissue = has_disease = has_celltype = has_pubmed = has_doctext = 0

    for f in files:
        d = load_json(f)
        org = d.get("organism", "")
        if org:
            organisms[org] += 1
        if d.get("tissues"):
            has_tissue += 1
        if d.get("diseases"):
            has_disease += 1
        if d.get("cell_types"):
            has_celltype += 1
        if d.get("pubmed"):
            has_pubmed += 1
        if d.get("document_text"):
            has_doctext += 1

    rows = [
        ("Total studies", n),
        ("Unique organisms", len(organisms)),
        ("Tissue annotations", f"{has_tissue} ({100*has_tissue/n:.1f}%)"),
        ("Disease annotations", f"{has_disease} ({100*has_disease/n:.1f}%)"),
        ("Cell type annotations", f"{has_celltype} ({100*has_celltype/n:.1f}%)"),
        ("PubMed linked", f"{has_pubmed} ({100*has_pubmed/n:.1f}%)"),
        ("Document text", f"{has_doctext} ({100*has_doctext/n:.1f}%)"),
    ]

    # Top organisms
    top_orgs = organisms.most_common(10)

    with open(OUT_DIR / "table1_corpus_stats.csv", "w") as f:
        f.write("Metric,Value\n")
        for label, val in rows:
            f.write(f"{label},{val}\n")
        f.write("\nOrganism,Count,Percentage\n")
        for org, cnt in top_orgs:
            f.write(f"{org},{cnt},{100*cnt/n:.1f}%\n")

    print(f"Table 1: {n} studies, {len(organisms)} organisms")
    for label, val in rows:
        print(f"  {label}: {val}")
    print("  Top organisms:", [(o, c) for o, c in top_orgs[:5]])
    return rows, top_orgs


# ═══════════════════════════════════════════════════════════════════════════
# Table 2: Query category distribution
# ═══════════════════════════════════════════════════════════════════════════
def table_2_query_distribution():
    """Query distribution table."""
    queries = load_json(QUERIES_PATH)
    categories = Counter()
    difficulties = Counter()
    for q in queries:
        categories[q["category"]] += 1
        difficulties[q["difficulty"]] += 1

    expected_gses = set()
    for q in queries:
        expected_gses.update(q.get("expected_gse", []))

    with open(OUT_DIR / "table2_query_distribution.csv", "w") as f:
        f.write("Category,Count\n")
        for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
            f.write(f"{cat},{cnt}\n")
        f.write(f"\nDifficulty,Count\n")
        for diff, cnt in sorted(difficulties.items()):
            f.write(f"{diff},{cnt}\n")
        f.write(f"\nTotal queries,{len(queries)}\n")
        f.write(f"Unique expected GSEs,{len(expected_gses)}\n")

    print(f"Table 2: {len(queries)} queries, {len(expected_gses)} unique expected GSEs")
    print(f"  Categories: {dict(categories)}")
    print(f"  Difficulties: {dict(difficulties)}")
    return categories, difficulties


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Radar chart — embedding models
# ═══════════════════════════════════════════════════════════════════════════
def figure_2_embedding_radar():
    """Radar chart comparing general vs biomedical embedding models."""
    data = load_json(RESULTS_DIR / "embedding_bench.json")

    # Categorize models
    bio_models = {"biolord-2023", "BiomedBERT-base", "BiomedBERT-large",
                  "medcpt-article", "medcpt-query", "specter2"}
    general_models = set(data.keys()) - bio_models

    # Metrics to plot (normalized 0-1 where higher is better)
    metrics = [
        ("R@50", lambda d: d.get("retrieval", {}).get("average", {}).get("r_at_50", 0)),
        ("MRR", lambda d: d.get("retrieval", {}).get("average", {}).get("mrr", 0)),
        ("nDCG@10", lambda d: d.get("retrieval", {}).get("average", {}).get("ndcg_at_10", 0)),
        ("Onto R@1", lambda d: d.get("ontology", {}).get("recall_at_1", 0)),
        ("Onto R@5", lambda d: d.get("ontology", {}).get("recall_at_5", 0)),
        ("Speed (norm)", lambda d: min(d.get("speed", {}).get("tokens_per_sec", 0) / 5000, 1.0)),
    ]

    metric_names = [m[0] for m in metrics]

    def avg_scores(model_set):
        scores = []
        for m in model_set:
            if m not in data:
                continue
            scores.append([fn(data[m]) for _, fn in metrics])
        return np.mean(scores, axis=0) if scores else np.zeros(len(metrics))

    gen_scores = avg_scores(general_models)
    bio_scores = avg_scores(bio_models)

    # Also plot top model
    top_model = max(data.keys(), key=lambda m: data[m].get("retrieval", {}).get("average", {}).get("r_at_50", 0))
    top_scores = [fn(data[top_model]) for _, fn in metrics]

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for scores, label, color, ls in [
        (gen_scores, "General (avg)", "#2196F3", "-"),
        (bio_scores, "Biomedical (avg)", "#FF9800", "--"),
        (top_scores, f"Best: {top_model}", "#F44336", "-"),
    ]:
        vals = scores.tolist() + [scores[0]] if hasattr(scores, 'tolist') else list(scores) + [scores[0]]
        ax.plot(angles, vals, linewidth=2, linestyle=ls, label=label, color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Embedding Model Comparison", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_embedding_radar.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_embedding_radar.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2: Radar chart saved ({top_model} is best)")


# ═══════════════════════════════════════════════════════════════════════════
# Table 4: Retrieval strategy comparison
# ═══════════════════════════════════════════════════════════════════════════
def table_4_retrieval():
    """Retrieval strategy comparison table."""
    data = load_json(RESULTS_DIR / "retrieval_bench.json")
    strategies = ["dense", "sparse", "hybrid", "hybrid+filter",
                  "hybrid+rerank", "hybrid+filter+rerank"]

    header = "Strategy,P@5,P@10,R@50,MRR,nDCG@10,Latency(ms)"
    rows = []
    for s in strategies:
        if s not in data:
            continue
        avg = data[s].get("average", {})
        rows.append(
            f"{s},{avg.get('p_at_5', 0):.4f},{avg.get('p_at_10', 0):.4f},"
            f"{avg.get('r_at_50', 0):.4f},{avg.get('mrr', 0):.4f},"
            f"{avg.get('ndcg_at_10', 0):.4f},{avg.get('avg_latency_ms', avg.get('latency_ms', 0)):.1f}"
        )

    with open(OUT_DIR / "table4_retrieval.csv", "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")

    # Also write by-difficulty breakdown
    with open(OUT_DIR / "table4_retrieval_by_difficulty.csv", "w") as f:
        f.write("Strategy,Difficulty,P@5,P@10,R@50,MRR,nDCG@10\n")
        for s in strategies:
            if s not in data:
                continue
            by_diff = data[s].get("by_difficulty", {})
            for diff in ["easy", "medium", "hard"]:
                if diff not in by_diff:
                    continue
                d = by_diff[diff]
                f.write(
                    f"{s},{diff},{d.get('p_at_5', 0):.4f},{d.get('p_at_10', 0):.4f},"
                    f"{d.get('r_at_50', 0):.4f},{d.get('mrr', 0):.4f},{d.get('ndcg_at_10', 0):.4f}\n"
                )

    print(f"Table 4: {len(rows)} strategies")
    for r in rows:
        print(f"  {r}")


# ═══════════════════════════════════════════════════════════════════════════
# Table 3 + Figure 3 + Figure 4: LLM benchmark analysis
# ═══════════════════════════════════════════════════════════════════════════
def compute_composite(d: dict) -> float:
    """Compute composite score from LLM bench result for one model config."""
    ta = d.get("task_a_parsing", {})
    tb = d.get("task_b_extraction", {})
    tc = d.get("task_c_ontology", {})
    td = d.get("task_d_answer", {})
    te = d.get("task_e_speed", {})

    # Task A: parsing EM (weight 0.20)
    parse_score = ta.get("exact_match", 0)

    # Task B: extraction avg F1 across fields (weight 0.20)
    eb = tb.get("average", {})
    f1s = []
    for field in ["tissues", "diseases", "cell_types"]:
        fdata = eb.get(field, {})
        f1s.append(fdata.get("f1", 0))
    extract_score = np.mean(f1s) if f1s else 0

    # Task C: ontology F1 (weight 0.20)
    onto_score = tc.get("f1", 0)

    # Task D: answer quality — avg of citation_recall and grounding (weight 0.30)
    answer_score = (td.get("citation_recall", 0) + td.get("grounding_rate", 0)) / 2

    # Task E: speed normalized (weight 0.10) — 100 tok/s = 1.0
    speed_score = min(te.get("tokens_per_sec", 0) / 100, 1.0)

    composite = (
        0.20 * parse_score
        + 0.20 * extract_score
        + 0.20 * onto_score
        + 0.30 * answer_score
        + 0.10 * speed_score
    )
    return composite


def table_3_llm_comparison():
    """Full LLM comparison matrix."""
    path = RESULTS_DIR / "llm_bench.json"
    if not path.exists():
        print("Table 3: SKIPPED — llm_bench.json not found")
        return None
    data = load_json(path)

    rows = []
    for label, d in sorted(data.items()):
        if "error" in str(d):
            continue
        ta = d.get("task_a_parsing", {})
        tb = d.get("task_b_extraction", {})
        tc = d.get("task_c_ontology", {})
        td = d.get("task_d_answer", {})
        te = d.get("task_e_speed", {})

        # Avg extraction F1
        eb = tb.get("average", {})
        f1s = [eb.get(f, {}).get("f1", 0) for f in ["tissues", "diseases", "cell_types"]]
        avg_f1 = np.mean(f1s) if f1s else 0

        composite = compute_composite(d)

        rows.append({
            "model": label,
            "family": d.get("family", "?"),
            "size_b": d.get("size_b", "?"),
            "quant": d.get("quant", "?"),
            "think": d.get("think_enabled", False),
            "parse_em": ta.get("exact_match", 0),
            "extract_f1": avg_f1,
            "onto_f1": tc.get("f1", 0),
            "cite_recall": td.get("citation_recall", 0),
            "grounding": td.get("grounding_rate", 0),
            "tok_s": te.get("tokens_per_sec", 0),
            "composite": composite,
        })

    rows.sort(key=lambda r: -r["composite"])

    with open(OUT_DIR / "table3_llm_comparison.csv", "w") as f:
        f.write("Model,Family,Size(B),Quant,Think,Parse_EM,Extract_F1,Onto_F1,"
                "Cite_Recall,Grounding,Tok/s,Composite\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['family']},{r['size_b']},{r['quant']},"
                f"{r['think']},{r['parse_em']:.4f},{r['extract_f1']:.4f},"
                f"{r['onto_f1']:.4f},{r['cite_recall']:.4f},{r['grounding']:.4f},"
                f"{r['tok_s']:.1f},{r['composite']:.4f}\n"
            )

    print(f"Table 3: {len(rows)} model configurations")
    for r in rows[:5]:
        print(f"  {r['model']}: composite={r['composite']:.3f}")

    return rows


def figure_3_composite_vs_size(rows):
    """Scatter plot: composite score vs model size."""
    if rows is None:
        print("Figure 3: SKIPPED — no LLM results")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for r in rows:
        size = r["size_b"]
        if isinstance(size, str):
            continue
        color = FAMILY_COLORS.get(r["family"], "#607D8B")
        marker = "^" if r["think"] else "o"
        ax.scatter(size, r["composite"], c=color, marker=marker, s=100,
                   edgecolors="black", linewidths=0.5, zorder=5)
        # Label — offset to avoid overlap
        offset = (5, 5) if r["composite"] > 0.5 else (5, -10)
        label = r["model"]
        if len(label) > 20:
            label = label[:18] + ".."
        ax.annotate(label, (size, r["composite"]), fontsize=7,
                    xytext=offset, textcoords="offset points")

    # Legend for families
    for family, color in FAMILY_COLORS.items():
        ax.scatter([], [], c=color, marker="o", s=60, label=family,
                   edgecolors="black", linewidths=0.5)
    ax.scatter([], [], c="gray", marker="^", s=60, label="think=ON",
               edgecolors="black", linewidths=0.5)
    ax.scatter([], [], c="gray", marker="o", s=60, label="think=OFF",
               edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Model Size (billion parameters)", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("LLM Composite Score vs. Model Size", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_composite_vs_size.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_composite_vs_size.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 3: Composite vs size scatter saved")


def figure_4_task_heatmap(rows):
    """Per-task heatmap: models × tasks."""
    if rows is None:
        print("Figure 4: SKIPPED — no LLM results")
        return

    # Sort by composite
    sorted_rows = sorted(rows, key=lambda r: -r["composite"])

    task_keys = ["parse_em", "extract_f1", "onto_f1", "cite_recall", "grounding"]
    task_labels = ["Parse EM", "Extract F1", "Ontology F1", "Cite Recall", "Grounding"]

    model_names = [r["model"] for r in sorted_rows]
    matrix = np.array([[r[k] for k in task_keys] for r in sorted_rows])

    fig, ax = plt.subplots(figsize=(8, max(6, len(model_names) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(task_labels)))
    ax.set_xticklabels(task_labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=8)

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(task_keys)):
            val = matrix[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("LLM Performance Heatmap (by task)", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_task_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_task_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 4: Task heatmap saved")


def table_5_top5(rows):
    """Top-5 model comparison table with per-task breakdown."""
    if rows is None:
        print("Table 5: SKIPPED — no LLM results")
        return
    top5 = sorted(rows, key=lambda r: -r["composite"])[:5]
    with open(OUT_DIR / "table5_top5.csv", "w") as f:
        f.write("Rank,Model,Family,Size(B),Composite,Parse_EM,Extract_F1,"
                "Onto_F1,Cite_Recall,Grounding,Tok/s\n")
        for i, r in enumerate(top5, 1):
            f.write(
                f"{i},{r['model']},{r['family']},{r['size_b']},"
                f"{r['composite']:.4f},{r['parse_em']:.4f},{r['extract_f1']:.4f},"
                f"{r['onto_f1']:.4f},{r['cite_recall']:.4f},{r['grounding']:.4f},"
                f"{r['tok_s']:.1f}\n"
            )
    print("Table 5: Top 5 models")
    for i, r in enumerate(top5, 1):
        print(f"  #{i}: {r['model']} (composite={r['composite']:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Context k vs quality curve
# ═══════════════════════════════════════════════════════════════════════════
def figure_5_context_curve():
    """Context k vs quality curve from context_bench.json."""
    path = RESULTS_DIR / "context_bench.json"
    if not path.exists():
        print("Figure 5: SKIPPED — context_bench.json not found")
        return

    data = load_json(path)
    summary = data.get("summary", {})

    # Parse k=N_format configs
    configs = {}
    for key, vals in summary.items():
        parts = key.split("_", 1)
        if len(parts) != 2 or not parts[0].startswith("k="):
            continue
        k = int(parts[0][2:])
        fmt = parts[1]
        if fmt not in configs:
            configs[fmt] = {"k": [], "recall": [], "precision": [], "tokens": []}
        configs[fmt]["k"].append(k)
        configs[fmt]["recall"].append(vals.get("avg_citation_recall", 0))
        configs[fmt]["precision"].append(vals.get("avg_citation_precision", 0))
        configs[fmt]["tokens"].append(vals.get("avg_context_tokens_approx", 0))

    if not configs:
        print("Figure 5: No valid context configs found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"full": "#2196F3", "structured": "#FF9800", "minimal": "#4CAF50"}
    for fmt, d in configs.items():
        # Sort by k
        order = np.argsort(d["k"])
        ks = [d["k"][i] for i in order]
        recalls = [d["recall"][i] for i in order]
        precisions = [d["precision"][i] for i in order]
        tokens = [d["tokens"][i] for i in order]

        c = colors.get(fmt, "gray")
        ax1.plot(ks, recalls, "o-", color=c, label=f"{fmt} (recall)", linewidth=2)
        ax1.plot(ks, precisions, "s--", color=c, label=f"{fmt} (precision)",
                 linewidth=1.5, alpha=0.7)
        ax2.plot(ks, tokens, "o-", color=c, label=fmt, linewidth=2)

    ax1.set_xlabel("k (number of retrieved documents)", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Citation Quality vs. k", fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.set_xlabel("k (number of retrieved documents)", fontsize=11)
    ax2.set_ylabel("Avg. Context Tokens", fontsize=11)
    ax2.set_title("Context Size vs. k", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Context Window Optimization", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_context_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_context_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 5: Context k vs quality curve saved")


# ═══════════════════════════════════════════════════════════════════════════
# Table S2: Context management strategies comparison
# ═══════════════════════════════════════════════════════════════════════════
def table_s2_context_management():
    """Context management strategies comparison."""
    path = RESULTS_DIR / "context_management_bench.json"
    if not path.exists():
        print("Table S2: SKIPPED — context_management_bench.json not found")
        return

    data = load_json(path)
    summary = data.get("summary", {})

    with open(OUT_DIR / "table_s2_context_management.csv", "w") as f:
        f.write("Strategy,Grounding,Cite_Precision,Cite_Recall,Avg_Tokens,Avg_Duration_ms\n")
        for name, vals in sorted(summary.items()):
            f.write(
                f"{name},"
                f"{vals.get('avg_grounding_rate', 0):.4f},"
                f"{vals.get('avg_citation_precision', 0):.4f},"
                f"{vals.get('avg_citation_recall', 0):.4f},"
                f"{vals.get('avg_context_tokens_approx', 0):.0f},"
                f"{vals.get('avg_duration_ms', 0):.0f}\n"
            )

    print(f"Table S2: {len(summary)} context management strategies")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: E2E pipeline comparison
# ═══════════════════════════════════════════════════════════════════════════
def figure_6_e2e_comparison():
    """E2E pipeline comparison bar chart."""
    path = RESULTS_DIR / "e2e_report.json"
    if not path.exists():
        print("Figure 6: SKIPPED — e2e_report.json not found")
        return

    data = load_json(path)
    configs = []
    for name, d in data.items():
        if not isinstance(d, dict):
            continue
        # Support both "metrics" and "average" keys
        m = d.get("metrics") or d.get("average", {})
        if not m:
            continue
        configs.append({
            "name": name,
            "r_at_50": m.get("r_at_50", 0),
            "mrr": m.get("mrr", 0),
            "cite_recall": m.get("citation_recall", 0),
            "grounding": m.get("grounding_rate", 0),
            "latency_s": m.get("latency_sec", m.get("avg_latency_ms", 0) / 1000),
        })

    if not configs:
        print("Figure 6: No valid E2E configs found")
        return

    names = [c["name"] for c in configs]
    x = np.arange(len(names))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - width, [c["r_at_50"] for c in configs], width, label="R@50", color="#2196F3")
    ax1.bar(x, [c["mrr"] for c in configs], width, label="MRR", color="#FF9800")
    ax1.bar(x + width, [c["cite_recall"] for c in configs], width, label="Cite Recall", color="#4CAF50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Quality Metrics", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, [c["latency_s"] for c in configs], 0.5, color="#F44336")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Latency (seconds)", fontsize=11)
    ax2.set_title("End-to-End Latency", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("End-to-End Pipeline Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_e2e_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_e2e_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 6: E2E comparison saved")


# ═══════════════════════════════════════════════════════════════════════════
# Embedding comparison table (supplementary)
# ═══════════════════════════════════════════════════════════════════════════
def table_s1_embedding_comparison():
    """Full embedding comparison table."""
    data = load_json(RESULTS_DIR / "embedding_bench.json")

    rows = []
    for model, d in data.items():
        ret = d.get("retrieval", {}).get("average", {})
        onto = d.get("ontology", {})
        spd = d.get("speed", {})
        rows.append({
            "model": model,
            "r_at_50": ret.get("r_at_50", 0),
            "mrr": ret.get("mrr", 0),
            "ndcg_at_10": ret.get("ndcg_at_10", 0),
            "onto_r1": onto.get("recall_at_1", 0),
            "onto_r5": onto.get("recall_at_5", 0),
            "tok_s": spd.get("tokens_per_sec", 0),
        })

    rows.sort(key=lambda r: -r["r_at_50"])

    with open(OUT_DIR / "table_s1_embedding_full.csv", "w") as f:
        f.write("Model,R@50,MRR,nDCG@10,Onto_R@1,Onto_R@5,Tok/s\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['r_at_50']:.4f},{r['mrr']:.4f},"
                f"{r['ndcg_at_10']:.4f},{r['onto_r1']:.4f},{r['onto_r5']:.4f},"
                f"{r['tok_s']:.1f}\n"
            )

    print(f"Table S1: {len(rows)} embedding models")
    for r in rows[:5]:
        print(f"  {r['model']}: R@50={r['r_at_50']:.3f}, MRR={r['mrr']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary JSON (machine-readable article data)
# ═══════════════════════════════════════════════════════════════════════════
def write_summary_json(corpus_stats, query_dist, llm_rows):
    """Write a summary JSON with key findings for easy reference."""
    summary = {
        "corpus": {
            "n_studies": corpus_stats[0][0][1] if corpus_stats else 0,
        },
        "queries": {
            "n_queries": sum(query_dist[0].values()) if query_dist else 0,
        },
        "embedding": {
            "best_model": None,
            "best_r_at_50": 0,
        },
        "retrieval": {
            "best_strategy": None,
            "best_r_at_50": 0,
        },
        "llm": {
            "n_configs": len(llm_rows) if llm_rows else 0,
            "top5": [],
        },
    }

    # Best embedding
    if (RESULTS_DIR / "embedding_bench.json").exists():
        edata = load_json(RESULTS_DIR / "embedding_bench.json")
        best = max(edata.items(), key=lambda x: x[1].get("retrieval", {}).get("average", {}).get("r_at_50", 0))
        summary["embedding"]["best_model"] = best[0]
        summary["embedding"]["best_r_at_50"] = best[1]["retrieval"]["average"]["r_at_50"]

    # Best retrieval
    if (RESULTS_DIR / "retrieval_bench.json").exists():
        rdata = load_json(RESULTS_DIR / "retrieval_bench.json")
        best_s = max(
            [(k, v) for k, v in rdata.items() if isinstance(v, dict) and "average" in v],
            key=lambda x: x[1]["average"].get("r_at_50", 0)
        )
        summary["retrieval"]["best_strategy"] = best_s[0]
        summary["retrieval"]["best_r_at_50"] = best_s[1]["average"]["r_at_50"]

    # Top 5 LLMs
    if llm_rows:
        top5 = sorted(llm_rows, key=lambda r: -r["composite"])[:5]
        summary["llm"]["top5"] = [
            {"model": r["model"], "composite": round(r["composite"], 4)}
            for r in top5
        ]

    with open(OUT_DIR / "article_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating article figures and tables")
    print("=" * 60)

    # Tables 1-2 (always available)
    corpus_stats = table_1_corpus_stats()
    print()
    query_dist = table_2_query_distribution()
    print()

    # Figure 2 + Table S1 (embedding)
    if (RESULTS_DIR / "embedding_bench.json").exists():
        figure_2_embedding_radar()
        table_s1_embedding_comparison()
        print()

    # Table 4 (retrieval)
    if (RESULTS_DIR / "retrieval_bench.json").exists():
        table_4_retrieval()
        print()

    # Tables 3, 5 + Figures 3, 4 (LLM)
    llm_rows = table_3_llm_comparison()
    if llm_rows:
        print()
        figure_3_composite_vs_size(llm_rows)
        figure_4_task_heatmap(llm_rows)
        table_5_top5(llm_rows)
    print()

    # Figure 5 (context) + Table S2 (context management)
    figure_5_context_curve()
    table_s2_context_management()
    print()

    # Figure 6 (E2E)
    figure_6_e2e_comparison()
    print()

    # Summary
    write_summary_json(corpus_stats, query_dist, llm_rows)

    print()
    print("=" * 60)
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
