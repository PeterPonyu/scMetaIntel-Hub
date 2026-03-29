#!/usr/bin/env python3
"""Generate article figures and tables from benchmark results.

Figure-task correspondence:
  Fig 1  - Study design diagram: tasks, methods, metrics, data
  Fig 2  - Embedding model comparison (radar + per-model bars)
  Fig 3  - Retrieval strategy comparison (6 strategies x metrics)
  Fig 4  - LLM per-task heatmap (models x sub-task metrics)
  Fig 5  - Context window optimisation (k x format curves)
  Fig 6  - End-to-end pipeline comparison (quality + latency)
  Fig 7  - Composite summary: LLM composite scores with component breakdown

Tables (CSV):
  Table 1  - Corpus statistics
  Table 2  - Query category distribution
  Table 3  - Full LLM comparison matrix
  Table 4  - Retrieval strategy comparison
  Table 5  - Top-5 model comparison
  Table S1 - Full embedding comparison
  Table S2 - Context management strategies

All figures use fig.add_axes() with manually calculated canvas positions.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
GT_DIR = ROOT / "benchmarks" / "ground_truth"
QUERIES_PATH = ROOT / "benchmarks" / "eval_queries.json"
OUT_DIR = ROOT / "article_figures"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from scmetaintel.config import LLM_MODELS

# ── Exclude disabled models globally ──
EXCLUDED_MODELS = {
    k for k, v in LLM_MODELS.items()
    if v.get("cpu_spill", False) or not v.get("enabled", True)
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_excluded(label: str) -> bool:
    """Check if a model label (possibly with +think suffix) is excluded."""
    base = label.removesuffix("+think")
    return base in EXCLUDED_MODELS


# ── Color palette ──
FAMILY_COLORS = {
    "qwen": "#2196F3",
    "gemma": "#FF9800",
    "mistral": "#9C27B0",
    "phi": "#4CAF50",
    "llama": "#F44336",
}


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

    top_orgs = organisms.most_common(10)

    with open(OUT_DIR / "table1_corpus_stats.csv", "w") as f:
        f.write("Metric,Value\n")
        for label, val in rows:
            f.write(f"{label},{val}\n")
        f.write("\nOrganism,Count,Percentage\n")
        for org, cnt in top_orgs:
            f.write(f"{org},{cnt},{100*cnt/n:.1f}%\n")

    print(f"Table 1: {n} studies, {len(organisms)} organisms")
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
    return categories, difficulties


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Study design diagram
# ═══════════════════════════════════════════════════════════════════════════
def figure_1_study_design():
    """Pipeline architecture diagram: tasks, benchmarks, methods, metrics, data."""
    fig_w, fig_h = 14, 9
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Title
    ax_title = fig.add_axes([0.0, 0.94, 1.0, 0.05])
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, "Figure 1: scMetaIntel-Hub Benchmarking Study Design",
                  ha="center", va="center", fontsize=20)

    # Main canvas
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.89])
    ax.set_xlim(0, 7.0); ax.set_ylim(0, 6.5)
    ax.axis("off")

    tasks = [
        {
            "label": "Task 1: Embedding Evaluation",
            "data": "2189 GSE metadata docs\n90 eval queries\n700 synonym pairs",
            "methods": "14 models\n(4 general, 10 biomedical)",
            "metrics": "R@50, MRR, nDCG@10\nOnto-R@1, Onto-MRR, Speed",
            "figure": "Fig 2",
            "color": "#E3F2FD",
        },
        {
            "label": "Task 2: Retrieval Strategy",
            "data": "Best embedding model\n(mxbai-embed-large)\n90 queries",
            "methods": "6 strategies:\ndense, sparse, hybrid\n+filter, +rerank",
            "metrics": "P@5, P@10, R@50\nMRR, nDCG@10, Latency",
            "figure": "Fig 3",
            "color": "#FFF3E0",
        },
        {
            "label": "Task 3: LLM Evaluation",
            "data": "5 sub-tasks (A\u2013E)\n13 models\nacross 5 families",
            "methods": "200 docs extraction\n200 docs ontology\n90 queries answer",
            "metrics": "Parse-EM, Extract-F1\nOnto-F1, Cite-Recall\nGrounding, Tok/s",
            "figure": "Fig 4",
            "color": "#F3E5F5",
        },
        {
            "label": "Task 4: Context Window",
            "data": "k = {3,5,10,15,20}\n3 formats: full,\nstructured, minimal",
            "methods": "15 k\u00d7format combos\n+ 3 system-prompt\nvariants",
            "metrics": "Cite-Recall, Cite-Prec\nGrounding, Context-Tok\nDuration",
            "figure": "Fig 5",
            "color": "#E8F5E9",
        },
        {
            "label": "Task 5: End-to-End Pipeline",
            "data": "4 pipeline configs\nbaseline vs optimised\n18 queries",
            "methods": "baseline, optimised_fast\noptimised_quality\nbalanced",
            "metrics": "R@50, MRR, Cite-Recall\nGrounding, Halluc-Rate\nLatency",
            "figure": "Fig 6",
            "color": "#FFEBEE",
        },
    ]

    col_x = [0.65, 2.0, 3.3, 4.6, 5.95]
    col_labels = ["Task", "Data", "Methods", "Metrics", "Figure"]
    for cx, cl in zip(col_x, col_labels):
        ax.text(cx, 6.2, cl, ha="center", va="center", fontsize=15,
                color="#333333")

    ax.plot([0.0, 7.0], [6.0, 6.0], color="#999999", linewidth=0.8)

    row_h = 1.05
    for i, t in enumerate(tasks):
        y_center = 6.0 - (i + 1) * row_h + row_h / 2

        # Task label box
        rect = mpatches.FancyBboxPatch(
            (0.02, y_center - 0.40), 1.30, 0.80,
            boxstyle="round,pad=0.05", facecolor=t["color"],
            edgecolor="#666666", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(0.67, y_center, t["label"], ha="center", va="center",
                fontsize=12, color="#333333")

        # Data / Methods / Metrics columns
        ax.text(2.0, y_center, t["data"], ha="center", va="center",
                fontsize=11, color="#333333", linespacing=1.3)
        ax.text(3.3, y_center, t["methods"], ha="center", va="center",
                fontsize=11, color="#333333", linespacing=1.3)
        ax.text(4.6, y_center, t["metrics"], ha="center", va="center",
                fontsize=11, color="#333333", linespacing=1.3)

        # Figure reference box
        fig_rect = mpatches.FancyBboxPatch(
            (5.6, y_center - 0.22), 0.7, 0.44,
            boxstyle="round,pad=0.05", facecolor="#E0E0E0",
            edgecolor="#999999", linewidth=0.8)
        ax.add_patch(fig_rect)
        ax.text(5.95, y_center, t["figure"], ha="center", va="center",
                fontsize=13, color="#333333")

        # Arrow
        ax.annotate("", xy=(5.6, y_center), xytext=(5.2, y_center),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2))

        if i < len(tasks) - 1:
            sep_y = y_center - row_h / 2
            ax.plot([0.0, 7.0], [sep_y, sep_y], color="#DDDDDD",
                    linewidth=0.5, linestyle="--")

    # Composite summary box
    comp_y = 6.0 - len(tasks) * row_h - 0.30
    comp_rect = mpatches.FancyBboxPatch(
        (0.3, comp_y - 0.28), 6.4, 0.56,
        boxstyle="round,pad=0.08", facecolor="#FFF9C4",
        edgecolor="#FBC02D", linewidth=1.5)
    ax.add_patch(comp_rect)
    ax.text(3.50, comp_y, "Fig 7: Composite Summary "
            "(Parse 20% + Extract 20% + Onto 20% + Answer 30% + Speed 10%)",
            ha="center", va="center", fontsize=12, color="#333333")

    fig.savefig(OUT_DIR / "fig1_study_design.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_study_design.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 1: Study design diagram saved")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Embedding benchmark (Bench 02)
# ═══════════════════════════════════════════════════════════════════════════
def figure_2_embedding_radar():
    """Left: category radar (6 retrieval+speed metrics).  Right: per-model bars."""
    data = load_json(RESULTS_DIR / "embedding_bench.json")

    bio_names = {"biolord-2023", "biomedbert-base", "biomedbert-large",
                 "medcpt-article", "medcpt-query", "specter2",
                 "sapbert", "pubmedbert-embed", "pubmedncl",
                 "s-pubmedbert-marco"}
    general_names = set(data.keys()) - bio_names

    # Compute max speed for normalisation
    max_speed = max(d.get("speed", {}).get("tokens_per_sec", 1) for d in data.values())

    metric_defs = [
        ("R@50",      lambda d: d.get("retrieval", {}).get("average", {}).get("r_at_50", 0)),
        ("MRR",       lambda d: d.get("retrieval", {}).get("average", {}).get("mrr", 0)),
        ("nDCG@10",   lambda d: d.get("retrieval", {}).get("average", {}).get("ndcg_at_10", 0)),
        ("Onto-R@1",  lambda d: d.get("ontology", {}).get("recall_at_1", 0)),
        ("Onto-MRR",  lambda d: d.get("ontology", {}).get("mrr", 0)),
        ("Speed",     lambda d: d.get("speed", {}).get("tokens_per_sec", 0) / max_speed),
    ]
    metric_names = [m[0] for m in metric_defs]

    def avg_scores(model_set):
        scores = []
        for m in model_set:
            if m not in data:
                continue
            scores.append([fn(data[m]) for _, fn in metric_defs])
        return np.mean(scores, axis=0) if scores else np.zeros(len(metric_defs))

    gen_scores = avg_scores(general_names)
    bio_scores = avg_scores(bio_names)
    top_model = max(data.keys(),
                    key=lambda m: data[m].get("retrieval", {}).get("average", {}).get("r_at_50", 0))
    top_scores = np.array([fn(data[top_model]) for _, fn in metric_defs])

    fig_w, fig_h = 16, 10
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Title
    ax_title = fig.add_axes([0.0, 0.94, 1.0, 0.05])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  "Figure 2: Embedding Model Evaluation (14 models, 90 queries)",
                  ha="center", va="center", fontsize=20)

    # ── Left panel: radar ──
    ax_radar = fig.add_axes([0.01, 0.14, 0.42, 0.76], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    for scores, label, color, ls in [
        (gen_scores, f"General avg (n={len(general_names & set(data.keys()))})", "#2196F3", "-"),
        (bio_scores, f"Biomedical avg (n={len(bio_names & set(data.keys()))})", "#FF9800", "--"),
        (top_scores, f"Best: {top_model}", "#F44336", "-"),
    ]:
        vals = list(scores) + [scores[0]]
        ax_radar.plot(angles, vals, linewidth=2.5, linestyle=ls, label=label, color=color)
        ax_radar.fill(angles, vals, alpha=0.10, color=color)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_names, fontsize=14)
    ax_radar.set_ylim(0, 1)
    ax_radar.tick_params(axis="y", labelsize=11)
    ax_radar.set_title("Category Comparison", fontsize=16, pad=22)
    ax_radar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                    fontsize=13, ncol=1, frameon=True, framealpha=0.9)

    # ── Right panel: per-model horizontal bar chart ──
    models_sorted = sorted(data.keys(),
                           key=lambda m: data[m].get("retrieval", {}).get("average", {}).get("r_at_50", 0),
                           reverse=True)

    n_models = len(models_sorted)
    ax_bars = fig.add_axes([0.54, 0.08, 0.45, 0.82])

    y_pos = np.arange(n_models)
    n_metrics = len(metric_defs)
    bar_width = 0.8 / n_metrics
    metric_colors = ["#2196F3", "#FF9800", "#9C27B0", "#E91E63", "#795548", "#4CAF50"]

    for j, (mname, fn) in enumerate(metric_defs):
        values = [fn(data[m]) for m in models_sorted]
        ax_bars.barh(y_pos + j * bar_width, values, bar_width * 0.9,
                     label=mname, color=metric_colors[j], alpha=0.85)

    ax_bars.set_yticks(y_pos + bar_width * (n_metrics - 1) / 2)
    ax_bars.set_yticklabels(models_sorted, fontsize=13)
    ax_bars.set_xlabel("Score (normalised)", fontsize=14)
    ax_bars.set_xlim(0, 1.05)
    ax_bars.set_title("Per-Model Scores (sorted by R@50)", fontsize=16)
    ax_bars.tick_params(axis="x", labelsize=12)
    ax_bars.grid(True, alpha=0.2, axis="x")
    ax_bars.invert_yaxis()
    ax_bars.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                   ncol=6, frameon=True, framealpha=0.9)

    fig.savefig(OUT_DIR / "fig2_embedding_radar.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_embedding_radar.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2: Embedding comparison saved ({len(models_sorted)} models, best={top_model})")

    # ── Table S1 ──
    rows = []
    for model in models_sorted:
        d = data[model]
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
            "onto_mrr": onto.get("mrr", 0),
            "tok_s": spd.get("tokens_per_sec", 0),
        })
    with open(OUT_DIR / "table_s1_embedding_full.csv", "w") as f:
        f.write("Model,R@50,MRR,nDCG@10,Onto_R@1,Onto_R@5,Onto_MRR,Tok/s\n")
        for r in rows:
            f.write(f"{r['model']},{r['r_at_50']:.4f},{r['mrr']:.4f},"
                    f"{r['ndcg_at_10']:.4f},{r['onto_r1']:.4f},"
                    f"{r['onto_r5']:.4f},{r['onto_mrr']:.4f},{r['tok_s']:.1f}\n")
    print(f"Table S1: {len(rows)} embedding models")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Retrieval strategy comparison (Bench 03)
# ═══════════════════════════════════════════════════════════════════════════
def figure_3_retrieval():
    """Grouped bar chart: 6 strategies × 5 quality metrics + latency side panel."""
    path = RESULTS_DIR / "retrieval_bench.json"
    if not path.exists():
        print("Figure 3: SKIPPED — retrieval_bench.json not found")
        return
    data = load_json(path)

    strategies = ["dense", "sparse", "hybrid", "hybrid+filter",
                  "hybrid+rerank", "hybrid+filter+rerank"]
    strategies = [s for s in strategies if s in data]

    quality_metrics = [
        ("P@5",      "p_at_5"),
        ("P@10",     "p_at_10"),
        ("R@50",     "r_at_50"),
        ("MRR",      "mrr"),
        ("nDCG@10",  "ndcg_at_10"),
    ]
    metric_colors = ["#2196F3", "#42A5F5", "#FF9800", "#F44336", "#9C27B0"]

    fig_w, fig_h = 16, 8
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Title
    ax_title = fig.add_axes([0.0, 0.93, 1.0, 0.06])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  "Figure 3: Retrieval Strategy Comparison (mxbai-embed-large, 90 queries)",
                  ha="center", va="center", fontsize=18)

    # ── Left panel: quality metrics grouped bars ──
    ax_qual = fig.add_axes([0.06, 0.12, 0.42, 0.76])

    x = np.arange(len(strategies))
    n_metrics = len(quality_metrics)
    total_bar_width = 0.75
    bar_w = total_bar_width / n_metrics

    for j, (mlabel, mkey) in enumerate(quality_metrics):
        vals = [data[s]["average"].get(mkey, 0) for s in strategies]
        offset = (j - n_metrics / 2 + 0.5) * bar_w
        bars = ax_qual.bar(x + offset, vals, bar_w * 0.9, label=mlabel,
                           color=metric_colors[j], alpha=0.85)
    ax_qual.set_xticks(x)
    short_labels = [s.replace("hybrid+filter+rerank", "hyb+flt+rrk")
                     .replace("hybrid+rerank", "hyb+rrk")
                     .replace("hybrid+filter", "hyb+flt")
                     for s in strategies]
    ax_qual.set_xticklabels(short_labels, fontsize=13, rotation=25, ha="right")
    ax_qual.set_ylabel("Score", fontsize=14)
    ax_qual.set_title("Quality Metrics", fontsize=15)
    ax_qual.tick_params(axis="y", labelsize=12)
    ax_qual.grid(True, alpha=0.2, axis="y")
    ax_qual.set_ylim(0, max(0.6, ax_qual.get_ylim()[1] * 1.15))

    ax_qual.legend(fontsize=12, ncol=5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.08), frameon=True, framealpha=0.9)

    # ── Right panel: latency bar ──
    ax_lat = fig.add_axes([0.52, 0.12, 0.46, 0.76])
    latencies = [data[s]["average"].get("avg_latency_ms",
                 data[s]["average"].get("latency_ms", 0)) for s in strategies]
    max_lat = max(latencies) if latencies else 1
    bars_lat = ax_lat.barh(np.arange(len(strategies)), latencies, 0.6,
                           color="#F44336", alpha=0.8)
    ax_lat.set_yticks(np.arange(len(strategies)))
    ax_lat.set_yticklabels(strategies, fontsize=13)
    ax_lat.set_xlabel("Latency (ms)", fontsize=14)
    ax_lat.set_title("Avg Latency", fontsize=15)
    ax_lat.tick_params(axis="x", labelsize=12)
    ax_lat.set_xlim(0, max_lat * 1.25)
    ax_lat.grid(True, alpha=0.2, axis="x")
    ax_lat.invert_yaxis()

    for bar, v in zip(bars_lat, latencies):
        ax_lat.text(bar.get_width() + max_lat * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f} ms", va="center", fontsize=12)

    fig.savefig(OUT_DIR / "fig3_retrieval_strategies.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_retrieval_strategies.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 3: Retrieval strategies saved ({len(strategies)} strategies)")

    # ── Table 4 ──
    header = "Strategy,P@5,P@10,R@50,MRR,nDCG@10,Latency(ms)"
    rows = []
    for s in strategies:
        avg = data[s].get("average", {})
        rows.append(
            f"{s},{avg.get('p_at_5', 0):.4f},{avg.get('p_at_10', 0):.4f},"
            f"{avg.get('r_at_50', 0):.4f},{avg.get('mrr', 0):.4f},"
            f"{avg.get('ndcg_at_10', 0):.4f},"
            f"{avg.get('avg_latency_ms', avg.get('latency_ms', 0)):.1f}"
        )
    with open(OUT_DIR / "table4_retrieval.csv", "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")

    # By-difficulty breakdown
    with open(OUT_DIR / "table4_retrieval_by_difficulty.csv", "w") as f:
        f.write("Strategy,Difficulty,P@5,P@10,R@50,MRR,nDCG@10\n")
        for s in strategies:
            by_diff = data[s].get("by_difficulty", {})
            for diff in ["easy", "medium", "hard"]:
                if diff not in by_diff:
                    continue
                d = by_diff[diff]
                f.write(
                    f"{s},{diff},{d.get('p_at_5', 0):.4f},{d.get('p_at_10', 0):.4f},"
                    f"{d.get('r_at_50', 0):.4f},{d.get('mrr', 0):.4f},{d.get('ndcg_at_10', 0):.4f}\n"
                )
    print(f"Table 4: {len(strategies)} strategies")
# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: LLM per-task heatmap (Bench 04) — detailed sub-metric view
# ═══════════════════════════════════════════════════════════════════════════
def _load_llm_rows():
    """Load and filter LLM bench data, excluding disabled models."""
    path = RESULTS_DIR / "llm_bench.json"
    if not path.exists():
        return None
    data = load_json(path)

    rows = []
    for label, d in sorted(data.items()):
        if "error" in str(d):
            continue
        if _is_excluded(label):
            continue
        ta = d.get("task_a_parsing", {})
        tb = d.get("task_b_extraction", {})
        tc = d.get("task_c_ontology", {})
        td = d.get("task_d_answer", {})
        te = d.get("task_e_speed", {})

        eb = tb.get("average", {})
        f1s = [eb.get(f, {}).get("f1", 0) for f in ["tissues", "diseases", "cell_types"]]
        avg_f1 = np.mean(f1s) if f1s else 0

        # Individual extraction fields
        tissue_f1 = eb.get("tissues", {}).get("f1", 0)
        disease_f1 = eb.get("diseases", {}).get("f1", 0)
        celltype_f1 = eb.get("cell_types", {}).get("f1", 0)

        # Composite
        parse_score = ta.get("exact_match", 0)
        onto_score = tc.get("f1", 0)
        answer_score = (td.get("citation_recall", 0) + td.get("grounding_rate", 0)) / 2
        speed_raw = te.get("tokens_per_sec", 0)
        speed_score = min(speed_raw / 100, 1.0)
        composite = (0.20 * parse_score + 0.20 * avg_f1 + 0.20 * onto_score
                     + 0.30 * answer_score + 0.10 * speed_score)

        rows.append({
            "model": label,
            "family": d.get("family", "?"),
            "size_b": d.get("size_b", "?"),
            "quant": d.get("quant", "?"),
            "think": d.get("think_enabled", False),
            "parse_em": parse_score,
            "tissue_f1": tissue_f1,
            "disease_f1": disease_f1,
            "celltype_f1": celltype_f1,
            "extract_f1": avg_f1,
            "onto_f1": onto_score,
            "cite_recall": td.get("citation_recall", 0),
            "cite_prec": td.get("citation_precision", 0),
            "grounding": td.get("grounding_rate", 0),
            "tok_s": speed_raw,
            "composite": composite,
        })

    rows.sort(key=lambda r: -r["composite"])
    return rows


def figure_4_llm_heatmap(rows):
    """Expanded heatmap: models × all sub-task metrics (not just 5 summary metrics)."""
    if not rows:
        print("Figure 4: SKIPPED — no LLM results")
        return

    sorted_rows = sorted(rows, key=lambda r: -r["composite"])

    # 10 detailed columns covering all sub-tasks
    task_keys = [
        "parse_em", "tissue_f1", "disease_f1", "celltype_f1",
        "onto_f1", "cite_recall", "cite_prec", "grounding",
    ]
    task_labels = [
        "Parse\nEM", "Tissue\nF1", "Disease\nF1", "CellType\nF1",
        "Onto\nF1", "Cite\nRecall", "Cite\nPrec", "Ground\nRate",
    ]
    task_groups = [
        ("Task A: Parsing", 0, 1),
        ("Task B: Extraction", 1, 4),
        ("Task C: Ontology", 4, 5),
        ("Task D: Answer Quality", 5, 8),
    ]

    model_names = [r["model"] for r in sorted_rows]
    n_models = len(model_names)
    n_cols = len(task_keys)

    matrix = np.array([[r[k] for k in task_keys] for r in sorted_rows])

    # Speed column (separate color scale)
    speed_vals = [r["tok_s"] for r in sorted_rows]
    composite_vals = [r["composite"] for r in sorted_rows]

    fig_w = 16
    fig_h = max(9, n_models * 0.50 + 3.0)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Title
    heatmap_bottom = 0.08
    heatmap_height = 0.74
    ax_title = fig.add_axes([0.0, 0.95, 1.0, 0.04])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  f"Figure 4: LLM Per-Task Performance ({n_models} configurations, 5 model families)",
                  ha="center", va="center", fontsize=20)

    # ── Main heatmap ──
    ax_heat = fig.add_axes([0.14, heatmap_bottom, 0.52, heatmap_height])
    im = ax_heat.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(task_labels, fontsize=12, ha="center")
    ax_heat.set_yticks(range(n_models))
    ax_heat.set_yticklabels(model_names, fontsize=13)

    for i in range(n_models):
        for j in range(n_cols):
            val = matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=11, color=color)

    # Task group brackets on top
    for glabel, gs, ge in task_groups:
        mid = (gs + ge - 1) / 2
        ax_heat.text(mid, -1.6, glabel, ha="center", va="center",
                     fontsize=12, color="#333333")
        ax_heat.plot([gs - 0.4, ge - 0.6], [-1.1, -1.1], color="#666666",
                     linewidth=1.5, clip_on=False)

    # Colorbar for heatmap
    cbar_ax = fig.add_axes([0.14, heatmap_bottom - 0.05, 0.52, 0.012])
    plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel("Score (0-1)", fontsize=12)
    cbar_ax.tick_params(labelsize=10)

    # ── Speed column ──
    ax_speed = fig.add_axes([0.70, heatmap_bottom, 0.08, heatmap_height])
    speed_arr = np.array(speed_vals).reshape(-1, 1)
    im_spd = ax_speed.imshow(speed_arr, cmap="Blues", aspect="auto",
                              vmin=0, vmax=max(speed_vals) * 1.1)
    ax_speed.set_xticks([0])
    ax_speed.set_xticklabels(["Tok/s"], fontsize=12)
    ax_speed.set_yticks([])
    for i, v in enumerate(speed_vals):
        ax_speed.text(0, i, f"{v:.0f}", ha="center", va="center",
                      fontsize=11, color="white" if v > max(speed_vals) * 0.6 else "black")
    ax_speed.text(0, -1.6, "Task E", ha="center", va="center",
                  fontsize=12, color="#333333", clip_on=False)

    # ── Composite column ──
    ax_comp = fig.add_axes([0.82, heatmap_bottom, 0.08, heatmap_height])
    comp_arr = np.array(composite_vals).reshape(-1, 1)
    im_comp = ax_comp.imshow(comp_arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax_comp.set_xticks([0])
    ax_comp.set_xticklabels(["Composite"], fontsize=12)
    ax_comp.set_yticks([])
    for i, v in enumerate(composite_vals):
        ax_comp.text(0, i, f"{v:.3f}", ha="center", va="center",
                     fontsize=11, color="white" if v > 0.6 else "black")

    fig.savefig(OUT_DIR / "fig4_llm_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_llm_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 4: LLM heatmap saved ({n_models} model configs)")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Context window optimisation (Bench 05)
# ═══════════════════════════════════════════════════════════════════════════
def figure_5_context_curve():
    """Three subplots: citation recall vs k, citation precision vs k, context tokens vs k."""
    path = RESULTS_DIR / "context_bench.json"
    if not path.exists():
        print("Figure 5: SKIPPED — context_bench.json not found")
        return

    data = load_json(path)
    summary = data.get("summary", {})

    configs = {}
    for key, vals in summary.items():
        parts = key.split("_", 1)
        if len(parts) != 2 or not parts[0].startswith("k="):
            continue
        k = int(parts[0][2:])
        fmt = parts[1]
        if fmt not in configs:
            configs[fmt] = {"k": [], "recall": [], "precision": [],
                            "grounding": [], "tokens": [], "duration": []}
        configs[fmt]["k"].append(k)
        configs[fmt]["recall"].append(vals.get("avg_citation_recall", 0))
        configs[fmt]["precision"].append(vals.get("avg_citation_precision", 0))
        configs[fmt]["grounding"].append(vals.get("avg_grounding_rate", 0))
        configs[fmt]["tokens"].append(vals.get("avg_context_tokens", 0))
        configs[fmt]["duration"].append(vals.get("avg_duration_ms", 0))

    if not configs:
        print("Figure 5: No valid context configs found")
        return

    fig_w, fig_h = 16, 8
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax_title = fig.add_axes([0.0, 0.93, 1.0, 0.06])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  "Figure 5: Context Window Optimisation (k x format, 15 queries)",
                  ha="center", va="center", fontsize=20)

    colors = {"full": "#2196F3", "structured": "#FF9800", "minimal": "#4CAF50"}
    markers = {"full": "o", "structured": "s", "minimal": "^"}

    axes_specs = [
        ([0.07, 0.10, 0.19, 0.78], "Cite Recall vs. k", "recall", "Score"),
        ([0.32, 0.10, 0.19, 0.78], "Cite Precision vs. k", "precision", "Score"),
        ([0.57, 0.10, 0.19, 0.78], "Grounding vs. k", "grounding", "Score"),
        ([0.82, 0.10, 0.16, 0.78], "Context Tokens vs. k", "tokens", "Tokens"),
    ]

    for spec, (pos, title, metric_key, ylabel) in enumerate(axes_specs):
        ax = fig.add_axes(pos)
        for fmt, d in configs.items():
            order = np.argsort(d["k"])
            ks = [d["k"][i] for i in order]
            vals = [d[metric_key][i] for i in order]
            c = colors.get(fmt, "gray")
            m = markers.get(fmt, "o")
            ax.plot(ks, vals, f"{m}-", color=c, label=fmt, linewidth=2.5, markersize=9)

        ax.set_xlabel("k", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(title, fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.2)
        if metric_key != "tokens":
            ax.set_ylim(0, 1.05)
        if spec == 0:
            ax.legend(fontsize=13, loc="lower left")

    fig.savefig(OUT_DIR / "fig5_context_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_context_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 5: Context curve saved")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: End-to-End pipeline comparison (Bench 07)
# ═══════════════════════════════════════════════════════════════════════════
def figure_6_e2e_comparison():
    """Full E2E comparison: quality metrics (R@50, MRR, Cite Recall, Grounding) + latency."""
    path = RESULTS_DIR / "e2e_report.json"
    if not path.exists():
        print("Figure 6: SKIPPED — e2e_report.json not found")
        return

    raw = load_json(path)
    configs = []
    for name, d in raw.items():
        if not isinstance(d, dict):
            continue
        m = d.get("metrics") or d.get("average", {})
        if not m:
            continue
        configs.append({
            "name": name,
            "r_at_50": m.get("r_at_50", 0),
            "mrr": m.get("mrr", 0),
            "cite_recall": m.get("citation_recall", 0),
            "grounding": m.get("grounding_rate", 0),
            "halluc": m.get("hallucination_rate", 0),
            "latency_s": m.get("latency_sec", m.get("avg_latency_ms", 0) / 1000),
            "config": d.get("config", {}),
        })

    if not configs:
        print("Figure 6: No valid E2E configs found")
        return

    names = [c["name"] for c in configs]
    n = len(names)

    fig_w, fig_h = 16, 10
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax_title = fig.add_axes([0.0, 0.94, 1.0, 0.05])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  "Figure 6: End-to-End Pipeline Comparison (4 configurations, 18 queries)",
                  ha="center", va="center", fontsize=20)

    # ── Top-left: quality metrics grouped bars ──
    ax_qual = fig.add_axes([0.06, 0.52, 0.52, 0.38])
    x = np.arange(n)
    qual_metrics = [
        ("Cite Recall", "cite_recall", "#4CAF50"),
        ("Grounding", "grounding", "#9C27B0"),
        ("Halluc Rate", "halluc", "#F44336"),
        ("R@50", "r_at_50", "#2196F3"),
    ]
    n_m = len(qual_metrics)
    bw = 0.75 / n_m
    for j, (ml, mk, mc) in enumerate(qual_metrics):
        vals = [c[mk] for c in configs]
        offset = (j - n_m / 2 + 0.5) * bw
        bars = ax_qual.bar(x + offset, vals, bw * 0.9, label=ml, color=mc, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax_qual.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=11)

    ax_qual.set_xticks(x)
    ax_qual.set_xticklabels(names, fontsize=14, rotation=12, ha="right")
    ax_qual.set_ylabel("Score", fontsize=15)
    ax_qual.set_title("Quality Metrics", fontsize=16)
    ax_qual.tick_params(labelsize=13)
    ax_qual.legend(fontsize=12, ncol=2, loc="upper right", frameon=True, framealpha=0.9)
    ax_qual.grid(True, alpha=0.2, axis="y")

    # ── Top-right: latency ──
    ax_lat = fig.add_axes([0.66, 0.52, 0.30, 0.38])
    lat_vals = [c["latency_s"] for c in configs]
    bar_colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
    bars = ax_lat.bar(np.arange(n), lat_vals, 0.6,
                      color=bar_colors[:n], alpha=0.85)
    ax_lat.set_xticks(np.arange(n))
    ax_lat.set_xticklabels(names, fontsize=12, rotation=15, ha="right")
    ax_lat.set_ylabel("Seconds", fontsize=14)
    ax_lat.set_title("End-to-End Latency", fontsize=15)
    ax_lat.tick_params(labelsize=12)
    ax_lat.grid(True, alpha=0.2, axis="y")
    for bar, v in zip(bars, lat_vals):
        ax_lat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}s", ha="center", va="bottom", fontsize=12)

    # ── Bottom: config details table ──
    ax_table = fig.add_axes([0.06, 0.04, 0.90, 0.42])
    ax_table.axis("off")
    ax_table.text(0.0, 1.0, "Pipeline Configuration Details:",
                  fontsize=14, va="top", transform=ax_table.transAxes)

    col_headers = ["Config", "Embedding", "Strategy", "LLM", "Parse Model",
                   "Context Fmt", "Context k"]
    col_x = [0.0, 0.14, 0.30, 0.44, 0.58, 0.72, 0.87]

    for j, h in enumerate(col_headers):
        ax_table.text(col_x[j], 0.85, h, fontsize=12,
                      va="top", transform=ax_table.transAxes,
                      color="#333333")

    for i, c in enumerate(configs):
        cfg = c["config"]
        vals = [
            c["name"],
            cfg.get("embedding", "?"),
            cfg.get("strategy", "?"),
            cfg.get("llm", "?"),
            str(cfg.get("parse_model", "none")),
            cfg.get("context_format", "?"),
            str(cfg.get("context_k", "?")),
        ]
        y = 0.70 - i * 0.15
        for j, v in enumerate(vals):
            ax_table.text(col_x[j], y, v, fontsize=11, va="top",
                          transform=ax_table.transAxes)

    fig.savefig(OUT_DIR / "fig6_e2e_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_e2e_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 6: E2E comparison saved ({n} configs)")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Composite summary (stacked breakdown)
# ═══════════════════════════════════════════════════════════════════════════
def figure_7_composite_summary(rows):
    """Horizontal stacked bar: composite score decomposed into 5 weighted components."""
    if not rows:
        print("Figure 7: SKIPPED — no LLM results")
        return

    sorted_rows = sorted(rows, key=lambda r: -r["composite"])

    model_names = [r["model"] for r in sorted_rows]
    n = len(model_names)

    # Compute weighted components
    components = []
    for r in sorted_rows:
        parse_w = 0.20 * r["parse_em"]
        extract_w = 0.20 * r["extract_f1"]
        onto_w = 0.20 * r["onto_f1"]
        answer_w = 0.30 * (r["cite_recall"] + r["grounding"]) / 2
        speed_w = 0.10 * min(r["tok_s"] / 100, 1.0)
        components.append([parse_w, extract_w, onto_w, answer_w, speed_w])

    components = np.array(components)
    comp_labels = ["Parse (0.20)", "Extract (0.20)", "Ontology (0.20)",
                   "Answer (0.30)", "Speed (0.10)"]
    comp_colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336"]

    fig_w = 16
    fig_h = max(9, n * 0.50 + 2.5)
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax_title = fig.add_axes([0.0, 1.0 - 1.0 / fig_h, 1.0, 1.0 / fig_h])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  "Figure 7: LLM Composite Score Breakdown",
                  ha="center", va="center", fontsize=20)

    bar_bottom = 1.5 / fig_h
    bar_height = 1.0 - bar_bottom - 1.6 / fig_h

    ax = fig.add_axes([0.16, bar_bottom, 0.74, bar_height])
    y_pos = np.arange(n)

    left = np.zeros(n)
    for j in range(5):
        ax.barh(y_pos, components[:, j], 0.7, left=left,
                label=comp_labels[j], color=comp_colors[j], alpha=0.85)
        for i in range(n):
            seg_w = components[i, j]
            if seg_w > 0.03:
                ax.text(left[i] + seg_w / 2, i, f"{seg_w:.2f}",
                        ha="center", va="center", fontsize=11, color="white")
        left += components[:, j]

    for i in range(n):
        total = components[i].sum()
        ax.text(total + 0.01, i, f"{total:.3f}",
                ha="left", va="center", fontsize=13)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=14)
    ax.set_xlabel("Composite Score", fontsize=16)
    ax.set_xlim(0, 1.0)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()

    fig.savefig(OUT_DIR / "fig7_composite_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig7_composite_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 7: Composite summary saved ({n} models)")


# ═══════════════════════════════════════════════════════════════════════════
# Tables 3, 5, S2
# ═══════════════════════════════════════════════════════════════════════════
def table_3_llm_comparison(rows):
    if not rows:
        print("Table 3: SKIPPED")
        return
    with open(OUT_DIR / "table3_llm_comparison.csv", "w") as f:
        f.write("Model,Family,Size(B),Quant,Think,Parse_EM,Tissue_F1,Disease_F1,"
                "CellType_F1,Extract_F1_avg,Onto_F1,Cite_Recall,Cite_Prec,"
                "Grounding,Tok/s,Composite\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['family']},{r['size_b']},{r['quant']},"
                f"{r['think']},{r['parse_em']:.4f},{r['tissue_f1']:.4f},"
                f"{r['disease_f1']:.4f},{r['celltype_f1']:.4f},{r['extract_f1']:.4f},"
                f"{r['onto_f1']:.4f},{r['cite_recall']:.4f},{r['cite_prec']:.4f},"
                f"{r['grounding']:.4f},{r['tok_s']:.1f},{r['composite']:.4f}\n"
            )
    print(f"Table 3: {len(rows)} model configurations")


def table_5_top5(rows):
    if not rows:
        print("Table 5: SKIPPED")
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


def table_s2_context_management():
    path = RESULTS_DIR / "context_management_bench.json"
    if not path.exists():
        print("Table S2: SKIPPED")
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
                f"{vals.get('avg_context_tokens', 0):.0f},"
                f"{vals.get('avg_duration_ms', 0):.0f}\n"
            )
    print(f"Table S2: {len(summary)} context management strategies")


# ═══════════════════════════════════════════════════════════════════════════
# Summary JSON
# ═══════════════════════════════════════════════════════════════════════════
def write_summary_json(corpus_stats, query_dist, llm_rows):
    summary = {
        "corpus": {"n_studies": corpus_stats[0][0][1] if corpus_stats else 0},
        "queries": {"n_queries": sum(query_dist[0].values()) if query_dist else 0},
        "embedding": {"best_model": None, "best_r_at_50": 0},
        "retrieval": {"best_strategy": None, "best_r_at_50": 0},
        "llm": {"n_configs": len(llm_rows) if llm_rows else 0, "top5": []},
    }

    if (RESULTS_DIR / "embedding_bench.json").exists():
        edata = load_json(RESULTS_DIR / "embedding_bench.json")
        best = max(edata.items(),
                   key=lambda x: x[1].get("retrieval", {}).get("average", {}).get("r_at_50", 0))
        summary["embedding"]["best_model"] = best[0]
        summary["embedding"]["best_r_at_50"] = best[1]["retrieval"]["average"]["r_at_50"]

    if (RESULTS_DIR / "retrieval_bench.json").exists():
        rdata = load_json(RESULTS_DIR / "retrieval_bench.json")
        candidates = [(k, v) for k, v in rdata.items()
                      if isinstance(v, dict) and "average" in v]
        if candidates:
            best_s = max(candidates, key=lambda x: x[1]["average"].get("r_at_50", 0))
            summary["retrieval"]["best_strategy"] = best_s[0]
            summary["retrieval"]["best_r_at_50"] = best_s[1]["average"]["r_at_50"]

    if llm_rows:
        top5 = sorted(llm_rows, key=lambda r: -r["composite"])[:5]
        summary["llm"]["top5"] = [
            {"model": r["model"], "composite": round(r["composite"], 4)}
            for r in top5
        ]

    with open(OUT_DIR / "article_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary JSON saved")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating article figures and tables")
    print(f"Excluded models: {EXCLUDED_MODELS}")
    print("=" * 60)

    # Tables 1-2
    corpus_stats = table_1_corpus_stats()
    print()
    query_dist = table_2_query_distribution()
    print()

    # Figure 1: Study design
    figure_1_study_design()
    print()

    # Figure 2 + Table S1 (embedding)
    if (RESULTS_DIR / "embedding_bench.json").exists():
        figure_2_embedding_radar()
        print()

    # Figure 3 + Table 4 (retrieval)
    if (RESULTS_DIR / "retrieval_bench.json").exists():
        figure_3_retrieval()
        print()

    # Load LLM data (used by Fig 4, 7 and Tables 3, 5)
    llm_rows = _load_llm_rows()

    # Figure 4 (LLM heatmap)
    if llm_rows:
        figure_4_llm_heatmap(llm_rows)
        table_3_llm_comparison(llm_rows)
        table_5_top5(llm_rows)
        print()

    # Figure 5 (context)
    figure_5_context_curve()
    table_s2_context_management()
    print()

    # Figure 6 (end-to-end)
    figure_6_e2e_comparison()
    print()

    # Figure 7 (composite summary)
    if llm_rows:
        figure_7_composite_summary(llm_rows)
        print()

    # Summary JSON
    write_summary_json(corpus_stats, query_dist, llm_rows)

    print()
    print("=" * 60)
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
