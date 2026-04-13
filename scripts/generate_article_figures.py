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


def _short_strategy_label(label: str) -> str:
    """Compact retrieval strategy labels for crowded axes."""
    return (label.replace("hybrid+filter+rerank", "hyb+flt+rrk")
            .replace("hybrid+rerank", "hyb+rrk")
            .replace("hybrid+filter", "hyb+flt"))


def _wrap_config_label(label: str) -> str:
    """Wrap snake_case config names for x/y tick labels."""
    return label.replace("_", "\n")


# ── Color palette (15 model families) ──
FAMILY_COLORS = {
    "qwen": "#2196F3",
    "gemma": "#FF9800",
    "mistral": "#9C27B0",
    "phi": "#4CAF50",
    "llama": "#F44336",
    "deepseek": "#00BCD4",
    "granite": "#795548",
    "falcon": "#607D8B",
    "aya": "#E91E63",
    "glm": "#3F51B5",
    "internlm": "#009688",
    "yi": "#FF5722",
    "solar": "#CDDC39",
    "exaone": "#8BC34A",
    "starcoder": "#FFC107",
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
def _draw_box(ax, x, y, w, h, text, facecolor, edgecolor="none",
              fontsize=13, fontweight="normal", text_color="#222222",
              linewidth=0, boxstyle="round,pad=0.06"):
    """Draw a rounded box with centered text and return the patch."""
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=boxstyle, facecolor=facecolor,
        edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            linespacing=1.4)
    return rect


def _draw_arrow(ax, x1, y1, x2, y2, color="#666666", lw=1.8):
    """Draw a connecting arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0"))


def figure_1_study_design():
    """Publication-ready study design overview with consistent typography."""
    fig_w, fig_h = 16.4, 13.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 16); ax.set_ylim(0, 13.4)
    ax.axis("off")

    title_fs = 22
    section_fs = 14.8
    body_lg_fs = 13.7
    body_md_fs = 12.8
    body_sm_fs = 11.6
    chip_fs = 9.3
    stage_fs = 11.8
    summary_fs = 11.4
    callout_fs = 11.2

    # ── Title ──
    ax.text(8.0, 12.78, "scMetaIntel-Hub: Comprehensive Benchmarking of Local LLMs\n"
            "for Single-Cell Genomics Metadata Intelligence",
        ha="center", va="center", fontsize=title_fs, fontweight="normal",
        color="#111111", linespacing=1.3)

    # Thin separator line
    ax.plot([1.4, 14.6], [12.08, 12.08], color="#D4D4D4", linewidth=0.9)

    # ================================================================
    # ROW 1 (y≈10.8): Data Foundation — two boxes
    # ================================================================
    data_y = 10.95
    _draw_box(ax, 4.5, data_y, 6.2, 1.24,
              "scRNA-seq Metadata Corpus\n"
              "2,189 GEO series across 43 organisms\n"
              "Curated tissue, disease & cell-type annotations",
          "#E3F2FD", fontsize=body_lg_fs)
    _draw_box(ax, 12.0, data_y, 5.4, 1.24,
              "Evaluation Framework\n"
              "171 hand-crafted queries (7 categories)\n"
              "27 public benchmark datasets",
          "#E3F2FD", fontsize=body_lg_fs)

    # Left stage label
    ax.text(0.55, data_y, "Data", ha="center", va="center",
        fontsize=stage_fs, fontweight="normal", color="#8A8A8A")

    # Arrows down
    _draw_arrow(ax, 4.5, data_y - 0.62, 4.5, 9.7)
    _draw_arrow(ax, 12.0, data_y - 0.62, 12.0, 9.7)

    # ================================================================
    # ROW 2 (y≈8.9): Embedding + Retrieval
    # ================================================================
    row2_y = 9.0
    _draw_box(ax, 4.5, row2_y, 6.2, 1.02,
              "Embedding Model Selection\n"
              "19 models: general-purpose & biomedical\n"
              "R@50, MRR, nDCG@10, ontology recall, speed",
          "#BBDEFB", fontsize=body_lg_fs)

    _draw_box(ax, 12.0, row2_y, 5.4, 1.02,
              "Retrieval Strategy Comparison\n"
              "Dense, sparse, hybrid, filtered, reranked\n"
              "11 reranker models evaluated",
          "#FFE0B2", fontsize=body_lg_fs)

    ax.annotate("best embedding", xy=(9.1, row2_y), xytext=(7.5, row2_y),
        fontsize=callout_fs, color="#1565C0", va="center",
        fontweight="medium",
            arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=1.3))

    ax.text(0.55, row2_y, "Stage 1\nRepresentation\n& Retrieval",
        ha="center", va="center", fontsize=stage_fs, fontweight="normal",
        color="#8A8A8A", linespacing=1.25)

    _draw_arrow(ax, 4.5, row2_y - 0.51, 8.0, 8.02)
    _draw_arrow(ax, 12.0, row2_y - 0.51, 8.0, 8.02)

    # ================================================================
    # ROW 3 (y≈7.0): LLM Evaluation — two sub-sections
    # ================================================================
    llm_y = 6.95
    llm_h = 2.0

    # Outer LLM box
    _draw_box(ax, 8.0, llm_y, 14.6, llm_h, "",
              "#F3E5F5")
    ax.text(8.0, llm_y + 0.72, "Large Language Model Evaluation  —  "
            "51 models across 15 architectural families",
        ha="center", va="center", fontsize=section_fs,
        fontweight="normal", color="#4A148C")

    # 8 domain-specific task chips in a 4×2 grid
    all_tasks = [
        ("Query\nParsing", "#CE93D8"), ("Metadata\nExtraction", "#BA68C8"),
        ("Ontology\nMapping", "#AB47BC"), ("Answer\nGeneration", "#9C27B0"),
        ("Inference\nSpeed", "#CE93D8"), ("Relevance\nJudgment", "#BA68C8"),
        ("Domain\nClassification", "#AB47BC"), ("Organism &\nModality", "#9C27B0"),
    ]
    chip_w2, chip_h2 = 1.6, 0.48

    ax.text(5.3, llm_y + 0.28, "Domain-Specific Tasks",
        ha="center", va="center", fontsize=body_sm_fs, color="#6A1B9A",
        fontweight="normal")
    xs_chip = [1.78, 3.46, 5.14, 6.82, 1.78, 3.46, 5.14, 6.82]
    ys_chip = [llm_y - 0.12] * 4 + [llm_y - 0.64] * 4
    for i, (label, color) in enumerate(all_tasks):
        _draw_box(ax, xs_chip[i], ys_chip[i], chip_w2, chip_h2, label,
          color, fontsize=chip_fs,
          fontweight="normal", text_color="white")

    # Public benchmarks (right portion)
    pub_x = 12.0
    ax.text(pub_x, llm_y + 0.28, "Public Benchmarks (27 datasets)",
        ha="center", va="center", fontsize=body_sm_fs, color="#6A1B9A",
        fontweight="normal")
    pub_items = [
        ("General & Reasoning\nMMLU, HellaSwag\nGSM8K, TruthfulQA", "#E1BEE7"),
        ("Biomedical\nPubMedQA, MedQA\nBioASQ, SciQ", "#D1C4E9"),
        ("Structured &\nTool-Use\nIFEval, NexusFC", "#EDE7F6"),
    ]
    for i, (label, color) in enumerate(pub_items):
        px = pub_x - 1.9 + i * 1.9
        _draw_box(ax, px, llm_y - 0.38, 1.8, 0.9, label,
                  color, fontsize=9.4,
                  text_color="#333333")

    ax.text(0.55, llm_y, "Stage 2\nGeneration\n& Reasoning",
        ha="center", va="center", fontsize=stage_fs, fontweight="normal",
        color="#8A8A8A", linespacing=1.25)

    # Arrows down — two paths
    _draw_arrow(ax, 5.5, llm_y - llm_h / 2, 5.5, 5.3)
    _draw_arrow(ax, 10.5, llm_y - llm_h / 2, 10.5, 5.3)

    # ================================================================
    # ROW 4 (y≈4.7): Ablation Studies
    # ================================================================
    row4_y = 4.7
    _draw_box(ax, 5.0, row4_y, 6.8, 1.06,
              "Ablation Studies\n"
              "Thinking mode on/off | Quantization Q4 vs Q8\n"
              "KV-cache precision | Context window sweep",
          "#C8E6C9", fontsize=body_lg_fs)

    _draw_box(ax, 12.5, row4_y, 4.6, 1.06,
              "End-to-End Pipeline\n"
              "Full RAG: embed → retrieve →\n"
              "rerank → generate → evaluate",
          "#FFCDD2", fontsize=body_lg_fs)

    ax.annotate("optimal config", xy=(10.3, row4_y), xytext=(8.25, row4_y),
        fontsize=callout_fs, color="#2E7D32", va="center",
        fontweight="medium",
            arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.3))

    ax.text(0.55, row4_y, "Stage 3\nOptimization",
        ha="center", va="center", fontsize=stage_fs, fontweight="normal",
        color="#8A8A8A", linespacing=1.25)

    # Arrows down
    _draw_arrow(ax, 5.0, row4_y - 0.53, 8.0, 3.58)
    _draw_arrow(ax, 12.5, row4_y - 0.53, 8.0, 3.58)

    # ================================================================
    # ROW 5 (y≈3.0): Recommendations & Deployment
    # ================================================================
    out_y = 3.0
    _draw_box(ax, 8.0, out_y, 14.6, 1.12,
              "", "#FFF9C4")

    ax.text(8.0, out_y + 0.24, "Recommendations for Local Deployment",
        ha="center", va="center", fontsize=section_fs,
        fontweight="normal", color="#E65100")

    findings = [
        "Compact models (4B–9B)\nmatch larger counterparts",
        "Think mode improves\nreasoning but hurts\nstructured output",
        "Q4 quantization retains\n>95% of Q8 accuracy",
        "Fully local inference\nwith no cloud dependency",
    ]
    x_positions = [2.5, 6.2, 10.2, 13.8]
    for fx, ftxt in zip(x_positions, findings):
        ax.text(fx, out_y - 0.24, f"\u2022 {ftxt}",
            ha="center", va="center", fontsize=body_sm_fs, color="#333333",
            linespacing=1.15)

    ax.text(0.55, out_y, "Output",
        ha="center", va="center", fontsize=stage_fs, fontweight="normal",
        color="#8A8A8A")

    # ================================================================
    # ROW 6 (y≈1.8): Scale summary bar
    # ================================================================
    sum_y = 1.62
    summary_items = [
        ("19\nEmbeddings", "#BBDEFB"),
        ("11\nRerankers", "#FFE0B2"),
        ("51\nLLMs", "#E1BEE7"),
        ("15\nFamilies", "#E1BEE7"),
        ("8 Domain\nTasks", "#D1C4E9"),
        ("27 Public\nDatasets", "#D1C4E9"),
        ("171\nQueries", "#B3E5FC"),
        ("2,189\nGSE Series", "#B3E5FC"),
    ]
    n_items = len(summary_items)
    total_w = 14.4
    item_w = total_w / n_items
    start_x = 8.0 - total_w / 2 + item_w / 2

    for i, (label, color) in enumerate(summary_items):
        ix = start_x + i * item_w
        _draw_box(ax, ix, sum_y, item_w * 0.92, 0.8, label,
                  color, fontsize=summary_fs,
                  fontweight="normal")

    # Thin line above summary
    ax.plot([0.8, 15.2], [sum_y + 0.54, sum_y + 0.54],
            color="#DDDDDD", linewidth=0.6)
    ax.text(8.0, sum_y + 0.7, "Benchmark Scale",
            ha="center", va="center", fontsize=body_sm_fs, color="#9B9B9B")

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
    gs = fig.add_gridspec(
        1, 2,
        left=0.05, right=0.98, top=0.88, bottom=0.17,
        width_ratios=[1.0, 1.15], wspace=0.48,
    )
    fig.suptitle(
        f"Figure 2: Embedding Model Evaluation ({len(data)} models, 90 queries)",
        fontsize=20, y=0.96,
    )

    # ── Left panel: radar ──
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
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
                    fontsize=12.5, ncol=1, frameon=False)

    # ── Right panel: per-model horizontal bar chart ──
    models_sorted = sorted(data.keys(),
                           key=lambda m: data[m].get("retrieval", {}).get("average", {}).get("r_at_50", 0),
                           reverse=True)

    n_models = len(models_sorted)
    ax_bars = fig.add_subplot(gs[0, 1])

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
    ax_bars.legend(fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                   ncol=3, frameon=False)

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
    gs = fig.add_gridspec(
        1, 2,
        left=0.06, right=0.98, top=0.86, bottom=0.18,
        width_ratios=[1.02, 1.12], wspace=0.30,
    )
    fig.suptitle(
        "Figure 3: Retrieval Strategy Comparison (mxbai-embed-large, 90 queries)",
        fontsize=18, y=0.96,
    )

    # ── Left panel: quality metrics grouped bars ──
    ax_qual = fig.add_subplot(gs[0, 0])

    x = np.arange(len(strategies))
    n_metrics = len(quality_metrics)
    total_bar_width = 0.75
    bar_w = total_bar_width / n_metrics

    for j, (mlabel, mkey) in enumerate(quality_metrics):
        vals = [data[s]["average"].get(mkey, 0) for s in strategies]
        offset = (j - n_metrics / 2 + 0.5) * bar_w
        ax_qual.bar(x + offset, vals, bar_w * 0.9, label=mlabel,
                    color=metric_colors[j], alpha=0.85)
    ax_qual.set_xticks(x)
    short_labels = [_short_strategy_label(s) for s in strategies]
    ax_qual.set_xticklabels(short_labels, fontsize=13, rotation=18, ha="right")
    ax_qual.set_ylabel("Score", fontsize=14)
    ax_qual.set_title("Quality Metrics", fontsize=15)
    ax_qual.tick_params(axis="y", labelsize=12)
    ax_qual.grid(True, alpha=0.2, axis="y")
    ax_qual.set_ylim(0, max(0.6, ax_qual.get_ylim()[1] * 1.15))

    ax_qual.legend(fontsize=11.5, ncol=3, loc="upper left",
                   bbox_to_anchor=(-0.01, -0.16), frameon=False)

    # ── Right panel: latency bar ──
    ax_lat = fig.add_subplot(gs[0, 1])
    latencies = [data[s]["average"].get("avg_latency_ms",
                 data[s]["average"].get("latency_ms", 0)) for s in strategies]
    max_lat = max(latencies) if latencies else 1
    bars_lat = ax_lat.barh(np.arange(len(strategies)), latencies, 0.6,
                           color="#F44336", alpha=0.8)
    ax_lat.set_yticks(np.arange(len(strategies)))
    ax_lat.set_yticklabels(short_labels, fontsize=13)
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
        if _is_excluded(label):
            continue
        # Skip only if core tasks (A: parsing) completely failed
        ta = d.get("task_a_parsing", {})
        if "error" in str(ta):
            continue
        tb = d.get("task_b_extraction", {})
        tc = d.get("task_c_ontology", {})
        td = d.get("task_d_answer", {})
        te = d.get("task_e_speed", {})
        tf = d.get("task_f_relevance", {})
        tg = d.get("task_g_domain", {})
        th = d.get("task_h_org_modality", {})

        eb = tb.get("average", {})
        f1s = [eb.get(f, {}).get("f1", 0) for f in ["tissues", "diseases", "cell_types"]]
        avg_f1 = np.mean(f1s) if f1s else 0

        # Individual extraction fields
        tissue_f1 = eb.get("tissues", {}).get("f1", 0)
        disease_f1 = eb.get("diseases", {}).get("f1", 0)
        celltype_f1 = eb.get("cell_types", {}).get("f1", 0)

        # Tasks F-H scores
        relevance_f1 = tf.get("f1", 0)
        domain_acc = tg.get("accuracy", 0)
        org_acc = th.get("organism_accuracy", 0)
        mod_f1 = th.get("modality", {}).get("f1", 0) if isinstance(th.get("modality"), dict) else 0

        # Composite (8 tasks, rebalanced weights)
        # answer_score uses cite_recall + cite_prec (not grounding, which is
        # near-duplicate — see STATUS_REPORT for the fix that separated them).
        parse_score = ta.get("exact_match", 0)
        onto_score = tc.get("f1", 0)
        cite_r = td.get("citation_recall", 0)
        cite_p = td.get("citation_precision", 0)
        answer_score = (cite_r + cite_p) / 2
        speed_raw = te.get("tokens_per_sec", 0)
        speed_score = min(speed_raw / 150, 1.0)  # cap at 150 tok/s (was 100)
        composite = (0.15 * parse_score + 0.15 * avg_f1 + 0.15 * onto_score
                     + 0.20 * answer_score + 0.05 * speed_score
                     + 0.10 * relevance_f1 + 0.10 * domain_acc
                     + 0.10 * (org_acc + mod_f1) / 2)

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
            "relevance_f1": relevance_f1,
            "domain_acc": domain_acc,
            "org_acc": org_acc,
            "mod_f1": mod_f1,
            "composite": composite,
        })

    rows.sort(key=lambda r: -r["composite"])
    return rows


def figure_4_llm_heatmap(rows):
    """Domain-only heatmap + speed/composite + composite breakdown bars."""
    if not rows:
        print("Figure 4: SKIPPED — no LLM results")
        return
    sorted_rows = sorted(rows, key=lambda r: -r["composite"])
    # Heatmap columns: dropped grounding (ceiling effect, ~60% at 1.0)
    # and kept cite_prec which now differs from grounding after the fix.
    task_keys = ["parse_em", "tissue_f1", "disease_f1", "celltype_f1",
                 "onto_f1", "cite_recall", "cite_prec",
                 "relevance_f1", "domain_acc", "org_acc", "mod_f1"]
    task_labels = ["Parse", "Tissue", "Disease", "Cell", "Onto",
                   "CiteR", "CiteP", "Relev", "Domain", "OrgAc", "ModF1"]
    task_groups = [("A", 0, 1), ("B: Extract", 1, 4), ("C", 4, 5),
                   ("D: Answer", 5, 7), ("F", 7, 8), ("G", 8, 9), ("H", 9, 11)]
    model_names = [r["model"] for r in sorted_rows]
    n = len(model_names); nc = len(task_keys)
    matrix = np.array([[r[k] for k in task_keys] for r in sorted_rows])
    speed_vals = [r["tok_s"] for r in sorted_rows]
    composite_vals = [r["composite"] for r in sorted_rows]
    components = np.array([[
        0.15 * r["parse_em"], 0.15 * r["extract_f1"], 0.15 * r["onto_f1"],
        0.20 * (r["cite_recall"] + r["cite_prec"]) / 2,
        0.05 * min(r["tok_s"] / 150, 1.0),
        0.10 * r["relevance_f1"], 0.10 * r["domain_acc"],
        0.10 * (r["org_acc"] + r["mod_f1"]) / 2,
    ] for r in sorted_rows])
    cl = ["Parse .15", "Extract .15", "Onto .15", "Answer .20",
          "Speed .05", "Relev .10", "Domain .10", "Org+Mod .10"]
    cc = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336",
          "#00BCD4", "#795548", "#607D8B"]
    row_h, brk_h, ttl_h, cb_h = 0.38, 1.2, 0.6, 0.5
    fig_h = ttl_h + brk_h + n * row_h + cb_h; fig_w = 24
    fig = plt.figure(figsize=(fig_w, fig_h))
    tf = ttl_h / fig_h; bf = brk_h / fig_h; cf = cb_h / fig_h
    hf = (n * row_h) / fig_h; hb = cf; ht = hb + hf
    ax_t = fig.add_axes([0, 1 - tf, 1, tf]); ax_t.axis("off")
    ax_t.text(0.5, 0.4, f"Domain-Specific LLM Performance & Composite "
              f"Breakdown ({n} configs, 8 tasks)",
              ha="center", va="center", fontsize=22, fontweight="bold")
    # Heatmap
    hl = 0.10; hw = 0.32
    ax = fig.add_axes([hl, hb, hw, hf])
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(nc))
    ax.set_xticklabels(task_labels, fontsize=13, rotation=55, ha="left")
    ax.xaxis.set_ticks_position("top"); ax.tick_params(axis="x", pad=1, length=2)
    ax.set_yticks(range(n))
    # Color model names by family for visual grouping
    ax.set_yticklabels(model_names, fontsize=12)
    for i, label in enumerate(ax.get_yticklabels()):
        fam = sorted_rows[i].get("family", "")
        label.set_color(FAMILY_COLORS.get(fam, "#333333"))
        if sorted_rows[i].get("think"):
            label.set_style("italic")
    for i in range(n):
        for j in range(nc):
            v = matrix[i, j]
            # Adaptive threshold: white text on dark cells
            c = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color=c)
    ab = fig.add_axes([hl, ht, hw, bf])
    ab.set_xlim(-0.5, nc - 0.5); ab.set_ylim(0, 1); ab.axis("off")
    for gl, gs, ge in task_groups:
        ab.text((gs + ge - 1) / 2, 0.88, gl, ha="center", va="center",
                fontsize=13, color="#333333", fontweight="bold")
        ab.plot([gs - 0.3, ge - 0.7], [0.72, 0.72], color="#666666", lw=1.5, clip_on=False)
    cb = fig.add_axes([hl, hb - cf * 0.5, hw, cf * 0.15])
    plt.colorbar(im, cax=cb, orientation="horizontal")
    cb.set_xlabel("Score (0–1)", fontsize=12); cb.tick_params(labelsize=10)
    # Speed + Composite
    sl = hl + hw + 0.004; sw = 0.020
    axs = fig.add_axes([sl, hb, sw, hf])
    sa = np.array(speed_vals).reshape(-1, 1)
    ms = max(speed_vals) if max(speed_vals) > 0 else 1
    axs.imshow(sa, cmap="Blues", aspect="auto", vmin=0, vmax=ms * 1.1)
    axs.set_xticks([0]); axs.set_xticklabels(["Tk/s"], fontsize=13)
    axs.xaxis.set_ticks_position("top"); axs.set_yticks([])
    for i, v in enumerate(speed_vals):
        axs.text(0, i, f"{v:.0f}", ha="center", va="center",
                 fontsize=10, color="white" if v > ms * 0.55 else "black")
    cpl = sl + sw + 0.003; cpw = 0.020
    axc = fig.add_axes([cpl, hb, cpw, hf])
    axc.imshow(np.array(composite_vals).reshape(-1, 1), cmap="RdYlGn",
               aspect="auto", vmin=0, vmax=1)
    axc.set_xticks([0]); axc.set_xticklabels(["Comp"], fontsize=13)
    axc.xaxis.set_ticks_position("top"); axc.set_yticks([])
    for i, v in enumerate(composite_vals):
        axc.text(0, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=10, color="white" if v > 0.55 else "black")
    # Stacked bars
    bl = cpl + cpw + 0.03; bw = 1.0 - bl - 0.005
    axb = fig.add_axes([bl, hb, bw, hf])
    yp = np.arange(n); la = np.zeros(n)
    for j in range(8):
        axb.barh(yp, components[:, j], 0.75, left=la, label=cl[j], color=cc[j], alpha=0.85)
        for i in range(n):
            sw2 = components[i, j]
            if sw2 > 0.035:
                axb.text(la[i] + sw2 / 2, i, f"{sw2:.2f}", ha="center", va="center",
                         fontsize=9, color="white")
        la += components[:, j]
    for i in range(n):
        axb.text(la[i] + 0.005, i, f"{la[i]:.3f}", ha="left", va="center",
                 fontsize=11, color="#333333")
    axb.set_yticks([]); axb.set_ylim(-0.5, n - 0.5)
    max_total = max(la)
    axb.set_xlabel("Composite Score", fontsize=14); axb.set_xlim(0, max_total + 0.08)
    axb.tick_params(labelsize=12); axb.invert_yaxis()
    axb.grid(True, alpha=0.15, axis="x")
    al = fig.add_axes([bl, ht, bw, bf]); al.axis("off")
    from matplotlib.patches import Patch
    al.legend(handles=[Patch(facecolor=cc[j], label=cl[j], alpha=0.85) for j in range(8)],
              fontsize=12, ncol=4, loc="center", frameon=False)
    # Add family legend at bottom
    from matplotlib.patches import Patch as FamPatch
    fam_in_data = sorted({r["family"] for r in sorted_rows if r["family"] in FAMILY_COLORS})
    fam_patches = [FamPatch(facecolor=FAMILY_COLORS[f], label=f.capitalize()) for f in fam_in_data]
    fig.legend(handles=fam_patches, fontsize=10, ncol=min(len(fam_patches), 8),
               loc="lower center", frameon=False, bbox_to_anchor=(0.25, 0.0))
    fig.savefig(OUT_DIR / "fig4_llm_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_llm_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 4: LLM domain heatmap + composite saved ({n} model configs)")

    # Think-mode delta table: compare +think vs base for all ablation pairs
    _generate_think_delta_table(rows)


def _generate_think_delta_table(rows):
    """Generate CSV showing think-mode impact (delta) for each ablation pair."""
    by_label = {r["model"]: r for r in rows}
    delta_metrics = ["parse_em", "extract_f1", "onto_f1", "cite_recall",
                     "relevance_f1", "domain_acc", "org_acc", "mod_f1", "composite"]
    lines = ["Model,Family,Size(B)," + ",".join(f"Δ_{m}" for m in delta_metrics)]

    for label, r in sorted(by_label.items()):
        if not label.endswith("+think"):
            continue
        base_label = label.removesuffix("+think")
        base = by_label.get(base_label)
        if not base:
            continue
        deltas = []
        for m in delta_metrics:
            d = r.get(m, 0) - base.get(m, 0)
            deltas.append(f"{d:+.4f}")
        lines.append(f"{base_label},{r['family']},{r['size_b']}," + ",".join(deltas))

    if len(lines) > 1:
        out = OUT_DIR / "table_s3_think_delta.csv"
        out.write_text("\n".join(lines) + "\n")
        print(f"Table S3: Think-mode deltas saved ({len(lines)-1} pairs)")
    else:
        print("Table S3: SKIPPED — no think ablation pairs found")


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

    fig_w, fig_h = 17, 6.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1, 4,
        left=0.06, right=0.98, top=0.78, bottom=0.12,
        width_ratios=[1.0, 1.0, 1.0, 0.88], wspace=0.34,
    )
    fig.suptitle(
        "Figure 5: Context Window Optimisation (k × format, 15 queries)",
        fontsize=20, y=0.97,
    )

    colors = {"full": "#2196F3", "structured": "#FF9800", "minimal": "#4CAF50"}
    markers = {"full": "o", "structured": "s", "minimal": "^"}

    axes_specs = [
        ("Cite Recall vs. k", "recall", "Score"),
        ("Cite Precision vs. k", "precision", "Score"),
        ("Grounding vs. k", "grounding", "Score"),
        ("Context Tokens vs. k", "tokens", "Tokens"),
    ]
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for ax, (title, metric_key, ylabel) in zip(axes, axes_specs):
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

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12.5, loc="upper center",
               ncol=len(handles), bbox_to_anchor=(0.5, 0.91), frameon=False)

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
    wrapped_names = [_wrap_config_label(name) for name in names]
    n = len(names)

    fig_w, fig_h = 16, 10.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2, 2,
        left=0.05, right=0.98, top=0.88, bottom=0.06,
        height_ratios=[1.0, 0.90], width_ratios=[1.65, 0.95],
        hspace=0.26, wspace=0.20,
    )
    fig.suptitle(
        f"Figure 6: End-to-End Pipeline Comparison ({n} configurations, 18 queries)",
        fontsize=20, y=0.96,
    )

    # ── Top-left: quality metrics grouped bars ──
    ax_qual = fig.add_subplot(gs[0, 0])
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
    ax_qual.set_xticklabels(wrapped_names, fontsize=13)
    ax_qual.set_ylabel("Score", fontsize=15)
    ax_qual.set_title("Quality Metrics", fontsize=16)
    ax_qual.tick_params(labelsize=13)
    max_score = max(c[mk] for c in configs for _, mk, _ in qual_metrics if c[mk] is not None)
    ax_qual.set_ylim(0, min(1.15, max_score + 0.08))
    ax_qual.legend(fontsize=12, ncol=2, loc="upper left", frameon=False)
    ax_qual.grid(True, alpha=0.2, axis="y")

    # ── Top-right: latency ──
    ax_lat = fig.add_subplot(gs[0, 1])
    lat_vals = [c["latency_s"] for c in configs]
    max_lat = max(lat_vals) if max(lat_vals) > 0 else 1
    bar_colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
    bars = ax_lat.barh(np.arange(n), lat_vals, 0.65,
                       color=bar_colors[:n], alpha=0.85)
    ax_lat.set_yticks(np.arange(n))
    ax_lat.set_yticklabels(wrapped_names, fontsize=12)
    ax_lat.set_xlabel("Seconds", fontsize=14)
    ax_lat.set_title("End-to-End Latency", fontsize=15)
    ax_lat.tick_params(axis="x", labelsize=12)
    ax_lat.set_xlim(0, max_lat * 1.18)
    ax_lat.grid(True, alpha=0.2, axis="x")
    ax_lat.invert_yaxis()
    for bar, v in zip(bars, lat_vals):
        ax_lat.text(v + max_lat * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}s", va="center", fontsize=12)

    # ── Bottom: config details table ──
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    ax_table.text(0.0, 1.0, "Pipeline Configuration Details:",
                  fontsize=14, va="top", transform=ax_table.transAxes)

    col_headers = ["Config", "Embedding", "Strategy", "LLM", "Parse Model",
                   "Context Fmt", "Context k"]
    table_rows = []
    for c in configs:
        cfg = c["config"]
        table_rows.append([
            c["name"],
            cfg.get("embedding", "?"),
            cfg.get("strategy", "?"),
            cfg.get("llm", "?"),
            str(cfg.get("parse_model", "None")),
            cfg.get("context_format", "?"),
            str(cfg.get("context_k", "?")),
        ])

    table = ax_table.table(
        cellText=table_rows,
        colLabels=col_headers,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        bbox=[0.0, 0.0, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.8 if row == 0 else 0.5)
        if row == 0:
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(color="#333333", fontweight="semibold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#FAFAFA")

    fig.savefig(OUT_DIR / "fig6_e2e_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_e2e_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 6: E2E comparison saved ({n} configs)")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Composite summary (stacked breakdown)
# ═══════════════════════════════════════════════════════════════════════════
def figure_7_public_benchmarks():
    """Heatmap of public benchmark results: models × individual datasets, grouped by category."""
    pub_path = RESULTS_DIR / "public_bench.json"
    if not pub_path.exists():
        print("Figure 7: SKIPPED — public_bench.json not found")
        return
    pub_data = load_json(pub_path)
    if not pub_data:
        print("Figure 7: SKIPPED — no public benchmark results")
        return

    # Dataset columns grouped by category
    dataset_groups = [
        ("General", ["mmlu", "hellaswag", "winogrande", "arc_challenge"]),
        ("Reasoning", ["arc_easy", "gsm8k", "truthfulqa"]),
        ("Biomedical", ["pubmedqa", "medqa", "medmcqa", "sciq", "bioasq_mini",
                        "mmlu_anatomy", "mmlu_clinical_knowledge",
                        "mmlu_college_biology", "mmlu_college_medicine",
                        "mmlu_medical_genetics", "mmlu_professional_medicine"]),
        ("Structured", ["ifeval", "json_mode_eval"]),
        ("Tool-Use", ["nexus_fc", "glaive_fc", "toolace"]),
        ("Commonsense", ["siqa", "openbookqa", "boolq"]),
        ("Other", ["squad_v2"]),
    ]

    # Short labels for column headers
    short_labels = {
        "mmlu": "MMLU", "hellaswag": "Hella", "winogrande": "Wino",
        "arc_challenge": "ARC-C", "arc_easy": "ARC-E", "gsm8k": "GSM8K",
        "truthfulqa": "TQA", "pubmedqa": "PubMd", "medqa": "MedQA",
        "medmcqa": "MedMC", "sciq": "SciQ", "bioasq_mini": "BioAQ",
        "mmlu_anatomy": "Anat", "mmlu_clinical_knowledge": "ClinK",
        "mmlu_college_biology": "ColBi", "mmlu_college_medicine": "ColMd",
        "mmlu_medical_genetics": "MedGn", "mmlu_professional_medicine": "ProMd",
        "ifeval": "IFEval", "json_mode_eval": "JSON",
        "nexus_fc": "NexFC", "glaive_fc": "GlaFC", "toolace": "ToolA",
        "siqa": "SIQA", "openbookqa": "OBQA", "boolq": "BoolQ",
        "squad_v2": "SQuAD",
    }

    # Flatten dataset list and build group bracket info
    all_ds = []
    group_brackets = []
    for gname, ds_list in dataset_groups:
        start = len(all_ds)
        all_ds.extend(ds_list)
        group_brackets.append((gname, start, len(all_ds)))

    # Sort models by average score across available datasets
    model_keys = list(pub_data.keys())
    model_avgs = []
    for mk in model_keys:
        scores = []
        for ds in all_ds:
            entry = pub_data[mk].get(ds, {})
            acc = entry.get("accuracy", entry.get("avg_score", np.nan))
            if not np.isnan(acc):
                scores.append(acc)
        model_avgs.append((mk, np.mean(scores) if scores else 0))
    model_avgs.sort(key=lambda x: -x[1])
    sorted_models = [m[0] for m in model_avgs]

    n = len(sorted_models)
    nd = len(all_ds)

    # Build matrix
    matrix = np.full((n, nd), np.nan)
    for i, mk in enumerate(sorted_models):
        for j, ds in enumerate(all_ds):
            entry = pub_data[mk].get(ds, {})
            matrix[i, j] = entry.get("accuracy", entry.get("avg_score", np.nan))

    # Category composite column
    cat_keys = ["general", "reasoning", "biomedical", "structured",
                "tool_use", "commonsense"]
    cat_labels = ["Gen", "Reas", "Bio", "Str", "Tool", "Com"]
    cat_matrix = np.full((n, len(cat_keys)), np.nan)
    for i, mk in enumerate(sorted_models):
        composites = pub_data[mk].get("composites", {})
        for j, ck in enumerate(cat_keys):
            cat_matrix[i, j] = composites.get(ck, np.nan)

    # Layout
    row_h = 0.42
    bracket_h = 1.2
    title_h = 0.6
    cbar_h = 0.5
    fig_h = title_h + bracket_h + n * row_h + cbar_h
    fig_w = max(16, nd * 0.50 + 6)
    fig = plt.figure(figsize=(fig_w, fig_h))

    tf = title_h / fig_h; bf = bracket_h / fig_h
    cf = cbar_h / fig_h; hf = (n * row_h) / fig_h
    hb = cf; ht = hb + hf

    ax_t = fig.add_axes([0, 1 - tf, 1, tf]); ax_t.axis("off")
    ax_t.text(0.5, 0.4,
              f"Public Benchmark Performance ({n} models, {nd} datasets)",
              ha="center", va="center", fontsize=20, fontweight="bold")

    # Main heatmap
    hl = 0.10; hw = 0.60
    ax = fig.add_axes([hl, hb, hw, hf])
    disp = np.where(np.isnan(matrix), -0.1, matrix)
    im = ax.imshow(disp, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    col_labels = [short_labels.get(ds, ds[:6]) for ds in all_ds]
    ax.set_xticks(range(nd))
    ax.set_xticklabels(col_labels, fontsize=12, rotation=65, ha="left")
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", pad=1, length=2)
    ax.set_yticks(range(n))

    # Model labels colored by family for visual grouping
    ax.set_yticklabels(sorted_models, fontsize=13)
    for i, label in enumerate(ax.get_yticklabels()):
        fam = pub_data[sorted_models[i]].get("family", "")
        label.set_color(FAMILY_COLORS.get(fam, "#333333"))

    for i in range(n):
        for j in range(nd):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="#AAAAAA")
            else:
                c = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color=c)

    # Brackets
    ab = fig.add_axes([hl, ht, hw, bf])
    ab.set_xlim(-0.5, nd - 0.5); ab.set_ylim(0, 1); ab.axis("off")
    for gname, gs, ge in group_brackets:
        mid = (gs + ge - 1) / 2
        ab.text(mid, 0.88, gname, ha="center", va="center",
                fontsize=13, color="#333333", fontweight="bold")
        ab.plot([gs - 0.3, ge - 0.7], [0.72, 0.72], color="#666666",
                linewidth=1.5, clip_on=False)

    # Colorbar
    cbar_ax = fig.add_axes([hl, hb - cf * 0.5, hw, cf * 0.15])
    plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel("Accuracy (0–1)", fontsize=14)
    cbar_ax.tick_params(labelsize=12)

    # Category summary columns
    cat_left = hl + hw + 0.015; cat_w = 0.14
    axc = fig.add_axes([cat_left, hb, cat_w, hf])
    cat_disp = np.where(np.isnan(cat_matrix), -0.1, cat_matrix)
    axc.imshow(cat_disp, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    axc.set_xticks(range(len(cat_labels)))
    axc.set_xticklabels(cat_labels, fontsize=12, rotation=55, ha="left")
    axc.xaxis.set_ticks_position("top")
    axc.tick_params(axis="x", pad=1, length=2)
    axc.set_yticks([])
    for i in range(n):
        for j in range(len(cat_keys)):
            v = cat_matrix[i, j]
            if np.isnan(v):
                axc.text(j, i, "—", ha="center", va="center",
                         fontsize=10, color="#AAAAAA")
            else:
                c = "white" if v > 0.55 else "black"
                axc.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=10, color=c)

    # Label above category columns
    ac = fig.add_axes([cat_left, ht, cat_w, bf])
    ac.set_xlim(-0.5, len(cat_keys) - 0.5); ac.set_ylim(0, 1); ac.axis("off")
    ac.text((len(cat_keys) - 1) / 2, 0.55, "Category Avg",
            ha="center", va="center", fontsize=13,
            color="#333333", fontweight="bold")

    fig.savefig(OUT_DIR / "fig7_public_benchmarks.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig7_public_benchmarks.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 7: Public benchmarks saved ({n} models, {nd} datasets)")


# ═══════════════════════════════════════════════════════════════════════════
# Figure S1: Context management strategies comparison
# ═══════════════════════════════════════════════════════════════════════════
def figure_s1_context_management():
    """Grouped bar chart: 15 context management strategies across 5 categories."""
    path = RESULTS_DIR / "context_management_bench.json"
    if not path.exists():
        print("Figure S1: SKIPPED — context_management_bench.json not found")
        return
    data = load_json(path)
    summary = data.get("summary", {})
    if not summary:
        print("Figure S1: SKIPPED — no summary data")
        return

    # Group strategies by category (prefix before first _)
    categories = {}
    for s, v in sorted(summary.items()):
        parts = s.split("_", 1)
        cat = parts[0].capitalize()
        label = parts[1] if len(parts) > 1 else s
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((label, v))

    metrics = [
        ("avg_grounding_rate", "Grounding", "#4CAF50"),
        ("avg_citation_precision", "Cite Precision", "#2196F3"),
        ("avg_citation_recall", "Cite Recall", "#FF9800"),
    ]

    # Two rows: 3 per row
    cat_list = list(categories.items())
    n_cats = len(cat_list)
    ncols = 3
    nrows = (n_cats + ncols - 1) // ncols
    fig, all_axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.5 * nrows),
                                 sharey=True)
    all_axes = all_axes.flatten() if nrows > 1 else (
        all_axes if n_cats > 1 else [all_axes])

    fig.suptitle("Context Management Strategy Comparison (15 strategies)",
                 fontsize=20, fontweight="bold", y=0.98)

    for ax_idx in range(nrows * ncols):
        ax = all_axes[ax_idx]
        if ax_idx >= n_cats:
            ax.axis("off")
            continue
        cat, strats = cat_list[ax_idx]
        n_strats = len(strats)
        n_metrics = len(metrics)
        bar_w = 0.25
        x = np.arange(n_strats)

        for m_idx, (mkey, mlabel, mcolor) in enumerate(metrics):
            vals = [s[1].get(mkey, 0) for s in strats]
            offset = (m_idx - n_metrics / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=mlabel,
                          color=mcolor, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=9, rotation=90)

        labels = [s[0].replace("_", "\n") for s in strats]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, ha="center")
        ax.set_title(cat, fontsize=16, fontweight="bold")
        ax.set_ylim(0, 1.18)
        ax.grid(True, alpha=0.15, axis="y")
        ax.tick_params(labelsize=11)
        if ax_idx % ncols == 0:
            ax.set_ylabel("Score", fontsize=14)
        if ax_idx == 0:
            ax.legend(fontsize=11, loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig_s1_context_management.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_s1_context_management.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"Figure S1: Context management saved ({len(summary)} strategies)")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# Figure S2 — Runtime Ablation Studies
# ═══════════════════════════════════════════════════════════════════════════
def figure_s2_ablation():
    path = RESULTS_DIR / "ablation_bench.json"
    if not path.exists():
        print("Fig S2 (ablation): SKIPPED — no ablation_bench.json")
        return

    with open(path) as f:
        abl = json.load(f)

    kv_data = abl.get("kv_cache", {})
    ctx_data = abl.get("context_length", {})
    if not kv_data and not ctx_data:
        print("Fig S2 (ablation): SKIPPED — empty data")
        return

    MODELS = ["qwen3-8b", "llama3.1-8b", "phi4-14b-q8", "gemma3-12b-q8", "qwen3.5-9b-q8"]
    MODEL_SHORT = ["Qwen3-8B", "Llama3.1-8B", "Phi4-14B", "Gemma3-12B", "Qwen3.5-9B"]
    KV_TYPES = ["f16", "q8_0", "q4_0"]
    KV_COLORS = ["#4393c3", "#74c476", "#fd8d3c"]
    CTX_LENGTHS = [2048, 4096, 8192, 16384]
    CTX_COLORS = ["#9ecae1", "#4292c6", "#08519c", "#08306b"]
    CTX_MARKERS = ["o", "s", "^", "D"]

    # ── Layout: 2 rows
    # Row 1: KV cache ablation — 3 metric sub-panels (A, D, F) side by side + VRAM
    # Row 2: Context length ablation — speed (tok/s) and answer quality (D)
    fig_w, fig_h = 22, 12
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Row positions (bottom of row from top)
    row1_b = 0.55   # KV cache row
    row2_b = 0.06   # Context row
    row_h = 0.36

    # ── ROW 1: KV cache ──
    # 4 sub-panels: Task A, Task D, Task F, VRAM
    kv_metrics = [
        ("task_a", "exact_match", "Parse Accuracy (Task A)"),
        ("task_d", "cite_recall",  "Citation Recall (Task D)"),
        ("task_f", "accuracy",     "Relevance Accuracy (Task F)"),
    ]
    n_m = len(MODELS)
    bar_w = 0.22
    group_w = bar_w * len(KV_TYPES) + 0.15
    total_w = n_m * group_w

    panel_w = 0.19
    panel_gap = 0.025
    vram_w = 0.16
    left_margin = 0.06

    for pi, (task_key, metric_key, panel_title) in enumerate(kv_metrics):
        pl = left_margin + pi * (panel_w + panel_gap)
        ax = fig.add_axes([pl, row1_b, panel_w, row_h])

        for mi, mk in enumerate(MODELS):
            vals = []
            for kv in KV_TYPES:
                key = f"{mk}@kv={kv}"
                v = kv_data.get(key, {}).get(task_key, {}).get(metric_key, np.nan)
                vals.append(float(v) if v is not None else np.nan)
            xs = [mi * group_w + ki * bar_w for ki in range(len(KV_TYPES))]
            for xi, (v, c) in enumerate(zip(vals, KV_COLORS)):
                ax.bar(xs[xi], v, width=bar_w * 0.9, color=c, edgecolor="white", lw=0.5)

        ax.set_xticks([mi * group_w + bar_w for mi in range(n_m)])
        ax.set_xticklabels(MODEL_SHORT, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=6)
        ax.yaxis.grid(True, alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # VRAM panel
    vram_l = left_margin + len(kv_metrics) * (panel_w + panel_gap)
    ax_v = fig.add_axes([vram_l, row1_b, vram_w, row_h])
    for mi, mk in enumerate(MODELS):
        vrams = []
        for kv in KV_TYPES:
            key = f"{mk}@kv={kv}"
            v = kv_data.get(key, {}).get("vram_mb", np.nan)
            vrams.append(float(v) / 1024 if v else np.nan)  # convert to GB
        xs = [mi * group_w + ki * bar_w for ki in range(len(KV_TYPES))]
        for xi, (v, c) in enumerate(zip(vrams, KV_COLORS)):
            ax_v.bar(xs[xi], v, width=bar_w * 0.9, color=c, edgecolor="white", lw=0.5, alpha=0.85)
    ax_v.set_xticks([mi * group_w + bar_w for mi in range(n_m)])
    ax_v.set_xticklabels(MODEL_SHORT, rotation=35, ha="right", fontsize=9)
    ax_v.set_ylabel("VRAM (GB)", fontsize=10)
    ax_v.set_title("VRAM Usage", fontsize=11, fontweight="bold", pad=6)
    ax_v.yaxis.grid(True, alpha=0.3, lw=0.5)
    ax_v.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_v.spines[spine].set_visible(False)

    # KV legend
    handles_kv = [mpatches.Patch(color=c, label=kv) for c, kv in zip(KV_COLORS, KV_TYPES)]
    fig.legend(handles=handles_kv, loc="upper center",
               bbox_to_anchor=(0.50, 0.98), ncol=3, fontsize=10,
               title="KV Cache Type", title_fontsize=10, frameon=False)

    # Row 1 section label
    fig.text(0.02, row1_b + row_h / 2, "A: KV Cache\nAblation",
             va="center", ha="center", fontsize=11, fontweight="bold",
             rotation=90, color="#333333")

    # ── ROW 2: Context length ──
    # Left: tok/s vs ctx, Right: Task D vs ctx
    ctx_panel_w = 0.38
    ctx_gap = 0.08
    ctx_l1 = left_margin + 0.05
    ctx_l2 = ctx_l1 + ctx_panel_w + ctx_gap

    ax_spd = fig.add_axes([ctx_l1, row2_b, ctx_panel_w, row_h])
    ax_qa  = fig.add_axes([ctx_l2, row2_b, ctx_panel_w, row_h])

    for mi, (mk, mshort) in enumerate(zip(MODELS, MODEL_SHORT)):
        spd_vals, qa_vals = [], []
        for ctx in CTX_LENGTHS:
            key = f"{mk}@ctx={ctx}"
            entry = ctx_data.get(key, {})
            spd = entry.get("task_e", {}).get("tok_per_sec", np.nan)
            qa  = entry.get("task_d", {}).get("cite_recall", np.nan)
            spd_vals.append(float(spd) if spd and spd == spd else np.nan)
            qa_vals.append(float(qa)  if qa  and qa  == qa  else np.nan)

        # Only plot ctx lengths that have data
        xs_spd = [CTX_LENGTHS[i] for i, v in enumerate(spd_vals) if not np.isnan(v)]
        ys_spd = [v for v in spd_vals if not np.isnan(v)]
        xs_qa  = [CTX_LENGTHS[i] for i, v in enumerate(qa_vals) if not np.isnan(v)]
        ys_qa  = [v for v in qa_vals if not np.isnan(v)]

        col = f"C{mi}"
        if xs_spd:
            ax_spd.plot(xs_spd, ys_spd, marker="o", color=col, lw=1.8, ms=6, label=mshort)
        if xs_qa:
            ax_qa.plot(xs_qa, ys_qa, marker="o", color=col, lw=1.8, ms=6, label=mshort)

    for ax, title, ylabel in [
        (ax_spd, "Throughput vs Context Length (Task E)", "Tokens / Second"),
        (ax_qa,  "Answer Quality vs Context Length (Task D)", "Citation Recall"),
    ]:
        ax.set_xlabel("Context Length (tokens)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks(CTX_LENGTHS)
        ax.set_xticklabels([str(c) for c in CTX_LENGTHS], fontsize=9)
        ax.yaxis.grid(True, alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best")

    ax_qa.set_ylim(0, 1.08)

    # Row 2 section label
    fig.text(0.02, row2_b + row_h / 2, "B: Context\nLength\nAblation",
             va="center", ha="center", fontsize=11, fontweight="bold",
             rotation=90, color="#333333")

    fig.suptitle("Figure S2 — Runtime Ablation Studies\n"
                 "(KV cache quantization and context length effects on quality, speed, and memory)",
                 y=0.995, fontsize=13, fontweight="bold", va="top")

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_s2_ablation.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Fig S2 (ablation): saved → {OUT_DIR}/fig_s2_ablation.pdf")


# Tables 3, 5, S2
# ═══════════════════════════════════════════════════════════════════════════
def table_3_llm_comparison(rows):
    if not rows:
        print("Table 3: SKIPPED")
        return
    with open(OUT_DIR / "table3_llm_comparison.csv", "w") as f:
        f.write("Model,Family,Size(B),Quant,Think,Parse_EM,Tissue_F1,Disease_F1,"
                "CellType_F1,Extract_F1_avg,Onto_F1,Cite_Recall,Cite_Prec,"
                "Grounding,Relev_F1,Domain_Acc,Org_Acc,Mod_F1,Tok/s,Composite\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['family']},{r['size_b']},{r['quant']},"
                f"{r['think']},{r['parse_em']:.4f},{r['tissue_f1']:.4f},"
                f"{r['disease_f1']:.4f},{r['celltype_f1']:.4f},{r['extract_f1']:.4f},"
                f"{r['onto_f1']:.4f},{r['cite_recall']:.4f},{r['cite_prec']:.4f},"
                f"{r['grounding']:.4f},{r['relevance_f1']:.4f},{r['domain_acc']:.4f},"
                f"{r['org_acc']:.4f},{r['mod_f1']:.4f},{r['tok_s']:.1f},"
                f"{r['composite']:.4f}\n"
            )
    print(f"Table 3: {len(rows)} model configurations")


def table_5_top5(rows):
    if not rows:
        print("Table 5: SKIPPED")
        return
    top5 = sorted(rows, key=lambda r: -r["composite"])[:5]
    with open(OUT_DIR / "table5_top5.csv", "w") as f:
        f.write("Rank,Model,Family,Size(B),Composite,Parse_EM,Extract_F1,"
                "Onto_F1,Cite_Recall,Cite_Prec,Relev_F1,Domain_Acc,Org_Acc,Tok/s\n")
        for i, r in enumerate(top5, 1):
            f.write(
                f"{i},{r['model']},{r['family']},{r['size_b']},"
                f"{r['composite']:.4f},{r['parse_em']:.4f},{r['extract_f1']:.4f},"
                f"{r['onto_f1']:.4f},{r['cite_recall']:.4f},{r['cite_prec']:.4f},"
                f"{r['relevance_f1']:.4f},{r['domain_acc']:.4f},{r['org_acc']:.4f},"
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
    figure_s1_context_management()
    table_s2_context_management()
    print()

    # Figure 6 (end-to-end)
    figure_6_e2e_comparison()
    print()

    # Figure 7 (public benchmarks)
    figure_7_public_benchmarks()
    print()

    # Figure S2 (ablation)
    figure_s2_ablation()
    print()

    # Summary JSON
    write_summary_json(corpus_stats, query_dist, llm_rows)

    print()
    print("=" * 60)
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
