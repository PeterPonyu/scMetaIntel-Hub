#!/usr/bin/env python3
"""Generate evaluation queries backed by real GSE ground truth from the corpus."""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "ground_truth"
OUT_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "eval_queries.json"


def load_corpus():
    docs = {}
    for fn in sorted(os.listdir(GT_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(GT_DIR / fn) as f:
            d = json.load(f)
            docs[d["gse_id"]] = d
    return docs


def build_indices(docs):
    org_gse = defaultdict(set)
    domain_gse = defaultdict(set)
    for gse, d in docs.items():
        org = d.get("organism", "")
        if org:
            org_gse[org].add(gse)
        dom = d.get("domain", "")
        if dom:
            domain_gse[dom].add(gse)
    return org_gse, domain_gse


def text_search(docs, org_gse, domain_gse, keywords, organism=None, domain=None, min_match=1, limit=5):
    candidates = set(docs.keys())
    if organism:
        candidates &= org_gse.get(organism, set())
    if domain:
        candidates &= domain_gse.get(domain, set())
    scored = []
    for gse in candidates:
        d = docs[gse]
        text = (d.get("title", "") + " " + d.get("summary", "")).lower()
        hits = sum(1 for kw in keywords if kw.lower() in text)
        if hits >= min_match:
            scored.append((gse, hits))
    scored.sort(key=lambda x: -x[1])
    return [s[0] for s in scored[:limit]]


def main():
    docs = load_corpus()
    org_gse, domain_gse = build_indices(docs)
    ts = lambda kw, **kv: text_search(docs, org_gse, domain_gse, kw, **kv)

    queries = []
    qid = 0

    # ===== Category 1: Basic tissue queries (easy) =====
    tissue_specs = [
        ("human brain scRNA-seq", ["brain", "neuron", "cortex"], "Homo sapiens", None, "brain"),
        ("human lung single-cell RNA-seq", ["lung", "pulmonary", "airway"], "Homo sapiens", None, "lung"),
        ("human liver single-cell transcriptomics", ["liver", "hepat"], "Homo sapiens", None, "liver"),
        ("human kidney scRNA-seq", ["kidney", "renal", "nephron"], "Homo sapiens", None, "kidney"),
        ("human heart single-cell sequencing", ["heart", "cardiac", "cardiomyocyte"], "Homo sapiens", None, "heart"),
        ("human pancreas single-cell RNA-seq", ["pancreas", "pancreatic", "islet"], "Homo sapiens", None, "pancreas"),
        ("human retina single-cell sequencing", ["retina", "retinal"], "Homo sapiens", None, "retina"),
        ("human skin single-cell RNA-seq", ["skin", "dermal", "epiderm"], "Homo sapiens", None, "skin"),
        ("human colon single-cell transcriptomics", ["colon", "colonic", "intestin"], "Homo sapiens", None, "colon"),
        ("human blood single-cell analysis", ["blood", "pbmc", "peripheral blood"], "Homo sapiens", None, "blood"),
        ("mouse brain scRNA-seq", ["brain", "neuron", "cortex"], "Mus musculus", None, "brain"),
        ("mouse lung single-cell RNA-seq", ["lung", "pulmonary"], "Mus musculus", None, "lung"),
        ("mouse liver single-cell RNA-seq", ["liver", "hepat"], "Mus musculus", None, "liver"),
        ("mouse kidney scRNA-seq datasets", ["kidney", "renal"], "Mus musculus", None, "kidney"),
        ("mouse spleen single-cell sequencing", ["spleen", "splenic"], "Mus musculus", None, "spleen"),
        ("mouse bone marrow scRNA-seq", ["bone marrow", "hematopoiet"], "Mus musculus", None, "bone marrow"),
    ]

    for query_text, keywords, organism, domain, tissue in tissue_specs:
        gses = ts(keywords, organism=organism, domain=domain, limit=3)
        if gses:
            qid += 1
            queries.append({
                "id": f"q{qid:02d}",
                "query": query_text,
                "expected_constraints": {"organism": organism, "tissue": tissue},
                "expected_gse": gses,
                "category": "basic",
                "difficulty": "easy",
            })

    # ===== Category 2: Disease-focused queries (medium) =====
    disease_specs = [
        ("lung cancer single-cell RNA-seq", ["lung", "cancer"], "Homo sapiens", "cancer", "cancer"),
        ("breast cancer scRNA-seq", ["breast", "cancer"], "Homo sapiens", "cancer", "cancer"),
        ("colorectal cancer single-cell transcriptomics", ["colon", "colorectal", "cancer"], "Homo sapiens", "cancer", "cancer"),
        ("pancreatic cancer scRNA-seq", ["pancrea", "cancer"], "Homo sapiens", "cancer", "cancer"),
        ("glioblastoma single-cell sequencing", ["glioblastoma", "glioma"], "Homo sapiens", "cancer", "glioblastoma"),
        ("melanoma single-cell RNA-seq", ["melanoma"], "Homo sapiens", "cancer", "melanoma"),
        ("leukemia scRNA-seq", ["leukemia"], "Homo sapiens", "cancer", "leukemia"),
        ("hepatocellular carcinoma scRNA-seq", ["hepatocellular", "liver cancer"], "Homo sapiens", "cancer", "hepatocellular carcinoma"),
        ("ovarian cancer single-cell sequencing", ["ovarian", "ovary", "cancer"], "Homo sapiens", "cancer", "ovarian cancer"),
        ("prostate cancer scRNA-seq", ["prostate", "cancer"], "Homo sapiens", "cancer", "prostate cancer"),
        ("COVID-19 single-cell transcriptomics", ["covid", "sars-cov"], "Homo sapiens", None, "COVID-19"),
        ("Alzheimer disease single-cell RNA-seq", ["alzheimer"], None, None, "Alzheimer disease"),
        ("liver fibrosis single-cell analysis", ["liver", "fibrosis"], None, None, "liver fibrosis"),
        ("kidney disease scRNA-seq", ["kidney", "disease", "nephropathy"], None, None, "kidney disease"),
        ("inflammatory bowel disease scRNA-seq", ["inflammatory bowel", "crohn", "colitis"], None, None, "IBD"),
    ]

    for query_text, keywords, organism, domain, disease in disease_specs:
        gses = ts(keywords, organism=organism, domain=domain, limit=3)
        if gses:
            qid += 1
            constraints = {"disease": disease}
            if organism:
                constraints["organism"] = organism
            queries.append({
                "id": f"q{qid:02d}",
                "query": query_text,
                "expected_constraints": constraints,
                "expected_gse": gses,
                "category": "disease",
                "difficulty": "medium",
            })

    # ===== Category 3: Cell-type focused (medium) =====
    celltype_specs = [
        ("T cell single-cell RNA-seq", ["t cell", "cd4", "cd8"]),
        ("macrophage scRNA-seq", ["macrophage"]),
        ("microglia single-cell transcriptomics", ["microglia"]),
        ("epithelial cell scRNA-seq", ["epithelial"]),
        ("fibroblast single-cell RNA-seq", ["fibroblast"]),
        ("dendritic cell scRNA-seq", ["dendritic"]),
        ("natural killer cell single-cell sequencing", ["natural killer", "nk cell"]),
        ("stem cell single-cell RNA-seq", ["stem cell"]),
        ("endothelial cell scRNA-seq", ["endothelial"]),
        ("B cell single-cell sequencing", ["b cell", "b-cell"]),
        ("astrocyte single-cell RNA-seq", ["astrocyte"]),
        ("neutrophil scRNA-seq", ["neutrophil"]),
    ]

    for query_text, keywords in celltype_specs:
        gses = ts(keywords, limit=3)
        if gses:
            qid += 1
            queries.append({
                "id": f"q{qid:02d}",
                "query": query_text,
                "expected_constraints": {"cell_type": keywords[0]},
                "expected_gse": gses,
                "category": "cell_type",
                "difficulty": "medium",
            })

    # ===== Category 4: Multi-constraint (hard) =====
    multi_specs = [
        ("human lung cancer T cell scRNA-seq", ["lung", "cancer", "t cell"], "Homo sapiens", "cancer"),
        ("mouse brain development single-cell RNA-seq", ["brain", "develop"], "Mus musculus", "development"),
        ("human liver cancer immune microenvironment scRNA-seq", ["liver", "cancer", "immune"], "Homo sapiens", "cancer"),
        ("tumor-infiltrating lymphocytes single-cell RNA-seq", ["tumor", "infiltrat", "lymphocyte"], "Homo sapiens", None),
        ("human bone marrow hematopoiesis scRNA-seq", ["bone marrow", "hematopoie"], "Homo sapiens", None),
        ("mouse intestine development single-cell sequencing", ["intestin", "develop"], "Mus musculus", None),
        ("human PBMC immune profiling scRNA-seq", ["pbmc", "immune"], "Homo sapiens", None),
        ("mouse tumor microenvironment scRNA-seq", ["tumor", "microenvironment"], "Mus musculus", "cancer"),
        ("human brain glioma immune cells scRNA-seq", ["brain", "gliom", "immune"], "Homo sapiens", "cancer"),
        ("mouse lung fibrosis single-cell RNA-seq", ["lung", "fibrosis"], "Mus musculus", None),
    ]

    for query_text, keywords, organism, domain in multi_specs:
        gses = ts(keywords, organism=organism, domain=domain, limit=3)
        if gses:
            qid += 1
            queries.append({
                "id": f"q{qid:02d}",
                "query": query_text,
                "expected_constraints": {"organism": organism},
                "expected_gse": gses,
                "category": "multi_constraint",
                "difficulty": "hard",
            })

    # ===== Category 5: Natural language / complex (hard) =====
    nl_specs = [
        ("What datasets study the tumor microenvironment in human cancers?", ["tumor microenvironment"], "Homo sapiens", "cancer"),
        ("Find single-cell studies of immune checkpoint therapy response", ["checkpoint", "immunotherapy"], None, None),
        ("Studies of cell differentiation at single-cell level", ["differentiat"], None, None),
        ("Datasets profiling spatial gene expression in tissues", ["spatial"], None, None),
        ("What scRNA-seq studies investigate drug resistance in cancer?", ["drug resistance", "resistance"], None, "cancer"),
        ("Find datasets studying cell-cell communication", ["cell-cell communication", "ligand-receptor", "signaling"], None, None),
        ("Single-cell multi-omics datasets combining RNA and ATAC", ["multi-omics", "multiome", "atac"], None, None),
        ("Studies of clonal evolution using single-cell sequencing", ["clonal", "evolution", "lineage tracing"], None, None),
    ]

    for query_text, keywords, organism, domain in nl_specs:
        gses = ts(keywords, organism=organism, domain=domain, limit=5)
        if gses:
            qid += 1
            constraints = {}
            if organism:
                constraints["organism"] = organism
            queries.append({
                "id": f"q{qid:02d}",
                "query": query_text,
                "expected_constraints": constraints,
                "expected_gse": gses,
                "category": "natural_language",
                "difficulty": "hard",
            })

    # ===== Category 6: Modality-specific (medium) =====
    mod_specs = [
        ("snRNA-seq datasets", ["single-nucleus", "snrna"]),
        ("spatial transcriptomics datasets", ["spatial", "visium"]),
        ("scATAC-seq datasets", ["atac-seq", "atac", "chromatin accessibility"]),
        ("CITE-seq protein and RNA datasets", ["cite-seq"]),
    ]

    for query_text, keywords in mod_specs:
        gses = ts(keywords, limit=3)
        if gses:
            qid += 1
            queries.append({
                "id": f"q{qid:02d}",
                "query": query_text,
                "expected_constraints": {"assay": keywords[0]},
                "expected_gse": gses,
                "category": "modality",
                "difficulty": "medium",
            })

    # ===== Category 7: Organism-specific (easy) =====
    org_specs = [
        ("zebrafish single-cell RNA-seq", ["zebrafish"]),
        ("rat single-cell RNA-seq", ["rat"]),
    ]
    for query_text, keywords in org_specs:
        gses = ts(keywords, limit=3)
        if gses:
            qid += 1
            queries.append({
                "id": f"q{qid:02d}",
                "query": query_text,
                "expected_constraints": {},
                "expected_gse": gses,
                "category": "organism",
                "difficulty": "easy",
            })

    # Summary
    print(f"Generated {len(queries)} eval queries")
    cats = Counter(q["category"] for q in queries)
    diffs = Counter(q["difficulty"] for q in queries)
    print(f"Categories: {dict(cats)}")
    print(f"Difficulties: {dict(diffs)}")
    print()
    for q in queries:
        n_gse = len(q["expected_gse"])
        print(f"  {q['id']}: [{q['difficulty']:6}] [{q['category']:18}] {q['query'][:55]:55s} -> {n_gse} GSEs")

    with open(OUT_PATH, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
