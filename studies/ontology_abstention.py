#!/usr/bin/env python3
"""Calibration study for the ontology abstention margin (issue #18).

Empirically grounds `ONTOLOGY_ABSTAIN_MARGIN`: using the repo's real Cell
Ontology (ontologies/cl.obo), it shows that the previous policy (cos>=0.65,
no margin) confidently maps out-of-ontology terms ("doublets", "cluster 7",
"low quality cells", ...) to a wrong CL id 100% of the time, and that a small
top1-top2 abstention margin drops that to 0% while retaining accuracy on real
cell types whose CL ids are externally known.

Run:  python studies/ontology_abstention.py
Requires: sentence-transformers + a downloadable encoder (default BAAI/bge-small-en-v1.5).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

for _k in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_k, None)

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

OBO = Path(__file__).resolve().parent.parent / "ontologies" / "cl.obo"
ENCODER = os.environ.get("ONTO_STUDY_ENCODER", "BAAI/bge-small-en-v1.5")

# Real cell types with externally-verifiable CL ids (check on OLS / EBI).
POSITIVES = {
    "hepatocyte": "CL:0000182", "natural killer cell": "CL:0000623",
    "macrophage": "CL:0000235", "B cell": "CL:0000236", "fibroblast": "CL:0000057",
    "neuron": "CL:0000540", "endothelial cell": "CL:0000115", "plasma cell": "CL:0000786",
    "regulatory T cell": "CL:0000815", "CD8-positive, alpha-beta T cell": "CL:0000625",
}
# Out-of-ontology terms that should abstain (no valid cell-type mapping).
NEGATIVES = ["doublets", "low quality cells", "unknown", "cluster 7",
             "ambient RNA", "debris", "unassigned", "mitochondrial-high cells"]


def parse_cl(path: Path):
    ids, names = [], []
    cur = nm = None
    syns: list[str] = []
    obs = False

    def flush():
        nonlocal cur, nm, syns, obs
        if cur and cur.startswith("CL:") and not obs and nm:
            for label in [nm] + syns:
                ids.append(cur)
                names.append(label)
        cur = nm = None
        syns = []
        obs = False

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                flush()
            elif line.startswith("id: "):
                cur = line[4:].strip()
            elif line.startswith("name: "):
                nm = line[6:].strip()
            elif line.startswith("synonym: "):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    syns.append(m.group(1))
            elif line.startswith("is_obsolete: true"):
                obs = True
        flush()
    return ids, names


def main():
    ids, names = parse_cl(OBO)
    print(f"CL terms (non-obsolete): {len(set(ids))} classes, {len(names)} name+synonym strings")
    model = SentenceTransformer(ENCODER)
    emb = model.encode(names, normalize_embeddings=True, batch_size=256, show_progress_bar=False)

    def normalize(term):
        q = model.encode([term], normalize_embeddings=True, show_progress_bar=False)[0]
        order = np.argsort(-(emb @ q))
        ranked, seen = [], set()
        for i in order[:50]:
            if ids[i] not in seen:
                seen.add(ids[i])
                ranked.append((ids[i], float((emb @ q)[i])))
            if len(ranked) >= 2:
                break
        id1, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        return id1, s1, s1 - s2

    pos = [(t, normalize(t), g) for t, g in POSITIVES.items()]
    neg = [(t, normalize(t)) for t in NEGATIVES]

    print(f"\n{'policy':34s} {'pos accuracy':>12s} {'neg false-map rate':>18s}")
    print("-" * 68)
    for thr, marg in [(0.65, 0.00), (0.65, 0.03), (0.65, 0.05), (0.70, 0.05)]:
        pa = np.mean([1.0 if (s >= thr and mg >= marg and i == g) else 0.0 for _, (i, s, mg), g in pos])
        fm = np.mean([1.0 if (s >= thr and mg >= marg) else 0.0 for _, (_id, s, mg) in neg])
        tag = "  <- current" if (thr, marg) == (0.65, 0.0) else ""
        print(f"cos>={thr:.2f}, margin>={marg:.2f}{tag:13s} {pa:12.2f} {fm:18.2f}")


if __name__ == "__main__":
    main()
