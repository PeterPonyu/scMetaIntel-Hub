"""Unit tests for the real BM25 sparse scorer (issue #17).

The previous "sparse" scorer was unweighted token-frequency overlap (no IDF, no
length normalization) and ranked ~10x worse than BM25 on a real qrels benchmark
(studies/bm25_retrieval.py). These tests pin the two properties that distinguish
real BM25 from the old overlap: relevance ranking and IDF weighting.
"""
from __future__ import annotations

import numpy as np

from scmetaintel.retrieve import _bm25_tokenize, _doc_text, build_bm25


def test_ranks_relevant_document_first():
    docs = [
        "lung fibrosis single cell rna sequencing of human samples",
        "mouse brain development cell atlas",
        "human liver hepatocyte expression study",
    ]
    bm25 = build_bm25(docs)
    scores = bm25.get_scores(_bm25_tokenize("lung fibrosis scRNA-seq"))
    assert int(np.argmax(scores)) == 0


def test_idf_beats_raw_overlap():
    # The old token-overlap scorer would prefer doc 0 (four matches of the common
    # word "the"). Real BM25 down-weights "the" via IDF and prefers doc 1, which
    # matches the rare, discriminative term.
    docs = ["the the the the generic cell", "the distinctive_marker_xyz"]
    bm25 = build_bm25(docs)
    scores = bm25.get_scores(_bm25_tokenize("the distinctive_marker_xyz"))
    assert scores[1] > scores[0]


def test_doc_text_fallbacks():
    assert _doc_text({"document_text": "a"}) == "a"
    assert _doc_text({"search_text": "b"}) == "b"
    assert _doc_text({"title": "c"}) == "c"
    assert _doc_text({}) == ""
