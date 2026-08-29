"""Unit tests for the ontology embedding-match abstention rule (issue #18).

Mirrors the calibration in studies/ontology_abstention.py: out-of-ontology terms
sit at a near-tie (tiny top1-top2 gap) and must abstain, while real terms with a
clear separation are accepted.
"""
from __future__ import annotations

from scmetaintel.config import ONTOLOGY_ABSTAIN_MARGIN, ONTOLOGY_SIMILARITY_THRESHOLD_LLM
from scmetaintel.ontology import _accept_embedding_match

THR = ONTOLOGY_SIMILARITY_THRESHOLD_LLM  # 0.65
MARG = ONTOLOGY_ABSTAIN_MARGIN           # 0.03


def test_accepts_confident_separated_match():
    # real term: high score, clear separation (e.g. hepatocyte, margin 0.12)
    assert _accept_embedding_match(1.0, 0.88, THR, MARG) is True


def test_rejects_below_similarity_floor():
    assert _accept_embedding_match(0.60, 0.10, THR, MARG) is False


def test_abstains_on_near_tie_even_above_floor():
    # out-of-ontology "doublets": top1=0.732, top2=0.728 -> margin 0.004 < 0.03
    assert _accept_embedding_match(0.732, 0.728, THR, MARG) is False
    # "low quality cells": 0.770 / 0.768 -> margin 0.002
    assert _accept_embedding_match(0.770, 0.768, THR, MARG) is False


def test_accepts_exactly_at_margin_boundary():
    assert _accept_embedding_match(0.70, 0.67, THR, MARG) is True  # margin == 0.03


def test_single_candidate_treated_as_full_margin():
    # top2 defaults to 0.0 upstream when only one candidate exists -> large margin
    assert _accept_embedding_match(0.66, 0.0, THR, MARG) is True
