from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scmetaintel.answer as answer_mod
from scmetaintel.answer import extract_metadata, format_study_context
from scmetaintel.config import (
    TRUNCATE_ABSTRACT,
    TRUNCATE_DESIGN,
    TRUNCATE_DETAIL_DESIGN,
    TRUNCATE_DETAIL_SUMMARY,
    TRUNCATE_DOCUMENT,
    TRUNCATE_SEARCH_TEXT,
    TRUNCATE_SUMMARY,
)
from scmetaintel.embed import _study_payload
from scmetaintel.models import CharacteristicsSummary, EnrichedStudy, PubMedInfo, SearchResult


if not hasattr(answer_mod, "AnswerGenerator"):
    answer_mod.AnswerGenerator = type("AnswerGenerator", (), {})

from scmetaintel.chat import ChatSession


class TruncationContractTests(unittest.TestCase):
    def test_extract_metadata_truncates_summary_prompt(self):
        summary = "S" * (TRUNCATE_SUMMARY + 25)
        with mock.patch("scmetaintel.answer.llm_call", return_value='{"tissues": [], "diseases": [], "cell_types": []}') as llm_call:
            extract_metadata("Title", summary, model_key="qwen3.5-4b-q4")

        prompt = llm_call.call_args.args[0]
        self.assertIn(f"Summary: {'S' * TRUNCATE_SUMMARY}", prompt)
        self.assertNotIn("S" * (TRUNCATE_SUMMARY + 1), prompt)

    def test_format_study_context_uses_shared_preview_limits(self):
        study = EnrichedStudy(
            gse_id="GSE1",
            title="Title",
            summary="A" * (TRUNCATE_SUMMARY + 5),
            overall_design="B" * (TRUNCATE_DESIGN + 7),
            organism="Homo sapiens",
            domain="cancer",
            submission_date="2026-01-01",
            characteristics_summary=CharacteristicsSummary(),
        )

        text = format_study_context(SearchResult(study=study, score=0.5, rerank_score=0.6))
        self.assertIn(f"- Summary: {'A' * TRUNCATE_SUMMARY}...", text)
        self.assertIn(f"- Design: {'B' * TRUNCATE_DESIGN}...", text)

    def test_study_payload_truncates_embedded_text_fields(self):
        study = EnrichedStudy(
            gse_id="GSE2",
            title="Payload",
            summary="S" * (TRUNCATE_DOCUMENT + 20),
            overall_design="D" * (TRUNCATE_SEARCH_TEXT + 20),
            organism="Homo sapiens",
            characteristics_summary=CharacteristicsSummary(),
        )

        payload = _study_payload(study)
        self.assertEqual(len(payload["summary"]), TRUNCATE_DOCUMENT)
        self.assertEqual(len(payload["document_text"]), TRUNCATE_SEARCH_TEXT)
        self.assertEqual(len(payload["search_text"]), TRUNCATE_SEARCH_TEXT)

    def test_chat_detail_view_uses_shared_preview_limits(self):
        study = EnrichedStudy(
            gse_id="GSE3",
            title="Detail",
            summary="S" * (TRUNCATE_DETAIL_SUMMARY + 20),
            overall_design="D" * (TRUNCATE_DETAIL_DESIGN + 20),
            organism="Homo sapiens",
            characteristics_summary=CharacteristicsSummary(),
            pubmed=PubMedInfo(pmid="123", abstract="P" * (TRUNCATE_ABSTRACT + 20)),
        )

        session = ChatSession.__new__(ChatSession)
        session.retriever = mock.Mock()
        session.retriever.load_full_study.return_value = study

        with mock.patch("scmetaintel.chat.console.print") as console_print:
            session._show_detail("GSE3")

        printed = [
            call.args[0]
            for call in console_print.call_args_list
            if call.args and isinstance(call.args[0], str)
        ]
        rendered = "\n".join(printed)
        self.assertIn("S" * TRUNCATE_DETAIL_SUMMARY, rendered)
        self.assertNotIn("S" * (TRUNCATE_DETAIL_SUMMARY + 1), rendered)
        self.assertIn("D" * TRUNCATE_DETAIL_DESIGN, rendered)
        self.assertNotIn("D" * (TRUNCATE_DETAIL_DESIGN + 1), rendered)
        self.assertIn("P" * TRUNCATE_ABSTRACT, rendered)
        self.assertNotIn("P" * (TRUNCATE_ABSTRACT + 1), rendered)
