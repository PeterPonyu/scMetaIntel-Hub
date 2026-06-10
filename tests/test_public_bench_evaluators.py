"""Unit tests for the public-benchmark constraint verifiers.

These guard the fix for the degenerate evaluators in
``benchmarks/05_bench_public.py`` (issue #20), where IFEval/JSON/function-call
checks credited models for constraints that were never actually verified.

The module has a numeric filename and runtime-only imports, so it is loaded by
path; only the pure verifier helpers are exercised (no LLM/Ollama calls).
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_bench_module():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "bench05_public", ROOT / "benchmarks" / "05_bench_public.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IfevalVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_bench_module()

    def test_unverifiable_constraint_is_excluded_not_passed(self):
        # The whole point of the fix: an instruction we cannot check must be
        # omitted, not counted as an automatic pass.
        verdicts = self.m.ifeval_verdicts(["detectable_format:made_up_thing"], [{}], "anything")
        self.assertEqual(verdicts, [])

    def test_no_comma_checked(self):
        self.assertEqual(self.m.ifeval_verdicts(["punctuation:no_comma"], [{}], "clean text"), [True])
        self.assertEqual(self.m.ifeval_verdicts(["punctuation:no_comma"], [{}], "a, b"), [False])

    def test_number_words_uses_kwargs(self):
        ids = ["length_constraints:number_words"]
        self.assertEqual(self.m.ifeval_verdicts(ids, [{"relation": "at least", "num_words": 3}], "one two three"), [True])
        self.assertEqual(self.m.ifeval_verdicts(ids, [{"relation": "at least", "num_words": 5}], "one two three"), [False])
        self.assertEqual(self.m.ifeval_verdicts(ids, [{"relation": "less than", "num_words": 3}], "one two"), [True])

    def test_number_words_without_kwargs_is_excluded(self):
        # Previously this defaulted to "pass"; now it is excluded when params are missing.
        self.assertEqual(self.m.ifeval_verdicts(["length_constraints:number_words"], [{}], "one two three"), [])

    def test_keywords_existence_and_forbidden(self):
        self.assertEqual(self.m.ifeval_verdicts(["keywords:existence"], [{"keywords": ["alpha", "beta"]}], "alpha and beta"), [True])
        self.assertEqual(self.m.ifeval_verdicts(["keywords:forbidden_words"], [{"forbidden_words": ["bad"]}], "all good"), [True])
        self.assertEqual(self.m.ifeval_verdicts(["keywords:forbidden_words"], [{"forbidden_words": ["bad"]}], "this is bad"), [False])

    def test_mixed_constraints_only_count_verifiable(self):
        ids = ["punctuation:no_comma", "detectable_format:made_up_thing"]
        # Only the no_comma constraint is verifiable -> single verdict.
        self.assertEqual(self.m.ifeval_verdicts(ids, [{}, {}], "no commas"), [True])


class JsonSchemaVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_bench_module()

    def test_none_fails(self):
        self.assertFalse(self.m.json_schema_pass(None, {"type": "object"}))

    def test_conforms(self):
        schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
        self.assertTrue(self.m.json_schema_pass({"name": "x"}, schema))

    def test_violation_is_rejected(self):
        schema = {"type": "object", "required": ["name"]}
        self.assertFalse(self.m.json_schema_pass({"other": 1}, schema))

    def test_string_schema_is_parsed(self):
        self.assertFalse(self.m.json_schema_pass({"other": 1}, '{"type": "object", "required": ["name"]}'))

    def test_no_schema_falls_back_to_valid_json(self):
        self.assertTrue(self.m.json_schema_pass({"anything": 1}, None))


class FunctionCallVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_bench_module()

    def test_no_gold_is_unverifiable(self):
        # Without a ground-truth function name, correctness can't be scored ->
        # None (caller excludes it) rather than crediting any 'word(...)'.
        self.assertIsNone(self.m.function_call_correct("foo(bar)", ""))

    def test_gold_match_json_and_call(self):
        self.assertTrue(self.m.function_call_correct('{"name": "get_weather", "arguments": {}}', "get_weather"))
        self.assertTrue(self.m.function_call_correct("I'll call get_weather(city='NYC')", "get_weather"))

    def test_gold_miss(self):
        self.assertFalse(self.m.function_call_correct("here is some prose (with parens)", "get_weather"))


if __name__ == "__main__":
    unittest.main()
