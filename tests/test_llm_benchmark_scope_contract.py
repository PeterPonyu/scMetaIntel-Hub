from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

spec = importlib.util.spec_from_file_location(
    "bench04_llm", ROOT / "benchmarks" / "04_bench_llm.py"
)
bench04 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bench04)


class LlmBenchmarkScopeContractTests(unittest.TestCase):
    def test_benchmark_runs_are_non_think_only(self):
        runs = bench04.build_benchmark_runs(["qwen3-8b", "llama3.1-8b"])
        self.assertEqual(
            runs,
            [("qwen3-8b", False, "qwen3-8b"), ("llama3.1-8b", False, "llama3.1-8b")],
        )

    def test_benchmark_runs_exclude_always_think_models(self):
        self.assertEqual(bench04.build_benchmark_runs(["deepseek-r1-7b"]), [])
