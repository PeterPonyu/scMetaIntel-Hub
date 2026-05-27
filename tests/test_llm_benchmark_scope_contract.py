from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

# `benchmarks/04_bench_llm.py` is private/untracked tooling. On a clean clone
# of the public repo it will not be present, so we skip these contract tests
# rather than fail with FileNotFoundError. When the private benchmark file is
# checked out locally the tests still run and exercise the contract.
_BENCH04_PATH = ROOT / "benchmarks" / "04_bench_llm.py"
_BENCH04_PRESENT = _BENCH04_PATH.is_file()

bench04 = None
if _BENCH04_PRESENT:
    spec = importlib.util.spec_from_file_location("bench04_llm", _BENCH04_PATH)
    bench04 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bench04)


@unittest.skipUnless(
    _BENCH04_PRESENT,
    f"private benchmark script not present at {_BENCH04_PATH}; "
    "skipping contract tests on clean public clone",
)
class LlmBenchmarkScopeContractTests(unittest.TestCase):
    def test_benchmark_runs_are_non_think_only(self):
        runs = bench04.build_benchmark_runs(["qwen3-8b", "llama3.1-8b"])
        self.assertEqual(
            runs,
            [("qwen3-8b", False, "qwen3-8b"), ("llama3.1-8b", False, "llama3.1-8b")],
        )

    def test_benchmark_runs_exclude_always_think_models(self):
        self.assertEqual(bench04.build_benchmark_runs(["deepseek-r1-7b"]), [])
