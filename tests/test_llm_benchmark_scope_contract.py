from pathlib import Path
import importlib.util
import sys
import types
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

def _load_bench04() -> types.ModuleType | None:
    if not _BENCH04_PRESENT:
        return None
    spec = importlib.util.spec_from_file_location("bench04_llm", _BENCH04_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load benchmark module spec from {_BENCH04_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench04 = _load_bench04()


@unittest.skipUnless(
    _BENCH04_PRESENT,
    f"private benchmark script not present at {_BENCH04_PATH}; "
    "skipping contract tests on clean public clone",
)
class LlmBenchmarkScopeContractTests(unittest.TestCase):
    def test_benchmark_runs_are_non_think_only(self):
        assert bench04 is not None
        runs = bench04.build_benchmark_runs(["qwen3-8b", "llama3.1-8b"])
        self.assertEqual(
            runs,
            [("qwen3-8b", False, "qwen3-8b"), ("llama3.1-8b", False, "llama3.1-8b")],
        )

    def test_benchmark_runs_exclude_always_think_models(self):
        assert bench04 is not None
        self.assertEqual(bench04.build_benchmark_runs(["deepseek-r1-7b"]), [])
