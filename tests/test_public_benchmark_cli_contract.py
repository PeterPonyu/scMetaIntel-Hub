from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

spec = importlib.util.spec_from_file_location(
    "bench05_public", ROOT / "benchmarks" / "05_bench_public.py"
)
public_bench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(public_bench)


class PublicBenchmarkCliContractTests(unittest.TestCase):
    def test_default_cli_uses_frozen_public_panel_and_sample_count(self):
        self.assertEqual(len(public_bench.FROZEN_PUBLIC_PANEL), 11)
        self.assertEqual(public_bench.FROZEN_PUBLIC_PANEL[0], "qwen2.5-0.5b")

    def test_public_runs_use_single_public_configuration_per_model(self):
        runs = public_bench.build_public_runs(["qwen3-8b", "llama3.1-8b"])
        self.assertEqual(
            runs,
            [("qwen3-8b", "qwen3-8b"), ("llama3.1-8b", "llama3.1-8b")],
        )
