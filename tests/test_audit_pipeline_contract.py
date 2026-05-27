"""Contract tests for the public audit pipeline wrapper guards."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_audit_pipeline.sh"


class AuditPipelineContractTests(unittest.TestCase):
    def run_script(self, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(env_overrides)
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_rejects_zero_poll_interval_before_polling(self):
        proc = self.run_script(BENCH_WAIT_POLL_SEC="0")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("BENCH_WAIT_POLL_SEC must be greater than 0", proc.stdout + proc.stderr)

    def test_rejects_non_integer_timeout(self):
        proc = self.run_script(BENCH_WAIT_TIMEOUT_SEC="not-a-number")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("BENCH_WAIT_TIMEOUT_SEC must be a non-negative integer", proc.stdout + proc.stderr)

    def test_missing_private_dependencies_fail_before_service_start(self):
        proc = self.run_script(BENCH_WAIT_TIMEOUT_SEC="0")
        self.assertEqual(proc.returncode, 2)
        combined = proc.stdout + proc.stderr
        self.assertIn("requires private/local scripts", combined)
        self.assertIn("scripts/surgical_rerun_a_d.py", combined)

    def test_timeout_diagnostics_do_not_log_full_command_line(self):
        marker = "05_bench_public_contract_marker"
        secret_arg = "SECRET_TOKEN_SHOULD_NOT_APPEAR"
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)", marker, secret_arg])
        try:
            proc = self.run_script(BENCH_WAIT_TIMEOUT_SEC="1", BENCH_WAIT_POLL_SEC="1")
        finally:
            sleeper.terminate()
            try:
                sleeper.wait(timeout=2)
            except subprocess.TimeoutExpired:
                sleeper.kill()
        combined = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 2, combined)
        self.assertIn("Matching process summaries", combined)
        self.assertNotIn(secret_arg, combined)

    def test_timeout_does_not_oversleep_large_poll_interval(self):
        marker = "05_bench_public_contract_marker"
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)", marker])
        try:
            start = time.monotonic()
            proc = self.run_script(BENCH_WAIT_TIMEOUT_SEC="1", BENCH_WAIT_POLL_SEC="30")
            elapsed = time.monotonic() - start
        finally:
            sleeper.terminate()
            try:
                sleeper.wait(timeout=2)
            except subprocess.TimeoutExpired:
                sleeper.kill()
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertLess(elapsed, 5, proc.stdout + proc.stderr)
        self.assertIn("timed out", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
