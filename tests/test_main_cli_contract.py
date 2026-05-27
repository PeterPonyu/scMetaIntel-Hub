"""Top-level scmetaintel CLI dispatch contract tests."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class MainCliContractTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "scmetaintel", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_top_level_help_exits_zero(self):
        proc = self.run_cli("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Usage:", proc.stdout)
        self.assertIn("geo", proc.stdout)

    def test_no_args_keeps_existing_nonzero_discovery_behavior(self):
        proc = self.run_cli()
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Usage:", proc.stdout)

    def test_chat_help_reaches_real_subcommand_parser(self):
        proc = self.run_cli("chat", "--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--no-stream", proc.stdout)
        self.assertIn("--debug", proc.stdout)

    def test_retrieve_help_reaches_real_subcommand_parser(self):
        proc = self.run_cli("retrieve", "--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--query", proc.stdout)
        self.assertIn("--top-k", proc.stdout)

    def test_malformed_command_does_not_execute_injection_like_args(self):
        proc = self.run_cli("☃", "--query", "'; rm -rf /")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Unknown command: ☃", proc.stdout)
        self.assertIn("Usage:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
