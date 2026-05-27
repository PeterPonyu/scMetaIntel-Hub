"""CLI smoke tests for the `scmetaintel chat` subcommand.

These tests guard against regressions like the one tracked in issue #5,
where `scmetaintel/chat.py` imported a name (`AnswerGenerator`) that did
not exist in `scmetaintel/answer.py`, breaking even `chat --help`.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ChatCliSmokeTests(unittest.TestCase):
    def test_chat_help_runs(self):
        """`python -m scmetaintel chat --help` must exit 0 and print usage."""
        proc = subprocess.run(
            [sys.executable, "-m", "scmetaintel", "chat", "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"chat --help exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}",
        )
        self.assertIn("usage", proc.stdout.lower() + proc.stderr.lower())

    def test_chat_module_imports(self):
        """Importing scmetaintel.chat must succeed (regression for issue #5)."""
        from scmetaintel import chat  # noqa: F401

        # The chat REPL constructs AnswerGenerator in ChatSession.__init__.
        # Confirm the symbol is importable through the same path chat.py uses.
        from scmetaintel.answer import AnswerGenerator  # noqa: F401

        self.assertTrue(hasattr(chat, "ChatSession"))


if __name__ == "__main__":
    unittest.main()
