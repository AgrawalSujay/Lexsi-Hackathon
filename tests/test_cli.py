"""Check command discovery and placeholder execution."""

import contextlib
import io
import subprocess
import sys
import unittest

from turnguard.cli import main

COMMANDS = (
    "run-episodes",
    "build-data",
    "freeze",
    "train",
    "eval-external",
    "report",
)


class CliTests(unittest.TestCase):
    def test_module_help_lists_every_command(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", "-m", "turnguard", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        for command in COMMANDS:
            with self.subTest(command=command):
                self.assertIn(command, result.stdout)

    def test_each_command_prints_placeholder(self) -> None:
        for command in COMMANDS[3:]:
            with self.subTest(command=command):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main([command]), 0)
                self.assertEqual(output.getvalue(), "not implemented yet\n")

    def test_unknown_command_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["unknown"])
        self.assertEqual(error.exception.code, 2)
