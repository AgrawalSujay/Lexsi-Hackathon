"""Check isolated run creation and optional environment metadata."""

import contextlib
import json
import platform
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from unittest.mock import Mock, patch

from turnguard.runs import _environment, new_run_dir


class RunDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(contextlib.chdir(temporary))
        clock = self.enterContext(patch("turnguard.runs.datetime"))
        clock.now.return_value = datetime(2026, 11, 7, 12, 30, tzinfo=timezone.utc)
        self.environment = {
            "python_version": "3.12.8",
            "packages": {},
            "gpu_name": None,
            "git_commit": None,
        }
        self.enterContext(
            patch("turnguard.runs._environment", return_value=self.environment)
        )

    def test_creates_run_directory_and_valid_environment(self) -> None:
        path = new_run_dir("A0", "CohereLabs/tiny-aya-global", "dev")
        timestamp = datetime(2026, 11, 7, 12, 30, tzinfo=timezone.utc)
        local_stamp = timestamp.astimezone().strftime("%Y%m%d-%H%M")
        self.assertEqual(
            path, Path("runs") / f"A0__CohereLabs-tiny-aya-global__dev__{local_stamp}"
        )
        self.assertTrue(path.is_dir())
        self.assertEqual(json.loads((path / "env.json").read_text()), self.environment)

    def test_collisions_append_suffix_without_overwriting(self) -> None:
        first = new_run_dir("A0", "model", "dev")
        sentinel = first / "env.json"
        sentinel.write_text('{"preserve": true}\n')
        second = new_run_dir("A0", "model", "dev")
        third = new_run_dir("A0", "model", "dev")
        self.assertEqual(second.name, first.name + "-2")
        self.assertEqual(third.name, first.name + "-3")
        self.assertEqual(json.loads(sentinel.read_text()), {"preserve": True})
        self.assertEqual(
            json.loads((second / "env.json").read_text()), self.environment
        )
        self.assertTrue(third.is_dir())

    def test_existing_file_is_also_a_collision(self) -> None:
        first = new_run_dir("A0", "model", "dev")
        (first.parent / (first.name + "-2")).write_text("keep")
        third = new_run_dir("A0", "model", "dev")
        self.assertEqual(third.name, first.name + "-3")

    def test_rejects_path_traversal_in_arm_and_split(self) -> None:
        for arm, split in (("../escape", "dev"), ("A0", "../escape")):
            with self.subTest(arm=arm, split=split):
                with self.assertRaises(ValueError):
                    new_run_dir(arm, "model", split)
        self.assertFalse(Path("runs").exists())


class EnvironmentTests(unittest.TestCase):
    def test_missing_optional_packages_and_git_are_null(self) -> None:
        with (
            patch(
                "turnguard.runs.metadata.version",
                side_effect=metadata.PackageNotFoundError,
            ),
            patch("turnguard.runs.subprocess.run", side_effect=FileNotFoundError),
            patch("turnguard.runs.importlib.import_module") as importer,
        ):
            environment = _environment()
        self.assertEqual(environment["python_version"], platform.python_version())
        self.assertEqual(environment["packages"], {})
        self.assertIsNone(environment["gpu_name"])
        self.assertIsNone(environment["git_commit"])
        importer.assert_not_called()

    def test_installed_versions_gpu_and_commit_are_recorded(self) -> None:
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "Fake test GPU"
        result = subprocess.CompletedProcess([], 0, stdout="abc123\n")
        with (
            patch("turnguard.runs.metadata.version", return_value="1.2.3"),
            patch("turnguard.runs.subprocess.run", return_value=result),
            patch("turnguard.runs.importlib.import_module", return_value=torch),
        ):
            environment = _environment()
        self.assertEqual(environment["packages"]["torch"], "1.2.3")
        self.assertEqual(environment["packages"]["curatorkit"], "1.2.3")
        self.assertEqual(environment["gpu_name"], "Fake test GPU")
        self.assertEqual(environment["git_commit"], "abc123")

    def test_cpu_only_torch_and_unborn_git_have_null_values(self) -> None:
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with (
            patch("turnguard.runs.metadata.version", return_value="1.2.3"),
            patch(
                "turnguard.runs.subprocess.run",
                side_effect=subprocess.CalledProcessError(128, "git"),
            ),
            patch("turnguard.runs.importlib.import_module", return_value=torch),
        ):
            environment = _environment()
        self.assertIsNone(environment["gpu_name"])
        self.assertIsNone(environment["git_commit"])
        torch.cuda.get_device_name.assert_not_called()
