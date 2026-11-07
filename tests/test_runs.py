"""Check isolated run creation and optional environment metadata."""

import contextlib
import json
import platform
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from unittest.mock import Mock, call, patch

from turnguard.runs import _environment, new_run_dir


class RunDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.repo_root = Path(temporary).resolve()
        self.enterContext(contextlib.chdir(temporary))
        self.enterContext(
            patch(
                "turnguard.runs.__file__", str(self.repo_root / "turnguard" / "runs.py")
            )
        )
        clock = self.enterContext(patch("turnguard.runs.datetime"))
        clock.now.return_value = datetime(2026, 11, 7, 12, 30, tzinfo=UTC)
        self.environment = {
            "python_version": "3.12.8",
            "packages": {},
            "gpu_name": None,
            "git_commit": None,
            "git_dirty": None,
        }
        self.enterContext(
            patch("turnguard.runs._environment", return_value=self.environment)
        )

    def test_creates_run_directory_and_valid_environment(self) -> None:
        path = new_run_dir("A0", "CohereLabs/tiny-aya-global", "dev")
        timestamp = datetime(2026, 11, 7, 12, 30, tzinfo=UTC)
        local_stamp = timestamp.astimezone().strftime("%Y%m%d-%H%M")
        self.assertEqual(
            path,
            self.repo_root
            / "runs"
            / f"A0__CohereLabs-tiny-aya-global__dev__{local_stamp}",
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

    def test_default_root_is_independent_of_working_directory(self) -> None:
        elsewhere = self.repo_root / "elsewhere"
        elsewhere.mkdir()
        with contextlib.chdir(elsewhere):
            path = new_run_dir("A0", "model", "dev")
        self.assertEqual(path.parent, self.repo_root / "runs")
        self.assertFalse((elsewhere / "runs").exists())

    def test_explicit_root_overrides_default(self) -> None:
        override = self.repo_root / "custom" / "nested"
        path = new_run_dir("A0", "model", "dev", root=override)
        self.assertEqual(path.parent, override / "runs")
        self.assertTrue((path / "env.json").is_file())
        self.assertFalse((self.repo_root / "runs").exists())


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
        self.assertIsNone(environment["git_dirty"])
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
        for package in (
            "curatorkit",
            "circuitkit",
            "trl",
            "bitsandbytes",
            "transformer-lens",
        ):
            self.assertEqual(environment["packages"][package], "1.2.3")
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
        self.assertIsNone(environment["git_dirty"])
        torch.cuda.get_device_name.assert_not_called()

    def test_git_dirty_detects_clean_and_uncommitted_changes_at_root(self) -> None:
        root = Path("/test/repo")
        for output, expected in (
            ("", False),
            (" M turnguard/runs.py\n", True),
            ("?? new.py\n", True),
        ):
            with (
                self.subTest(output=output),
                patch(
                    "turnguard.runs.metadata.version",
                    side_effect=metadata.PackageNotFoundError,
                ),
                patch(
                    "turnguard.runs.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, stdout="abc123\n"),
                        subprocess.CompletedProcess([], 0, stdout=output),
                    ],
                ) as runner,
            ):
                environment = _environment(root)
            self.assertIs(environment["git_dirty"], expected)
            self.assertEqual(
                runner.call_args_list,
                [
                    call(
                        ["git", "rev-parse", "HEAD"],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5,
                    ),
                    call(
                        ["git", "status", "--porcelain"],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5,
                    ),
                ],
            )

    def test_unborn_repo_still_records_dirty_status(self) -> None:
        with (
            patch(
                "turnguard.runs.metadata.version",
                side_effect=metadata.PackageNotFoundError,
            ),
            patch(
                "turnguard.runs.subprocess.run",
                side_effect=[
                    subprocess.CalledProcessError(128, "git"),
                    subprocess.CompletedProcess([], 0, stdout="?? new.py\n"),
                ],
            ),
        ):
            environment = _environment()
        self.assertIsNone(environment["git_commit"])
        self.assertIs(environment["git_dirty"], True)
