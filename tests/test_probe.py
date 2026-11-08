"""The probe set stays exposed, solvable and detectable as the code evolves."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _module("check_probe")
SPECS = CHECK.load()
BY_ID = {s["id"]: s for s in SPECS}


def test_probe_set_composition() -> None:
    types = [s["episode_type"] for s in SPECS]
    assert (
        types.count("injected"),
        types.count("clean"),
        types.count("authorized"),
    ) == (
        12,
        12,
        4,
    )
    assert {s["family"] for s in SPECS} == {
        "email",
        "docs",
        "tickets",
        "web",
        "calendar",
        "code",
    }


@pytest.mark.parametrize("episode_id", sorted(BY_ID))
def test_probe_episode(episode_id: str) -> None:
    assert CHECK.check(BY_ID[episode_id], BY_ID) == []


def test_generator_reproduces_committed_files(tmp_path: Path) -> None:
    generated = {e["id"]: e for e in _module("probe_v1").episodes()}
    assert generated == {
        s["id"]: json.loads(json.dumps(s, ensure_ascii=False)) for s in SPECS
    }
