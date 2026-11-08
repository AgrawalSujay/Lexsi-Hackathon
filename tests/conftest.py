"""Bank-wide validation fixtures with five default injected bases, no --limit."""

import shutil
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def bank_root(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "turnguard/data"
    shutil.copytree(repo / "turnguard/data/templates", data / "templates")
    shutil.copy(repo / "turnguard/data/injections.yaml", data / "injections.yaml")
    (tmp_path / "configs").mkdir()
    config = yaml.safe_load((repo / "configs/data.yaml").read_text())
    for target in config["targets"].values():
        target["injected"] = target["clean"] = 5
    (tmp_path / "configs/data.yaml").write_text(yaml.safe_dump(config))
    return tmp_path
