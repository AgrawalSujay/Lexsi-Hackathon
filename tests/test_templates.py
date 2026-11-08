"""The Step 2 template formats render into valid, solvable, detectable episodes."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_prototype.py"


def test_every_rendered_template_episode_passes_the_route_checks() -> None:
    spec = importlib.util.spec_from_file_location("render_prototype", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
