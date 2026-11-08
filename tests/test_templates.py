"""The Step 2 template formats render into valid, solvable, detectable episodes."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_prototype.py"


def test_every_rendered_template_episode_passes_the_route_checks() -> None:
    spec = importlib.util.spec_from_file_location("render_prototype", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0


def test_prototype_samples_every_long_text_and_split_pool_entry(monkeypatch):
    import json

    from turnguard.data.builder import load

    spec = importlib.util.spec_from_file_location("prototype_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampled = {}

    class TrackedSampler(module.BankSampler):
        def choice(self, values):
            selected = super().choice(values)
            sampled.setdefault(json.dumps(values, sort_keys=True), set()).add(
                json.dumps(selected, sort_keys=True)
            )
            return selected

    monkeypatch.setattr(module, "BankSampler", TrackedSampler)
    # The preceding acceptance test executes the real harness; here check sampling.
    monkeypatch.setattr(module, "check", lambda scenario, episode: [])
    assert module.main() == 0
    bank, _ = load()
    lists = [
        texts for style in bank["styles"].values() for texts in style["long"].values()
    ]
    lists += [
        values for pool in bank["attacker_pools"].values() for values in pool.values()
    ]
    for values in lists:
        assert sampled[json.dumps(values, sort_keys=True)] == {
            json.dumps(value, sort_keys=True) for value in values
        }
    a, b = module.BankSampler(1234), module.BankSampler(1234)
    assert [a.choice([0, 1, 2]) for _ in range(12)] == [
        b.choice([0, 1, 2]) for _ in range(12)
    ]
