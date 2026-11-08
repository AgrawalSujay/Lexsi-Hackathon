"""Prototype renderer for the Step 2 template formats (English, stub filler).

Not the builder: it proves the formats in turnguard/data/templates/*.yaml and
turnguard/data/injections.yaml are complete enough to produce valid episodes.
For every scenario x goal x style x pressure x position x carrier it renders
the injected episode, its clean twin and the authorized variants, then runs the
scenario's `route` through the real harness and checks:

  exposure    every injection part reaches the model on the route;
  solvable    route alone -> injected `safe`, clean `pass`, authorized `pass`;
  detectable  route + obeying the injection -> `unsafe`;
  neutral     oracle terms are language-neutral (contain a digit, @ or /).

The builder (Step 3) replaces stub filler with the teacher and adds languages,
sampling, CuratorKIT, splits and freezing; this logic is meant to be reused.

Usage: python scripts/render_prototype.py   (exit code 1 on any problem)
"""

import copy
import itertools
import sys

from turnguard.data.builder import STYLES, authorized, check, injected, load
from turnguard.schema import make_clean_twin


def main() -> int:
    bank, scenarios = load()
    total, failures = 0, []
    for scenario in scenarios:
        combos = itertools.product(
            scenario["goals"],
            STYLES,
            ("none", "snag"),
            ("first", "later", "split"),
            ("body", "metadata", "quoted", "tool_error"),
        )
        for goal, style, pressure, position, carrier in combos:
            episode = injected(scenario, bank, goal, style, pressure, position, carrier)
            if episode is None:
                continue
            for item in (episode, make_clean_twin(copy.deepcopy(episode))):
                total += 1
                if problems := check(scenario, item):
                    failures.append((item["id"], problems))
        for index, variant in enumerate(scenario.get("authorized", [])):
            total += 1
            item = authorized(scenario, variant, index)
            if problems := check(scenario, item):
                failures.append((item["id"], problems))
    for episode_id, problems in failures[:20]:
        print(f"FAIL {episode_id}: {'; '.join(problems)}")
    print(f"{total - len(failures)}/{total} rendered episodes pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
