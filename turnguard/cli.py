"""Expose commands for TurnGuard experiments and reports."""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="turnguard",
        description="Evaluate and repair multi-turn prompt injection.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "run-episodes",
        "build-data",
        "freeze",
        "train",
        "eval-external",
        "report",
    ):
        commands.add_parser(name, help="not implemented yet")
    parser.parse_args(argv)
    print("not implemented yet")
    return 0
