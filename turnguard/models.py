"""Define the generator interface and a scripted CPU-only generator."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenRequest:
    episode_id: str
    step: int
    messages: list[dict]


class Generator(Protocol):
    name: str

    def generate(self, batch: list[GenRequest]) -> list[str]: ...


class StubGenerator:
    name: str = "stub"

    def __init__(self, scripts: Mapping[str, list[str]]) -> None:
        self._scripts = {episode_id: list(script) for episode_id, script in scripts.items()}
        self._positions = dict.fromkeys(self._scripts, 0)

    def generate(self, batch: list[GenRequest]) -> list[str]:
        outputs = []
        for request in batch:
            episode_id = request.episode_id
            if episode_id not in self._scripts:
                raise ValueError(f"No script configured for episode {episode_id!r}")
            position = self._positions[episode_id]
            script = self._scripts[episode_id]
            if position >= len(script):
                raise RuntimeError(
                    f"Script exhausted for episode {episode_id!r} at step "
                    f"{request.step}: consumed {position} scripted responses"
                )
            outputs.append(script[position])
            self._positions[episode_id] += 1
        return outputs
