"""Resolve one arm configuration for agent and plain-chat evaluations alike."""

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArmConfig:
    name: str
    system_prompt: str | None = None
    intervention_factory: str | None = None
    intervention_kwargs: dict = field(default_factory=dict)
    adapter: str | None = None

    def apply(self, generator: Any) -> None:
        """Attach adapters first, then build the shared HF intervention context."""
        if self.adapter:
            from peft import PeftModel

            generator.model = PeftModel.from_pretrained(
                generator.model,
                self.adapter,
                is_trainable=False,
            )
            generator.model.eval()
        if self.intervention_factory:
            module, separator, name = self.intervention_factory.partition(":")
            if not separator:
                raise ValueError("intervention_factory must be module:callable")
            factory = getattr(importlib.import_module(module), name)
            intervention = factory(generator=generator, **self.intervention_kwargs)
            if not callable(intervention):
                raise ValueError(
                    "Arm factory must return an intervention context factory"
                )
            generator.intervention = intervention


def resolve_arm(name: str, config: dict) -> ArmConfig:
    """Non-A0 arms must declare their actual behavior; a label is insufficient."""
    spec = config.get("arms", {}).get(name)
    if name == "A0":
        if spec and any(spec.values()):
            raise ValueError("A0 must have no system prompt, intervention or adapter")
        return ArmConfig(name)
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"Arm {name} is not configured; supply it in the arms config")
    allowed = {
        "system_prompt",
        "intervention_factory",
        "intervention_kwargs",
        "adapter",
    }
    if extra := spec.keys() - allowed:
        raise ValueError(f"Unknown arm fields: {sorted(extra)}")
    for key in ("system_prompt", "intervention_factory", "adapter"):
        if spec.get(key) is not None and not isinstance(spec[key], str):
            raise ValueError(f"{key} must be a string or null")
    if not isinstance(spec.get("intervention_kwargs", {}), dict):
        raise ValueError("intervention_kwargs must be a mapping")
    if not any(
        spec.get(key) for key in ("system_prompt", "intervention_factory", "adapter")
    ):
        raise ValueError(f"Arm {name} declares no behavior")
    return ArmConfig(name, **spec)
