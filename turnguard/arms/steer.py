"""A2 CAA: train-only token-window extraction and SafeTune runtime steering."""

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from turnguard.config import load_config
from turnguard.data.pairs import load_training_pairs
from turnguard.models import HFGenerator
from turnguard.runs import _environment

VERSION = "caa-v1"
COMPLETION_TOKENS = 8


def resolve_model_revision(generator, model_id: str) -> str | None:
    """The exact Hub commit of the loaded weights.

    transformers 5 no longer always sets ``config._commit_hash``; fall back to
    the snapshot directory the weights were loaded from (HF cache layout
    ``.../snapshots/<sha>/``), and only then to the Hub API.
    """
    revision = getattr(getattr(generator.model, "config", None), "_commit_hash", None)
    if isinstance(revision, str) and revision:
        return revision
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
        if isinstance(cached, str) and "/snapshots/" in cached:
            return cached.split("/snapshots/")[1].split("/")[0]
    except Exception:  # noqa: BLE001 - fall through to the Hub API
        pass
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:  # noqa: BLE001
        return None


def decoder_layers(model: Any) -> Any:
    from safetune._refusal_helpers import _get_decoder_layers

    layers = _get_decoder_layers(model)
    if not layers:
        raise ValueError("Cannot locate decoder layers")
    return layers


def caa_class() -> type:
    from safetune.interventions.steer.caa import CAAModel

    return CAAModel


def parse_layers(value: str) -> list[int]:
    try:
        layers = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(
            "Layers must be comma-separated nonnegative integers"
        ) from error
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise ValueError("Layers must be unique nonnegative integers")
    return layers


def window_ids(
    tokenizer: Any, row: dict, count: int = COMPLETION_TOKENS
) -> tuple[list[int], list[int], list[int]]:
    """Use an equal completion prefix length on both sides of each contrast."""
    prompt = tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True
    )
    chosen = tokenizer(row["chosen"], add_special_tokens=False)["input_ids"]
    rejected = tokenizer(row["rejected"], add_special_tokens=False)["input_ids"]
    n = min(count, len(chosen), len(rejected))
    if not prompt or n < 1:
        raise ValueError("Contrast needs a prompt and two nonempty completions")
    return prompt, chosen[:n], rejected[:n]


def extract_vectors(
    generator, rows: list[dict], layers: list[int]
) -> tuple[dict, dict]:
    """Capture only pooled residuals; no logits or full hidden-state stacks."""
    import torch

    if not rows:
        raise ValueError("No contrast pairs")
    blocks = decoder_layers(generator.model)
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(i < 0 or i >= len(blocks) for i in layers)
    ):
        raise ValueError(
            f"Layers must be unique decoder block indices 0..{len(blocks) - 1}"
        )
    if hasattr(generator.model, "peft_config"):
        raise ValueError("Extract A2 vectors from the unadapted base model")
    totals = {layer: None for layer in layers}
    capture, span = {}, [0, 0]
    handles = []

    def capture_layer(layer):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            capture[layer] = (
                hidden[0, span[0] : span[1], :].float().mean(dim=0).detach().cpu()
            )

        return hook

    tokens, identical = 0, 0
    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(capture_layer(layer)))
        backbone = generator.model.base_model
        with torch.inference_mode():
            for row in rows:
                prompt, chosen, rejected = window_ids(generator.tokenizer, row)
                identical += chosen == rejected
                span[:] = [len(prompt) - 1, len(prompt) + len(chosen)]
                pair = []
                for completion in (chosen, rejected):
                    capture.clear()
                    ids = torch.tensor(
                        [prompt + completion], device=generator.model.device
                    )
                    backbone(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                    )
                    if set(capture) != set(layers):
                        raise ValueError("Not all requested decoder hooks ran")
                    pair.append(dict(capture))
                    tokens += len(prompt) + len(completion)
                for layer in layers:
                    delta = pair[0][layer] - pair[1][layer]
                    totals[layer] = (
                        delta if totals[layer] is None else totals[layer] + delta
                    )
    finally:
        for handle in handles:
            handle.remove()
    vectors = {layer: unit_vector(value / len(rows)) for layer, value in totals.items()}
    return vectors, {
        "pairs": len(rows),
        "forward_tokens": tokens,
        "identical_completion_prefix_pairs": identical,
        "raw_norms": {
            layer: (value / len(rows)).norm().item() for layer, value in totals.items()
        },
    }


def unit_vector(value: Any) -> Any:
    import torch

    vector = value.detach().float().cpu()
    norm = vector.norm().item()
    if (
        vector.ndim != 1
        or not math.isfinite(norm)
        or norm <= 0
        or not torch.isfinite(vector).all().item()
    ):
        raise ValueError("Steering vector must be finite, nonzero and one-dimensional")
    return vector / norm


def build_vectors(
    pairs: str | Path,
    *,
    model: str = "headline",
    layers: list[int] | None = None,
    root: Path | None = None,
    generator=None,
    dry_run: bool = False,
) -> dict:
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/models.yaml")
    model_id = config["models"][model]
    layers = [10, 14, 18, 22] if layers is None else layers
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise ValueError("Layers must be unique nonnegative integers")
    source = Path(pairs).expanduser().resolve()
    source = source / "pairs.jsonl" if source.is_dir() else source
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = load_training_pairs(pairs)  # Must precede every optional GPU import/load.
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise ValueError("Pairs changed while loading")
    provenance = {
        "version": VERSION,
        "model_id": model_id,
        "model_key": model,
        "pairs_sha256": before,
        "pairs": len(rows),
        "split": "train",
        "layers": layers,
        "layer_indexing": "zero-based decoder block output",
        "pooling": (
            "equal-weight mean of last prompt token and up to 8 "
            "completion tokens per pair side"
        ),
        "completion_tokens": COMPLETION_TOKENS,
        "direction": "chosen minus rejected",
        "normalization": "unit L2 per layer",
        "seed": 1234,
        "sources": dict(Counter(row["source"] for row in rows)),
        "estimated_forward_tokens": sum(
            2 * sum(len(m["content"]) for m in r["prompt"]) // 4 + 16 for r in rows
        ),
    }
    if dry_run:
        return provenance
    generator = generator or HFGenerator(model_id, seed=1234)
    if generator.name != model_id:
        raise ValueError("Extraction generator model does not match configuration")
    revision = resolve_model_revision(generator, model_id)
    if not revision:
        raise ValueError("A2 extraction requires a resolved model revision")
    started = time.perf_counter()
    vectors, stats = extract_vectors(generator, rows, layers)
    stats["elapsed_seconds"] = time.perf_counter() - started
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise ValueError("Pairs changed during vector extraction")
    provenance.update(
        extraction=stats, environment=_environment(root), model_revision=revision
    )
    import torch

    output = root / "steering/A2" / f"{model}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vectors, "provenance": provenance}, output)
    report = {
        **provenance,
        "vector": str(output),
        "vector_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def last_token_class(base: type) -> type:
    """Reuse SafeTune lifecycle/layer placement, restrict its additive hook."""

    class CurrentTokenCAA(base):
        def _make_hook(self, vec):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                vector = vec.to(dtype=hidden.dtype, device=hidden.device)
                if hidden.ndim != 3 or hidden.shape[-1] != vector.shape[0]:
                    raise ValueError("Steering hidden/vector shape mismatch")
                # Prefill changes only the last prompt position (predicting the
                # first token); cached decode changes its current token only.
                result = hidden.clone()
                result[:, -1:, :] += self.strength * vector
                return (result,) + output[1:] if isinstance(output, tuple) else result

            return hook

    return CurrentTokenCAA


def make_hook(
    *,
    generator: Any,
    vector: str,
    layer: int,
    alpha: float,
    vector_sha256: str | None = None,
) -> Any:
    import torch

    if type(layer) is not int or not math.isfinite(alpha):
        raise ValueError("Layer must be an integer and alpha finite")
    path = Path(vector).expanduser()
    if (
        vector_sha256 is not None
        and hashlib.sha256(path.read_bytes()).hexdigest() != vector_sha256
    ):
        raise ValueError("Selected steering vector checksum mismatch")
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        expected = json.loads(sidecar.read_text())["vector_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Steering vector checksum mismatch")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    provenance = artifact["provenance"]
    if provenance.get("version") != VERSION or provenance.get("split") != "train":
        raise ValueError("A2 requires train-only CAA provenance")
    if provenance["model_id"] != generator.name:
        raise ValueError("Steering vector model does not match generator")
    revision = resolve_model_revision(generator, generator.name)
    if (
        provenance.get("model_revision") is not None
        and revision != provenance["model_revision"]
    ):
        raise ValueError("Steering vector model revision does not match generator")
    if (
        layer < 0
        or layer >= len(decoder_layers(generator.model))
        or layer not in artifact["vectors"]
    ):
        raise ValueError("Requested layer absent from model or steering artifact")
    unit = unit_vector(artifact["vectors"][layer])
    wrapper = last_token_class(caa_class())
    return lambda: wrapper(generator.model, vectors={layer: unit}, strength=alpha)
