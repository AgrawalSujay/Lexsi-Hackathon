"""A4 preference training; optional GPU packages load only after data preflight."""

import copy
import hashlib
import importlib
import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from turnguard.config import load_config
from turnguard.data.pairs import load_training_pairs
from turnguard.protocol import Action, parse
from turnguard.runs import RunTimer, new_run_dir

logger = logging.getLogger(__name__)


def chosen_kind(row: dict) -> str:
    return "ACTION" if isinstance(parse(row["chosen"]), Action) else "FINAL"


def split_pairs(rows: list[dict], fraction: float = 0.05) -> tuple[list, list]:
    """Seeded, kind-stratified holdout, rounded up to one whole pair."""
    if len(rows) < 2 or not 0 < fraction < 1:
        raise ValueError(
            "Need at least two pairs and a holdout fraction between 0 and 1"
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate pair IDs")
    rng = random.Random(1234)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        groups[chosen_kind(row)].append(row)
    n = min(len(rows) - 1, max(1, math.ceil(len(rows) * fraction)))
    counts = {k: int(len(v) * n / len(rows)) for k, v in groups.items()}
    for key in sorted(
        groups, key=lambda k: (-(len(groups[k]) * n / len(rows) - counts[k]), k)
    ):
        if sum(counts.values()) == n:
            break
        counts[key] += 1
    train, held = [], []
    for key, group in sorted(groups.items()):
        rng.shuffle(group)
        held.extend(group[: counts[key]])
        train.extend(group[counts[key] :])
    return sorted(train, key=lambda r: r["id"]), sorted(held, key=lambda r: r["id"])


def conversational(rows: list[dict]) -> list[dict]:
    # Preserve the run's model-format messages; never add a second system prompt
    # or use CST's optional opposite-prompt/preference-inverting formatter.
    return [
        {
            "prompt": copy.deepcopy(row["prompt"]),
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }
        for row in rows
    ]


def trainer_api() -> tuple[type, type, str]:
    """Use SafeTune's DPO path; do not hide missing dependencies with a fallback."""
    try:
        module = importlib.import_module("safetune.interventions.harden.cst")
    except ModuleNotFoundError as error:
        if error.name not in (
            "safetune.interventions.harden.cst",
            "safetune.interventions.harden",
        ):
            raise ImportError(
                "A4 requires the gpu extra, including SafeTune and TRL"
            ) from error
        from trl import DPOConfig, DPOTrainer

        logger.warning("SafeTune has no CST module; falling back to trl.DPOTrainer")
        return DPOTrainer, DPOConfig, "trl.DPOTrainer (SafeTune CST absent)"
    if getattr(module, "_TRL_IMPORT_ERROR", None) is not None:
        raise ImportError(
            "SafeTune CSTTrainer cannot import TRL"
        ) from module._TRL_IMPORT_ERROR
    return (
        module.CSTTrainer,
        module.CSTConfig,
        "safetune.interventions.harden.cst.CSTTrainer",
    )


def named_metrics_trainer(base: type) -> type:
    """TRL 1.14 logs every eval subset's rewards under the same generic name."""

    class NamedMetricsTrainer(base):
        def log(self, logs, start_time=None):
            prefix = next(
                (
                    key[:-5]
                    for key in logs
                    if key.startswith("eval_") and key.endswith("_loss")
                ),
                None,
            )
            if prefix and hasattr(self, "_metrics"):
                bucket = self._metrics["eval"]
                label = prefix.removeprefix("eval_")
                for key, values in list(bucket.items()):
                    del bucket[key]
                    bucket[f"{label}_{key}"] = values
            return super().log(logs, start_time)

    return NamedMetricsTrainer


def _gpu_trainer(config: dict, train: list, held: list, run: Path):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from safetune.runner.utils.model_utils import load_model, load_tok
    from transformers import set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("A4 requires a bf16-capable CUDA GPU; use --dry-run on CPU")
    base, config_class, api = trainer_api()
    set_seed(1234)
    tok = load_tok(config["model_id"])
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(config["model_id"], dtype=torch.bfloat16, device="cuda")
    model.config.use_cache = False
    targets = config["target_modules"]
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if missing := set(targets) - present:
        raise ValueError(f"LoRA target modules missing from model: {sorted(missing)}")
    peft = LoraConfig(
        r=config["rank"],
        lora_alpha=config["alpha"],
        lora_dropout=0.0,
        target_modules=targets,
        task_type="CAUSAL_LM",
        bias="none",
    )
    args = config_class(
        output_dir=str(run),
        learning_rate=config["learning_rate"],
        beta=config["beta"],
        num_train_epochs=config["epochs"],
        max_length=config["max_length"],
        bf16=True,
        fp16=False,
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_steps=10 if config["smoke"] else -1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        seed=1234,
        data_seed=1234,
        prediction_loss_only=True,
        precompute_ref_log_probs=False,
        truncation_mode="keep_end",
        use_liger_kernel=False,
    )
    evals = {"all": Dataset.from_list(conversational(held))}
    for kind in ("FINAL", "ACTION"):
        subset = [row for row in held if chosen_kind(row) == kind]
        if subset:
            evals[kind] = Dataset.from_list(conversational(subset))
    trainer = named_metrics_trainer(base)(
        model=model,
        ref_model=None,
        args=args,
        processing_class=tok,
        peft_config=peft,
        train_dataset=Dataset.from_list(conversational(train)),
        eval_dataset=evals,
    )
    if trainer.ref_model is not None or not hasattr(trainer.model, "disable_adapter"):
        raise ValueError("A4 must use the adapter-disabled base as its DPO reference")
    # Keep every completion token and a common recent prompt for both sides.
    # Never let TRL silently drop long prompts or truncate away the decision.
    tokenization = {}

    def budget(dataset, name):
        count = 0

        def trim(row):
            nonlocal count
            completion = max(len(row["chosen_ids"]), len(row["rejected_ids"]))
            available = config["max_length"] - completion
            if available < 1:
                raise ValueError(f"{name}: completion exceeds max_length")
            prompt = row["prompt_ids"]
            if len(prompt) > available:
                count += 1
                prompt = prompt[-available:]
            return {"prompt_ids": prompt}

        result = dataset.map(trim)
        tokenization[name] = {"n": len(dataset), "left_truncated_prompts": count}
        return result

    if len(trainer.train_dataset) != len(train):
        raise ValueError("Trainer unexpectedly dropped training pairs")
    trainer.train_dataset = budget(trainer.train_dataset, "train")
    for name, dataset in trainer.eval_dataset.items():
        expected = (
            held if name == "all" else [r for r in held if chosen_kind(r) == name]
        )
        if len(dataset) != len(expected):
            raise ValueError(f"Trainer unexpectedly dropped {name} held-out pairs")
        trainer.eval_dataset[name] = budget(dataset, name)
    trainer.turnguard_tokenization = tokenization
    logger.info("A4 context budgeting (completion tokens unchanged): %s", tokenization)
    return trainer, api


def train_a4(
    pairs: str | Path,
    *,
    model: str = "headline",
    arm: str = "A4",
    smoke: bool = False,
    dry_run: bool = False,
    output: str | Path | None = None,
    root: Path | None = None,
    trainer_factory=None,
) -> dict:
    """Preflight data on CPU; construct/train a GPU trainer only when requested."""
    if arm != "A4":
        raise ValueError("Only A4 preference training is implemented")
    rows = load_training_pairs(pairs)  # Includes copied/renamed dev-data refusal.
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    settings = load_config(root / "configs/arms.yaml")["training"]["A4"]
    models = load_config(root / "configs/models.yaml")["models"]
    if model not in models:
        raise ValueError(f"Unknown model: {model}")
    if smoke:
        # Deterministic, small smoke population; real training always uses all.
        rows = sorted(
            rows, key=lambda r: hashlib.sha256(("1234" + r["id"]).encode()).hexdigest()
        )[:20]
    train, held = split_pairs(rows, settings["holdout_fraction"])
    target = Path(output) if output else root / settings["adapters"][model]
    if smoke and output is None:
        target = target.with_name(target.name + "-smoke")
    config = {
        **settings,
        "model_id": models[model],
        "arm": arm,
        "seed": 1234,
        "smoke": smoke,
        "adapter": str(target.resolve()),
        "pairs": str(Path(pairs).resolve()),
        "pairs_sha256": hashlib.sha256(
            (
                Path(pairs) / "pairs.jsonl" if Path(pairs).is_dir() else Path(pairs)
            ).read_bytes()
        ).hexdigest(),
    }
    if smoke:
        config["gradient_accumulation_steps"] = 1
    counts = {
        "all": len(held),
        **{k: sum(chosen_kind(r) == k for r in held) for k in ("FINAL", "ACTION")},
    }
    report = {
        "config": config,
        "train_n": len(train),
        "heldout_n": counts,
        "chosen_n": dict(Counter(chosen_kind(row) for row in rows)),
        "train_ids": [r["id"] for r in train],
        "heldout_ids": [r["id"] for r in held],
        "estimated_tokens_per_epoch": sum(
            (
                sum(len(m["content"]) for m in r["prompt"]) * 2
                + len(r["chosen"])
                + len(r["rejected"])
            )
            // 4
            for r in train
        ),
        "token_estimate_method": (
            "characters/4; policy pairs only; reference adds another forward"
        ),
    }
    if dry_run:
        return report
    if target.exists():
        raise ValueError(f"Adapter output already exists; choose --output: {target}")
    timer = RunTimer()
    run = new_run_dir(arm, models[model], "train-a4", root=root)
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run / "pair_split.json").write_text(json.dumps(report, indent=2) + "\n")
    trainer = None
    try:
        trainer, api = (trainer_factory or _gpu_trainer)(config, train, held, run)
        trainer.train()
        metrics = trainer.evaluate()
        # Named eval subsets make FINAL vs ACTION reward accuracy visible at
        # each epoch and in the final summary, independently of terse answers.
        held_metrics = {}
        for kind, n in counts.items():
            loss = metrics.get(f"eval_{kind}_loss")
            accuracy = metrics.get(f"eval_{kind}_rewards/accuracies")
            if n and (loss is None or accuracy is None):
                raise ValueError(f"Missing held-out loss/reward accuracy for {kind}")
            held_metrics[kind] = {"n": n, "loss": loss, "reward_accuracy": accuracy}
        trainer.save_model(str(target))
        trainer.processing_class.save_pretrained(str(target))
        trainer.save_state()
        (run / "training_log.json").write_text(
            json.dumps(trainer.state.log_history, indent=2) + "\n"
        )
        report.update(
            tokenization=getattr(trainer, "turnguard_tokenization", {}),
            trainer_api=api,
            reference="adapter-disabled base",
            heldout=held_metrics,
            run_dir=str(run),
            adapter=str(target),
        )
        (run / "metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        (run / "arms.yaml").write_text(
            yaml.safe_dump({"arms": {"A4": {"adapter": str(target.resolve())}}})
        )
        return report
    except Exception as error:
        (run / "error.json").write_text(json.dumps({"error": str(error)}) + "\n")
        raise
    finally:
        if trainer is not None:
            (run / "training_log.json").write_text(
                json.dumps(trainer.state.log_history, indent=2) + "\n"
            )
        timer.finish(run)
