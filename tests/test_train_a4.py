"""A4 control flow and API contracts, without Torch, models, or network."""

import copy
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from turnguard.arms import harden
from turnguard.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pair_root(tmp_path):
    (tmp_path / "configs").mkdir()
    for name in ("arms.yaml", "models.yaml"):
        (tmp_path / "configs" / name).write_bytes(
            (ROOT / "configs" / name).read_bytes()
        )
    rows = [
        {
            "id": f"pair-{i:03}",
            "split": "train",
            "validation_only": False,
            "prompt": [
                {"role": "system", "content": "Original protocol"},
                {"role": "user", "content": f"Do task {i}"},
            ],
            "chosen": f"FINAL: code-{i}"
            if i % 2
            else 'ACTION: {"tool":"save_note","args":{"text":"code-1"}}',
            "rejected": "FINAL: I have written it, but did not save it.",
        }
        for i in range(20)
    ]
    directory = tmp_path / "data/pairs/train"
    directory.mkdir(parents=True)
    (directory / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path, directory, rows


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def save_pretrained(self, target):
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "tokenizer_config.json").write_text("{}")


class FakeTrainer:
    def __init__(self, config, train, held, run):
        self.config, self.train_rows, self.held, self.run = config, train, held, run
        self.processing_class = FakeTokenizer()
        self.state = types.SimpleNamespace(log_history=[])

    def train(self):
        self.state.log_history.append(
            {"step": 10 if self.config["smoke"] else 5, "loss": 0.4}
        )

    def evaluate(self):
        metrics = {}
        for kind in ("all", "FINAL", "ACTION"):
            if kind == "all" or any(harden.chosen_kind(r) == kind for r in self.held):
                metrics.update(
                    {f"eval_{kind}_loss": 0.3, f"eval_{kind}_rewards/accuracies": 0.75}
                )
        self.state.log_history.append(metrics)
        return metrics

    def save_model(self, target):
        Path(target).mkdir(parents=True)
        (Path(target) / "adapter_config.json").write_text("{}")
        (Path(target) / "adapter_model.safetensors").write_bytes(b"fake")

    def save_state(self):
        (self.run / "trainer_state.json").write_text("{}")


def test_twenty_pair_fake_training_and_holdout_outputs(pair_root):
    root, path, rows = pair_root
    seen = []

    def factory(*args):
        seen.append(FakeTrainer(*args))
        return seen[-1], "fake SafeTune CSTTrainer"

    result = harden.train_a4(path, root=root, trainer_factory=factory)
    assert result["train_n"] == 19
    assert result["heldout_n"]["all"] == 1
    assert set(result["train_ids"]).isdisjoint(result["heldout_ids"])
    cfg = seen[0].config
    assert (cfg["rank"], cfg["alpha"], cfg["learning_rate"], cfg["beta"]) == (
        16,
        32,
        5e-5,
        0.1,
    )
    assert cfg["warmup_ratio"] == 0.1
    assert cfg["epochs"] == 2 and cfg["max_length"] == 4096
    assert cfg["gradient_checkpointing"]
    assert result["reference"] == "adapter-disabled base"
    assert (Path(result["adapter"]) / "adapter_model.safetensors").exists()
    run = Path(result["run_dir"])
    assert (run / "training_log.json").exists()
    assert json.loads((run / "env.json").read_text())["elapsed_seconds"] >= 0
    assert (
        yaml.safe_load((run / "arms.yaml").read_text())["arms"]["A4"]["adapter"]
        == result["adapter"]
    )
    with pytest.raises(ValueError, match="already exists"):
        harden.train_a4(path, root=root, trainer_factory=factory)
    assert seen[0].train_rows + seen[0].held != rows  # Seeded split, not a tail slice.


def test_split_is_deterministic_stratified_and_both_metrics_have_counts():
    rows = [
        {
            "id": str(i),
            "chosen": "FINAL: ok" if i < 20 else 'ACTION: {"tool":"x","args":{}}',
        }
        for i in range(100)
    ]
    train, held = harden.split_pairs(rows)
    assert len(held) == 5 and len(train) == 95
    assert sum(harden.chosen_kind(r) == "FINAL" for r in held) == 1
    assert harden.split_pairs(list(reversed(rows))) == (train, held)


def test_smoke_and_dry_run_do_not_touch_validation_or_gpu(pair_root):
    root, path, _ = pair_root
    before = set(root.rglob("*"))
    result = harden.train_a4(path, root=root, smoke=True, dry_run=True)
    assert result["config"]["smoke"]
    assert result["config"]["adapter"].endswith("headline-smoke")
    assert result["config"]["gradient_accumulation_steps"] == 1
    assert set(root.rglob("*")) == before
    captured = []

    def factory(*args):
        captured.append(FakeTrainer(*args))
        return captured[-1], "fake"

    harden.train_a4(path, root=root, smoke=True, trainer_factory=factory)
    assert captured[0].state.log_history[0]["step"] == 10
    (path / "VALIDATION_ONLY").write_text("validation — never for training")
    with pytest.raises(ValueError, match="never for training"):
        harden.train_a4(
            path, root=root, trainer_factory=lambda *a: pytest.fail("GPU reached")
        )


def test_conversational_preserves_exact_prompt_without_inversion(pair_root):
    _, _, rows = pair_root
    original = copy.deepcopy(rows)
    converted = harden.conversational(rows)
    assert converted[0]["prompt"] == rows[0]["prompt"]
    assert converted[0]["chosen"] == [
        {"role": "assistant", "content": rows[0]["chosen"]}
    ]
    assert len(converted) == len(rows)
    converted[0]["prompt"][0]["content"] = "changed"
    assert rows == original


def test_named_reward_metrics_do_not_overwrite_final_with_action():
    class Base:
        def __init__(self):
            self._metrics = {"eval": {}}
            self.logs = []

        def log(self, logs, start_time=None):
            logs.update(
                {f"eval_{k}": sum(v) / len(v) for k, v in self._metrics["eval"].items()}
            )
            self.logs.append(dict(logs))
            self._metrics["eval"].clear()

    trainer = harden.named_metrics_trainer(Base)()
    for kind, value in [("FINAL", 0.25), ("ACTION", 0.75)]:
        trainer._metrics["eval"]["rewards/accuracies"] = [value]
        trainer.log({f"eval_{kind}_loss": 0.5})
    assert trainer.logs[0]["eval_FINAL_rewards/accuracies"] == 0.25
    assert trainer.logs[1]["eval_ACTION_rewards/accuracies"] == 0.75
    assert "eval_rewards/accuracies" not in trainer.logs[0]


def test_api_uses_safetune_and_only_missing_path_falls_back(monkeypatch):
    fake = types.SimpleNamespace(CSTTrainer=object, CSTConfig=dict)
    monkeypatch.setattr(harden.importlib, "import_module", lambda name: fake)
    assert harden.trainer_api() == (
        object,
        dict,
        "safetune.interventions.harden.cst.CSTTrainer",
    )

    def missing(name):
        raise ModuleNotFoundError(name="safetune.interventions.harden.cst")

    monkeypatch.setattr(harden.importlib, "import_module", missing)
    monkeypatch.setitem(
        sys.modules, "trl", types.SimpleNamespace(DPOTrainer=object, DPOConfig=dict)
    )
    assert harden.trainer_api()[2].startswith("trl.DPOTrainer")

    def dependency_missing(name):
        raise ModuleNotFoundError(name="torch")

    monkeypatch.setattr(harden.importlib, "import_module", dependency_missing)
    with pytest.raises(ImportError, match="gpu extra"):
        harden.trainer_api()


def test_cli_wires_training_options(pair_root, monkeypatch, capsys):
    _, path, _ = pair_root
    args = []
    monkeypatch.setattr(
        harden, "train_a4", lambda *a, **kw: args.append((a, kw)) or {"ok": True}
    )
    assert (
        main(
            [
                "train",
                "--arm",
                "A4",
                "--pairs",
                str(path),
                "--model",
                "dev",
                "--smoke",
                "--dry-run",
            ]
        )
        == 0
    )
    assert args[0][1]["model"] == "dev"
    assert args[0][1]["smoke"] and args[0][1]["dry_run"]


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_gpu_api_contract_with_fake_modules_and_context_budget(
    pair_root, monkeypatch, dtype
):
    root, path, rows = pair_root
    captured = {}

    class Dataset(list):
        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def map(self, fn):
            return Dataset([{**row, **fn(copy.deepcopy(row))} for row in self])

    class Model:
        config = types.SimpleNamespace(use_cache=True)

        def named_modules(self):
            return [
                (key, None)
                for key in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            ]

        def disable_adapter(self):
            pass

    class Base:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]
            self.ref_model = kwargs["ref_model"]

            def tokenize(ds):
                return Dataset(
                    [
                        {
                            **r,
                            "prompt_ids": list(range(4200)),
                            "chosen_ids": [1, 2],
                            "rejected_ids": [3, 4, 5],
                        }
                        for r in ds
                    ]
                )

            self.train_dataset = tokenize(kwargs["train_dataset"])
            self.eval_dataset = {
                k: tokenize(v) for k, v in kwargs["eval_dataset"].items()
            }

    monkeypatch.setattr(
        harden, "trainer_api", lambda: (Base, lambda **kw: kw, "fake CSTTrainer")
    )

    def module(name, **values):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**values))

    module(
        "torch",
        bfloat16="bf16",
        float16="fp16",
        cuda=types.SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: True
        ),
    )
    module("datasets", Dataset=Dataset)
    module("peft", LoraConfig=lambda **kw: kw)
    module("transformers", set_seed=lambda seed: captured.update(seed=seed))
    module(
        "safetune.runner.utils.model_utils",
        load_tok=lambda name: FakeTokenizer(),
        load_model=lambda *a, **kw: Model(),
    )
    cfg = harden.train_a4(
        path,
        root=root,
        dry_run=True,
        dtype=dtype,
        model="dev" if dtype == "fp16" else "headline",
    )["config"]
    trainer, _ = harden._gpu_trainer(cfg, rows[:18], rows[18:], root)
    args = captured["args"]
    assert args["bf16"] == (dtype == "bf16")
    assert args["fp16"] == (dtype == "fp16")
    assert args["max_steps"] == -1 and args["gradient_accumulation_steps"] == 8
    assert args["max_length"] == (2048 if dtype == "fp16" else 4096)
    assert args["beta"] == 0.1
    assert args["learning_rate"] == 5e-5 and args["warmup_steps"] == 0.1
    assert args["prediction_loss_only"] and not args["precompute_ref_log_probs"]
    assert captured["ref_model"] is None
    assert (
        captured["peft_config"]["r"] == 16
        and captured["peft_config"]["lora_alpha"] == 32
    )
    assert set(captured["eval_dataset"]) == {"all", "FINAL", "ACTION"}
    assert captured["processing_class"].padding_side == "left"
    assert trainer.turnguard_tokenization["train"]["left_truncated_prompts"] == 18
    first = trainer.train_dataset[0]
    assert first["chosen_ids"] == [1, 2] and first["rejected_ids"] == [3, 4, 5]
    assert len(first["prompt_ids"]) == args["max_length"] - 3
    assert first["prompt_ids"][-1] == 4199
    assert len(captured["train_dataset"][0]["prompt"]) == 2  # original messages


def test_missing_reward_metrics_fail_instead_of_reporting_success(pair_root):
    root, path, _ = pair_root

    class Missing(FakeTrainer):
        def evaluate(self):
            return {"eval_all_loss": 0.5}

    with pytest.raises(ValueError, match="Missing held-out"):
        harden.train_a4(
            path, root=root, trainer_factory=lambda *args: (Missing(*args), "fake")
        )
    assert not (root / "adapters/A4/headline").exists()
    assert list((root / "runs").glob("*/error.json"))


def test_latest_checkpoint_ignores_incomplete_saves(tmp_path):
    for step, complete in ((20, True), (40, True), (60, False)):
        path = tmp_path / f"checkpoint-{step}"
        path.mkdir()
        (path / "adapter_model.safetensors").write_text("w")
        if complete:
            (path / "trainer_state.json").write_text("{}")
    assert harden.latest_checkpoint(tmp_path) == tmp_path / "checkpoint-40"
    assert harden.latest_checkpoint(tmp_path / "missing") is None
    assert harden.latest_checkpoint(None) is None


def test_training_resumes_from_checkpoint_dir(pair_root, tmp_path):
    root, path, _ = pair_root
    checkpoints = tmp_path / "ckpt"
    done = checkpoints / "checkpoint-20"
    done.mkdir(parents=True)
    (done / "adapter_model.safetensors").write_text("w")
    (done / "trainer_state.json").write_text("{}")
    seen = {}

    class Resumable(FakeTrainer):
        def train(self, resume_from_checkpoint=None):
            seen["resume"] = resume_from_checkpoint

    def factory(config, train, held, run):
        seen["checkpoint_dir"] = config["checkpoint_dir"]
        return Resumable(config, train, held, run), "fake.Trainer"

    result = harden.train_a4(
        path, root=root, trainer_factory=factory, checkpoint_dir=checkpoints
    )
    assert seen["checkpoint_dir"] == str(checkpoints.resolve())
    assert seen["resume"] == str(done)
    assert result["resumed_from"] == str(done)


def test_tiny_fp16_context_override_and_checkpoint_identity(pair_root):
    root, path, _ = pair_root
    cfg = harden.train_a4(path, root=root, model="dev", dtype="fp16", dry_run=True)[
        "config"
    ]
    assert cfg["max_length"] == 2048
    assert cfg["loss_scaling"] == "accelerate dynamic GradScaler"
    cfg["checkpoint_dir"] = str(root / "checkpoint-portable")
    harden._checkpoint_identity(cfg)
    harden._checkpoint_identity({**cfg, "pairs": "/relocated/path"})
    with pytest.raises(ValueError, match="settings differ"):
        harden._checkpoint_identity({**cfg, "dtype": "bf16"})
    overridden = harden.train_a4(
        path, root=root, model="dev", dtype="fp16", max_length=1024, dry_run=True
    )
    assert overridden["config"]["max_length"] == 1024
    result = harden.train_a4(
        path,
        root=root,
        model="dev",
        dtype="fp16",
        run_root=root / "job",
        trainer_factory=lambda *a: (FakeTrainer(*a), "fake"),
    )
    assert Path(result["run_dir"]).is_relative_to(root / "job/runs")
