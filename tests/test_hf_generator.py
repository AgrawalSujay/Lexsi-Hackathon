"""Test HF adapter contracts using Python doubles: no torch, GPU, or downloads."""

import subprocess
import sys
import types
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from turnguard.models import GenRequest, HFGenerator


class Tensor:
    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.device = None

    def to(self, device: str) -> "Tensor":
        self.device = device
        return self

    def __getitem__(self, key: tuple) -> "Tensor":
        row_slice, column_slice = key
        return Tensor([row[column_slice] for row in self.rows[row_slice]])


class Tokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 99
    eos_token = "EOS"

    def __init__(self) -> None:
        self.templates = []
        self.encodings = []
        self.decoded = []

    @property
    def pad_token(self) -> str:
        return "PAD"

    @pad_token.setter
    def pad_token(self, token: str) -> None:
        assert token == self.eos_token
        self.pad_token_id = self.eos_token_id

    def apply_chat_template(self, messages: list, **kwargs: object) -> str:
        self.templates.append((messages, kwargs))
        return messages[-1]["content"]

    def __call__(self, prompts: list[str], **kwargs: object) -> dict:
        assert self.padding_side == "left"
        width = max(len(prompt) for prompt in prompts)
        tokens = Tensor(
            [[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in prompts]
        )
        mask = Tensor(
            [[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in prompts]
        )
        self.encodings.append((prompts, kwargs, tokens))
        return {"input_ids": tokens, "attention_mask": mask}

    def batch_decode(self, tensor: Tensor, **kwargs: object) -> list[str]:
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        self.decoded.extend(tensor.rows)
        return [f" FINAL: response {row[0]} " for row in tensor.rows]


@pytest.fixture
def dependencies(monkeypatch: pytest.MonkeyPatch) -> tuple:
    tokenizer = Tokenizer()
    model = Mock()
    model.device = "cpu-double"
    calls = []
    contexts = {"inference": False, "entries": 0}

    @contextmanager
    def inference_mode():
        contexts["inference"] = True
        contexts["entries"] += 1
        try:
            yield
        finally:
            contexts["inference"] = False

    def generate(**kwargs: object) -> Tensor:
        assert contexts["inference"]
        assert kwargs["input_ids"].device == model.device
        assert kwargs["attention_mask"].device == model.device
        calls.append(kwargs)
        return Tensor(
            [row + [10 + index] for index, row in enumerate(kwargs["input_ids"].rows)]
        )

    model.generate.side_effect = generate
    torch = types.ModuleType("torch")

    class FakeOOM(RuntimeError):
        pass

    torch.cuda = types.SimpleNamespace(OutOfMemoryError=FakeOOM, empty_cache=Mock())
    torch.manual_seed = Mock()
    torch.inference_mode = inference_mode
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("safetune", "safetune.runner", "safetune.runner.utils"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    helpers = types.ModuleType("safetune.runner.utils.model_utils")
    helpers.load_tok = Mock(return_value=tokenizer)
    helpers.load_model = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    return tokenizer, model, calls, helpers, torch, contexts


def requests() -> list[GenRequest]:
    return [
        GenRequest(str(index), 1, [{"role": "user", "content": text}])
        for index, text in enumerate(("a", "longer", "last"))
    ]


def test_loaders_padding_greedy_microbatching_and_continuation_slicing(
    dependencies: tuple,
) -> None:
    tokenizer, model, calls, helpers, torch, contexts = dependencies
    generator = HFGenerator("configured-model-id", batch_size=2)
    assert generator.name == "configured-model-id"
    helpers.load_tok.assert_called_once_with("configured-model-id")
    helpers.load_model.assert_called_once_with("configured-model-id")
    torch.manual_seed.assert_called_once_with(1234)
    model.eval.assert_called_once()
    assert generator.generate(requests()) == [
        " FINAL: response 10 ",
        " FINAL: response 11 ",
        " FINAL: response 10 ",
    ]
    assert [call["input_ids"].shape[0] for call in calls] == [2, 1]
    assert tokenizer.decoded == [[10], [11], [10]]
    assert contexts["entries"] == 2
    for call in calls:
        assert call["max_new_tokens"] == 256
        assert call["do_sample"] is False
        assert call["num_beams"] == call["num_return_sequences"] == 1
        assert call["return_dict_in_generate"] is False
        assert call["stop_strings"] == ["\nTOOL RESULT", "\nUser:"]
        assert call["tokenizer"] is tokenizer
        assert not any("FINAL" in stop for stop in call["stop_strings"])
    for _, kwargs in tokenizer.templates:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
    for _, kwargs, _ in tokenizer.encodings:
        assert kwargs == {
            "padding": True,
            "truncation": False,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
    assert generator.generate([]) == []
    assert len(calls) == 2


def test_intervention_wraps_each_generate_and_cleans_up_on_error(
    dependencies: tuple,
) -> None:
    _, model, _, _, _, _ = dependencies
    active = {"inside": False, "entered": 0, "exited": 0}

    @contextmanager
    def intervention():
        active["inside"] = True
        active["entered"] += 1
        try:
            yield
        finally:
            active["inside"] = False
            active["exited"] += 1

    old_generate = model.generate.side_effect

    def generate(**kwargs: object) -> Tensor:
        assert active["inside"]
        return old_generate(**kwargs)

    model.generate.side_effect = generate
    generator = HFGenerator("configured-id", batch_size=1, intervention=intervention)
    generator.generate(requests())
    assert active == {"inside": False, "entered": 3, "exited": 3}
    model.generate.side_effect = RuntimeError("generation failed")
    with pytest.raises(RuntimeError):
        generator.generate(requests()[:1])
    assert active == {"inside": False, "entered": 4, "exited": 4}


def test_missing_padding_token_uses_eos(dependencies: tuple) -> None:
    tokenizer = dependencies[0]
    tokenizer.pad_token_id = None
    generator = HFGenerator("configured-id")
    assert generator.tokenizer.pad_token_id == 99


@pytest.mark.parametrize("keyword", ["batch_size", "max_new_tokens"])
def test_invalid_sizes_fail_before_optional_imports(keyword: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HFGenerator("configured-id", **{keyword: 0})


def test_importing_models_never_imports_gpu_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import turnguard.models; "
            "assert 'torch' not in sys.modules; "
            "assert 'safetune' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_oom_once_retries_smaller_chunks_in_order(
    dependencies: tuple, caplog: pytest.LogCaptureFixture
) -> None:
    _, model, _, _, torch, _ = dependencies
    original = model.generate.side_effect
    sizes = []

    def generate(**kwargs: object) -> Tensor:
        sizes.append(kwargs["input_ids"].shape[0])
        if len(sizes) == 1:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return original(**kwargs)

    model.generate.side_effect = generate
    generator = HFGenerator("configured-id", batch_size=2)
    outputs = generator.generate(requests())
    assert sizes == [2, 1, 1, 1]
    assert outputs == [" FINAL: response 10 "] * 3
    torch.cuda.empty_cache.assert_called_once()
    assert "batch size 2 -> 1" in caplog.text


def test_persistent_oom_halves_to_one_then_propagates(
    dependencies: tuple, caplog: pytest.LogCaptureFixture
) -> None:
    _, model, _, _, torch, _ = dependencies
    sizes = []

    def generate(**kwargs: object) -> Tensor:
        sizes.append(kwargs["input_ids"].shape[0])
        raise torch.cuda.OutOfMemoryError("still out of memory")

    model.generate.side_effect = generate
    generator = HFGenerator("configured-id", batch_size=4)
    batch = requests() + requests()[:1]
    with pytest.raises(torch.cuda.OutOfMemoryError, match="still out of memory"):
        generator.generate(batch)
    assert sizes == [4, 2, 1]
    assert torch.cuda.empty_cache.call_count == 3
    assert "batch size 4 -> 2" in caplog.text
    assert "batch size 2 -> 1" in caplog.text


def test_explicit_fp16_reaches_safetune_without_bf16(dependencies):
    _, _, _, helpers, torch, _ = dependencies
    torch.float16 = object()
    generator = HFGenerator("tiny-aya", dtype="fp16")
    helpers.load_model.assert_called_once_with("tiny-aya", dtype=torch.float16)
    assert generator.dtype == "fp16"
    assert not hasattr(torch, "bfloat16")
