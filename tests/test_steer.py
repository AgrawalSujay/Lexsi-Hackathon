"""A2 math, token placement and runtime lifecycle, with NumPy tensor doubles."""

import json
import pickle
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from turnguard.arms import steer

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    device = "cpu"

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return Tensor(self.data.astype(np.float32))

    def clone(self):
        return Tensor(self.data.copy())

    def to(self, **kwargs):
        return Tensor(self.data.astype(kwargs.get("dtype", self.dtype)))

    def mean(self, dim):
        return Tensor(self.data.mean(axis=dim))

    def norm(self):
        return Tensor(np.linalg.norm(self.data))

    def all(self):
        return Tensor(self.data.all())

    def item(self):
        return self.data.item()

    def __getitem__(self, index):
        return Tensor(self.data[index])

    def __setitem__(self, index, value):
        self.data[index] = value.data if isinstance(value, Tensor) else value

    def __add__(self, value):
        return Tensor(self.data + (value.data if isinstance(value, Tensor) else value))

    def __sub__(self, value):
        return Tensor(self.data - value.data)

    def __truediv__(self, value):
        return Tensor(self.data / value)

    def __rmul__(self, value):
        return Tensor(value * self.data)


class Block:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        handle = types.SimpleNamespace(active=True)
        handle.remove = lambda: setattr(handle, "active", False)
        self.hooks.append((handle, hook))
        return handle

    def apply(self, value):
        for handle, hook in self.hooks:
            if handle.active:
                result = hook(self, (), value)
                if result is not None:
                    value = result
        return value


class Model:
    device = "cpu"

    def __init__(self):
        self.blocks = [Block(), Block()]
        self.forwards = []
        self.base_model = self
        self.fail = False

    def __call__(self, input_ids, attention_mask, use_cache):
        assert not use_cache
        self.forwards.append(input_ids.data.copy())
        if self.fail:
            raise RuntimeError("fake forward failure")
        ids = input_ids.data
        hidden = Tensor(np.stack([ids, 2 * ids], axis=-1))
        for i, block in enumerate(self.blocks):
            if i:
                hidden = Tensor(hidden.data + np.stack([ids * 0, ids], axis=-1))
            hidden = block.apply(hidden)
        return hidden


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        return [10, 20]

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": [3, 5] if text == "FINAL: yes" else [1, 1]}


class CAA:
    def __init__(self, model, vectors, strength):
        self.model, self.vectors, self.strength = model, vectors, strength
        self.handles = []

    def __enter__(self):
        for layer, vector in self.vectors.items():
            self.handles.append(
                self.model.blocks[layer].register_forward_hook(self._make_hook(vector))
            )
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


@pytest.fixture
def fake_gpu(monkeypatch):
    def save(value, path):
        Path(path).write_bytes(pickle.dumps(value))

    def load(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return pickle.loads(Path(path).read_bytes())

    torch = types.SimpleNamespace(
        tensor=lambda value, **kw: Tensor(value),
        ones_like=lambda value: Tensor(np.ones_like(value.data)),
        isfinite=lambda value: Tensor(np.isfinite(value.data)),
        inference_mode=nullcontext,
        save=save,
        load=load,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(steer, "decoder_layers", lambda model: model.blocks)
    monkeypatch.setattr(steer, "caa_class", lambda: CAA)
    monkeypatch.setattr(steer, "_environment", lambda root: {"fake": True})
    return torch


@pytest.fixture
def pair_source(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.yaml").write_bytes(
        (ROOT / "configs/models.yaml").read_bytes()
    )
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    rows = [
        {
            "id": str(i),
            "source": "injected-synthesized",
            "split": "train",
            "validation_only": False,
            "prompt": [{"role": "user", "content": "task"}],
            "chosen": "FINAL: yes",
            "rejected": "FINAL: no",
        }
        for i in range(2)
    ]
    (pairs / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    generator = types.SimpleNamespace(
        name="CohereLabs/aya-expanse-8b", model=Model(), tokenizer=Tokenizer()
    )
    return tmp_path, pairs, rows, generator


def test_chosen_minus_rejected_window_math_and_normalization(fake_gpu, pair_source):
    _, _, rows, generator = pair_source
    vectors, stats = steer.extract_vectors(generator, rows, [0, 1])
    assert len(generator.model.forwards) == 4
    assert stats["forward_tokens"] == 16
    assert stats["raw_norms"][0] == pytest.approx(np.sqrt(20))
    np.testing.assert_allclose(vectors[0].data, np.array([1, 2]) / np.sqrt(5))
    np.testing.assert_allclose(
        vectors[1].data, np.array([1, 3]) / np.sqrt(10), rtol=1e-6
    )
    assert all(not h.active for b in generator.model.blocks for h, _ in b.hooks)
    generator.model.fail = True
    with pytest.raises(RuntimeError, match="fake forward"):
        steer.extract_vectors(generator, rows, [0, 1])
    assert all(not h.active for b in generator.model.blocks for h, _ in b.hooks)


def test_completion_window_is_equal_length_and_at_most_eight():
    class Tok(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(12 if text == "chosen" else 3))}

    assert steer.window_ids(
        Tok(), {"prompt": [], "chosen": "chosen", "rejected": "rejected"}
    ) == ([10, 20], [0, 1, 2], [0, 1, 2])
    assert (
        len(
            steer.window_ids(
                Tok(), {"prompt": [], "chosen": "chosen", "rejected": "chosen"}
            )[1]
        )
        == 8
    )


def test_artifact_provenance_runtime_prefill_decode_and_cleanup(fake_gpu, pair_source):
    root, path, _, generator = pair_source
    report = steer.build_vectors(path, root=root, generator=generator, layers=[0, 1])
    assert report["split"] == "train" and report["completion_tokens"] == 8
    assert report["pairs"] == 2 and report["environment"]["fake"]
    hook = steer.make_hook(
        generator=generator, vector=report["vector"], layer=0, alpha=4
    )
    original = Tensor(np.zeros((2, 3, 2)))
    with hook():
        result = generator.model.blocks[0].apply((original, "cache"))
        assert result[1] == "cache"
        np.testing.assert_array_equal(result[0].data[:, :-1], 0)
        np.testing.assert_allclose(
            result[0].data[:, -1],
            np.tile(4 * np.array([1, 2]) / np.sqrt(5), (2, 1)),
            rtol=1e-6,
        )
        current = generator.model.blocks[0].apply(Tensor(np.zeros((2, 1, 2))))
        assert np.count_nonzero(current.data) == 4
    np.testing.assert_array_equal(original.data, 0)
    assert all(not h.active for b in generator.model.blocks for h, _ in b.hooks)
    with pytest.raises(RuntimeError):
        with hook():
            raise RuntimeError("fake OOM")
    assert all(not h.active for b in generator.model.blocks for h, _ in b.hooks)
    with hook():  # The retry installs a fresh context, without double-addition.
        result = generator.model.blocks[0].apply(Tensor(np.zeros((1, 1, 2))))
        assert np.linalg.norm(result.data) == pytest.approx(4)


def test_runtime_refuses_wrong_model_layer_or_modified_file(fake_gpu, pair_source):
    root, path, _, generator = pair_source
    report = steer.build_vectors(path, root=root, generator=generator, layers=[0])
    generator.name = "another-model"
    with pytest.raises(ValueError, match="model does not match"):
        steer.make_hook(generator=generator, vector=report["vector"], layer=0, alpha=4)
    generator.name = report["model_id"]
    with pytest.raises(ValueError, match="Selected steering vector checksum"):
        steer.make_hook(
            generator=generator,
            vector=report["vector"],
            layer=0,
            alpha=4,
            vector_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="layer absent"):
        steer.make_hook(generator=generator, vector=report["vector"], layer=1, alpha=4)
    Path(report["vector"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        steer.make_hook(generator=generator, vector=report["vector"], layer=0, alpha=4)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_heldout_pairs_refused_before_any_gpu_load(pair_source, monkeypatch, split):
    root, path, rows, _ = pair_source
    for row in rows:
        row["split"] = split
    (path / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(
        steer, "HFGenerator", lambda *a, **kw: pytest.fail("GPU reached")
    )
    with pytest.raises(ValueError, match="never for training"):
        steer.build_vectors(path, root=root)


def test_dry_run_no_optional_imports_or_writes(pair_source):
    root, path, _, _ = pair_source
    report = steer.build_vectors(path, root=root, dry_run=True)
    assert report["layers"] == [10, 14, 18, 22]
    assert not (root / "steering").exists()


@pytest.mark.parametrize("value", ["", "-1", "1,1", "x,2"])
def test_invalid_layers(value):
    with pytest.raises(ValueError):
        steer.parse_layers(value)


@pytest.mark.parametrize("value", [[0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_vectors(fake_gpu, value):
    with pytest.raises(ValueError, match="finite, nonzero"):
        steer.unit_vector(Tensor(value))
