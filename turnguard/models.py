"""Define the generator interface and a scripted CPU-only generator."""

import logging
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


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
        self._scripts = {
            episode_id: list(script) for episode_id, script in scripts.items()
        }
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


STOP_STRINGS = ["\nTOOL RESULT", "\nUser:"]


class HFGenerator:
    """Generate greedy continuations through SafeTune's shared model loaders."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        intervention: Callable[[], AbstractContextManager] | None = None,
        *,
        seed: int = 1234,
        dtype: str | None = None,
    ) -> None:
        if dtype not in (None, "fp16", "bf16", "fp32"):
            raise ValueError("dtype must be fp16, bf16 or fp32")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        # Optional dependencies are touched only when an HF generator is created.
        try:
            import torch
            from safetune.runner.utils.model_utils import load_model, load_tok
        except ImportError as error:
            raise ImportError(
                "HF generation requires the gpu extra: pip install -e '.[gpu]'"
            ) from error
        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.intervention = intervention
        self._torch = torch
        torch.manual_seed(seed)
        self.tokenizer = load_tok(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has neither a padding token nor an EOS token"
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if dtype is None:
            self.model = load_model(model_id)
        else:
            torch_dtype = getattr(
                torch, {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[dtype]
            )
            self.model = load_model(model_id, dtype=torch_dtype)
        self.dtype = dtype
        self.model.eval()

    def generate(self, batch: list[GenRequest]) -> list[str]:
        outputs = []
        start = 0
        batch_size = self.batch_size
        while start < len(batch):
            chunk = batch[start : start + batch_size]
            try:
                decoded = self._generate_chunk(chunk)
            except self._torch.cuda.OutOfMemoryError:
                if len(chunk) == 1:
                    self._torch.cuda.empty_cache()
                    raise
                smaller = max(1, len(chunk) // 2)
            else:
                outputs.extend(decoded)
                start += len(chunk)
                continue
            # Leave the exception handler first: its traceback can otherwise
            # retain the failed helper's CUDA tensors while clearing the cache.
            self._torch.cuda.empty_cache()
            logger.warning(
                "CUDA OOM: retrying chunk at offset %s with batch size %s -> %s",
                start,
                len(chunk),
                smaller,
            )
            batch_size = smaller
        return outputs

    def _generate_chunk(self, chunk: list[GenRequest]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for request in chunk
        ]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        # Decoder-only output includes the entire left-padded input width,
        # not each example's unpadded prompt length.
        width = encoded["input_ids"].shape[1]
        context = (
            self.intervention() if self.intervention is not None else nullcontext()
        )
        with self._torch.inference_mode(), context:
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=False,
                # Stop when the model starts inventing a tool result or the next
                # user turn. Never stop on FINAL: an answer may follow a preamble.
                stop_strings=STOP_STRINGS,
                tokenizer=self.tokenizer,
            )
        decoded = self.tokenizer.batch_decode(
            generated[:, width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != len(chunk):
            raise ValueError("Model must return one continuation per request")
        return decoded
