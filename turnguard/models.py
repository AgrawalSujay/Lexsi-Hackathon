"""Define the generator interface and a scripted CPU-only generator."""

import json
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


class UnsupportedDtypeError(ValueError):
    """A safe-to-display precision preflight error, before model generation."""


def validate_cuda_dtype(torch, dtype: str, *, require_cuda: bool = False) -> None:
    """Reject unsupported bf16 before loading weights; fp16 uses CUDA loss scaling."""
    if dtype not in ("fp16", "bf16", "fp32"):
        raise ValueError("dtype must be fp16, bf16 or fp32")
    if require_cuda and not torch.cuda.is_available():
        raise ValueError("A4 requires a CUDA GPU; use --dry-run on CPU")
    if dtype == "bf16" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise UnsupportedDtypeError(
            "bf16 is not supported on this GPU; use --dtype fp16 on T4"
        )


class APIGenerator:
    """Greedy chat completions from any OpenAI-compatible endpoint (stdlib only).

    Responses are cached on disk by request hash, so an interrupted or repeated
    run never pays twice. Stop strings are sent and also enforced client-side.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        *,
        max_new_tokens: int = 256,
        concurrency: int = 4,
        rpm: float | None = None,
        cache_dir: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 6,
        extra_body: dict | None = None,
    ) -> None:
        if not model or not api_base or not api_key:
            raise ValueError("API generation needs a model, base URL and key")
        self.name = model
        self.api_base = api_base.rstrip("/")
        self._key = api_key
        self.max_new_tokens = max_new_tokens
        self.concurrency = max(1, concurrency)
        self.interval = 60.0 / rpm if rpm else 0.0
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.extra_body = dict(extra_body or {})
        self.calls = 0
        self.cached = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        import threading

        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "stop": STOP_STRINGS,
            **self.extra_body,
        }

    def _throttle(self) -> None:
        import time

        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self.interval
        if wait:
            time.sleep(wait)

    def _post(self, payload: dict) -> dict:
        import time
        import urllib.error
        import urllib.request

        body = json.dumps(payload).encode()
        for attempt in range(self.max_retries + 1):
            self._throttle()
            request = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                    return json.loads(reply.read())
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or error.code >= 500
                if not retryable or attempt == self.max_retries:
                    detail = error.read()[:300].decode("utf-8", "replace")
                    raise RuntimeError(f"API error {error.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.max_retries:
                    raise RuntimeError(f"API unreachable: {error}") from None
            time.sleep(min(60, 2**attempt * 2))
        raise RuntimeError("unreachable")

    def _one(self, request: GenRequest) -> str:
        import hashlib
        from pathlib import Path

        payload = self._payload(request.messages)
        key = hashlib.sha256(
            json.dumps([self.api_base, payload], sort_keys=True).encode()
        ).hexdigest()
        path = Path(self.cache_dir) / f"{key}.json" if self.cache_dir else None
        if path is not None and path.exists():
            try:
                text = json.loads(path.read_text("utf-8"))["text"]
            except (ValueError, KeyError, TypeError):
                logger.warning("Ignoring unreadable API cache entry %s", path.name)
            else:
                with self._lock:
                    self.cached += 1
                return text
        data = self._post(payload)
        text = data["choices"][0]["message"].get("content") or ""
        # Reasoning models may inline their hidden reasoning; grade the reply only.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].lstrip()
        for stop in STOP_STRINGS:
            text = text.split(stop)[0]
        with self._lock:
            self.calls += 1
            for k in self.usage:
                self.usage[k] += int((data.get("usage") or {}).get(k) or 0)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Identical requests can run concurrently (episodes sharing a prefix),
            # so each writer needs its own temporary file before the atomic rename.
            import os
            import threading

            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"text": text}, ensure_ascii=False), "utf-8")
            tmp.replace(path)
        return text

    def generate(self, batch: list[GenRequest]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.concurrency) as pool:
            outputs = list(pool.map(self._one, batch))
        logger.info(
            "API calls %s (cached %s), tokens %s", self.calls, self.cached, self.usage
        )
        return outputs


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
        if dtype is not None:
            validate_cuda_dtype(torch, dtype)
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
