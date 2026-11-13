import json

from turnguard.models import APIGenerator, GenRequest


def test_api_generator_truncates_caches_and_counts(tmp_path, monkeypatch):
    sent = []

    def fake_post(self, payload):
        sent.append(payload)
        return {
            "choices": [{"message": {"content": "CALL x {}\nTOOL RESULT: fake"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

    monkeypatch.setattr(APIGenerator, "_post", fake_post)
    gen = APIGenerator("m", "https://example.invalid/v1", "k", cache_dir=str(tmp_path))
    req = [GenRequest("e1", 1, [{"role": "user", "content": "hi"}])]
    assert gen.generate(req) == ["CALL x {}"]
    assert sent[0]["temperature"] == 0 and sent[0]["stop"]
    assert gen.generate(req) == ["CALL x {}"]
    assert len(sent) == 1 and gen.calls == 1 and gen.cached == 1
    assert gen.usage == {"prompt_tokens": 10, "completion_tokens": 3}
    cached = list(tmp_path.glob("*.json"))
    assert len(cached) == 1 and "k" not in json.loads(cached[0].read_text())


def test_api_generator_requires_settings():
    import pytest

    with pytest.raises(ValueError):
        APIGenerator("", "https://example.invalid/v1", "k")
