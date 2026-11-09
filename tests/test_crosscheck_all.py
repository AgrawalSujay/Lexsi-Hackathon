"""Split-isolated censuses, original filler briefs and shared GLM cache reuse."""

import copy
import hashlib
import json
import shutil

import pytest
from test_crosscheck import limiter, response, snapshot
from test_crosscheck import translation_build as translation_build
from test_drop_episodes import release as release
from test_drop_episodes import write_ids

from turnguard.cli import main
from turnguard.data.crosscheck import (
    GLMJudge,
    canonical,
    collect_filler,
    collect_items,
    crosscheck,
    summarize_filler,
)
from turnguard.data.drop_episodes import drop_episodes


def add_split(root, split):
    source = root / "data/v1/test"
    target = source.with_name(split)
    if split != "test":
        shutil.copytree(source, target)
        path = target / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["split"] = split
        path.write_text("".join(canonical(row) + "\n" for row in rows))
    return target


def add_filler(root, split="test", language="hi"):
    directory = add_split(root, split)
    fields = {"/files/filler-42.txt": "Invoice 42: the due date is Friday."}
    text = fields["/files/filler-42.txt"]
    brief = {
        "about": "Invoice 42 due Friday",
        "kind": "invoice_note",
        "language": language,
        "include_exactly_once": ["42", "Friday"],
    }
    cache = root / "data/cache"
    cache.mkdir(exist_ok=True)
    blocks = []
    for number in (1, 2):
        payload = {
            "task": "filler",
            "field": "/files/filler-42.txt",
            **brief,
            "scenario": f"scenario-{number}",
        }
        request = {"messages": [{"role": "user", "content": canonical(payload)}]}
        key = hashlib.sha256(canonical(request).encode()).hexdigest()
        (cache / f"{key}.json").write_text(
            canonical({"request": request, "text": text})
        )
        blocks.append(
            {
                "id": f"block-{number}",
                "scenario": payload["scenario"],
                "language": language,
                "fields": fields,
                "cache_keys": [key],
            }
        )
    rows = [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row["meta"]["filler_block"] = (
            "block-2" if row["episode_type"] == "authorized" else "block-1"
        )
        row["initial_state"].setdefault("files", {})["filler-42.txt"] = text
    (directory / "episodes.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in rows)
    )
    (directory / "filler_blocks.jsonl").write_text(
        "".join(canonical(block) + "\n" for block in blocks)
    )
    return directory, text


@pytest.fixture(autouse=True)
def teacher_settings(monkeypatch):
    monkeypatch.setenv("CK_API_BASE", "https://api.example.com")
    monkeypatch.setenv("CK_API_KEY", "secret")


@pytest.mark.parametrize("split", ["dev", "train"])
def test_collection_is_split_specific_and_census_label(translation_build, split):
    add_split(translation_build, split)
    shutil.rmtree(translation_build / "data/v1/test")
    shutil.rmtree(translation_build / "data/v1/test_cueflip")
    items, population = collect_items(translation_build, split)
    assert len(items) == 5
    assert set(population["episodes"]) == {split}
    assert all(item["test_cueflip_episode_ids"] == [] for item in items)
    result = crosscheck(root=translation_build, split=split, n=1000, dry_run=True)
    assert result["selection"] == "census" and result["census_n"] == 5
    assert "sample_n" not in result and "sample" not in result


def test_dev_then_train_reuses_shared_cache(translation_build):
    for split in ("dev", "train"):
        add_split(translation_build, split)
    calls = []

    def transport(*args):
        calls.append(args)
        return response()

    first = crosscheck(
        root=translation_build,
        split="dev",
        n=1000,
        transport=transport,
        limiter=limiter(),
    )
    second = crosscheck(
        root=translation_build,
        split="train",
        n=1000,
        transport=transport,
        limiter=limiter(),
    )
    assert len(calls) == 5
    assert first["usage"]["new_requests"]["completion_tokens"] == 400
    assert second["usage"]["new_requests"]["completion_tokens"] == 0
    for split in ("dev", "train"):
        output = translation_build / "data/crosscheck" / split
        assert (output / "verdicts.jsonl").is_file()
        assert not (output / "cache").exists()


def test_legacy_test_cache_is_reused_without_changes(translation_build):
    add_split(translation_build, "dev")
    items, _ = collect_items(translation_build, "dev")
    cache = translation_build / "data/crosscheck/test/cache"
    judge = GLMJudge(
        cache,
        model="openai/glm-5.3",
        api_base="https://api.example.com",
        api_key="secret",
        transport=lambda *_: response(),
        limiter=limiter(),
    )
    for item in items:
        judge.judge(item)
    before = snapshot(cache.parent)
    report = crosscheck(
        root=translation_build,
        split="dev",
        n=999,
        transport=lambda *_: pytest.fail("already cached"),
        limiter=limiter(),
    )
    assert report["estimate"]["teacher_calls"] == 0
    assert snapshot(cache.parent) == before


@pytest.mark.parametrize("split", ["dev", "train", "test"])
@pytest.mark.parametrize("filler", [False, True])
def test_read_only_root_cli_makes_no_writes(translation_build, split, filler, capsys):
    add_filler(translation_build, split)
    if split == "test":
        # Cue-flip records need their own block provenance for filler auditing.
        shutil.rmtree(translation_build / "data/v1/test_cueflip")
    before = snapshot(translation_build)
    args = [
        "crosscheck",
        "--split",
        split,
        "--dry-run",
        "--n",
        "999",
        "--root",
        str(translation_build),
    ]
    if filler:
        args.append("--filler")
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["selection"] == "census"
    assert report["estimate"]["teacher_calls"] == (1 if filler else 5)
    assert snapshot(translation_build) == before


def test_root_override_requires_dry_run(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["crosscheck", "--split", "train", "--root", str(tmp_path)])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("language", ["en", "hi", "hinglish"])
def test_filler_collects_original_brief_and_all_block_uses(translation_build, language):
    add_filler(translation_build, "test", language)
    shutil.rmtree(translation_build / "data/v1/test_cueflip")
    before = snapshot(translation_build)
    items, population = collect_filler(translation_build, "test")
    assert len(items) == 1
    item = items[0]
    assert item["brief"] == {
        "about": "Invoice 42 due Friday",
        "kind": "invoice_note",
        "language": language,
        "include": ["42", "Friday"],
    }
    assert len(item["filler_blocks"]) == 2
    assert item["episode_count"] == 3
    assert population["unique_filler_items"] == 1
    assert "injection" not in item["text"]
    assert snapshot(translation_build) == before


def test_filler_outputs_and_summary_do_not_replace_translations(translation_build):
    add_filler(translation_build, "test")
    shutil.rmtree(translation_build / "data/v1/test_cueflip")
    crosscheck(
        root=translation_build,
        n=999,
        transport=lambda *_: response(),
        limiter=limiter(),
    )
    output = translation_build / "data/crosscheck/test"
    before = {
        name: (output / name).read_bytes()
        for name in ("items.jsonl", "verdicts.jsonl", "disagreements.jsonl")
    }

    def filler_transport(request, secret):
        payload = json.loads(request["messages"][-1]["content"])
        assert set(payload) == {"text", "brief"}
        assert payload["brief"]["include"] == ["42", "Friday"]
        return response(
            '{"matches_brief":true,"natural":false,'
            '"contains_instruction":true,"issue":"extra instruction"}'
        )

    report = crosscheck(
        root=translation_build,
        filler=True,
        n=999,
        transport=filler_transport,
        limiter=limiter(),
    )
    assert report["overall"]["contains_instruction"]["k"] == 1
    assert report["overall"]["passes_audit"]["rate"] == 0
    assert all((output / name).read_bytes() == data for name, data in before.items())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["translations"]["overall"]["faithful"]["rate"] == 1
    assert summary["filler"]["selection"] == "census"
    rows = [
        json.loads(line)
        for line in (output / "filler_disagreements.jsonl").read_text().splitlines()
    ]
    assert len(rows[0]["filler_blocks"]) == 2 and len(rows[0]["episode_ids"]) == 3
    assert (output / "filler_verdicts.jsonl").is_file()
    filler_summary = copy.deepcopy(summary["filler"])
    crosscheck(
        root=translation_build,
        n=999,
        transport=lambda *_: pytest.fail("cached"),
        limiter=limiter(),
    )
    assert json.loads((output / "summary.json").read_text())["filler"] == filler_summary


def test_filler_test_includes_matching_cueflip_ids(translation_build):
    test_dir, _ = add_filler(translation_build, "test")
    cue = translation_build / "data/v1/test_cueflip"
    shutil.copyfile(test_dir / "filler_blocks.jsonl", cue / "filler_blocks.jsonl")
    rows = [
        json.loads(line) for line in (cue / "episodes.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row["meta"]["filler_block"] = "block-1"
    (cue / "episodes.jsonl").write_text("".join(canonical(row) + "\n" for row in rows))
    items, _ = collect_filler(translation_build, "test")
    assert len(items[0]["test_cueflip_episode_ids"]) == 3


def test_filler_wrong_schema_never_passes(translation_build, tmp_path):
    add_filler(translation_build, "dev")
    item = collect_filler(translation_build, "dev")[0][0]
    judge = GLMJudge(
        tmp_path / "audit-cache",
        model="openai/glm-5.3",
        api_base="base",
        api_key="secret",
        transport=lambda *_: response(),
        limiter=limiter(),
    )
    row = judge.judge(item)
    assert "error" in row and row["matches_brief"] is None
    assert summarize_filler([item], [row])["overall"]["passes_audit"]["n"] == 0


@pytest.mark.parametrize("split", ["dev", "train"])
def test_drop_all_splits_uses_filler_disagreements_and_keeps_trail(release, split):
    directory = add_split(release, split)
    output = release / "data/crosscheck" / split
    output.mkdir(parents=True)
    (output / "filler_disagreements.jsonl").write_text(
        canonical(
            {
                "key": "filler-42",
                "issue": "accidental instruction",
                "episode_ids": ["base-1"],
            }
        )
        + "\n"
    )
    ids = release / "ids.txt"
    ids.write_text("base-1\n")
    result = drop_episodes(split, ids, root=release)
    assert result["removed"] == 2
    assert all(row["item_key"] == "filler-42" for row in result["dropped"])
    assert (directory / "dropped.jsonl").is_file()
    assert json.loads((directory / "summary.json").read_text())["counts"] == {
        "injected": 1,
        "clean": 1,
        "authorized": 1,
    }
    before = snapshot(directory)
    (directory / "FROZEN").write_text("locked")
    with pytest.raises(ValueError, match="frozen"):
        drop_episodes(split, write_ids(release, "base-2"), root=release)
    assert {k: v for k, v in snapshot(directory).items() if k != "FROZEN"} == before


def test_live_incomplete_legacy_cache_is_read_only(translation_build):
    from turnguard.data.crosscheck import request_for, request_key

    add_split(translation_build, "dev")
    items, _ = collect_items(translation_build, "dev")
    cache = translation_build / "data/crosscheck/test/cache"
    cache.mkdir(parents=True)
    request = request_for(items[0], "openai/glm-5.3", "https://api.example.com")
    path = cache / f"{request_key(request)}.json"
    path.write_text('{"request":')  # Writer has not finished this entry yet.
    before = snapshot(cache.parent)
    result = crosscheck(root=translation_build, split="dev", n=999, dry_run=True)
    assert result["estimate"]["teacher_calls"] == 5
    assert snapshot(cache.parent) == before


def test_missing_original_filler_brief_fails_loudly(translation_build):
    directory, _ = add_filler(translation_build, "dev")
    block = json.loads((directory / "filler_blocks.jsonl").read_text().splitlines()[0])
    (translation_build / "data/cache" / f"{block['cache_keys'][0]}.json").unlink()
    with pytest.raises(ValueError, match="Missing filler brief cache"):
        collect_filler(translation_build, "dev")


def test_empty_translation_population_is_a_census(translation_build):
    directory = add_split(translation_build, "dev")
    rows = [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row["lang"] = {"conversation": "en", "injection": "en"}
    (directory / "episodes.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in rows)
    )
    result = crosscheck(root=translation_build, split="dev", n=999, dry_run=True)
    assert result["selection"] == "census" and result["census_n"] == 0
    assert result["estimate"]["teacher_calls"] == 0
