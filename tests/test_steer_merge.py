"""Shard selection compares inputs, never cross-GPU tensor byte equality."""

import hashlib
import json

import pytest
import yaml

from turnguard.arms.steer_merge import merge_grid_runs
from turnguard.arms.steer_tune import GRID, RULE, extraction_inputs, grid_partition


def make_shards(tmp_path):
    paths = []
    for i in range(1, 4):
        path = tmp_path / f"shard-{i}"
        vector = path / "steering/A2/headline.pt"
        vector.parent.mkdir(parents=True)
        # Deliberately different bytes: realistic GPU numeric differences.
        vector.write_bytes(f"vector-extracted-in-shard-{i}".encode())
        digest = hashlib.sha256(vector.read_bytes()).hexdigest()
        provenance = dict(
            version="caa-v1",
            pairs_sha256="pairs",
            model_id="model",
            model_key="headline",
            model_revision="revision-1",
            layers=[10, 14, 18, 22],
            completion_tokens=8,
            pooling="window",
            layer_indexing="zero-based",
            direction="chosen minus rejected",
            normalization="unit L2",
            environment={"git_commit": "code-commit", "git_dirty": False},
            vector_sha256=digest,
        )
        vector.with_suffix(".json").write_text(json.dumps(provenance))
        points = [
            dict(
                layer=layer,
                alpha=alpha,
                asr=0.1,
                asr_relayed=0.1,
                asr_incl_relay=0.01 if (layer, alpha) == (18, 8) else 0.2,
                clean_task_success=0.8,
                authorized_followed=0.7,
                ungraded=0,
                vector_sha256=digest,
                run_dir="evaluated-point",
            )
            for layer, alpha in grid_partition(f"{i}/3")
        ]
        report = dict(
            model_id="model",
            model_key="headline",
            dataset={"sha256": "dev"},
            selected_ids=["fixed-ids"],
            composition={"injected": 50, "clean": 50, "authorized": 50},
            baseline={
                "asr_incl_relay": 0.3,
                "clean_task_success": 0.8,
                "authorized_followed": 0.7,
            },
            baseline_source_sha256={"episodes.jsonl": "A0"},
            grading_version="c8-v1",
            rule=RULE,
            decoding={"do_sample": False},
            protocol_version="v2.4",
            vector=provenance,
            grid_shard={"index": i, "count": 3},
            points=points,
        )
        (path / "summary.json").write_text(json.dumps(report))
        paths.append(path)
    return paths


def test_grid_partition_is_disjoint_complete_and_bounded():
    for count in (1, 2, 3):
        parts = [grid_partition(f"{i}/{count}") for i in range(1, count + 1)]
        assert set().union(*(set(p) for p in parts)) == set(GRID)
        assert sum(len(p) for p in parts) == 6
    assert grid_partition(None) == GRID
    with pytest.raises(ValueError, match="at most 3"):
        grid_partition("1/4")


def test_merge_uses_winning_shards_vector_bytes_and_no_torch(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    paths = make_shards(tmp_path)
    result = merge_grid_runs(paths, output=tmp_path / "merged")
    assert result["chosen"]["layer"] == 18 and result["chosen"]["alpha"] == 8
    assert result["vector_source_run"] == "shard-1"
    expected = paths[0] / "steering/A2/headline.pt"
    actual = tmp_path / "merged/steering/A2/headline.pt"
    assert actual.read_bytes() == expected.read_bytes()
    settings = yaml.safe_load((tmp_path / "merged/selected_arm.yaml").read_text())
    args = settings["arms"]["A2"]["intervention_kwargs"]
    assert args["vector_sha256"] == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert (
        len(
            {
                json.loads((p / "summary.json").read_text())["vector"]["vector_sha256"]
                for p in paths
            }
        )
        == 3
    )


@pytest.mark.parametrize(
    "field",
    [
        "pairs_sha256",
        "model_id",
        "model_revision",
        "layers",
        "completion_tokens",
        "pooling",
    ],
)
def test_changed_extraction_inputs_refused(tmp_path, field):
    paths = make_shards(tmp_path)
    report = json.loads((paths[1] / "summary.json").read_text())
    report["vector"][field] = "different"
    (paths[1] / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different extraction"):
        merge_grid_runs(paths, output=tmp_path / "bad")


def test_commit_missing_revision_incomplete_and_duplicate_points_refused(tmp_path):
    paths = make_shards(tmp_path)
    with pytest.raises(ValueError, match="All grid shards"):
        merge_grid_runs(paths[:2], output=tmp_path / "missing")
    with pytest.raises(ValueError, match="Duplicate grid run"):
        merge_grid_runs([paths[0], paths[0]], output=tmp_path / "duplicate")
    report = json.loads((paths[1] / "summary.json").read_text())
    report["vector"]["environment"]["git_commit"] = "other-code"
    (paths[1] / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different extraction"):
        merge_grid_runs(paths, output=tmp_path / "code")
    report["vector"]["model_revision"] = None
    with pytest.raises(ValueError, match="Missing extraction input"):
        extraction_inputs(report["vector"])


def test_point_vector_mismatch_and_corruption_refused(tmp_path):
    paths = make_shards(tmp_path)
    report = json.loads((paths[0] / "summary.json").read_text())
    report["points"][0]["vector_sha256"] = "other-vector"
    (paths[0] / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evaluated with a different vector"):
        merge_grid_runs(paths, output=tmp_path / "bad-point")
    (paths[0] / "steering/A2/headline.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        merge_grid_runs(paths, output=tmp_path / "bad-file")


@pytest.mark.parametrize(
    "field",
    [
        "dataset",
        "selected_ids",
        "baseline_source_sha256",
        "grading_version",
        "decoding",
    ],
)
def test_changed_eval_inputs_refused(tmp_path, field):
    paths = make_shards(tmp_path)
    report = json.loads((paths[1] / "summary.json").read_text())
    report[field] = "other-input"
    (paths[1] / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different extraction/dev/baseline"):
        merge_grid_runs(paths, output=tmp_path / "bad")
