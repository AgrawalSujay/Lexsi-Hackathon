"""CPU-only grid merge: match inputs and select the winning shard's own vector."""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import yaml

from turnguard.arms.steer_tune import (
    GRID,
    RULE,
    extraction_inputs,
    grid_partition,
    pick_point,
    selected_config,
)
from turnguard.runs import RunTimer, new_run_dir


def merge_grid_runs(
    runs: list[Path], *, output: Path | None = None, root: Path | None = None
) -> dict:
    if not runs or len(runs) > 3:
        raise ValueError("Merge requires one to three grid run directories")
    paths = [Path(p).resolve() for p in runs]
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Duplicate grid run names")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate grid run directories")
    reference, points, shards, source_vectors = None, [], set(), {}
    common = (
        "model_id",
        "model_key",
        "dataset",
        "selected_ids",
        "composition",
        "baseline",
        "baseline_source_sha256",
        "grading_version",
        "rule",
        "decoding",
        "protocol_version",
    )
    count = None
    for path in paths:
        report = json.loads((path / "summary.json").read_text())
        if report["model_id"] != report["vector"].get("model_id") or report[
            "model_key"
        ] != report["vector"].get("model_key"):
            raise ValueError("Grid shard has different extraction model input")
        identity = {key: report[key] for key in common}
        identity["extraction"] = extraction_inputs(report["vector"])
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(
                "Grid shards have different extraction/dev/baseline inputs"
            )
        shard = report["grid_shard"]
        if count is None:
            count = shard["count"]
        if count != shard["count"] or shard["index"] in shards:
            raise ValueError("Duplicate or incompatible grid shard")
        shards.add(shard["index"])
        expected = grid_partition(f"{shard['index']}/{count}")
        actual = [(p["layer"], p["alpha"]) for p in report["points"]]
        if actual != expected or report["rule"] != RULE:
            raise ValueError("Incomplete, duplicate or altered grid point set/rule")
        key = report["model_key"]
        if key not in ("dev", "headline"):
            raise ValueError("Unknown vector model key")
        vector = path / "steering/A2" / f"{key}.pt"
        digest = hashlib.sha256(vector.read_bytes()).hexdigest()
        sidecar = json.loads(vector.with_suffix(".json").read_text())
        if (
            digest != report["vector"]["vector_sha256"]
            or digest != sidecar["vector_sha256"]
        ):
            raise ValueError("Shard vector checksum mismatch")
        if extraction_inputs(sidecar) != identity["extraction"]:
            raise ValueError("Shard sidecar input provenance differs")
        for point in report["points"]:
            if point["vector_sha256"] != digest:
                raise ValueError("Point was evaluated with a different vector")
            points.append({**point, "source_run": path.name})
        source_vectors[path.name] = vector
    if count != len(paths) or shards != set(range(1, count + 1)):
        raise ValueError("All grid shards must be present before selection")
    if {(p["layer"], p["alpha"]) for p in points} != set(GRID) or len(points) != 6:
        raise ValueError("Selection requires all six distinct grid points")
    chosen = pick_point(points, reference["baseline"])
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    timer = RunTimer()
    if output is None:
        output = new_run_dir(
            "A2",
            reference["model_id"],
            f"steer-merged-{uuid.uuid4().hex[:8]}",
            root=root,
        )
    else:
        output = Path(output).resolve()
        if any(output.is_relative_to(path) for path in paths):
            raise ValueError("Merge output cannot be inside a source run")
        output.mkdir(parents=True, exist_ok=False)
        (output / "env.json").write_text("{}")
    vector = source_vectors[chosen["source_run"]]
    key = reference["model_key"]
    destination = output / "steering/A2" / f"{key}.pt"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(vector, destination)
    shutil.copyfile(vector.with_suffix(".json"), destination.with_suffix(".json"))
    selected = selected_config(key, chosen, chosen["vector_sha256"])
    (output / "selected_arm.yaml").write_text(yaml.safe_dump(selected))
    destination.with_suffix(".yaml").write_text(yaml.safe_dump(selected))
    result = {
        **reference,
        "points": points,
        "chosen": chosen,
        "run_dir": str(output),
        "vector_source_run": chosen["source_run"],
        "vector_sha256": chosen["vector_sha256"],
        "source_runs": [str(p) for p in paths],
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    timer.finish(output)
    return result
