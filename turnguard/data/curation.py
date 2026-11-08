"""CuratorKIT adapters: filler-only dedup, non-rewriting gates, lossless export."""

import copy
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from curatorkit.gates.schema import SchemaGate
from curatorkit.interfaces import BaseExporter, BaseGate, BaseNormalizer, BaseReader
from curatorkit.manifest import (
    DatasetCardGenerator,
    ProvenanceManifest,
    build_provenance,
)
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.pipeline import Pipeline
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

from turnguard.schema import validate_episode


def projection(value) -> str:
    """All visible state/turn strings and dictionary keys; no hash/provenance fields."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(projection(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(
            str(key) + "\n" + projection(child) for key, child in value.items()
        )
    return str(value)


@dataclass
class FillerBlock:
    id: str
    scenario: str
    variant: int
    language: str
    state: dict
    fields: dict[str, str]
    cache_keys: list[str]

    def sample(self) -> DataSample:
        # Only GENERATED values enter the comparison. No IDs, literal fixtures,
        # injection text, schema keys, route or user requests dilute similarity.
        return DataSample(
            id=self.id,
            source_uri=self.scenario,
            task_type="language_modeling",
            output="\n\n".join(self.fields.values()),
            metadata={
                "filler_block": {
                    "id": self.id,
                    "scenario": self.scenario,
                    "variant": self.variant,
                    "language": self.language,
                }
            },
        )


class MemoryReader(BaseReader):
    def __init__(self, samples: list[DataSample]):
        self.samples = samples

    def read(self) -> tuple[list[DataSample], list[RejectedSample]]:
        return self.samples, []


class TrackedDedup(BaseNormalizer):
    """Stock CK deduplicators omit rejection rows; retain their removed IDs."""

    def __init__(self, normalizer, removals: list[dict]):
        self.normalizer = normalizer
        self.removals = removals
        self._display_name = type(normalizer).__name__

    def run(self, samples: list[DataSample]) -> list[DataSample]:
        passed = self.normalizer.run(samples)
        keep = {sample.id for sample in passed}
        for sample in samples:
            if sample.id not in keep:
                self.removals.append({"block": sample.id, "reason": self._display_name})
        return passed


class EpisodeGate(BaseGate):
    def run(
        self, samples: list[DataSample]
    ) -> tuple[list[DataSample], list[RejectedSample]]:
        twins = {sample.id: sample.metadata["episode"] for sample in samples}
        passed, rejected = [], []
        for sample in samples:
            problems = validate_episode(sample.metadata["episode"], twins=twins)
            if problems:
                rejected.append(
                    RejectedSample(
                        **sample.model_dump(),
                        rejection_reason="episode_schema:" + "; ".join(problems),
                        rejecting_step="EpisodeGate",
                    )
                )
            else:
                passed.append(sample)
        return passed, rejected


class EpisodeExporter(BaseExporter):
    """Export exactly the nested episode; CK envelope is lineage, never dataset data."""

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "episodes.jsonl").write_text(
            "".join(
                json.dumps(
                    sample.metadata["episode"], ensure_ascii=False, sort_keys=True
                )
                + "\n"
                for sample in samples
            ),
            encoding="utf-8",
        )


class CuratorPass:
    def __init__(self, config: dict, *, clean: bool = False):
        if clean:
            raise ValueError(
                "TurnGuard requires clean=False to preserve exact episode text"
            )
        try:
            from curatorkit.hygiene.secrets import SecretsGate

            self.secrets = SecretsGate(fields=["output"])
        except ImportError as error:
            raise ImportError(
                "CuratorKIT SecretsGate needs detect-secrets; install the data extra "
                "or .venv/bin/python -m pip install 'detect-secrets>=1.4'"
            ) from error
        self.config = {**config, "clean": False, "pii_pseudonymizer": False}
        self.filler_result = None
        self.result = None
        self.removals = []
        self.block_samples = {}

    def _dedup_stage(self, samples: list[DataSample], removals: list[dict]):
        return Pipeline(
            [
                MemoryReader(samples),
                SchemaGate(
                    required_fields=["output", "metadata"],
                    min_tokens=1,
                    max_tokens=1_000_000,
                ),
                TrackedDedup(ExactDeduplicator(), removals),
                TrackedDedup(MinHashDeduplicator(**self.config["minhash"]), removals),
            ]
        ).run()

    def dedup(self, blocks: list[FillerBlock]) -> dict:
        """Two passes, so that only cross-scenario repetition can stop a build.

        1. Within each (scenario, language): near-identical filler variants
           collapse to one ("variant_collapse"). That only means fewer versions
           of the same scenario, which is expected at temperature 0, so it does
           not count toward the stop threshold.
        2. Across all survivors: a block duplicating another scenario's filler is
           removed and counts toward `max_removal_rate`, because that means the
           teacher is repeating itself across different contexts.
        """
        self.removals = []
        groups: dict[tuple[str, str], list[DataSample]] = {}
        for block in blocks:
            groups.setdefault((block.scenario, block.language), []).append(
                block.sample()
            )
        collapsed: list[dict] = []
        survivors: list[DataSample] = []
        for group in groups.values():
            found: list[dict] = []
            result = self._dedup_stage(group, found)
            survivors.extend(result.passed)
            collapsed.extend(
                {"block": row["block"], "reason": "variant_collapse:" + row["reason"]}
                for row in found
            )
            collapsed.extend(
                {"block": s.id, "reason": s.rejection_reason} for s in result.rejected
            )
        self.filler_result = self._dedup_stage(survivors, self.removals)
        for sample in self.filler_result.rejected:
            self.removals.append(
                {"block": sample.id, "reason": sample.rejection_reason}
            )
        counted = len(self.removals)
        self.removals.extend(collapsed)
        self.block_samples = {sample.id: sample for sample in self.filler_result.passed}
        return {
            "blocks": len(blocks),
            "removed": len(self.removals),
            "variant_collapsed": len(collapsed),
            "cross_scenario_removed": counted,
            "removal_rate": counted / len(blocks) if blocks else 0,
            "removals": self.removals,
            "stopped": bool(
                blocks and counted / len(blocks) > self.config["max_removal_rate"]
            ),
        }

    def gate(
        self, episodes: list[dict], sources: dict[str, str]
    ) -> tuple[list[dict], list[dict]]:
        samples = []
        for episode in episodes:
            block = episode["meta"]["filler_block"]
            sample = DataSample(
                id=episode["id"],
                source_uri=episode["meta"]["scenario"],
                task_type="language_modeling",
                output=projection([episode["user_turns"], episode["initial_state"]]),
                metadata={"episode": copy.deepcopy(episode)},
                provenance_chain=copy.deepcopy(
                    self.block_samples[block].provenance_chain
                ),
            )
            for path, digest in sources.items():
                sample.append_provenance(
                    ProvenanceRecord(
                        step_name="TurnGuardAssembly",
                        step_version="1",
                        config_hash=digest,
                        notes={
                            "source_file": path,
                            "sha256": digest,
                            "filler_block": block,
                        },
                    )
                )
            samples.append(sample)
        self.result = Pipeline(
            [
                MemoryReader(samples),
                EpisodeGate(),
                SchemaGate(
                    required_fields=["output", "metadata"],
                    min_tokens=1,
                    max_tokens=1_000_000,
                ),
                self.secrets,
            ]
        ).run()
        bad_blocks = {
            sample.metadata["episode"]["meta"]["filler_block"]
            for sample in self.result.rejected
        }
        keep = []
        for sample in self.result.passed:
            if sample.metadata["episode"]["meta"]["filler_block"] in bad_blocks:
                self.result.rejected.append(
                    RejectedSample(
                        **sample.model_dump(),
                        rejection_reason="filler_group_rejected",
                        rejecting_step="FillerGroupClosure",
                    )
                )
            else:
                keep.append(sample)
        self.result.stage_counts["FillerGroupClosure"] = {
            "input_count": len(self.result.passed),
            "output_count": len(keep),
            "rejected_count": len(self.result.passed) - len(keep),
        }
        self.result.passed = keep
        rejected = [
            {
                "id": sample.id,
                "block": sample.metadata["episode"]["meta"]["filler_block"],
                "reasons": [sample.rejection_reason],
            }
            for sample in self.result.rejected
        ]
        return [sample.metadata["episode"] for sample in keep], rejected

    def write(
        self, output: Path, report: dict, rejected: list[dict], sources: dict
    ) -> None:
        EpisodeExporter().export(self.result.passed, output)
        self.result.stage_counts["EpisodeExporter"] = {
            "exported_count": len(self.result.passed)
        }
        for name, counts in self.filler_result.stage_counts.items():
            self.result.stage_counts["Filler/" + name] = counts
        params = {
            "curator": self.config,
            "split": report["split"],
            "teacher": report["teacher"],
            "source_sha256": sources,
            "dedup": report["dedup"],
        }
        provenance = build_provenance(
            "turnguard-build-data",
            inputs=[
                {"path": path, "sha256": digest} for path, digest in sources.items()
            ],
            params=params,
        )
        manifest = ProvenanceManifest(
            self.result,
            pipeline_config_hash=hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest(),
            output_dir=output,
            provenance=provenance,
        )
        manifest.write()
        path = output / "manifest.json"
        data = json.loads(path.read_text())
        data["turnguard"] = {
            "clean": False,
            "pii_pseudonymizer": False,
            "counts": report["counts"],
            "dedup": report["dedup"],
            "rejected": len(rejected),
            "source_sha256": sources,
        }
        data["tool_versions"]["detect-secrets"] = version("detect-secrets")
        if report["teacher"] == "stub":
            # Stub artifacts remain byte-reproducible; real runs retain CK timing.
            data["run_timestamp"] = "1970-01-01T00:00:00Z"
            data["wall_clock_seconds"] = 0
            provenance.pop("created_at", None)
            data["provenance"] = provenance
        path.write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "lexsi_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        card = DatasetCardGenerator().generate(
            data, output, pipeline_name="TurnGuard", write_readme=False
        )
        with card.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## TurnGuard invariants\n\n"
                "PII rewriting is disabled: sandbox addresses, identifiers and "
                "filenames are intentional oracle/relay tokens. Rewriting them "
                "would change the experiment. SecretsGate rejects credentials; "
                "it never redacts. clean=False preserves exact text.\n\n"
                "Dedup compares generated filler blocks "
                "(scenario × variant × language), not episodes. "
                "Dependent base/twin/authorized episodes stay together.\n\n"
                "Filler blocks: "
                f"{report['dedup']['variant_collapsed']} near-identical versions of "
                "the same scenario merged (their episodes use the surviving "
                f"version); {report['dedup']['cross_scenario_removed']}/"
                f"{report['dedup']['blocks']} removed as cross-scenario duplicates "
                f"({report['dedup']['removal_rate']:.1%}); the build stops above 5%.\n"
            )
        # Card and all other artifacts precede checksums; omit checksum itself.
        manifest.write_checksums(
            sorted(
                p
                for p in output.rglob("*")
                if p.is_file() and p.name != "checksums.txt"
            )
        )
