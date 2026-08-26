"""THE LOCAL ARTIFACT INDEX — this machine's inventory of installed checkpoints.

It answers ONE question: what does the ref `org/repo[@release]` mean on THIS disk — which
store root, which snapshot per component, which construction config, which custody, which
uploader-minted stamps. A row is a projection of what TensorFS's ingest border hands back,
never a second description of the bytes: `bytes` is copied from the header, and the index
holds no tensor facts of its own.

IT MOVED OUT OF THE CLI LAYER (#567e), and that is a correction rather than a shuffle. The
old header called ref resolution "a CLI-layer capability the runtime CORE does not have",
because the only resolver that had ever existed was the one that also wrote the record. On
a rented pod those are different machines: the owner's index cannot answer what a ref means
on the pod's disk, and the worker — which is `internal` — is the thing that has to ask.
#565b's rule is RESOLUTION IS THE RESOLVER'S OWN, and an index only the record-WRITER may
read is a resolver bolted to the wrong machine.

What the law protected is still true and is unmoved: the fill plane and the author surface
are handed exact store roots and exact snapshot digests, and neither can name a ref. The
resolving happens once, at the boundary where a binding record becomes a prepare — one
layer below where it used to, on the machine that is actually going to serve.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cozy_runtime.internal.exits import Exit


class ArtifactRefError(ValueError):
    """A string that is not a ref reached the index. It carries the CLI's own words so the
    one caller that can be handed a human's typo renders them without inventing a second
    message for one refusal."""

    def __init__(self, message: str, remedy: str, exit_code: Exit = Exit.usage) -> None:
        super().__init__(message)
        self.remedy = remedy
        self.exit_code = exit_code


SCHEMA = "cozy.runtime.LocalArtifact/1"
DIRECTORY = "artifacts"
#: `org/repo`, optionally `@release`. A bare ref resolves on tensorhub only (never a foreign
#: fallback), which is a `pull` concern; here it is only the index key's grammar.
REF = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*(@[A-Za-z0-9][A-Za-z0-9._-]*)?$")


@dataclass(frozen=True, slots=True)
class Artifact:
    """One installed checkpoint, in checkpoint-and-lane vocabulary."""

    ref: str
    store: str
    config: str
    snapshots: Mapping[str, str]
    lane: str = "plain"
    bytes: int = 0
    tensors: int = 0
    variant: str = "sm89"
    custody: str = "canonical"
    """`canonical` (the border classified every key) or `local` (an un-bordered local bind:
    unknown extra keys warn, are ignored and are confessed — §1.1's custody tier)."""
    vram_floor_bytes: int = 0
    """The PHYSICAL FLOOR this artifact declares — activations + scratch + one component
    chunk. Below it no rung exists and a local run exits 14 with the quantified shortfall.
    A deployment fact, which locally means an installed-artifact fact; 0 takes the CLI's
    conservative default rather than asserting a floor the border never measured."""
    extra_key_markers: tuple[str, ...] = field(default_factory=tuple)
    """Known marker rules an extra key may match, e.g. `v_pred`. A match is transcribed or
    refused NAMING the marker; it is never ignored."""
    stamps: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """THE CHECKPOINT'S UPLOADER-MINTED STAMP SET, as the hub's catalog lane carries it
    (#567d). Key -> the canonically ordered set of values, verified at publish against the
    border's own derivations; the runtime interprets none of them and only JOINS a model
    class's declared keywords against them, failing closed on absence.

    It lives HERE — beside store, snapshots and custody — because a stamp is a fact about
    the checkpoint that the resolver of a ref has to hand over with the rest of them. The
    tensor bytes never carried it and never should: `cozytensors/1` holds configs,
    encodings and components, and a fact the hub verifies is not a fact the header can
    assert. Absent means this row was written before its lane had stamps, which fails every
    stamped model class closed — exactly as unstamped should."""

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(sorted(self.snapshots))

    @property
    def primary(self) -> str:
        """The snapshot a single-component reader falls back to. Deterministic by order."""
        return self.snapshots[self.components[0]]

    def document(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "ref": self.ref,
            "store": self.store,
            "config": self.config,
            "snapshots": dict(sorted(self.snapshots.items())),
            "lane": self.lane,
            "bytes": self.bytes,
            "tensors": self.tensors,
            "variant": self.variant,
            "custody": self.custody,
            "vram_floor_bytes": self.vram_floor_bytes,
            "extra_key_markers": list(self.extra_key_markers),
            "stamps": {k: list(v) for k, v in sorted(self.stamps.items())},
        }


def _slug(ref: str) -> str:
    return ref.replace("/", "__").replace("@", "--")


def directory(cozy_home: Path) -> Path:
    return cozy_home / DIRECTORY


def install(cozy_home: Path, artifact: Artifact) -> Path:
    """Record one installed artifact. Idempotent: identical facts rewrite the same bytes."""
    if not REF.match(artifact.ref):
        raise ArtifactRefError(
            f"{artifact.ref!r} is not a ref: org/repo[@release]",
            "refs are org/repo with an optional @release; a filesystem path is admitted "
            "through `cozy-runtime ingest` first",
        )
    target = directory(cozy_home)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{_slug(artifact.ref)}.json"
    path.write_text(json.dumps(artifact.document(), indent=2, sort_keys=True))
    return path


def rows(cozy_home: Path, prefix: str = "") -> list[Artifact]:
    target = directory(cozy_home)
    if not target.is_dir():
        return []
    out: list[Artifact] = []
    for path in sorted(target.glob("*.json")):
        doc = json.loads(path.read_text())
        if doc.get("schema") != SCHEMA:
            continue
        if prefix and not str(doc["ref"]).startswith(prefix):
            continue
        out.append(
            Artifact(
                ref=str(doc["ref"]),
                store=str(doc["store"]),
                config=str(doc["config"]),
                snapshots=dict(doc["snapshots"]),
                lane=str(doc.get("lane", "plain")),
                bytes=int(doc.get("bytes", 0)),
                tensors=int(doc.get("tensors", 0)),
                variant=str(doc.get("variant", "sm89")),
                custody=str(doc.get("custody", "canonical")),
                vram_floor_bytes=int(doc.get("vram_floor_bytes", 0)),
                extra_key_markers=tuple(doc.get("extra_key_markers", ())),
                stamps={
                    str(k): tuple(str(x) for x in v) for k, v in (doc.get("stamps") or {}).items()
                },
            )
        )
    return out


def find(cozy_home: Path, ref: str) -> Artifact | None:
    """Exact ref, then the bare `org/repo` prefix when only one release is installed."""
    installed = rows(cozy_home)
    for row in installed:
        if row.ref == ref:
            return row
    if "@" not in ref:
        matches = [row for row in installed if row.ref.split("@")[0] == ref]
        if len(matches) == 1:
            return matches[0]
    return None


def remove(cozy_home: Path, ref: str) -> bool:
    """Drop the index row. Returns whether anything was there — an absent ref is a 0."""
    path = directory(cozy_home) / f"{_slug(ref)}.json"
    if not path.is_file():
        return False
    path.unlink()
    return True
