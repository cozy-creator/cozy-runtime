"""THE LOCAL ARTIFACT INDEX — this machine's inventory of installed checkpoints.

It answers ONE question: what does the ref `org/model[@release]` mean on THIS disk — which
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

from cozy_runtime.internal import cache
from cozy_runtime.internal.exits import Exit


class ArtifactRefError(ValueError):
    """A string that is not a ref reached the index. It carries the CLI's own words so the
    one caller that can be handed a human's typo renders them without inventing a second
    message for one refusal."""

    def __init__(self, message: str, remedy: str, exit_code: Exit = Exit.usage) -> None:
        super().__init__(message)
        self.remedy = remedy
        self.exit_code = exit_code


#: `org/model`, optionally `@release`. A bare ref resolves on tensorhub only (never a foreign
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

    def view(self) -> dict[str, object]:
        """Human/CLI projection of this row; never stored or used as identity."""

        return {
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
            "stamps": {key: list(values) for key, values in sorted(self.stamps.items())},
        }


def install(cozy_home: Path, artifact: Artifact) -> Path:
    """Record one installed artifact as a typed local row."""
    if not REF.match(artifact.ref):
        raise ArtifactRefError(
            f"{artifact.ref!r} is not a model ref: org/model[@release]",
            "model refs are org/model with an optional @release; a filesystem path is admitted "
            "through `cozy-runtime ingest` first",
        )
    with cache.connect(cozy_home) as database:
        database.execute(
            """
            INSERT INTO artifacts (
                ref, store_root, config, snapshots_json, lane, stored_bytes, tensors,
                variant, custody, vram_floor_bytes, extra_key_markers_json, stamps_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ref) DO UPDATE SET
                store_root=excluded.store_root,
                config=excluded.config,
                snapshots_json=excluded.snapshots_json,
                lane=excluded.lane,
                stored_bytes=excluded.stored_bytes,
                tensors=excluded.tensors,
                variant=excluded.variant,
                custody=excluded.custody,
                vram_floor_bytes=excluded.vram_floor_bytes,
                extra_key_markers_json=excluded.extra_key_markers_json,
                stamps_json=excluded.stamps_json
            """,
            (
                artifact.ref,
                artifact.store,
                artifact.config,
                json.dumps(dict(sorted(artifact.snapshots.items())), separators=(",", ":")),
                artifact.lane,
                artifact.bytes,
                artifact.tensors,
                artifact.variant,
                artifact.custody,
                artifact.vram_floor_bytes,
                json.dumps(list(artifact.extra_key_markers), separators=(",", ":")),
                json.dumps(
                    {key: list(values) for key, values in sorted(artifact.stamps.items())},
                    separators=(",", ":"),
                ),
            ),
        )
        database.commit()
    return cozy_home / cache.FILENAME


def rows(cozy_home: Path, prefix: str = "") -> list[Artifact]:
    out: list[Artifact] = []
    with cache.connect(cozy_home) as database:
        selected = database.execute(
            """
            SELECT ref, store_root, config, snapshots_json, lane, stored_bytes, tensors,
                   variant, custody, vram_floor_bytes, extra_key_markers_json, stamps_json
            FROM artifacts
            ORDER BY ref
            """,
        )
        for row in selected:
            ref = str(row["ref"])
            if prefix and not ref.startswith(prefix):
                continue
            try:
                snapshots = json.loads(str(row["snapshots_json"]))
                markers = json.loads(str(row["extra_key_markers_json"]))
                stamps = json.loads(str(row["stamps_json"]))
            except ValueError as exc:
                raise cache.CacheError(
                    f"{cozy_home / cache.FILENAME}: malformed artifact row"
                ) from exc
            if (
                not isinstance(snapshots, dict)
                or not isinstance(markers, list)
                or not isinstance(stamps, dict)
                or not all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in snapshots.items()
                )
                or not all(isinstance(value, str) for value in markers)
                or not all(
                    isinstance(key, str)
                    and isinstance(values, list)
                    and all(isinstance(value, str) for value in values)
                    for key, values in stamps.items()
                )
            ):
                raise cache.CacheError(f"{cozy_home / cache.FILENAME}: malformed artifact row")
            out.append(
                Artifact(
                    ref=ref,
                    store=str(row["store_root"]),
                    config=str(row["config"]),
                    snapshots=snapshots,
                    lane=str(row["lane"]),
                    bytes=int(row["stored_bytes"]),
                    tensors=int(row["tensors"]),
                    variant=str(row["variant"]),
                    custody=str(row["custody"]),
                    vram_floor_bytes=int(row["vram_floor_bytes"]),
                    extra_key_markers=tuple(markers),
                    stamps={key: tuple(values) for key, values in stamps.items()},
                )
            )
    return out


def find(cozy_home: Path, ref: str) -> Artifact | None:
    """Exact ref, then the bare `org/model` prefix when only one release is installed."""
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
    with cache.connect(cozy_home) as database:
        removed = database.execute("DELETE FROM artifacts WHERE ref = ?", (ref,)).rowcount
        database.commit()
    return bool(removed)
