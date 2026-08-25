"""The LOCAL artifact index — the CLI layer's "install local snapshot refs" power.

cozy-runtime-cli.md §Local authority names this as a CLI-layer capability the runtime CORE
does not have: the core is handed exact store roots and snapshot digests and never resolves
a ref. This module is where a ref (`org/repo[@release]`) becomes those exact facts, and it
is the one vocabulary `list`, `rm` and `run` share.

A row is a projection of what TensorFS's ingest border hands back, never a second
description of the bytes: store root, one snapshot digest per component, the immutable
construction config, and the byte total the header already counted. The index holds no
tensor facts of its own — `bytes` is copied from the header, and `list` reports one artifact
size exactly as the CLI vocabulary requires.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cozy_runtime.cli.io import CliError
from cozy_runtime.internal.exits import Exit

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
    extra_key_markers: tuple[str, ...] = field(default_factory=tuple)
    """Known marker rules an extra key may match, e.g. `v_pred`. A match is transcribed or
    refused NAMING the marker; it is never ignored."""

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
            "extra_key_markers": list(self.extra_key_markers),
        }


def _slug(ref: str) -> str:
    return ref.replace("/", "__").replace("@", "--")


def directory(cozy_home: Path) -> Path:
    return cozy_home / DIRECTORY


def install(cozy_home: Path, artifact: Artifact) -> Path:
    """Record one installed artifact. Idempotent: identical facts rewrite the same bytes."""
    if not REF.match(artifact.ref):
        raise CliError(
            "usage",
            f"{artifact.ref!r} is not a ref: org/repo[@release]",
            "refs are org/repo with an optional @release; a filesystem path is admitted "
            "through `cozy-runtime ingest` first",
            Exit.usage,
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
                extra_key_markers=tuple(doc.get("extra_key_markers", ())),
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
