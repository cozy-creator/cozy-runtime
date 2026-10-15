"""Where a Load's admission table comes from: release data, else this machine's measurement.

What the decode suite (`probe`) answers is a fact about an implementation on a card
architecture under one torch minor and one Runtime source (`IDENTITY_SOURCES`): driver,
power cap and which card of the arch move only the micro-benchmark, which nothing ranks on.
The release gates measure one table per architecture they run on
(`python -m cozy_runtime.cli.qualify <dir>`) and the wheel ships them (`SHIPPED`), so a fresh
pod measures nothing. A card no gate measured, or a Runtime whose identity sources moved since,
measures what it is asked once per machine (concurrent executors wait for the first) and keeps
it under `COZY_HOME`. This module is outside `IDENTITY_SOURCES`: where a table comes from does
not change what it says.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.internal.encoding import (
    Capabilities,
    DeviceFacts,
    Provider,
    RuntimeIdentity,
    implementation_digest,
    measure_runtime,
)
from cozy_runtime.internal.probe import (
    Observation,
    QualificationResult,
    admission_table,
    cells,
    leaf_speedups,
    make_observation,
    qualify,
    settled,
)

#: The destinations a shipped table is measured over: what checkpoints load into.
TABLE_DTYPES = ("bfloat16", "float16", "float32")

#: The tables the release gates measured, one per card architecture and torch minor.
SHIPPED = Path(__file__).resolve().parent / "qualification_tables"

#: Bumped when a table's MEANING changes in a way its name does not catch.
TABLE_FORMAT = 2

#: How many tables one machine keeps: a floor under growth across Runtime sources.
KEEP = 64


@dataclass(frozen=True, slots=True)
class Qualification:
    """One admission table, and where it came from: `shipped`, `kept` or `measured`."""

    capabilities: Capabilities
    result: QualificationResult
    key: str
    source: str
    note: str = ""

    def document(self) -> dict[str, str | int | dict[str, float]]:
        return {
            "key": self.key,
            "source": self.source,
            "suite_ms": self.result.suite_ms,
            "observations": len(self.result.observations),
            "leaf_speedup_x": leaf_speedups(self.result),
            "note": self.note,
        }


def table_name(device: DeviceFacts, runtime: RuntimeIdentity) -> str:
    """What a table answers for: the card architecture, torch's minor version and the source
    of every decode, leaf, oracle and admission rule."""
    minor = ".".join(runtime.torch_build.split("+")[0].split(".")[:2])
    digest = runtime.source_digest.removeprefix("sha256:")[:16]
    return f"{device.predicate}-torch{minor}-{digest}-v{TABLE_FORMAT}"


def _cell(one: Observation) -> tuple[str, str, str]:
    return one.encoding_spec_id, one.implementation_digest, one.output_dtype


def _covering(
    table: QualificationResult | None,
    name: str,
    providers: Mapping[str, tuple[Provider, ...]],
    device: DeviceFacts,
    output_dtypes: Sequence[str],
) -> QualificationResult | None:
    """`table`'s observations for exactly these cells, or None where it is not this name's
    table or leaves one unmeasured."""
    if table is None or table_name(table.device, table.runtime) != name:
        return None
    kept = {_cell(one): one for one in table.observations}
    observations = []
    for encoding, provider, dtype in cells(providers, output_dtypes):
        evidence = settled(provider, device)
        found = (
            kept.get((encoding, implementation_digest(provider), dtype))
            if evidence is None
            else make_observation(encoding, provider, device, dtype, evidence)
        )
        if found is None:
            return None
        observations.append(found)
    return msgspec.structs.replace(table, observations=observations)


def _read(path: Path | None) -> QualificationResult | None:
    if path is None:
        return None
    # stdlib JSON both ways: a failed native route can measure a NaN or infinite deviation,
    # which `msgspec.json` would write as null and then refuse to read back.
    try:
        return msgspec.convert(json.loads(path.read_bytes()), QualificationResult)
    except (OSError, ValueError, msgspec.ValidationError):
        return None


def write_table(path: Path, table: QualificationResult) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(msgspec.to_builtins(table)))
    os.replace(temporary, path)


@contextlib.contextmanager
def _held(cache: str, name: str) -> Iterator[tuple[Path | None, str]]:
    """This machine's table, held: one executor measures, the others wait for its table (an
    executor that wedges is ended by its machine, which releases the lock)."""
    if not cache:
        yield None, "no cache directory"
        return
    path = Path(cache) / f"{name}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(path.with_suffix(".lock"), "a+")  # noqa: SIM115
    except OSError as exc:
        yield None, f"not kept: {type(exc).__name__}"
        return
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield path, ""


def qualified(
    torch: Any,
    providers: Mapping[str, tuple[Provider, ...]],
    device: DeviceFacts,
    output_dtypes: Sequence[str],
    *,
    release: str,
    runtime: RuntimeIdentity | None = None,
    cache: str = "",
) -> Qualification:
    """The admission table for these inputs: from the table the release gates measured on
    this card architecture, else from this machine's table under `cache`, else measured now
    and added to it. A filesystem failure measures and keeps nothing; it never refuses."""
    if runtime is None:
        runtime = measure_runtime(torch, release)
    name = table_name(device, runtime)
    shipped = _covering(_read(SHIPPED / f"{name}.json"), name, providers, device, output_dtypes)
    if shipped is not None:
        return Qualification(admission_table(shipped), shipped, name, "shipped")
    with _held(cache, name) as (path, note):
        kept = _read(path)
        result = _covering(kept, name, providers, device, output_dtypes)
        if result is not None:
            return Qualification(admission_table(result), result, name, "kept")
        _, result = qualify(
            torch, providers, device, output_dtypes, release=release, runtime=runtime
        )
        if path is not None:
            known = (
                kept.observations if kept and table_name(kept.device, kept.runtime) == name else []
            )
            new = {_cell(one) for one in result.observations}
            table = msgspec.structs.replace(
                result,
                observations=[*result.observations, *(o for o in known if _cell(o) not in new)],
            )
            try:
                write_table(path, table)
                stored = sorted(path.parent.glob("*.json"), key=lambda one: one.stat().st_mtime)
                for old in stored[: max(len(stored) - KEEP, 0)]:
                    old.unlink()
            except OSError as exc:
                note = f"not kept: {type(exc).__name__}"
    return Qualification(admission_table(result), result, name, "measured", note)
