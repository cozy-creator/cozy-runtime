"""Kernel qualification is offline release data, never a consumer experiment.

Each table binds implementation/source identity, GPU architecture and Torch minor. A missing
cell cannot trigger a benchmark or optimistic admission: an existing supported route serves,
or the model receives an explicit capability refusal. Development uses cli.qualify to measure
and publish the tables that subsequent consumers read.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.internal import arch
from cozy_runtime.internal.encoding import (
    Capabilities,
    DeviceFacts,
    Provider,
    RuntimeIdentity,
    implementation_digest,
    measure_runtime,
)
from cozy_runtime.internal.encoding.selection import reference_decoder
from cozy_runtime.internal.probe import (
    FamilyDefaultEvidence,
    Observation,
    QualificationResult,
    ReferenceEvidence,
    UnmeasuredEvidence,
    admission_table,
    cells,
    leaf_speedups,
    make_observation,
    settled,
)

#: The destinations a shipped table is measured over: what checkpoints load into.
TABLE_DTYPES = ("bfloat16", "float16", "float32")

#: The tables the release gates measured, one per card architecture and torch minor.
SHIPPED = Path(__file__).resolve().parent / "qualification_tables"

#: Bumped when a table's MEANING changes in a way its name does not catch.
TABLE_FORMAT = 3

#: Native leaves whose lane is one tensor-core format: on a card whose row has it, the family's
#: default GEMM serves where no offline verdict refines the choice (prebuilt, else cuBLASLt).
#: The destinations an 8-bit GEMM writes.
HALF = ("float16", "bfloat16")
LANE_FORMATS = {
    "int8-convrot256-native": "int8",
    "fp8-rowwise-native": "fp8",
    "mxfp8-native": "fp8",
}


@dataclass(frozen=True, slots=True)
class Qualification:
    """Offline facts or declared reference decoding; inference never measures kernels."""

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
    return one.scheme, one.implementation_digest, one.output_dtype


def _covering(
    table: QualificationResult | None,
    name: str,
    providers: Mapping[str, tuple[Provider, ...]],
    device: DeviceFacts,
    output_dtypes: Sequence[str],
) -> QualificationResult | None:
    """Keep matching offline cells; absent cells refuse without running a consumer probe."""
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
        # The current NVFP4 oracle observes one dispatch arm. A W4A16 fallback does not
        # qualify optional kitchen W4A4, nor its tensor-global activation quantizer. Keep
        # the declared decode route until offline evidence covers the complete dispatcher.
        if provider.suite == "nvfp4-native":
            found = None
        observations.append(
            found if found is not None else _missing(encoding, provider, device, dtype)
        )
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


def _missing(encoding: str, provider: Provider, device: DeviceFacts, dtype: str) -> Observation:
    evidence = settled(provider, device)
    if evidence is None and provider.suite == "nvfp4-native":
        evidence = UnmeasuredEvidence(
            note="NVFP4 native dispatch lacks separate offline qualification for kitchen "
            "W4A4 and both W4A16 arms; the declared reference decoder serves"
        )
    if evidence is None and dtype in HALF and (family := family_default(provider, device)):
        evidence = FamilyDefaultEvidence(
            note=f"no offline verdict refines {provider.name} on {device.predicate}; the "
            f"{family} family default serves it"
        )
    if evidence is None and reference_decoder(provider, device, dtype):
        evidence = ReferenceEvidence(
            note=f"shipped {provider.name} format decoder into {dtype}, then ordinary float GEMM; "
            "role, parameter, geometry and device checks apply; timing and numeric axes unmeasured"
        )
    return make_observation(
        encoding,
        provider,
        device,
        dtype,
        evidence
        or UnmeasuredEvidence(
            note=f"no shipped offline qualification for {provider.name} on {device.predicate} "
            f"with output {dtype}; qualify this implementation in development"
        ),
    )


def family_default(provider: Provider, device: DeviceFacts) -> str:
    """The tensor-core family whose default GEMM serves `provider`'s lane on `device`, or ""
    where the card lacks the lane's format (only there does a declared decoder serve it)."""
    lane = LANE_FORMATS.get(provider.suite)
    row = arch.row(device.sm) if device.kind == "cuda" else arch.NONE
    return row.family if lane is not None and lane in row.formats() else ""


def qualified(
    torch: Any,
    providers: Mapping[str, tuple[Provider, ...]],
    device: DeviceFacts,
    output_dtypes: Sequence[str],
    *,
    release: str,
    runtime: RuntimeIdentity | None = None,
    tables: Path | None = None,
) -> Qualification:
    """Read offline release facts, or admit only routes whose correctness needs no probe.

    Missing fast-kernel facts remain explicit refusals. The ordinary resolver can choose a
    shipped format decoder into the declared dtype followed by plain GEMM; those contracts
    carry no invented performance or accuracy measurements. `tables` also lets release
    tooling inspect an offline oracle's output.
    """
    if runtime is None:
        runtime = measure_runtime(torch, release)
    name = table_name(device, runtime)
    directory = SHIPPED if tables is None else tables
    shipped = _covering(_read(directory / f"{name}.json"), name, providers, device, output_dtypes)
    if shipped is not None:
        return Qualification(admission_table(shipped), shipped, name, "shipped")
    result = QualificationResult(
        tuple_key=device.tuple_key(),
        device=device,
        device_predicate=device.predicate,
        runtime_release=release,
        runtime=runtime,
        suite_ms=0,
        observations=[
            _missing(encoding, provider, device, dtype)
            for encoding, provider, dtype in cells(providers, output_dtypes)
        ],
    )
    return Qualification(
        admission_table(result), result, name, "supported", "no matching offline table"
    )
