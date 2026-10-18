"""The leaf speedup the probe banks reaches the capability record and the qualification
document (cr-135).

The input is the recorded real probe `tests/observations/probe-cuda-sm89.json`. It was
written by runtime 0.0.36 under that build's spec digests, so `rekeyed()` moves its rowwise
observations onto the scheme this build keys on; the card's measurements are untouched.
`sm86()` is the release gate's sm86 table of runtime 0.28, rekeyed the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import msgspec

from cozy_runtime.internal import probe, qualification
from cozy_runtime.internal.encoding import implementation_digest, launch_providers
from cozy_runtime.internal.probe import QualificationResult

OBSERVATIONS = Path(__file__).resolve().parent / "observations"
OBSERVATION = OBSERVATIONS / "probe-cuda-sm89.json"

#: The two `fp8-rowwise/1` spec digests runtime 0.0.36 reviewed: both `fp8-rowwise` here.
RECORDED_SPECS = {
    "sha256:8c86b26daec5bcd287401d9873b70aa3594855f97ac4d01cf75cc2bafd5d50b5": "fp8.rowwise",
    "sha256:91ddc2a73ba2778b344afc5e78b4addee9174a015cfff02f17cfbb629ad5ef78": "fp8.rowwise",
}

LEAF = "cozy.fp8-rowwise.native-leaf/1"


def recorded() -> QualificationResult:
    return msgspec.convert(json.loads(OBSERVATION.read_bytes()), QualificationResult)


def rekeyed() -> QualificationResult:
    """The recorded rowwise observations on this build's digests, measurements verbatim."""
    result = recorded()
    by_name = {provider.name: provider for provider in launch_providers()["fp8.rowwise"]}
    return msgspec.structs.replace(
        result,
        observations=[
            msgspec.structs.replace(
                o,
                scheme=RECORDED_SPECS[o.scheme],
                implementation_digest=implementation_digest(by_name[o.implementation]),
            )
            for o in result.observations
            if o.scheme in RECORDED_SPECS and o.implementation in by_name
        ],
    )


def sm86() -> QualificationResult:
    """The sm86 table as `qualification.qualified` serves it: settled cells evidenced anew."""
    table = msgspec.convert(
        json.loads((OBSERVATIONS / "qualified-cuda-sm86.json").read_bytes()), QualificationResult
    )
    name = qualification.table_name(table.device, table.runtime)
    measured = {o.scheme for o in table.observations}
    providers = {k: v for k, v in launch_providers().items() if k in measured}
    covered = qualification._covering(
        table, name, providers, table.device, qualification.TABLE_DTYPES
    )
    assert covered is not None
    return covered


# ------------------------------------------------------------------------- the record


def test_the_probe_mints_the_leaf_speedup_onto_the_record_and_nothing_else() -> None:
    """Every passed encoded-GEMM observation carries its bench; every decode route carries
    NOT MEASURED, spelled -1.0 exactly as the two axes beside it spell it."""
    table = probe.admission_table(rekeyed())
    leaves = [r for r in table.records if r.delivery_route == "encoded_gemm"]
    assert {r.leaf_speedup_x for r in leaves} == {0.6122, 0.7683}, leaves
    assert leaves and all(r.document()["leaf_speedup_x"] == r.leaf_speedup_x for r in leaves)
    floors = [r for r in table.records if r.delivery_route != "encoded_gemm"]
    assert floors and all(r.leaf_speedup_x == -1.0 for r in floors)


def test_the_qualification_document_names_the_worst_leaf_speedup_per_implementation() -> None:
    """One implementation measured under two spec digests reports the WORSE of the two: the
    number the boot tail prints is the one an operator would want to be alarmed by."""
    result = recorded()
    document = qualification.Qualification(
        probe.admission_table(result), result, "key", "kept"
    ).document()
    assert document["leaf_speedup_x"] == {f"{LEAF}@bfloat16": 0.6122}
    # A document with no passed leaf carries an empty map, not a missing key.
    floors_only = msgspec.structs.replace(
        result,
        observations=[o for o in result.observations if o.delivery_route != "encoded_gemm"],
    )
    assert probe.leaf_speedups(floors_only) == {}
