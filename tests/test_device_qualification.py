"""The OBSERVED (encoding, device) qualification this runtime reports to TensorFS.

Executability is two independent questions. Whether bytes can be READ is the format axis and
TensorFS answers it from the encoding rule. Whether an encoding is QUALIFIED on a card is
this axis, and only the process holding the card can answer it -- so TensorFS owns the
admission rule, this runtime owns the observation, and the hub joins them. A COMPATIBLE
format grade is not executability.

The document `observed_capability_records` emits is TensorFS's input shape and it is strict:
four keys per record, canonical bytes, printable ASCII, no trailing byte. A document that
misses any of those is refused WHOLE at the border, so one bad record hides every good one,
and the failure reads to the hub as "this card cannot run anything".

The input here is a RECORDED REAL PROBE, not a fixture anybody wrote: `tests/observations/`
holds the exact result document `probe.qualify` returned on an NVIDIA RTX 4070 Laptop (sm89,
driver 595.84, torch 2.13.0+cu130), 27 observations, 19 of them passing. That is the same
discipline `internal/encoding/vectors/` already uses -- real bytes, recorded once, replayed
by the suite -- and it is what lets this run in every `uv run pytest`, on CI, with no GPU and
no torch. Nothing here skips: a skip is indistinguishable from a pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest
import tensorfs

from cozy_runtime.derive import plan
from cozy_runtime.internal import canonical
from cozy_runtime.internal.encoding import (
    SPEC_MXFP8,
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
    SPEC_VECTORS,
    DeviceFacts,
    launch_providers,
)
from cozy_runtime.internal.probe import (
    DecodeEvidence,
    Observation,
    QualificationResult,
    VerbatimEvidence,
    case_dtype,
    observed_capability_records,
    qualify,
    spec_vector_document,
)
from test_derive_quantize_facade import mint, run

#: The exact keys TensorFS's `CapabilityRecords::parse_observed` requires, all of them, on
#: every record. It reads with unknown fields refused, so a surplus key and a missing key are
#: the same refusal.
RECORD_KEYS = {"device", "encoding", "implementation", "note"}

OBSERVATION = Path(__file__).resolve().parent / "observations" / "probe-cuda-sm89.json"


def probe_result() -> QualificationResult:
    return msgspec.convert(json.loads(OBSERVATION.read_bytes()), QualificationResult)


def test_the_recorded_observation_is_a_real_measured_probe() -> None:
    """The input is evidence, so it has to carry what it was measured on."""
    result = probe_result()
    assert result.device_predicate == "cuda.sm89"
    assert result.device.sm == 89
    assert result.device.kind == "cuda"
    # Both halves of a capability claim have an identity: the card, and the build that ran on
    # it. A digest here proves the recording came out of `runtime_source_digest`, which is
    # computed from installed bytes -- it is NOT compared to this build's, because the
    # recording is of the build that measured, not of the one replaying it.
    assert result.runtime.source_digest.startswith("sha256:")
    assert result.runtime.torch_build.startswith("2.")
    assert len(result.observations) == 27
    assert sum(1 for o in result.observations if o.evidence.passed) == 19


def test_every_record_carries_exactly_the_four_keys() -> None:
    document = observed_capability_records(probe_result())
    assert set(document) == {"records"}
    assert document["records"], "the recorded probe qualified something"
    for record in document["records"]:
        assert set(record) == RECORD_KEYS
        assert record["device"] and record["implementation"], "both are required non-empty"
        assert record["encoding"].startswith("sha256:") and len(record["encoding"]) == 71


def test_the_bytes_are_canonical_with_no_trailing_byte() -> None:
    """TensorFS parses these bytes with `parse_canonical`: a trailing newline refuses."""
    raw = canonical.write(observed_capability_records(probe_result()))
    assert raw[-1:] == b"}"
    assert raw.isascii()
    # The whole definition of canonical: the bytes are the encoding of their own content.
    assert canonical.write(canonical.parse(raw)) == raw
    assert canonical.parse_canonical(raw) == observed_capability_records(probe_result())


def test_one_record_per_encoding_and_device() -> None:
    """The suite mints a record per output dtype and implementation; the RULE admits on the
    pair, so the projection collapses to it and names every implementation that earned it."""
    result = probe_result()
    document = observed_capability_records(result)
    pairs = [(r["encoding"], r["device"]) for r in document["records"]]
    assert len(pairs) == len(set(pairs))
    assert pairs == sorted(pairs), "records are ordered by the pair, not by probe order"
    minted = {
        (o.encoding_spec_id, o.device_predicate) for o in result.observations if o.evidence.passed
    }
    # Every collapsed pair traces back to a passing observation, and every implementation
    # named in a record is one that actually minted for that pair.
    assert set(pairs) <= minted
    for record in document["records"]:
        for named in record["implementation"].split(" "):
            assert named.split("@")[-1] in {
                o.implementation_digest
                for o in result.observations
                if (o.encoding_spec_id, o.device_predicate)
                == (record["encoding"], record["device"])
                and o.evidence.passed
            }


def test_a_failed_observation_is_never_reported() -> None:
    """The recorded probe carries a real failure: mxfp8 decoded to float16, whose range
    cannot hold the MX domain (442 normal and 420 subnormal elements wrong). The encoding is
    still reported for the dtypes that DID pass, and float16 is not among them -- so the
    dtype set a record names is exactly the reportable passing set, never the attempted one.
    """
    result = probe_result()
    assert [o for o in result.observations if not o.evidence.passed], (
        "the recording includes failures on purpose"
    )
    for record in observed_capability_records(result)["records"]:
        pair = (record["encoding"], record["device"])
        assert _note_dtypes(record) == {
            o.output_dtype
            for o in result.observations
            if (o.encoding_spec_id, o.device_predicate) == pair and _reportable(o)
        }
    mxfp8 = _mxfp8(result)
    reported = {r["encoding"]: r for r in observed_capability_records(result)["records"]}
    assert "float16" not in _note_dtypes(reported[mxfp8]), "float16 mxfp8 decode FAILED here"
    assert "bfloat16" in _note_dtypes(reported[mxfp8])


def test_only_evidence_that_actually_ran_crosses_the_boundary() -> None:
    """A `verbatim` route's evidence is byte identity and travels. Any other route must have
    RUN the spec's producer-mined vectors: `_spec_vectors` passes a spec with no vendored
    document as SKIPPED, which is right for a local table built beside the synthetic oracle
    that did run, and is not a claim to hand another process -- and it is what keeps a spec
    TensorFS carries vectorless out of a document it would refuse whole."""
    result = probe_result()
    reported = {
        (r["encoding"], r["device"]) for r in observed_capability_records(result)["records"]
    }
    for observation in result.observations:
        if not observation.evidence.passed:
            continue
        pair = (observation.encoding_spec_id, observation.device_predicate)
        if not _reportable(observation):
            assert pair not in reported or _also_ran(result, pair), (
                f"{pair} was reported on evidence that did not run"
            )


def _note_dtypes(record: dict[str, Any]) -> set[str]:
    """The output dtypes a record's note names. Parsed rather than matched: `bfloat16`
    CONTAINS `float16`, and a substring test here silently passed the one case this file
    exists to catch."""
    body = record["note"].split("output dtypes ", 1)[1].split(";", 1)[0]
    return {token.strip() for token in body.split(",") if token.strip()}


def _reportable(observation: Observation) -> bool:
    evidence = observation.evidence
    if not evidence.passed:
        return False
    if isinstance(evidence, VerbatimEvidence):
        return True
    return isinstance(evidence, DecodeEvidence) and not evidence.spec_vectors.skipped


def _mxfp8(result: QualificationResult) -> str:
    for observation in result.observations:
        if "mxfp8" in observation.implementation:
            return observation.encoding_spec_id
    raise AssertionError("the recording carries an mxfp8 observation")


def _also_ran(result: QualificationResult, pair: tuple[str, str]) -> bool:
    """Another observation for the same pair whose evidence DID run. One dtype skipping its
    vectors does not unqualify a pair another dtype measured."""
    return any(
        (o.encoding_spec_id, o.device_predicate) == pair and _reportable(o)
        for o in result.observations
    )


def test_the_document_carries_no_per_run_measurement() -> None:
    """A registered artifact that churns without a change of meaning is one nobody can diff.

    The probe's `evidence_digest` covers the micro-benchmark and therefore moves on every
    run; the document must depend only on the card class, the runtime build and which dtypes
    passed. Every digest it carries is an encoding's or an implementation's -- both of which
    are over structure and over no timing.
    """
    result = probe_result()
    document = observed_capability_records(result)
    stable = {r["encoding"] for r in document["records"]} | {
        o.implementation_digest for o in result.observations
    }
    found = set(re.findall(r"sha256:[0-9a-f]{64}", canonical.write(document).decode()))
    assert found and found <= stable, f"a per-run digest reached the document: {found - stable}"


def _torch_here() -> str:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"torch must be importable to run a provider over the mined bytes: {exc}"
    return ""


needs_torch = pytest.mark.skipif(bool(_torch_here()), reason=_torch_here() or "")


def test_every_vendored_document_is_readable_by_the_reader_that_runs_it() -> None:
    """The mined bytes and the suite's reader have to name the logical dtype the SAME WAY.

    They stopped: TensorFS 0.3 renamed the CozyTensors header's `logical.dtype` to
    `logical_dtype`, the rename was applied to this reader too, and these documents are not
    headers — they are producer-mined vector artifacts, pinned by digest, whose `logical`
    object still spells it `dtype`. Every case then matched nothing, `_spec_vectors`
    reported SKIPPED, `_reportable` dropped the skip, and the observed document handed to
    TensorFS lost every non-verbatim record on every card. This arm reads the documents
    through the SAME expression the suite does, so the two cannot drift apart again.
    """
    assert SPEC_VECTORS, "the reviewed table pins at least one mined document"
    for digest in SPEC_VECTORS:
        document = spec_vector_document(digest)
        assert document is not None, f"{digest} is pinned and its bytes must verify"
        dtypes = {case_dtype(case) for case in document.cases}
        assert dtypes and None not in dtypes, (
            f"{digest}: the mined cases declare {dtypes} — the reader read no dtype out of "
            "bytes that carry one"
        )


@needs_torch
def test_the_mined_cases_actually_run_against_the_shipped_decoder() -> None:
    """The whole point of the vectors: a real provider decodes the producer's own bytes.

    Runs on CPU with no card, because the arithmetic is the same one the accelerator
    qualifies and the question here is whether the check RUNS at all — a skipped check is
    the failure this file exists to catch.
    """
    import torch

    from cozy_runtime.internal.encoding import SPEC_ROWWISE, RowwiseDequant
    from cozy_runtime.internal.probe import _spec_vectors

    provider = RowwiseDequant(SPEC_ROWWISE)
    report = _spec_vectors(torch, provider, torch.device("cpu"))
    assert not report.skipped, report
    assert report.passed, report
    assert report.cases, "the bf16 case the mined document carries has to have run"


@needs_torch
def test_a_float16_fp8_lane_fits_the_card_its_decode_qualified_on(tmp_path: Path) -> None:
    """Run 1312: an fp8-rowwise SDXL lane on an RTX 3090 refused `encoding_unqualified ...
    no qualification observation on cuda.sm86 (qualified on: cpu)`. SDXL fills float16 and
    the rowwise and mxfp8 documents mine f32 and bf16 cases, so filtering the mined cases by
    the requested dtype skipped the vector check, `_reportable` dropped the skip, and TensorFS
    received no record. The decode floors are `any`: the real suite runs here on CPU, and its
    observations are keyed to sm86 because the decode arithmetic is the card's."""
    import torch

    values = np.arange(256, dtype="<f2").reshape(8, 32) / 256
    store, source = mint(tmp_path, {"unet": {"a.weight": ("f16", (8, 32), values.tobytes())}})
    _, lane = run(tmp_path, store, source, plan(("unet",), "fp8-rowwise/1"))
    header = store.manifest(lane.manifest.digest)["header"]

    cpu = DeviceFacts("cpu", "cpu", 0, "", "default")
    _, measured = qualify(torch, launch_providers(), cpu, ["float16"], release="test")
    result = msgspec.structs.replace(
        measured,
        device_predicate="cuda.sm86",
        observations=[
            msgspec.structs.replace(o, device_predicate="cuda.sm86") for o in measured.observations
        ],
    )
    document = observed_capability_records(result)
    reported = {r["encoding"] for r in document["records"]}
    assert {SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM, SPEC_MXFP8} <= reported

    verdict = tensorfs.fit(
        [tensorfs.TensorRequirement("unet", "a.weight", [8, 32], None)],
        header,
        custody="canonical",
        encoded_leaves=True,
        device="cuda.sm86",
        observations=canonical.write(document),
    )
    assert verdict["ok"], verdict
