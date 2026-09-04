"""Per-file model source preparation, through the real compiled TensorFS boundary.

Nothing here is mocked. A real Store, the real protobuf request the pod host sends, and the
real pyo3 call — which is the whole point, because the failure this file exists to prevent
was a SIGNATURE drift between two repositories that no amount of Python-side unit testing
would have caught. cozy-runtime called `store.prepare_model_source` with six positional
arguments against a binding that took four, so every `hf://` model source preparation died
with a TypeError after the pod had already spent hours downloading the weights.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.worker import model_source_prepare
from cozy_runtime.protocol import worker_pb2 as pb

_DIGEST = bytes(range(32))
_SOURCE_URI = "hf://acme/model@" + "a" * 40

# SELF-ARMING, like the drift job. The per-file binding lands in TensorFS first and is only
# reachable here once a wheel carrying it is installed; until then these tests would be
# testing the absence of the thing they exist to test. They arm themselves the moment the
# right wheel is present rather than being remembered later.
_binding = getattr(tensorfs.Store, "prepare_model_source", None)
_BINDING = str(getattr(_binding, "__text_signature__", "") or "")
_PER_FILE_BINDING = "roster" in _BINDING and "landed" in _BINDING
_needs_binding = pytest.mark.skipif(
    not _PER_FILE_BINDING,
    reason=(
        "installed tensorfs has no per-file model source binding "
        f"(signature: {_BINDING or 'unknown'}); needs the wheel carrying tensorfs#99"
    ),
)


def _header(keys: dict[str, int]) -> bytes:
    """A safetensors `u64le length || JSON` prefix, exactly as a ranged GET returns it."""
    fields = []
    offset = 0
    for key, size in keys.items():
        fields.append(
            f'"{key}":{{"data_offsets":[{offset},{offset + size}],'
            f'"dtype":"F32","shape":[{size // 4}]}}'
        )
        offset += size
    body = ("{" + ",".join(fields) + "}").encode()
    return struct.pack("<Q", len(body)) + body


def _member(
    tmp_path: Path, member: str, *, sizes: dict[str, int], land: bool
) -> tuple[pb.LocalModelSourceFile, int]:
    header = _header(sizes)
    header_path = tmp_path / f"{member}.header"
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_bytes(header)

    length = len(header) + sum(sizes.values())
    carrier = tmp_path / member
    carrier.parent.mkdir(parents=True, exist_ok=True)
    if land:
        carrier.write_bytes(header + bytes(sum(sizes.values())))

    return (
        pb.LocalModelSourceFile(
            member=member,
            object_id="sha256:" + f"{abs(hash(member)):064x}"[:64],
            length=length,
            path=str(carrier),
            header_path=str(header_path),
        ),
        length,
    )


def _request(tmp_path: Path, *, landed: list[str]) -> pb.PrepareModelSourceRequest:
    first, _ = _member(
        tmp_path, "encoder.safetensors", sizes={"weight": 64}, land="encoder.safetensors" in landed
    )
    second, _ = _member(
        tmp_path, "decoder.safetensors", sizes={"weight": 64}, land="decoder.safetensors" in landed
    )
    return pb.PrepareModelSourceRequest(
        operation_id="per-file-proof",
        source_selection_digest=_DIGEST,
        profiles=[pb.ModelSourceProfile(slot="only", profile="acme/absent/1")],
        roster=[second, first] if second.member < first.member else [first, second],
        source_uri=_SOURCE_URI,
        declared_license="apache-2.0",
        landed=landed,
    )


@_needs_binding
def test_a_partly_landed_roster_reaches_tensorfs_and_is_answered_typed(tmp_path: Path) -> None:
    """The regression guard for the cross-repo signature drift.

    One of two members has landed and the other has no file on disk at all — the ask a
    per-file preparation actually makes, and the one the whole-selection shape could not
    express. It must reach TensorFS's own validator and come back a TYPED refusal (the
    profile is not in the builtin reviewed registry), never a TypeError from the binding.
    """
    store_root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(store_root))

    result = model_source_prepare.prepare_model_source(
        _request(tmp_path, landed=["encoder.safetensors"]), tensorfs_root=store_root
    )

    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    # Reached the compiled planner and was refused on its merits, not on its shape.
    assert result.safe_code.startswith("model_source_tensorfs_")
    assert "unregistered" in result.safe_code or "fingerprint" in result.safe_code


@_needs_binding
def test_a_member_the_supervisor_calls_landed_must_really_be_there(tmp_path: Path) -> None:
    """Arrival is evidence, and evidence is checked."""
    store_root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(store_root))

    # Says both landed; only one was written.
    request = _request(tmp_path, landed=["encoder.safetensors"])
    request.landed.append("decoder.safetensors")

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=store_root)
    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code == "model_source_file_unavailable"


def test_a_landed_member_off_the_roster_is_a_wrong_ask_not_a_partial_one(tmp_path: Path) -> None:
    store_root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(store_root))

    request = _request(tmp_path, landed=["encoder.safetensors"])
    request.landed.append("nowhere.safetensors")

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=store_root)
    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code == "model_source_landed_invalid"


@_needs_binding
def test_every_roster_row_must_carry_a_header(tmp_path: Path) -> None:
    """A plan is a pure function of headers, so a roster missing one can plan nothing."""
    store_root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(store_root))

    request = _request(tmp_path, landed=["encoder.safetensors"])
    Path(request.roster[0].header_path).unlink()

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=store_root)
    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code == "model_source_header_unavailable"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"replayed": False, "complete": False, "spent": ["encoder.safetensors"]},
            pb.MODEL_SOURCE_PREPARE_OUTCOME_ADVANCED,
        ),
    ],
)
def test_an_incomplete_pass_answers_advanced_and_names_what_may_be_dropped(
    tmp_path: Path, value: dict[str, object], expected: int
) -> None:
    """ADVANCED is not a verdict: work was journalled and carriers are still owed.

    `spent` is the answer that matters — the supervisor drops those carriers now, which is
    what makes a raw source carrier transient by construction, and so what makes its
    prohibition on the shared cache affordable.
    """
    request = _request(tmp_path, landed=["encoder.safetensors"])
    result = model_source_prepare._result(request, value)

    assert result.outcome == expected
    assert not result.sources
    assert list(result.spent) == ["encoder.safetensors"]
