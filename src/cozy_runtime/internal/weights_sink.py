"""Execution identity and protocol projections for native TensorFS output receipts.

TensorFS owns source and writer APIs. Runtime binds native facts to the accepted
execution and converts retained native receipts into its existing protocol documents.
The native source API supplies all reader metadata and bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._weights import (
    WeightsReceipt,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.base_observation import Observation
    from cozy_runtime.internal.hostfacts import HostFacts

_OUTPUT_SLOT = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_EXCHANGE_KINDS = frozenset({"declaration", "receipt", "derivation", "native-receipt"})


def work_fingerprint(
    spec: Mapping[str, Any],
    environment_content_digest: str,
    base: Observation,
    device: HostFacts,
) -> str:
    """Parent-authored numerical work identity, independent of attempt/custody clocks.

    The original exact InvocationSpec still scopes the transaction. Its deadline and
    publication grant are authority, not inputs to arithmetic. CUDA architecture/driver
    deliberately participate: AdaLN tables computed on another architecture are not
    presumed bit-identical. Device UUID, memory, paths and boot facts never participate.
    """
    from cozy_runtime.internal.base_observation import protected_distribution

    job = spec.get("job")
    if not isinstance(job, Mapping) or not environment_content_digest:
        raise CapabilityError(
            "weights work has no selected job environment", code="weights_work_identity"
        )
    if device.backend == "cuda" and (device.gpu_sm <= 0 or not device.driver_version):
        raise CapabilityError(
            "CUDA work has no measured architecture/driver", code="weights_work_identity"
        )
    facts = {
        "job": {key: job[key] for key in ("build_id", "job_descriptor_id")},
        "environment_digest": spec.get("environment_digest", ""),
        "environment_content_digest": environment_content_digest,
        "payload_digest": spec["payload_digest"],
        "inputs": spec.get("inputs", []),
        "outputs": spec.get("outputs", []),
        "interpreter": base.interpreter_identity,
        "distributions": {
            name: version for name, version in base.distributions if protected_distribution(name)
        },
        "torch_build": base.accelerator_build,
        "cuda_runtime": base.cuda_version,
        "device": {
            "backend": device.backend or "cpu",
            "architecture": device.gpu_sm,
            "driver": device.driver_version,
        },
    }
    return (
        "sha256:"
        + hashlib.sha256(
            b"cozy.runtime.derived-work/1\0" + canonical_json.encode(facts)
        ).hexdigest()
    )


def exchange_file(spool: Path, output_slot: str, kind: str) -> Path:
    """One worker-chosen spool file; no child-authored path crosses the seam."""
    if _OUTPUT_SLOT.fullmatch(output_slot) is None or kind not in _EXCHANGE_KINDS:
        raise CapabilityError("weights exchange identity is invalid", code="weights_host_identity")
    return spool / f"weights-{output_slot}-{kind}.canonical"


def write_exchange_file(spool: Path, output_slot: str, kind: str, data: bytes) -> Path:
    """Publish immutable exact bytes into the already-brokered attempt spool."""
    target = exchange_file(spool, output_slot, kind)
    try:
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        target.chmod(0o444)
    except FileExistsError:
        if target.read_bytes() != data:
            raise CapabilityError(
                f"weights {kind} replay changed exact bytes",
                code="weights_receipt_mismatch" if kind == "receipt" else "weights_intent_conflict",
            ) from None
    return target


def receipt_from_native(
    output_slot: str,
    transaction_id: str,
    facts: dict[str, Any],
    *,
    replayed: bool,
    request_id: str,
) -> WeightsReceipt:
    raw = canonical_json.encode(facts)
    receipt_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return WeightsReceipt(
        output_slot=output_slot,
        weights_transaction_id=transaction_id,
        tensorfs_receipt_digest=receipt_digest,
        tensorfs_receipt=raw,
        replayed=replayed,
        _artifact=ModelArtifact(
            request_id,
            output_slot,
            ObjectRef("sha256:" + facts["manifest"]["sha256"], facts["manifest"]["length"]),
            receipt_digest,
        ),
    )


def weights_transaction_id(
    owner_scope: str, request_id: str, invocation_spec_digest: str, output_slot: str
) -> str:
    """Runtime alone assigns this ID for creation and finalization; hosts record and echo it."""
    subject = canonical_json.encode(
        {
            "owner_authority_scope": owner_scope,
            "request_id": request_id,
            "invocation_spec_digest": invocation_spec_digest,
            "output_slot": output_slot,
        }
    )
    return "sha256:" + hashlib.sha256(b"cozy.runtime.artifact-transaction\0" + subject).hexdigest()


def protocol_receipt(
    receipt: WeightsReceipt,
    *,
    owner_scope: str,
    request_id: str,
    invocation_spec_digest: str,
) -> tuple[pb.WeightsReceiptRef, bytes, bytes]:
    """Wrap exact TensorFS receipt bytes in Worker Protocol WeightsReceipt/1."""
    body = pb.WeightsReceipt(
        owner_authority_scope=owner_scope,
        request_id=request_id,
        invocation_spec_digest=invocation_spec_digest,
        output_slot=receipt.output_slot,
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=receipt.tensorfs_receipt_digest,
        tensorfs_receipt_canonical_bytes=receipt.tensorfs_receipt,
    )
    canonical_bytes, digest = documents.identity(body)
    return (
        pb.WeightsReceiptRef(
            weights_receipt_digest=digest,
            weights_receipt_canonical_bytes=canonical_bytes,
        ),
        canonical_bytes,
        digest,
    )
