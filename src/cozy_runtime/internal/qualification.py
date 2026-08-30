"""The one stored Runtime qualification document.

The command that produced a qualification supplies the context and request version.  Only the
successful, independently retained evidence needs a portable discriminator.  Environment, model,
native-wheel, and execution qualification therefore share this one closed envelope instead of
each growing request, result, inspection, and observation formats of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from cozy_runtime.internal import canonical

FORMAT = "cozy.runtime.Qualification/1"
Kind = Literal["environment", "execution", "model", "native-wheel"]
KINDS = frozenset({"environment", "execution", "model", "native-wheel"})
EVIDENCE_FIELDS = {
    # Environment qualification never repeats the measured environment.  It identifies the one
    # EnvironmentReceipt that passed the proof seat; those receipt bytes remain the observation.
    "environment": frozenset({"environment_receipt"}),
    "execution": frozenset(
        {
            "accelerator_qualification",
            "attempt_metrics",
            "attempt_outcome_digest",
            "attempt_plan_digest",
            "base_worker_profile",
            "binding_plan_digest",
            "boot_warmup",
            "control_runtime_digest",
            "entrypoint",
            "environment_spec_digest",
            "execution_started",
            "installed_environment_receipt_digest",
            "invocation_spec_digest",
            "logical_model_digest",
            "model_construction_contract_digests",
            "object_set_digest",
            "output_manifest",
            "overlay_content_digest",
            "package_binding_release_digest",
            "package_descriptor_digest",
            "package_execution_digest",
            "package_release_id",
            "payload_digest",
            "placement_set_digest",
            "result_digest",
            "result_schema_digest",
            "status",
            "worker_resources",
        }
    ),
    "model": frozenset(
        {
            "application",
            "environment_spec_digest",
            "functions",
            "installed_environment_receipt",
            "package_descriptor",
            "package_release_id",
            "project_wheel_digest",
        }
    ),
    "native-wheel": frozenset(
        {
            "base_worker_profile",
            "host_capability",
            "host_qualification",
            "installed_environment_receipt_digest",
            "operator_observation",
            "overlay_content_digest",
            "package_environment_spec_digest",
            "static_inspection",
            "wheelhouse_manifest_digest",
        }
    ),
}


class QualificationError(Exception):
    """A malformed qualification document."""


def document(kind: Kind, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build one qualification without preserving a nested legacy report envelope."""

    if kind not in KINDS:
        raise QualificationError(f"unknown qualification kind {kind!r}")
    _closed_evidence(kind, evidence)
    return {"evidence": dict(evidence), "format": FORMAT, "kind": kind}


def read(raw: bytes) -> tuple[Kind, dict[str, Any]]:
    """Read exact canonical qualification bytes and return their typed evidence."""

    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise QualificationError(str(exc)) from exc
    if not isinstance(value, dict) or set(value) != {"evidence", "format", "kind"}:
        raise QualificationError("qualification fields are not closed")
    kind = value.get("kind")
    evidence = value.get("evidence")
    if value.get("format") != FORMAT or kind not in KINDS or not isinstance(evidence, dict):
        raise QualificationError("qualification format, kind, or evidence is invalid")
    typed_kind = cast(Kind, kind)
    _closed_evidence(typed_kind, evidence)
    return typed_kind, dict(evidence)


def _closed_evidence(kind: Kind, evidence: Mapping[str, Any]) -> None:
    expected = EVIDENCE_FIELDS[kind]
    if set(evidence) != expected:
        raise QualificationError(
            f"{kind} evidence fields={sorted(evidence)} expected={sorted(expected)}"
        )
