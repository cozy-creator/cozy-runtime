"""Canonical release-document adapter for the offline environment materializer.

Tensorhub owns the document schemas and delivery grants.  This module owns the one Runtime
join from their already-fetched exact bytes to :class:`environment.EnvironmentInput`.  Local
paths are deliberately function arguments: they never enter a canonical release document.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical, environment

ENDPOINT_BUNDLE_FORMAT = "tensorhub.endpoint_bundle/1"
ENVIRONMENT_SPEC_FORMAT = "cozy.worker.v1.EndpointEnvironmentSpec/1"
RESOLVED_WHEEL_SET_FORMAT = "ResolvedWheelSet/1"
WHEELHOUSE_MANIFEST_FORMAT = "WheelhouseManifest/1"


class LaunchRefusal(Exception):
    """A fail-closed release/grant join verdict with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def environment_input(
    *,
    environment_spec_bytes: bytes,
    endpoint_bundle_bytes: bytes,
    resolved_wheel_set_bytes: bytes,
    wheelhouse_manifest_bytes: bytes,
    wheel_paths: Mapping[str, Path],
) -> environment.EnvironmentInput:
    """Join exact th-046 documents and fetched wheel paths without resolving anything."""

    spec = _document(environment_spec_bytes, "environment_spec")
    bundle = _document(endpoint_bundle_bytes, "endpoint_bundle")
    resolved = _document(resolved_wheel_set_bytes, "resolved_wheel_set")
    wheelhouse = _document(wheelhouse_manifest_bytes, "wheelhouse_manifest")
    _closed(
        spec,
        {
            "endpoint_bundle_digest",
            "format",
            "platform_target",
            "project_wheel_digest",
            "wheelhouse_manifest_digest",
        },
        "environment_spec",
    )
    if spec["format"] != ENVIRONMENT_SPEC_FORMAT:
        raise LaunchRefusal("environment_spec_format", str(spec["format"]))
    _closed(
        bundle,
        {
            "descriptor",
            "endpoint_release_id",
            "evaluated_config",
            "format",
            "project_wheel",
            "resolved_wheel_set",
            "wheelhouse_manifest",
        },
        "endpoint_bundle",
    )
    if bundle["format"] != ENDPOINT_BUNDLE_FORMAT:
        raise LaunchRefusal("endpoint_bundle_format", str(bundle["format"]))
    _closed(
        resolved,
        {"format", "lock_digest", "platform_target", "wheelhouse_manifest_digest", "wheels"},
        "resolved_wheel_set",
    )
    if resolved["format"] != RESOLVED_WHEEL_SET_FORMAT:
        raise LaunchRefusal("resolved_wheel_set_format", str(resolved["format"]))
    _closed(wheelhouse, {"format", "platform_target", "wheels"}, "wheelhouse_manifest")
    if wheelhouse["format"] != WHEELHOUSE_MANIFEST_FORMAT:
        raise LaunchRefusal("wheelhouse_manifest_format", str(wheelhouse["format"]))

    _require_digest(spec["endpoint_bundle_digest"], "environment_spec.endpoint_bundle_digest")
    _require_digest(spec["project_wheel_digest"], "environment_spec.project_wheel_digest")
    _require_digest(
        spec["wheelhouse_manifest_digest"], "environment_spec.wheelhouse_manifest_digest"
    )
    _matches(endpoint_bundle_bytes, str(spec["endpoint_bundle_digest"]), "endpoint_bundle")
    _matches(
        wheelhouse_manifest_bytes,
        str(spec["wheelhouse_manifest_digest"]),
        "wheelhouse_manifest",
    )
    _ref_matches(
        bundle["resolved_wheel_set"], resolved_wheel_set_bytes, "resolved_wheel_set"
    )
    _ref_matches(
        bundle["wheelhouse_manifest"], wheelhouse_manifest_bytes, "wheelhouse_manifest"
    )
    if bundle["wheelhouse_manifest"] != {
        "digest": spec["wheelhouse_manifest_digest"],
        "length": len(wheelhouse_manifest_bytes),
    }:
        raise LaunchRefusal(
            "wheelhouse_manifest_ref_mismatch", "bundle and environment spec disagree"
        )
    if resolved["wheelhouse_manifest_digest"] != spec["wheelhouse_manifest_digest"]:
        raise LaunchRefusal(
            "wheelhouse_manifest_ref_mismatch", "resolved set and environment spec disagree"
        )
    if not isinstance(spec["platform_target"], dict) or not (
        spec["platform_target"] == resolved["platform_target"] == wheelhouse["platform_target"]
    ):
        raise LaunchRefusal("platform_target_mismatch", "release documents disagree")

    project = _wheel(bundle["project_wheel"], "project")
    if project.digest != spec["project_wheel_digest"]:
        raise LaunchRefusal("project_wheel_mismatch", project.digest)
    wheels = [project]
    wheels.extend(_wheels(resolved["wheels"], "bundle"))
    wheels.extend(_wheels(wheelhouse["wheels"], "wheelhouse"))
    digests = [wheel.digest for wheel in wheels]
    distributions = [wheel.distribution for wheel in wheels]
    if len(digests) != len(set(digests)):
        raise LaunchRefusal("wheel_digest_duplicate", "one digest has multiple owners")
    if len(distributions) != len(set(distributions)):
        raise LaunchRefusal("wheel_distribution_duplicate", "one distribution has multiple owners")
    if set(wheel_paths) != set(digests):
        missing = sorted(set(digests) - set(wheel_paths))
        extra = sorted(set(wheel_paths) - set(digests))
        raise LaunchRefusal(
            "wheel_grant_set_mismatch", f"missing={missing[:1]} extra={extra[:1]}"
        )
    grants = tuple(
        environment.BlobGrant(wheel.digest, wheel.length, wheel_paths[wheel.digest])
        for wheel in wheels
    )
    return environment.EnvironmentInput(
        environment_spec_bytes=environment_spec_bytes,
        environment_spec_digest=_digest(environment_spec_bytes),
        endpoint_bundle_bytes=endpoint_bundle_bytes,
        wheelhouse_manifest_bytes=wheelhouse_manifest_bytes,
        wheels=tuple(wheels),
        grants=grants,
    )


def materialize_exact(
    source: environment.EnvironmentInput,
    root: Path,
    *,
    uv: Path,
    expected_receipt_bytes: bytes,
) -> environment.InstalledEnvironment:
    """Materialize and require byte identity with th-046's proof-seat receipt."""

    try:
        expected = _document(expected_receipt_bytes, "expected_environment_receipt")
    except LaunchRefusal:
        raise
    if expected.get("format") != environment.RECEIPT_FORMAT:
        raise LaunchRefusal("expected_environment_receipt_format", str(expected.get("format")))
    expected_digest = _digest(expected_receipt_bytes)
    try:
        installed = environment.materialize(
            source,
            root,
            uv=uv,
            expected_receipt_digest=expected_digest,
        )
    except environment.EnvironmentRefusal as exc:
        raise LaunchRefusal(exc.code, exc.detail) from exc
    if installed.receipt_bytes != expected_receipt_bytes:
        raise LaunchRefusal(
            "expected_environment_receipt_bytes_mismatch",
            f"measured {installed.receipt_digest}, exact expected bytes differ",
        )
    return installed


def _document(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise LaunchRefusal(f"{name}_invalid", str(exc)) from exc
    if not isinstance(value, dict):
        raise LaunchRefusal(f"{name}_invalid", "document is not an object")
    return value


def _closed(value: dict[str, Any], keys: set[str], name: str) -> None:
    if set(value) != keys:
        raise LaunchRefusal(
            f"{name}_invalid", f"fields={sorted(value)} expected={sorted(keys)}"
        )


def _ref_matches(value: object, raw: bytes, name: str) -> None:
    if not isinstance(value, dict) or set(value) != {"digest", "length"}:
        raise LaunchRefusal(f"{name}_ref_invalid", "reference is not {digest,length}")
    digest = value.get("digest")
    length = value.get("length")
    _require_digest(digest, f"{name}.digest")
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise LaunchRefusal(f"{name}_ref_invalid", "length is not positive")
    if length != len(raw):
        raise LaunchRefusal(f"{name}_length_mismatch", f"measured {len(raw)}, expected {length}")
    _matches(raw, str(digest), name)


def _matches(raw: bytes, digest: str, name: str) -> None:
    if _digest(raw) != digest:
        raise LaunchRefusal(
            f"{name}_digest_mismatch", f"measured {_digest(raw)}, expected {digest}"
        )


def _wheels(value: object, owner: str) -> list[environment.WheelInventoryEntry]:
    if not isinstance(value, list):
        raise LaunchRefusal(f"{owner}_wheel_inventory_invalid", "wheels is not a list")
    wheels = [_wheel(row, owner) for row in value]
    names = [wheel.distribution for wheel in wheels]
    if names != sorted(names) or len(names) != len(set(names)):
        raise LaunchRefusal(
            f"{owner}_wheel_inventory_invalid", "distributions are not sorted unique"
        )
    return wheels


def _wheel(value: object, owner: str) -> environment.WheelInventoryEntry:
    keys = {"digest", "distribution", "filename", "length", "tags", "version"}
    if not isinstance(value, dict) or set(value) != keys:
        raise LaunchRefusal(f"{owner}_wheel_invalid", "wheel row has the wrong fields")
    digest = value["digest"]
    _require_digest(digest, f"{owner}_wheel.digest")
    strings = (value["distribution"], value["filename"], value["version"])
    if not all(isinstance(item, str) and item for item in strings):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "wheel strings are empty or invalid")
    length = value["length"]
    tags = value["tags"]
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise LaunchRefusal(f"{owner}_wheel_invalid", "length is not positive")
    if (
        not isinstance(tags, list)
        or not tags
        or not all(isinstance(tag, str) and tag for tag in tags)
        or tags != sorted(set(tags))
    ):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "tags are not non-empty sorted unique")
    return environment.WheelInventoryEntry(
        distribution=value["distribution"],
        version=value["version"],
        filename=value["filename"],
        digest=str(digest),
        length=length,
        tags=tuple(tags),
        owner=owner,
    )


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or canonical.DIGEST_RE.fullmatch(value) is None:
        raise LaunchRefusal(f"{name}_invalid", "not sha256:<64 lowercase hex>")


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
