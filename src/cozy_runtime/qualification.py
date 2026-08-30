"""The one stored Runtime qualification decision.

Proof-seat evidence is normalized Hub state, not a portable report hierarchy. Once all required
environment, model, native-wheel, and execution checks pass, Tensorhub authors this two-field
decision over the exact selected facts. PlacementSet carries its ref; Runtime recomputes the
selection digest before trusting it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from cozy_runtime.internal import canonical

FORMAT = "cozy.runtime.Qualification/1"


class QualificationError(Exception):
    """A malformed or mismatched qualification decision."""


def selection_digest(
    *,
    bindings_digest: str,
    environment_digest: str,
    environment_receipt: Mapping[str, Any],
    package_descriptor: Mapping[str, Any],
    package: Mapping[str, Any],
) -> str:
    """Hash the closed selection preimage. Qualification and placement id are excluded."""

    for name, value in (
        ("bindings_digest", bindings_digest),
        ("environment_digest", environment_digest),
    ):
        if canonical.DIGEST_RE.fullmatch(value) is None:
            raise QualificationError(f"{name} is not sha256:<64 lowercase hex>")
    refs = {
        "environment_receipt": _ref(environment_receipt, "environment_receipt"),
        "package_descriptor": _ref(package_descriptor, "package_descriptor"),
    }
    preimage = canonical.write(
        {
            "bindings_digest": bindings_digest,
            "environment_digest": environment_digest,
            "package": _package(package),
            **refs,
        }
    )
    return "sha256:" + hashlib.sha256(preimage).hexdigest()


def document(selection: str) -> dict[str, str]:
    if canonical.DIGEST_RE.fullmatch(selection) is None:
        raise QualificationError("selection_digest is not sha256:<64 lowercase hex>")
    return {"format": FORMAT, "selection_digest": selection}


def read(raw: bytes) -> str:
    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise QualificationError(str(exc)) from exc
    if not isinstance(value, dict) or set(value) != {"format", "selection_digest"}:
        raise QualificationError("qualification fields are not closed")
    selection = value.get("selection_digest")
    if value.get("format") != FORMAT or not isinstance(selection, str):
        raise QualificationError("qualification format or selection digest is invalid")
    if canonical.DIGEST_RE.fullmatch(selection) is None:
        raise QualificationError("selection_digest is not sha256:<64 lowercase hex>")
    return selection


def _ref(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if set(value) != {"digest", "length"}:
        raise QualificationError(f"{name} is not a closed Ref")
    digest = value.get("digest")
    length = value.get("length")
    if (
        not isinstance(digest, str)
        or canonical.DIGEST_RE.fullmatch(digest) is None
        or not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
    ):
        raise QualificationError(f"{name} is not a positive exact Ref")
    return {"digest": digest, "length": length}


def _package(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"package", "release", "release_digest", "project_wheel"}:
        raise QualificationError("package is not a closed PackageSelection")
    identity = value.get("package")
    release = value.get("release")
    release_digest = value.get("release_digest")
    wheel = value.get("project_wheel")
    if (
        not isinstance(identity, str)
        or identity.count("/") != 1
        or not all(identity.split("/"))
        or not isinstance(release, str)
        or not release
        or not isinstance(release_digest, str)
        or canonical.DIGEST_RE.fullmatch(release_digest) is None
        or not isinstance(wheel, Mapping)
    ):
        raise QualificationError("package identity, release, or provenance is invalid")
    if set(wheel) != {"distribution", "filename", "import_roots", "ref", "tags", "version"}:
        raise QualificationError("project_wheel is not a closed WheelFact")
    project = {
        "distribution": wheel.get("distribution"),
        "filename": wheel.get("filename"),
        "import_roots": wheel.get("import_roots"),
        "ref": _ref(wheel["ref"], "project_wheel.ref"),
        "tags": wheel.get("tags"),
        "version": wheel.get("version"),
    }
    if (
        project["distribution"] != identity.split("/", 1)[1]
        or project["version"] != release
        or not isinstance(project["filename"], str)
        or not isinstance(project["import_roots"], list)
        or not isinstance(project["tags"], list)
    ):
        raise QualificationError("project_wheel does not match package release")
    return {
        "package": identity,
        "project_wheel": project,
        "release": release,
        "release_digest": release_digest,
    }
