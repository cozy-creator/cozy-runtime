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
    package_release: Mapping[str, Any],
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
        "package_release": _ref(package_release, "package_release"),
    }
    preimage = canonical.write(
        {
            "bindings_digest": bindings_digest,
            "environment_digest": environment_digest,
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
