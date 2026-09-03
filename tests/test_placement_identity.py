"""Placement identity is the content prepared, never a credential.

xs-014 finding 1: the seed used to be the raw document bytes — whose per-signing
expiry and worker pins re-identified the SAME package on every re-signed converge,
defeating ledger reuse and (pre co-tenancy fix) making one package's identity depend
on its neighbors. Those fields are deleted (owner ruling 2026-09-03), so a download
set is now a pure function of its content; identity still derives from the named
facts, because that is what identity means.
"""

from __future__ import annotations

from cozy_runtime.internal.worker.package_prepare import placement_identity

_MODELS = [
    {
        "manifest": "sha256:aa",
        "model": "paul/anima",
        "package": "paul/anima",
        "release": "1.0.0",
        "slot": "generate.models.model",
    },
    {
        "manifest": "sha256:bb",
        "model": "paul/sdxl",
        "package": "paul/anima",
        "release": "1.0.0",
        "slot": "generate.models.refiner",
    },
]


def test_identity_ignores_credential_clocks() -> None:
    # Two download sets for the same content are identical bytes now, and even a
    # document that still carried an expiry or worker pins could not reach the
    # derivation: it reads named facts, never the bytes.
    first = placement_identity("paul/anima", "1.0.0", "sha256:cc", _MODELS)
    second = placement_identity("paul/anima", "1.0.0", "sha256:cc", list(_MODELS))
    assert first == second


def test_identity_ignores_model_row_order() -> None:
    assert placement_identity(
        "paul/anima", "1.0.0", "sha256:cc", list(reversed(_MODELS))
    ) == placement_identity("paul/anima", "1.0.0", "sha256:cc", _MODELS)


def test_identity_tracks_the_selection() -> None:
    base = placement_identity("paul/anima", "1.0.0", "sha256:cc", _MODELS)
    assert placement_identity("paul/anima", "1.0.0", "sha256:cc", _MODELS[:1]) != base
    assert placement_identity("paul/anima", "1.0.1", "sha256:cc", _MODELS) != base
    assert placement_identity("paul/anima", "1.0.0", "sha256:dd", _MODELS) != base
