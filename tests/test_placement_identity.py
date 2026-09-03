"""Placement identity is the content prepared, never the credential.

xs-014 finding 1: the seed used to be the raw delegation bytes — whose
per-signing expiry and worker pins re-identified the SAME package on every
re-signed converge, defeating ledger reuse and (pre co-tenancy fix) making
one package's identity depend on its neighbors.
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
    # Two delegations for the same content differ only in expiry/pins —
    # which never enter the derivation at all.
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
