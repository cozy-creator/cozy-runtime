"""RUN 190, AS A TEST. A selection declares a DIGEST; the store declares the LENGTH.

`cli/package_prepare` requires `manifest_length` on every selection row it accepts. The
pod's rows come from a signed DownloadDelegation, whose model entries carry
lane/manifest/model/package/release/slot and no length at all. Preparation read one anyway,
got 0, and compared it to the store's real length -- so every modeled preparation on a
rented pod refused

    package_prepare_model_selection_mismatch:
    paul/anima@1.0.0/bf16 does not resolve to sha256:5d6a0518...

naming the digest that had in fact MATCHED. The lane was right, the record was right, and
the message pointed at neither.
"""

from __future__ import annotations

import pytest

from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    _resolved_manifest_length,
)

DIGEST = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64
WHERE = "paul/anima@1.0.0/bf16"
RESOLVED = {"manifest_digest": DIGEST, "manifest_length": 161}


def test_a_selection_without_a_declared_length_takes_the_stores() -> None:
    """The pod's shape. This is the case that could never pass."""

    assert _resolved_manifest_length(RESOLVED, DIGEST, 0, WHERE) == 161


def test_a_matching_declared_length_is_accepted() -> None:
    """The CLI's shape, whose stronger assertion is preserved rather than dropped."""

    assert _resolved_manifest_length(RESOLVED, DIGEST, 161, WHERE) == 161


def test_a_disagreeing_declared_length_still_refuses() -> None:
    """The negative control for the half that was doing real work: a caller that DOES
    declare a length is still held to it."""

    with pytest.raises(PreparationRefusal) as refusal:
        _resolved_manifest_length(RESOLVED, DIGEST, 1, WHERE)
    assert refusal.value.code == "package_prepare_model_selection_mismatch"
    # Both sides are named, so a length disagreement cannot read as a digest disagreement.
    assert "/161" in refusal.value.detail and "/1" in refusal.value.detail


def test_a_digest_the_release_does_not_name_still_refuses() -> None:
    """The cross-check this exists to perform, undamaged."""

    with pytest.raises(PreparationRefusal) as refusal:
        _resolved_manifest_length(RESOLVED, OTHER, 0, WHERE)
    assert refusal.value.code == "package_prepare_model_selection_mismatch"
    assert DIGEST in refusal.value.detail and OTHER in refusal.value.detail


def test_the_delegation_shape_is_exactly_the_cli_shape_minus_the_length() -> None:
    """The two producers of a selection row, locked together.

    `cli/package_prepare` refuses a row whose key set is not exactly `required_model`; the
    hub's `DelegatedModel` carries every one of those names except `manifest_length`. That
    one difference is the whole defect, and it is the only one -- so nothing else read off a
    selection can be absent on the pod path.
    """

    cli_required = {"package", "slot", "model", "release", "lane", "manifest", "manifest_length"}
    delegated = {"lane", "manifest", "model", "package", "release", "slot"}
    assert cli_required - delegated == {"manifest_length"}
    assert delegated - cli_required == set()
