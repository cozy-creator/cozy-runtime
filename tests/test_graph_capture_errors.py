"""Consumed capture errors must release their callers' leases without waiting for cyclic GC."""

from __future__ import annotations

import weakref

from cozy_runtime.graphs import _capture_failure


class Lease:
    pass


def chained_failure() -> tuple[RuntimeError, weakref.ReferenceType[Lease]]:
    lease = Lease()
    try:
        raise ValueError("inner capture failure")
    except ValueError as cause:
        try:
            raise RuntimeError("capture refusal\nsecond line") from cause
        except RuntimeError as failure:
            return failure, weakref.ref(lease)


def test_consumed_capture_failure_releases_caller_lease_and_keeps_its_diagnostic() -> None:
    failure, lease = chained_failure()
    cause = failure.__cause__
    assert lease() is not None
    assert _capture_failure(failure) == "RuntimeError: capture refusal"
    assert lease() is None
    assert failure.__cause__ is cause
    assert str(failure) == "capture refusal\nsecond line"


def test_consumed_capture_failure_handles_cyclic_exception_links() -> None:
    failure, lease = chained_failure()
    cause = failure.__cause__
    assert cause is not None
    cause.__context__ = failure
    assert lease() is not None
    assert _capture_failure(failure) == "RuntimeError: capture refusal"
    assert lease() is None
    assert failure.__cause__ is cause and cause.__context__ is failure


def test_consumed_capture_group_releases_nested_exception_frames() -> None:
    failure, lease = chained_failure()
    group = ExceptionGroup("capture errors", [failure])
    assert lease() is not None
    assert _capture_failure(group) == "ExceptionGroup: capture errors (1 sub-exception)"
    assert lease() is None
    assert group.exceptions == (failure,)
