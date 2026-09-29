"""Real TensorFS reads exercise speculative host credits and cancellation; no GPU use."""

from pathlib import Path

import pytest

from cozy_runtime.internal.memory import Machine
from cozy_runtime.internal.worker.page_warm import warm_component
from test_weights_resume_native import _committed


def test_store_prefetch_releases_its_host_promise(tmp_path: Path) -> None:
    store, _, _, receipt = _committed(tmp_path)
    policy = Machine({}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30)
    report = warm_component(
        str(store.root),
        receipt.artifact.manifest.digest,
        "model",
        reserve=lambda n: policy.reserve_host("prefetch", n),
        release=lambda: policy.release_host("prefetch"),
        wanted=lambda: True,
    )
    assert report["bytes"] == 2048 and "skipped" not in report
    assert not policy.host_reserved
    assert store.manifest(receipt.artifact.manifest.digest)


def test_superseded_read_returns_slots_and_host_credit(tmp_path: Path) -> None:
    store, _, _, receipt = _committed(tmp_path)
    policy = Machine({}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30)
    checks = 0

    def wanted() -> bool:
        nonlocal checks
        checks += 1
        if checks > 1:
            assert not policy.host_reserved  # Materialized slots already count in cgroup usage.
        return checks == 1

    report = warm_component(
        str(store.root),
        receipt.artifact.manifest.digest,
        "model",
        reserve=lambda n: policy.reserve_host("old-queue", n),
        release=lambda: policy.release_host("old-queue"),
        wanted=wanted,
    )
    assert checks >= 2 and "queue changed" in report["skipped"]
    assert not policy.host_reserved
    # The interrupted read neither disposes the artifact nor prevents its next lease.
    again = warm_component(
        str(store.root),
        receipt.artifact.manifest.digest,
        "model",
        reserve=lambda n: policy.reserve_host("new-queue", n),
        release=lambda: policy.release_host("new-queue"),
        wanted=lambda: True,
    )
    assert again["bytes"] == 2048 and not policy.host_reserved


def test_host_denial_skips_prefetch_without_claiming_a_load(tmp_path: Path) -> None:
    store, _, _, receipt = _committed(tmp_path)
    policy = Machine({}, lambda _: None, lambda _: None, host_available=lambda: 0)
    report = warm_component(
        str(store.root),
        receipt.artifact.manifest.digest,
        "model",
        reserve=lambda n: policy.reserve_host("denied", n),
        release=lambda: policy.release_host("denied"),
        wanted=lambda: True,
    )
    assert "skipped" in report and "bytes" not in report
    assert not policy.host_reserved


def test_failed_slot_allocation_releases_the_promise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cozy_runtime.internal.worker import page_warm

    store, _, _, receipt = _committed(tmp_path)
    policy = Machine({}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30)

    def fail(nbytes: int) -> bytearray:
        assert policy.host_reserved == {"prefetch": nbytes}
        raise MemoryError("slot allocation refused")

    monkeypatch.setattr(page_warm, "bytearray", fail, raising=False)
    with pytest.raises(MemoryError, match="slot allocation refused"):
        warm_component(
            str(store.root),
            receipt.artifact.manifest.digest,
            "model",
            reserve=lambda n: policy.reserve_host("prefetch", n),
            release=lambda: policy.release_host("prefetch"),
            wanted=lambda: True,
        )
    assert not policy.host_reserved
