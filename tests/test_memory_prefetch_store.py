"""Real TensorFS reads exercise speculative host credits and cancellation; no GPU use."""

from pathlib import Path

import pytest

from cozy_runtime.internal import fill
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
        return checks <= 2

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


def _multi_window_store(tmp_path: Path) -> tuple[str, str, int]:
    """More than one existing native slot, with real immutable bytes and no tensor device."""
    import tensorfs
    from tensorfs.derived import Derivation, Part, Target, Tensor

    from native_weights import NativeExecution

    store = tensorfs.Store.init(tmp_path / "large-store")
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    size = 4 * fill.slot_bytes(fill.WINDOW_BYTES)
    request = Derivation(
        {},
        {
            "model": Target(
                add={"weight": Tensor("u8", (size,), plain, {"value": Part("u8", (size,))})}
            )
        },
        {},
        (("model", "weight"),),
    )
    host = NativeExecution(store, tmp_path, "large", {}, {"result": size})
    with host.client.open_output("result", request) as writer:
        writer.add_part("model", "weight", "value", b"x" * size)
        facts = writer.commit()
    return str(store.root), "sha256:" + str(facts["manifest"]["sha256"]), size


@pytest.mark.parametrize("extra", [0, 2 << 20])
def test_multi_window_optional_budget_reports_partial_not_full_readiness(
    tmp_path: Path,
    extra: int,
) -> None:
    from cozy_runtime.internal import fill

    root, manifest, size = _multi_window_store(tmp_path)
    policy = Machine({}, lambda _: None, lambda _: None, host_available=lambda: 1 << 30)
    budget = 2 * fill.slot_bytes(fill.WINDOW_BYTES) + extra
    report = warm_component(
        root,
        manifest,
        "model",
        reserve=lambda n: policy.reserve_host("partial", n),
        release=lambda: policy.release_host("partial"),
        wanted=lambda: True,
        allowance=lambda _: budget,
    )
    assert 0 < report["bytes"] <= budget < size
    assert not report["complete"] and "headroom" in report["skipped"]
    assert report["batches"] >= 2
    assert not policy.host_reserved
    # A prefix is not a full warming promise; normal demand still sees every immutable byte.
    checkpoint = fill.Checkpoint(root, manifest)
    with checkpoint.acquire() as lease:
        assert lease.live


def test_optional_budget_below_one_native_quantum_does_not_allocate(tmp_path: Path) -> None:
    store, _, _, receipt = _committed(tmp_path)

    def reserve(_nbytes: int) -> bool:
        pytest.fail("an underfunded native read allocated its slot")

    report = warm_component(
        str(store.root),
        receipt.artifact.manifest.digest,
        "model",
        reserve=reserve,
        release=lambda: None,
        wanted=lambda: True,
        allowance=lambda _: 1,
    )
    assert "skipped" in report and "bytes" not in report


def test_old_store_capability_skips_without_ordinary_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    def ordinary(*_args: object) -> None:
        pytest.fail("optional read used ordinary verification")

    monkeypatch.setattr(
        fill,
        "store",
        lambda _: SimpleNamespace(acquire_cozytensors=ordinary, manifest=ordinary),
    )
    report = warm_component(
        "store",
        "manifest",
        "model",
        reserve=lambda _: True,
        release=lambda: None,
        wanted=lambda: True,
    )
    assert report["skipped"] == "verified-only acquisition unavailable"
