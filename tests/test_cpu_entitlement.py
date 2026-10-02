"""Real cgroup control-file fixtures using the shared containment mount resolver."""
from pathlib import Path

import pytest

from cozy_runtime.internal import proctree
from cozy_runtime.internal.model_sources import VerificationLimits


def hierarchy(tmp_path, monkeypatch, *, unified, quota="1020000", parent_quota="-1"):
    root = tmp_path / "cgroup"
    root.mkdir()
    leaf = root / "pod"
    leaf.mkdir()
    membership = tmp_path / "membership"
    mountinfo = tmp_path / "mountinfo"
    if unified:
        (root / "cgroup.controllers").write_text("cpu memory")
        membership.write_text("0::/pod\n")
        (root / "cpu.max").write_text("max 100000")
        (leaf / "cpu.max").write_text(f"{quota} 100000")
    else:
        membership.write_text("2:cpu,cpuacct:/tenant/pod\n3:memory:/tenant/pod\n")
        mountinfo.write_text(f"19 1 0:1 /tenant {root} rw - cgroup cgroup rw,cpu,cpuacct\n")
        for directory, value in ((root, parent_quota), (leaf, quota)):
            (directory / "cpu.cfs_quota_us").write_text(value)
            (directory / "cpu.cfs_period_us").write_text("100000")
    real_path = Path

    def redirected(value, *args):
        mapping = {"/sys/fs/cgroup": root, "/proc/self/cgroup": membership,
                   "/proc/self/mountinfo": mountinfo}
        return real_path(mapping.get(str(value), value), *args)

    monkeypatch.setattr(proctree, "Path", redirected)
    monkeypatch.setattr(proctree.os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(proctree, "available_host_bytes", lambda: 8 << 30)
    return root, leaf


@pytest.mark.parametrize("unified", [False, True])
def test_visible_cpu_quota_uses_affinity_and_bounded_native_verifier(
    tmp_path, monkeypatch, unified
):
    hierarchy(tmp_path, monkeypatch, unified=unified)
    assert proctree.available_cpus() == pytest.approx(10.2)
    limits = VerificationLimits.for_process()
    assert limits.workers == 4 and limits.pending_fds == 8
    assert limits.pending_bytes == 128 << 20


def test_v1_parent_quota_and_fractional_core_bound_workers(tmp_path, monkeypatch):
    hierarchy(tmp_path, monkeypatch, unified=False, quota="1360000", parent_quota="250000")
    assert proctree.available_cpus() == pytest.approx(2.5)
    assert VerificationLimits.for_process().workers == 1


@pytest.mark.parametrize("quota", ["broken", "0", "-2"])
def test_unreadable_or_invalid_quota_retains_single_worker(tmp_path, monkeypatch, quota):
    hierarchy(tmp_path, monkeypatch, unified=False, quota=quota)
    assert proctree.available_cpus() == 1
    assert VerificationLimits.for_process().workers == 1


def test_low_live_memory_keeps_fd_and_byte_window_bounded(tmp_path, monkeypatch):
    hierarchy(tmp_path, monkeypatch, unified=False)
    monkeypatch.setattr(proctree, "available_host_bytes", lambda: 1 << 20)
    limits = VerificationLimits.for_process()
    assert limits.workers == 1 and limits.pending_fds == 2
    assert limits.pending_bytes == 64 << 10


def test_v1_limits_drive_real_native_verification_and_corruption_rejection(tmp_path, monkeypatch):
    import hashlib
    import os

    from tensorfs import plane as native_plane

    from cozy_runtime.author._executor_requests import SourceBlob
    from cozy_runtime.internal.fill import Checkpoint
    from cozy_runtime.internal.model_sources import BrokeredModelSources
    from test_machine_model_sources import owner_model

    hierarchy(tmp_path, monkeypatch, unified=False)
    _store, manifest, lease, files, header, _data = owner_model(tmp_path)
    calls = []

    def exchange(request, into):
        assert into is SourceBlob
        path = header if request.role == "header" else files[request.name]
        body = path.read_bytes()
        calls.append(request)
        return SourceBlob(ok=True, sha256=hashlib.sha256(body).hexdigest(), length=len(body),
                          descriptor=os.open(path, os.O_RDONLY | os.O_CLOEXEC))

    provider = BrokeredModelSources(exchange)
    assert provider.limits.workers == 4
    checkpoint = Checkpoint("", manifest, source_provider=provider)
    plan = checkpoint.read_plan((("linear", "weight"),), 4 << 20, ("linear",))
    native = native_plane.Plane(devices=[], readers=1)
    try:
        source = provider.source(native, manifest, plan)
        assert source is not None and provider.descriptors
        assert any(request.role == "object" for request in calls)
        provider.close()
        del source
        object_id, path = next(iter(files.items()))
        original_length = path.stat().st_size
        path.chmod(0o644)
        path.write_bytes(b"z" * original_length)
        path.chmod(0o444)

        def corrupt(request, into):
            reply = exchange(request, into)
            if request.role == "object":
                return SourceBlob(ok=True, sha256=object_id.removeprefix("sha256:"),
                                  length=reply.length, descriptor=reply.descriptor)
            return reply

        broken = BrokeredModelSources(corrupt)
        assert broken.limits.workers == 4
        with pytest.raises(native_plane.Invalid, match="identity"):
            broken.source(native, manifest, plan)
        assert not broken.descriptors
        broken.close()
        assert "libcuda.so" not in Path("/proc/self/maps").read_text()
    finally:
        provider.close()
        native.close()
        lease.release()
