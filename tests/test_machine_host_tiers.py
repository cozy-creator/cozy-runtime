"""Actual SDK socket/typed partition descriptors; no author package or CUDA execution."""
import fcntl
import hashlib
import os
import socket
import threading
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author._executor_requests import Answer, HostTierPlan, HostTierPrepare, decode
from cozy_runtime.internal import canonical, host_tiers
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Load
from cozy_runtime.internal.seam import MAX_FRAME, Channel


def plan(count=1):
    regions = tuple((f"unet/layer{i}.weight",) for i in range(count))
    manifest = "sha256:" + "f" * 64
    return HostTierPlan(manifest=manifest, name="construction/unet",
                        layout=canonical.digest({"manifest": manifest, "regions": regions}),
                        traversal=tuple(("unet", f"layer{i}.weight") for i in range(count)),
                        components=("unet",), regions=regions)


def test_large_typed_roster_is_sealed_and_control_stays_small():
    value = plan(3000)
    with host_tiers.sealed_plan(value) as request:
        assert request.length > MAX_FRAME
        assert len(msgspec.json.encode(request)) < MAX_FRAME
        content = os.pread(request.memfd, request.length, 0)
        assert msgspec.json.decode(content, type=HostTierPlan) == value
        assert hashlib.sha256(content).hexdigest() == request.sha256
        assert fcntl.fcntl(request.memfd, host_tiers._GET_SEALS) == host_tiers._SEALS
        assert not os.get_inheritable(request.memfd)
        with pytest.raises(OSError):
            os.pwrite(request.memfd, b"x", 0)
        with pytest.raises(OSError):
            os.ftruncate(request.memfd, 0)
    with pytest.raises(OSError):
        os.fstat(request.memfd)


@pytest.mark.parametrize("owner_ok", [True, False])
def test_existing_durable_seam_transfers_real_plan_and_older_owner_falls_back(tmp_path, owner_ok):
    left, right = socket.socketpair()
    executor = Executor(Channel(left), tmp_path / "attempt")
    owner = Channel(right)
    errors = []
    received = []

    def serve():
        try:
            frame = owner.recv()
            request = decode(frame)
            assert isinstance(request, HostTierPrepare)
            fd = owner.recv_memfd()
            try:
                content = os.pread(fd, request.length, 0)
                received.append(msgspec.json.decode(content, type=HostTierPlan))
                assert hashlib.sha256(content).hexdigest() == request.sha256
                assert not os.get_inheritable(fd)
                with pytest.raises(OSError):
                    os.pwrite(fd, b"x", 0)
            finally:
                os.close(fd)
            owner.send({"event": "answer", "seq": frame["seq"], "ok": owner_ok,
                        "code": "" if owner_ok else "unknown_durable_request",
                        "detail": "", "future_advisory": 101})
        except Exception as exc:
            errors.append(exc)
        finally:
            owner.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        # Actual Executor method and framing, not a replacement function invocation.
        executor._prepare_tier(plan())
        thread.join()
        assert not errors and received == [plan()]
        assert host_tiers.CAPABILITY in executor.hello()["memory"]
        maps = Path("/proc/self/maps").read_text()
        assert "libcuda.so" not in maps and "libnvidia-ml" not in maps
    finally:
        left.close()
        right.close()
        thread.join()


def test_omitted_load_selector_keeps_baseline_and_unknown_fields_are_harmless():
    old = msgspec.json.decode(b'{"cmd":"load","construction":"a"}', type=Load)
    assert not old.host_tier_owner
    new = msgspec.json.decode(
        b'{"cmd":"load","construction":"a","host_tier_owner":true,"future":123}', type=Load
    )
    assert new.host_tier_owner
    encoded = msgspec.json.encode(Answer(ok=True))
    assert msgspec.json.decode(encoded, type=Answer).ok
