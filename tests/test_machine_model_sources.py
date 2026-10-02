"""Real owner store + typed socket/fd broker + native descriptor plane, CPU only."""
from __future__ import annotations

import hashlib
import io
import mmap
import os
import socket
import threading
from pathlib import Path

import pytest
import tensorfs
from tensorfs import plane as native_plane

from cozy_runtime.author._executor_requests import (
    ModelSourceRead,
    SourceBlob,
    decode,
    encode,
)
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Load
from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.model_sources import (
    CAPABILITY,
    BrokeredModelSources,
    VerificationLimits,
    available,
)
from cozy_runtime.internal.seam import Channel, SeamError


def owner_model(root: Path):
    store = tensorfs.Store.init(root / "owner")
    data = b"".join((n % 251).to_bytes(1) for n in range(65540))
    plain = next(value for name, value in tensorfs.seed_digests() if name == "plain/1")
    targets = {"linear": {"drop": [], "add": {"weight": {
        "logical_dtype": "u8", "shape": [len(data)], "encoding": plain,
        "parts": {"value": {"dtype": "u8", "shape": [len(data)]}},
    }}}}
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    writer = store.begin_derived(digest, 1, {}, targets, {}, [("linear", "weight")],
                                 2 * len(data), work_fingerprint=digest)
    writer.add_part("linear", "weight", "value", io.BytesIO(data))
    manifest = writer.commit()["manifest"]["sha256"]
    header = store.manifest(manifest)["header"]
    lease = store.acquire_cozytensors(manifest)
    files = {}
    for object_id, length in lease.objects:
        buffer = bytearray(length)
        lease.read_into(object_id, length, 0, length, buffer)
        path = root / object_id.removeprefix("sha256:")
        path.write_bytes(buffer)
        path.chmod(0o444)
        files[object_id] = path
    header_file = root / "header"
    header_file.write_bytes(header)
    header_file.chmod(0o444)
    return store, manifest, lease, files, header_file, data


def test_existing_checkpoint_metadata_and_host_plane_use_machine_descriptors(tmp_path: Path):
    if not available():
        pytest.skip("native descriptor-source dependency candidate not installed")
    _store, manifest, lease, files, header_file, data = owner_model(tmp_path)
    local, remote = socket.socketpair()
    owner = Channel(remote)
    executor = Executor(Channel(local), tmp_path / "executor")
    requests = []
    errors = []

    def serve():
        try:
            while (frame := owner.recv()) is not None:
                request = decode(frame)
                assert isinstance(request, ModelSourceRead) and request.manifest == manifest
                requests.append(request)
                path = header_file if request.role == "header" else files[request.name]
                content = path.read_bytes()
                answer = SourceBlob(ok=True, sha256=hashlib.sha256(content).hexdigest(),
                                    length=len(content))
                owner.send({"event": "answer", "seq": frame["seq"],
                            **encode(answer), "descriptor": True, "future_advisory": {"v": 99}})
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    owner.send_readonly_file(fd)
                finally:
                    os.close(fd)
        except Exception as exc:
            errors.append(exc)

    serving = threading.Thread(target=serve)
    serving.start()
    provider = BrokeredModelSources(executor._durable)
    catalog = tmp_path / "owner/tensorfs.sqlite"
    before = catalog.read_bytes()
    try:
        checkpoint = Checkpoint(tmp_path / "must-not-open", manifest, source_provider=provider)
        assert checkpoint.store is None and not (tmp_path / "must-not-open").exists()
        assert checkpoint.has("linear")
        rows = checkpoint.rows("linear")
        assert rows[0].shape == (len(data),) and rows[0].nbytes == len(data)
        assert checkpoint.model_assets() == {}
        plan = checkpoint.read_plan([("linear", "weight")], 4096, ["linear"])
        p = native_plane.Plane(devices=[], readers=2, direct_io=False)
        source = provider.source(p, manifest, plan)
        ws = p.register("linear", source, plan, [["linear/weight"]])
        p.set_pinned_budget(ws.nbytes)
        p.want(ws, "pinned").wait()
        with mmap.mmap(ws.host_fd, ws.nbytes, access=mmap.ACCESS_READ) as mapping:
            part = ws.parts[0]
            assert mapping[part["offset"]:part["offset"] + part["nbytes"]] == data
        second = provider.source(p, manifest, plan)
        assert sum(request.role == "header" for request in requests) == 1
        assert sum(request.role == "object" for request in requests) == 1
        assert len(provider.descriptors) == 1  # only the selected data object, not header/siblings
        assert catalog.read_bytes() == before
        assert "libcuda" not in Path("/proc/self/maps").read_text()
        assert CAPABILITY in executor.hello()["memory"]
        ws.close()
        del source, second
        provider.close()
        p.close()
        assert not provider.descriptors
    finally:
        local.close()
        serving.join()
        remote.close()
        lease.release()
    assert not errors


def test_new_load_flag_is_optional_and_additive():
    assert Load(construction="legacy").descriptor_sources is False
    load = Load(construction="owned", descriptor_sources=True)
    assert load.descriptor_sources is True


def test_unqualified_group_source_mode_refuses_before_device_preparation(tmp_path: Path):
    local, remote = socket.socketpair()
    try:
        executor = Executor(Channel(local), tmp_path, world=2)
        reply = executor.load(Load(construction="group", descriptor_sources=True))
        assert not reply["ok"] and reply["code"] == "model_source_unavailable"
        assert executor.torch is None and not executor.started
        assert "libcuda" not in Path("/proc/self/maps").read_text()
    finally:
        local.close()
        remote.close()


def test_readonly_file_transport_rejects_writable_and_keeps_caller_ownership(tmp_path: Path):
    local, remote = socket.socketpair()
    try:
        with (tmp_path / "file").open("w+b") as file:
            file.write(b"source")
            file.flush()
            with pytest.raises(SeamError, match="readonly regular"):
                Channel(local).send_readonly_file(file.fileno())
            assert os.fstat(file.fileno()).st_size == 6
    finally:
        local.close()
        remote.close()


def test_malformed_source_answer_closes_received_capability(tmp_path: Path):
    local, remote = socket.socketpair()
    path = tmp_path / "readonly"
    path.write_bytes(b"source")
    path.chmod(0o444)
    owner = Channel(remote)
    executor = Executor(Channel(local), tmp_path)
    before = len(list(Path("/proc/self/fd").iterdir()))

    def answer():
        frame = owner.recv()
        owner.send({"event": "answer", "seq": frame["seq"], "ok": True,
                    "sha256": "invalid", "length": 6, "descriptor": True})
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            owner.send_readonly_file(fd)
        finally:
            os.close(fd)

    serving = threading.Thread(target=answer)
    serving.start()
    try:
        with pytest.raises(SeamError, match="malformed"):
            executor._durable(ModelSourceRead(manifest="m", role="header"), SourceBlob)
        serving.join()
        assert len(list(Path("/proc/self/fd").iterdir())) == before
    finally:
        local.close()
        remote.close()


def test_single_worker_oversized_object_keeps_full_checks_and_coalesces(tmp_path: Path):
    _store, manifest, lease, files, header_file, data = owner_model(tmp_path)
    requests = []
    caller = threading.get_ident()

    def exchange(request, into):
        assert threading.get_ident() == caller  # all control exchanges stay serialized here
        assert into is SourceBlob
        requests.append(request)
        path = header_file if request.role == "header" else files[request.name]
        body = path.read_bytes()
        return SourceBlob(ok=True, sha256=hashlib.sha256(body).hexdigest(), length=len(body),
                          descriptor=os.open(path, os.O_RDONLY | os.O_CLOEXEC))

    before = len(list(Path("/proc/self/fd").iterdir()))
    provider = BrokeredModelSources(exchange, limits=VerificationLimits(1, 1, 1024))
    checkpoint = Checkpoint("", manifest, source_provider=provider)
    plan = checkpoint.read_plan([("linear", "weight")], 4096, ["linear"])
    p = native_plane.Plane(devices=[], readers=1, direct_io=False)
    source = provider.source(p, manifest, plan)
    second = provider.source(p, manifest, plan)
    assert sum(request.role == "object" for request in requests) == 1
    ws = p.register("linear", source, plan, [["linear/weight"]])
    p.set_pinned_budget(ws.nbytes)
    p.want(ws, "pinned").wait()
    with mmap.mmap(ws.host_fd, ws.nbytes, access=mmap.ACCESS_READ) as mapping:
        part = ws.parts[0]
        assert mapping[part["offset"]:part["offset"] + part["nbytes"]] == data
    ws.close()
    del source, second
    provider.close()
    p.close()
    assert len(list(Path("/proc/self/fd").iterdir())) == before
    lease.release()


def test_parallel_verification_rejects_corrupt_grant_and_closes_all_received_fds(tmp_path: Path):
    _store, manifest, lease, files, header_file, _data = owner_model(tmp_path)
    original = next(iter(files))
    wrong = tmp_path / "corrupt-immutable-export"
    wrong.write_bytes(b"x" * files[original].stat().st_size)
    wrong.chmod(0o444)
    received = []

    def exchange(request, into):
        assert into is SourceBlob
        path = header_file if request.role == "header" else wrong
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        received.append(fd)
        sha256 = (hashlib.sha256(header_file.read_bytes()).hexdigest()
                  if request.role == "header" else request.name.removeprefix("sha256:"))
        return SourceBlob(ok=True, sha256=sha256, length=path.stat().st_size, descriptor=fd)

    before = len(list(Path("/proc/self/fd").iterdir()))
    provider = BrokeredModelSources(exchange, limits=VerificationLimits(2, 4, 128 << 20))
    checkpoint = Checkpoint("", manifest, source_provider=provider)
    plan = checkpoint.read_plan([("linear", "weight")], 4096, ["linear"])
    p = native_plane.Plane(devices=[], readers=1, direct_io=False)
    with pytest.raises(native_plane.Invalid, match="identity"):
        provider.source(p, manifest, plan)
    assert not provider.descriptors
    assert len(list(Path("/proc/self/fd").iterdir())) == before
    for fd in received:
        with pytest.raises(OSError):
            os.fstat(fd)
    provider.close()
    p.close()
    lease.release()


def test_parallel_failure_drains_pending_imports_without_partial_cache(tmp_path: Path):
    store = tensorfs.Store.init(tmp_path / "owner")
    plain = next(value for name, value in tensorfs.seed_digests() if name == "plain/1")
    count, length = 6, 8 << 20
    payloads = {f"w{n}": n.to_bytes(8, "little") + b"w" * (length - 8) for n in range(count)}
    targets = {"many": {"drop": [], "add": {
        key: {"logical_dtype": "u8", "shape": [length], "encoding": plain,
              "parts": {"value": {"dtype": "u8", "shape": [length]}}}
        for key in payloads
    }}}
    digest = "sha256:" + hashlib.sha256(b"bounded pending native failure fixture").hexdigest()
    writer = store.begin_derived(digest, 1, {}, targets, {}, [("many", key) for key in payloads],
                                 count * length * 2, work_fingerprint=digest)
    for key, body in payloads.items():
        writer.add_part("many", key, "value", io.BytesIO(body))
    manifest = writer.commit()["manifest"]["sha256"]
    header = store.manifest(manifest)["header"]
    lease = store.acquire_cozytensors(manifest)
    exported = {}
    for object_id, object_length in lease.objects:
        body = bytearray(object_length)
        lease.read_into(object_id, object_length, 0, object_length, body)
        path = tmp_path / object_id.removeprefix("sha256:")
        path.write_bytes(body)
        path.chmod(0o444)
        exported[object_id] = path
    corrupted = tmp_path / "bad-export"
    checkpoint_header = tmp_path / "header"
    checkpoint_header.write_bytes(header)
    checkpoint_header.chmod(0o444)
    offered = []

    def exchange(request, into):
        assert into is SourceBlob
        if request.role == "header":
            path, sha = checkpoint_header, hashlib.sha256(header).hexdigest()
        else:
            if not offered:
                corrupted.write_bytes(b"z" * request.length)
                corrupted.chmod(0o444)
            path = corrupted if not offered else exported[request.name]
            sha = request.name.removeprefix("sha256:")
            offered.append(request)
        return SourceBlob(ok=True, sha256=sha, length=path.stat().st_size,
                          descriptor=os.open(path, os.O_RDONLY | os.O_CLOEXEC))

    baseline = len(list(Path("/proc/self/fd").iterdir()))
    provider = BrokeredModelSources(exchange, limits=VerificationLimits(2, 4, 32 << 20))
    checkpoint = Checkpoint("", manifest, source_provider=provider)
    plan = checkpoint.read_plan([("many", key) for key in payloads], 4 << 20, ["many"])
    p = native_plane.Plane(devices=[], readers=1, direct_io=False)
    with pytest.raises(native_plane.Invalid, match="identity"):
        provider.source(p, manifest, plan)
    assert len(offered) >= 2  # more than the failing import was actually submitted
    assert not provider.descriptors
    assert len(list(Path("/proc/self/fd").iterdir())) == baseline
    # A retained exception traceback cannot keep completed-but-unpublished mappings alive.
    assert not any(str(tmp_path) in row and row.split()[1] == "r--p"
                   for row in Path("/proc/self/maps").read_text().splitlines())
    provider.close()
    p.close()
    lease.release()
