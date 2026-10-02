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
from cozy_runtime.internal.model_sources import CAPABILITY, BrokeredModelSources, available
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
