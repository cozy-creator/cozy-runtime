"""Inline config reads preserve exact source authorization and native custody."""

import json
import os
import signal
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from cozy_runtime.author import CapabilityError, Loader, Model, WeightsSink
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._services import Attempt
from cozy_runtime.internal.weights_sink import WeightsTransactionHost, source_config
from cozy_runtime.internal.weights_writer import StoreProxy, WriterBroker
from cozy_runtime.internal.worker.grants import MODEL_PREFIX
from test_derived_config_runtime import _released_source
from test_model_runtime_closure import _CONFIG


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("config inspection does not construct a model")


def test_config_reads_only_named_configs_of_granted_native_sources(tmp_path: Path) -> None:
    store, manifest, length = _released_source(tmp_path)
    host = WeightsTransactionHost(
        store=store,
        owner_scope="config-proof",
        request_id="config-proof",
        invocation_spec_digest="sha256:" + "51" * 32,
        work_fingerprint="sha256:" + "52" * 32,
        writer_session_id=1,
        allowed_sources={manifest: length},
        output_bounds={},
    )
    source = _derive_model(Source, manifest)
    sink = WeightsSink(
        Attempt("config-proof", tmp_path / "spool"),
        {"source": source},
        {},
        None,
        source_config=host.config,
    )
    assert sink.config(source, "unet") == _CONFIG
    with pytest.raises(CapabilityError, match="not a granted model"):
        sink.config(_derive_model(Source, manifest), "unet")
    with pytest.raises(CapabilityError, match="no declared config"):
        sink.config(source, "ungranted")
    with pytest.raises(CapabilityError, match="immutable grant"):
        host.config("sha256:" + "ff" * 32, "unet")
    host.allowed_sources[manifest] = length + 1
    with pytest.raises(CapabilityError, match="TensorFS source config refused"):
        sink.config(source, "unet")


def test_executor_config_uses_parent_store_and_exact_input_grant(tmp_path: Path) -> None:
    store, manifest, length = _released_source(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    attempt = SimpleNamespace(
        request_id="parent-config",
        attempt=1,
        state="running",
        canceling="",
        spool=spool,
        spec={
            "inputs": [{"input_id": MODEL_PREFIX + "source", "digest": manifest, "length": length}]
        },
    )
    broker = WriterBroker(lambda: store, store_root=tmp_path / "store")
    # The proxy has no native Store: this read must cross the parent control lane.
    proxy = StoreProxy(None, spool, lambda kind, frame: broker.handle(attempt, frame))
    source = _derive_model(Source, manifest)
    sink = WeightsSink(
        Attempt("parent-config", spool), {"source": source}, {}, None, source_config=proxy.config
    )
    assert sink.config(source, "unet") == _CONFIG
    assert list(spool.iterdir()) == []
    with pytest.raises(CapabilityError) as failure:
        proxy.config("sha256:" + "ff" * 32, "unet")
    assert failure.value.code == "weights_source_ungranted"
    with pytest.raises(CapabilityError) as failure:
        sink.config(source, "missing")
    assert failure.value.code == "weights_config_ungranted"
    attempt.spec["inputs"][0]["length"] += 1
    with pytest.raises(CapabilityError) as failure:
        sink.config(source, "unet")
    assert failure.value.code == "weights_source_refused"
    attempt.spec["inputs"][0]["length"] = length
    attempt.canceling = "canceled"
    with pytest.raises(CapabilityError) as failure:
        sink.config(source, "unet")
    assert failure.value.code == "weights_writer_closed"


@pytest.mark.skipif(os.geteuid() != 0, reason="separate executor UID requires root/container")
def test_parent_config_read_preserves_repository_permissions(tmp_path: Path) -> None:
    store, manifest, length = _released_source(tmp_path)
    # pytest's root temp parents normally have0700; expose traversal only for this
    # isolated container proof, while keeping the canonical repository owner-only.
    for path in (tmp_path, *tmp_path.parents):
        if path == Path("/tmp"):
            break
        path.chmod(0o755)
    repository = tmp_path / "store" / "repos" / "test" / "model.json"
    os.chown(repository, 65532, 65532)
    repository.chmod(0o600)
    spool = tmp_path / "spool"
    spool.mkdir()
    os.chown(spool, 65533, 65533)
    attempt = SimpleNamespace(
        request_id="uid-config",
        attempt=1,
        state="running",
        canceling="",
        spool=spool,
        spec={
            "inputs": [{"input_id": MODEL_PREFIX + "source", "digest": manifest, "length": length}]
        },
    )
    broker = WriterBroker(lambda: store, store_root=tmp_path / "store")
    parent, child = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        parent.close()
        os.setgroups([])
        os.setgid(65533)
        os.setuid(65533)
        channel = child.makefile("rwb", buffering=0)
        try:
            try:
                repository.read_bytes()
            except PermissionError:
                pass
            else:
                raise AssertionError("executor can read private repository")

            with pytest.raises(CapabilityError, match="Permission denied"):
                source_config(store, manifest, length, "unet")

            def exchange(kind: str, frame: dict[str, object]) -> dict[str, object]:
                assert kind == "weights_writer"
                channel.write(json.dumps(frame).encode() + b"\n")
                response = json.loads(channel.readline())
                assert isinstance(response, dict)
                return response

            proxy = StoreProxy(None, spool, exchange)
            assert proxy.config(manifest, "unet") == _CONFIG
            try:
                proxy.config("sha256:" + "ff" * 32, "unet")
            except CapabilityError as exc:
                assert exc.code == "weights_source_ungranted"
            else:
                raise AssertionError("ungranted source was served")
            channel.write(b'{"finished":true}\n')
            os._exit(0)
        except BaseException as exc:
            try:
                channel.write(json.dumps({"error": repr(exc)}).encode() + b"\n")
            finally:
                os._exit(1)
    child.close()
    parent.settimeout(10)
    channel = parent.makefile("rwb", buffering=0)
    try:
        while raw := channel.readline():
            frame = json.loads(raw)
            if "finished" in frame or "error" in frame:
                assert frame == {"finished": True}
                break
            channel.write(json.dumps(broker.handle(attempt, frame)).encode() + b"\n")
    except BaseException:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise
    finally:
        channel.close()
        parent.close()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert repository.stat().st_uid == 65532
    assert repository.stat().st_mode & 0o777 == 0o600
    assert list(spool.iterdir()) == []
