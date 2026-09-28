"""Inline config reads preserve exact source authorization and native custody."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tensorfs.derived import SourceCapability

from cozy_runtime.author import CapabilityError
from cozy_runtime.author._executor_requests import Opened, WriterSource
from cozy_runtime.internal.weights_writer import ExecutionStorage, WriterBroker
from cozy_runtime.internal.worker.grants import MODEL_PREFIX
from durable_seam import seam
from test_derived_config_runtime import _released_source
from test_model_runtime_closure import _CONFIG
from weights_channel import broker_lane


def test_native_source_fd_uses_exact_grant_and_is_fenced_with_attempt(tmp_path: Path) -> None:
    store, manifest, length = _released_source(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    attempt = SimpleNamespace(
        request_id="native-source",
        attempt=1,
        digest=b"\x00" * 32,
        state="running",
        canceling="",
        spool=spool,
        spec={
            "inputs": [{"input_id": MODEL_PREFIX + "source", "digest": manifest, "length": length}]
        },
    )
    broker = WriterBroker(lambda: store, store_root=Path(store.root))
    client = ExecutionStorage(spool, seam(broker_lane(broker, attempt)), {})
    try:
        with client.source(manifest) as source:
            inspection = source.inspect()
            assert inspection.source.manifest == manifest
            assert inspection.source.length == length
            assert inspection.configs["unet"] == _CONFIG
            assert inspection.components
        with client.source(manifest) as selected:
            assert selected.inspect(configs=["unet"]).configs == {"unet": _CONFIG}
        with client.source(manifest) as missing, pytest.raises(Exception) as refused_config:
            missing.inspect(configs=["missing"])
        assert getattr(refused_config.value, "code", "") == "MISSING_FIELD"
        assert list(spool.iterdir()) == []
        with pytest.raises(CapabilityError) as refused:
            client.source("sha256:" + "ff" * 32)
        assert refused.value.code == "weights_source_ungranted"

        answer = client._call(WriterSource(manifest=manifest), Opened)
        with SourceCapability.from_fd(manifest, length + 1, answer.descriptor) as forged:
            with pytest.raises(Exception) as refused_native:
                forged.inspect()
            assert getattr(refused_native.value, "code", "") == "TRANSACTION_CONFLICT"

        # Native source validation independently verifies the worker's exact admitted length.
        attempt.spec["inputs"][0]["length"] += 1
        with client.source(manifest) as stale, pytest.raises(Exception) as absent:
            stale.inspect()
        assert getattr(absent.value, "code", "")
        attempt.spec["inputs"][0]["length"] = length
        outstanding = client.source(manifest)
        outstanding.inspect()
        broker.close_attempt(attempt)
        with pytest.raises((EOFError, OSError)):
            outstanding.inspect()
        assert broker.sources == {}
        attempt.canceling = "canceled"
        with pytest.raises(CapabilityError) as refused:
            client.source(manifest)
        assert refused.value.code == "weights_writer_closed"
    finally:
        broker.close()


@pytest.mark.skipif(os.geteuid() != 0, reason="separate executor UID requires root/container")
def test_native_source_fd_preserves_repository_permissions(tmp_path: Path) -> None:
    store, manifest, length = _released_source(tmp_path)
    for path in (tmp_path, *tmp_path.parents):
        if path == Path("/tmp"):
            break
        path.chmod(0o755)
    repository = tmp_path / "store" / "repos" / "test" / "model.json"
    os.chown(repository, 65532, 65532)
    repository.chmod(0o600)
    spool = tmp_path / "spool"
    spool.mkdir()
    attempt = SimpleNamespace(
        request_id="uid-source",
        attempt=1,
        digest=b"\x00" * 32,
        state="running",
        canceling="",
        spool=spool,
        spec={
            "inputs": [{"input_id": MODEL_PREFIX + "source", "digest": manifest, "length": length}]
        },
    )
    broker = WriterBroker(lambda: store, store_root=Path(store.root))
    client = ExecutionStorage(spool, seam(broker_lane(broker, attempt)), {})
    fd = client._call(WriterSource(manifest=manifest), Opened).descriptor
    code = """
import os, sys
from pathlib import Path
from tensorfs.derived import SourceCapability
from tensorfs._source_channel import SourceClient
os.setgroups([]); os.setgid(65533); os.setuid(65533)
try:
    Path(sys.argv[1]).read_bytes()
except PermissionError:
    pass
else:
    raise AssertionError('executor read a private repository')
with SourceCapability.from_fd(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])) as source:
    assert source.inspect(configs=['unet']).configs['unet'].hex() == sys.argv[5]
"""
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(repository),
                manifest,
                str(length),
                str(fd),
                _CONFIG.hex(),
            ],
            pass_fds=(fd,),
            check=True,
            timeout=10,
        )
    finally:
        os.close(fd)
        broker.close()
    assert repository.stat().st_uid == 65532
    assert repository.stat().st_mode & 0o777 == 0o600
    assert list(spool.iterdir()) == []
