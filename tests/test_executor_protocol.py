"""Exercise real executor hello frames independently of package release metadata."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.seam import EXECUTOR_PROTOCOL_REVISION, Channel, listener
from cozy_runtime.internal.worker.child import ExecutorProtocolMismatch, require_executor_protocol


@pytest.mark.parametrize(
    "revision,version,accepted",
    [
        (1, "9.8.7", True),
        (2, "0.18.33", False),
        (None, "0.18.33", False),
        (None, "0.18.20", False),
    ],
)
def test_real_executor_hello_uses_protocol_contract(
    tmp_path: Path, revision: int | None, version: str, accepted: bool
) -> None:
    socket_path = str(tmp_path / "seam")
    server = listener(socket_path)
    server.settimeout(15)  # fixture deadlock bound; no product lifetime policy
    program = f"""
import sys
from cozy_runtime.internal import executor
executor.RUNTIME_VERSION = {version!r}
executor.EXECUTOR_PROTOCOL_REVISION = {revision!r}
if executor.EXECUTOR_PROTOCOL_REVISION is None:
    send = executor.Channel.send
    def legacy(self, message):
        message.pop('executor_protocol_revision', None)
        send(self, message)
    executor.Channel.send = legacy
raise SystemExit(executor.main(sys.argv[1:]))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, "--socket", socket_path, "--root", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    channel = None
    try:
        connection, _ = server.accept()
        channel = Channel(connection)
        channel.send({"cmd": "hello"})
        hello = channel.recv(timeout=15, total_timeout=15)
        assert hello is not None and hello["ok"]
        assert hello["runtime_version"] == version
        if accepted:
            require_executor_protocol(hello)
            channel.send({"cmd": "shutdown"})
            shutdown = channel.recv(timeout=15)
            assert shutdown is not None and shutdown["ok"]
        else:
            with pytest.raises(ExecutorProtocolMismatch):
                require_executor_protocol(hello)
            # Unsupported IPC receives no work; EOF retires only this executor.
            channel.close()
            channel = None
        assert process.wait(timeout=15) == 0
    finally:
        if channel is not None:
            channel.close()
        server.close()
        if process.poll() is None:
            process.terminate()
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr.decode()


def test_protocol_rejects_malformed_or_unreviewed_hello() -> None:
    for revision in (True, "1", 0, EXECUTOR_PROTOCOL_REVISION + 1):
        with pytest.raises(ExecutorProtocolMismatch):
            require_executor_protocol(
                {"executor_protocol_revision": revision, "runtime_version": "0.18.33"}
            )
    for version in ("", "unknown", "0.18.29", "0.18.34"):
        with pytest.raises(ExecutorProtocolMismatch):
            require_executor_protocol({"runtime_version": version})
    # Revision 1 without its stated native interfaces predates 0.18.51.
    with pytest.raises(ExecutorProtocolMismatch, match=r"cozy-runtime>=0\.18\.51"):
        require_executor_protocol(
            {"executor_protocol_revision": EXECUTOR_PROTOCOL_REVISION, "runtime_version": "0.18.49"}
        )


def test_an_executor_maps_installed_distributions_once() -> None:
    """Diffusers and Transformers both ask at import; the executor walks the RECORDs once and
    each caller still gets the same answer as a fresh walk, in a copy of its own."""
    program = """
import importlib.metadata as metadata
from cozy_runtime.internal import executor
fresh = metadata.packages_distributions()
walks = []
walk = metadata.distributions
metadata.distributions = lambda **kw: walks.append(1) or walk(**kw)
executor._remember_distributions()
first = metadata.packages_distributions()
first[next(iter(first))].append("mutated")
second = metadata.packages_distributions()
assert first != second == fresh and len(walks) == 1, (len(walks), second == fresh)
"""
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
