"""Exercise real executor hello frames independently of package release metadata."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.seam import EXECUTOR_PROTOCOL_REVISION, Channel, listener
from cozy_runtime.internal.worker.child import ExecutorProtocolMismatch, require_executor_protocol


@pytest.mark.parametrize(
    "revision,version,accepted",
    [
        (EXECUTOR_PROTOCOL_REVISION, "9.8.7", True),
        (1, "0.18.86", True),
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
    # Revision 1 with native interfaces but without additive memory fields remains compatible.
    require_executor_protocol(
        {
            "executor_protocol_revision": EXECUTOR_PROTOCOL_REVISION,
            "runtime_version": "0.18.49",
            "native_interfaces": {},
        }
    )
    # The native-interface field itself remains a required protocol guard.
    with pytest.raises(ExecutorProtocolMismatch):
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


@pytest.mark.parametrize("model_bearing", [False, True])
@pytest.mark.parametrize("preparation_coverage", [False, True])
def test_import_only_start_defers_warmup_and_device_initialization(
    tmp_path: Path, model_bearing: bool, preparation_coverage: bool
) -> None:
    python = os.environ.get("COZY_CPU_TEST_PYTHON") if model_bearing else sys.executable
    if python is None:
        pytest.skip("CPU Torch interpreter not selected")
    (tmp_path / "import_start_fixture.py").write_text(
        "from pathlib import Path\n"
        "import msgspec\n"
        "from cozy_runtime.author import App\n"
        "app = App()\n"
        "class Request(msgspec.Struct):\n    pass\n"
        "class Result(msgspec.Struct):\n    pass\n"
        "@app.entrypoint\n"
        "def compute(payload: Request) -> Result:\n    return Result()\n"
        "def warmup():\n    Path(__file__).with_suffix('.warmed').write_text('warm')\n"
    )
    program = """
import socket, sys
from pathlib import Path
import msgspec
from cozy_runtime.internal import executor_commands, package_interface
from cozy_runtime.internal import executor as runtime
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.executor_commands import Probe, Residency, Start, Vacate, Warm
from cozy_runtime.internal.seam import Channel, EXECUTOR_MEMORY_CAPABILITIES
from cozy_runtime.internal.worker.child import require_executor_protocol
root, model_bearing = Path(sys.argv[1]), sys.argv[2] == 'True'
preparation_coverage = sys.argv[3] == 'True'
if not preparation_coverage:
    runtime.EXECUTOR_MEMORY_CAPABILITIES = frozenset({'store_read_gate'})
sys.path.insert(0, str(root))
_capture_seal()
body = package_interface.build(discover_installed('import_start_fixture:app'))
if model_bearing:
    body['entrypoints'][0]['models'] = [
        {'path': 'compute.models.model', 'class': 'M', 'component_use': {}}
    ]
interface = root / 'interface.json'
interface.write_bytes(package_interface.canonical_bytes(body))
ours, theirs = socket.socketpair()
executor = Executor(Channel(ours), root)
try:
    hello = executor.hello()
    require_executor_protocol(hello)
    assert ('store_read_prepare' in hello['memory']) == preparation_coverage
    # The current Worker announces more operations than this older peer reports.
    # Start still succeeds: announced support is not a mandatory peer bitset.
    command = Start(application='import_start_fixture:app', package_interface=str(interface),
                    import_only=True, memory_capabilities=tuple(EXECUTOR_MEMORY_CAPABILITIES))
    assert executor_commands.decode(executor_commands.encode(command)) == command
    assert not executor_commands.decode({'cmd': 'start'}).import_only
    first = executor.handle(command)
    assert first['ok'] and first['imported'] and not first['started'] and not first['warmed']
    assert first['device_total_bytes'] == first['device_free_bytes'] == -1
    assert first['memory'] == {'allocated_bytes': 0, 'reserved_bytes': 0, 'overhead_bytes': 0}
    assert first['rank_memory'] == {'0': first['memory']}
    assert not executor._device_ready and executor.planner is None
    assert executor.handle(command)['reused']
    assert executor.handle(Warm())['skipped'] == 'no started device process'
    assert executor.handle(Probe())['ok']
    assert executor.handle(Residency())['free_bytes'] == -1
    assert executor.handle(Vacate())['vacated']['freed_bytes'] == 0
    assert not (root / 'import_start_fixture.warmed').exists()
    if model_bearing:
        import torch
        assert executor.torch is torch and not torch.cuda.is_initialized()
    else:
        started = executor.handle(msgspec.structs.replace(command, import_only=False))
        assert started['started'] and started['warmed']
        assert (root / 'import_start_fixture.warmed').read_text() == 'warm'
finally:
    executor.channel.close()
    theirs.close()
"""
    done = subprocess.run(
        [python, "-c", program, str(tmp_path), str(model_bearing), str(preparation_coverage)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,  # fixture deadlock bound, not product policy
    )
    assert done.returncode == 0, done.stdout + done.stderr
