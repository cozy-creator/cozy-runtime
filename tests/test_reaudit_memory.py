"""CPU-only re-audit regressions for host credit and reusable spread refusals."""

from pathlib import Path

import pytest

from cozy_runtime.internal import memory, proctree


@pytest.mark.parametrize(
    ("current", "inactive", "expected"),
    [(900, 600, 700), (1100, 600, 500), (900, 2000, 1000), (900, -20, 100)],
)
def test_cgroup_file_cache_is_reclaimable_but_never_exceeds_usage(
    monkeypatch: pytest.MonkeyPatch, current: int, inactive: int, expected: int
) -> None:
    files = {
        "/proc/meminfo": "MemAvailable: 10 kB\n",
        "/proc/self/cgroup": "0::/tenant\n",
        "/sys/fs/cgroup/tenant/memory.max": "1000",
        "/sys/fs/cgroup/tenant/memory.current": str(current),
        "/sys/fs/cgroup/tenant/memory.stat": f"inactive_file {inactive}\n",
        "/sys/fs/cgroup/memory.max": "max",
    }
    monkeypatch.setattr(
        Path, "is_file", lambda path: str(path) == "/sys/fs/cgroup/cgroup.controllers"
    )
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    assert proctree.available_host_bytes() == expected


def test_nested_v1_cgroup_limits_and_file_cache_are_both_accounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "/proc/meminfo": "MemAvailable: 10 kB\n",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "2000",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1000",
        "/sys/fs/cgroup/memory/memory.stat": "total_inactive_file 0\n",
        "/sys/fs/cgroup/memory/parent/memory.limit_in_bytes": "900",
        "/sys/fs/cgroup/memory/parent/memory.usage_in_bytes": "700",
        "/sys/fs/cgroup/memory/parent/memory.stat": "total_inactive_file 300\n",
        "/sys/fs/cgroup/memory/parent/child/memory.limit_in_bytes": "1000",
        "/sys/fs/cgroup/memory/parent/child/memory.usage_in_bytes": "800",
        "/sys/fs/cgroup/memory/parent/child/memory.stat": "total_inactive_file 100\n",
    }
    mount = proctree._CgroupMount(
        Path("/"), Path("/sys/fs/cgroup/memory"), frozenset({"memory"})
    )
    monkeypatch.setattr(Path, "is_file", lambda _: False)
    monkeypatch.setattr(Path, "read_text", lambda path: files[str(path)])
    monkeypatch.setattr(
        proctree, "_cgroup_v1_controller_path", lambda _: Path("/sys/fs/cgroup/memory/parent/child")
    )
    monkeypatch.setattr(proctree, "_cgroup_v1_mounts", lambda: (mount,))
    assert proctree.available_host_bytes() == 300


@pytest.mark.parametrize("code", ["device_shortfall", "host_shortfall", "device_out_of_memory"])
def test_only_reusable_memory_refusals_can_decline_a_spread(code: str) -> None:
    refusal = memory.MemoryRefusal(code, "capacity", 20, 10, reusable=True)
    assert memory.Planner.spread_decline(refusal) == f"{code}: capacity"
    refusal.reusable = False
    assert memory.Planner.spread_decline(refusal) is None
    assert memory.Planner.spread_decline(RuntimeError("device lost")) is None


def test_unknown_legacy_usage_is_charged_once_under_a_budget() -> None:
    from types import SimpleNamespace

    machine = memory.Machine({0: 800}, lambda _ordinal: (500, 1000), lambda _note: None)
    legacy = SimpleNamespace(
        slot="legacy",
        ordinals=(0,),
        held={},
        unattributed={0: 500},
        overhead={},
    )
    current = SimpleNamespace(
        slot="current",
        ordinals=(0,),
        held={},
        unattributed={},
        overhead={},
    )
    assert machine.quote(current, [legacy, current]) == {0: 500}


def test_modeled_invoke_refills_disk_weights_and_records_the_invoke_leg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU model stands in for a CUDA executor's refill, placement and invoke legs."""
    torch = pytest.importorskip("torch")
    from cozy_runtime.author import pure

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, device="meta"))

        @pure
        def forward(self, value: object) -> object:
            return value + self.weight.sum()

    root = Model()
    owner = memory.Planner(torch, torch.device("cpu"))
    loaded: list[str] = []

    class Store:
        def load(self, name: str) -> None:
            loaded.append(name)
            root.weight = torch.nn.Parameter(torch.ones(2))

        def load_unit(self, name: str, unit: object) -> None:
            raise AssertionError("this component is restored as a whole")

        def drop(self, name: str) -> None:
            raise AssertionError("the modeled invoke must not evict its refilled weights")

        def drop_unit(self, unit: object) -> None:
            raise AssertionError("the modeled invoke must not evict its refilled weights")

        def unit_loads(self, name: str, keys: frozenset[str]) -> bool:
            return False

        def transient(self, name: str) -> int:
            return 0

    held = owner.adopt("c", "model", root, Store(), "disk")
    report = owner.restore_resident("c", ())
    assert report == {"restored": {"model": 8}, "held": {}, "restored_bytes": 8}
    assert loaded == ["model"] and held.part.on("device") == 8 and owner.refilled == 8

    # Model the device accounting while retaining real CPU execution and the real operation
    # counters. No CUDA context, allocator, or machine is needed for this policy proof.
    owner.cuda = True
    monkeypatch.setattr(owner, "allocatable", lambda: 1 << 30)
    monkeypatch.setattr(owner, "_allocated", lambda: 0)
    monkeypatch.setattr(owner, "_reserved", lambda: 0)
    monkeypatch.setattr(owner, "_apply", lambda *_: None)
    monkeypatch.setattr(owner, "_sample", lambda: None)
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(owner, "_arm", lambda *_: memory.Snapshot([], [], (), (), lambda: (), None))
    with torch.no_grad():
        value = owner.run(held, "forward", root.forward, (torch.tensor(3.0),), {})
    assert value == 5
    counts = owner.operations["c/model.forward"]
    assert counts["placement_calls"] == counts["invoke_calls"] == 1
    assert owner.refilled == 8 and held.part.on("device") == 8


def test_revision_one_capabilities_are_optional_and_independent() -> None:
    from cozy_runtime.internal.seam import EXECUTOR_PROTOCOL_REVISION
    from cozy_runtime.internal.worker.child import require_executor_protocol

    assert EXECUTOR_PROTOCOL_REVISION == 1
    require_executor_protocol({"executor_protocol_revision": 1, "native_interfaces": {}})
    require_executor_protocol(
        {"executor_protocol_revision": 1, "native_interfaces": {}, "memory": ["vacate_ranks"]}
    )
    assert memory.Machine.executor_placement(()) == "all_resident"
    assert memory.Machine.executor_placement(("vacate_ranks",)) == "all_resident"
    assert memory.Machine.executor_placement(("plan",)) == "component_staged"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("version", ["0.18.86", "0.18.87"])
def test_real_revision_one_worker_executor_start_and_restore(
    reverse: bool, version: str, tmp_path: Path
) -> None:
    import os
    import subprocess
    import sys

    from test_cross_version_seam import CACHE, PYTHON, _lock, _uv
    from test_end_to_end import NO_EXECUTOR

    if NO_EXECUTOR:
        pytest.skip(NO_EXECUTOR)
    environment = CACHE / "reaudit-peers" / version
    python = environment / "bin/python"
    if not python.exists():
        environment.parent.mkdir(parents=True, exist_ok=True)
        _uv("venv", "--python", str(PYTHON), str(environment), cwd=environment.parent)
        requirements = environment / "requirements.txt"
        requirements.write_bytes(_lock(version))
        _uv(
            "pip",
            "install",
            "--python",
            str(python),
            "--require-hashes",
            "--requirements",
            str(requirements),
            cwd=environment,
        )
    script = """
import json, os, shutil, subprocess, sys
from pathlib import Path
from cozy_runtime.internal import child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.executor_commands import (
    Activate, Binding, Invoke, Load, PrepareRequest, Probe, Restore, Start,
)
root = Path(sys.argv[1])
uv = shutil.which('uv')
assert uv is not None
def _uv(*args, cwd):
    done = subprocess.run(
        [uv, *args], cwd=cwd, env=os.environ.copy(), capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-4000:]
project = root / 'reaudit_app'
project.mkdir()
(project / 'pyproject.toml').write_text('''
[project]
name = "reaudit-app"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["reaudit_app.py"]
''')
(project / 'reaudit_app.py').write_text('''
import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    n: int
class Result(msgspec.Struct):
    value: int
@app.entrypoint
def generate(ctx: Context, payload: Request) -> Result:
    return Result(payload.n + 1)
''')
dist = root / 'dist'
dist.mkdir()
_uv('build', '--wheel', '--out-dir', str(dist), str(project), cwd=project)
wheel = next(dist.glob('reaudit_app-*.whl'))
for interpreter in {sys.executable, sys.argv[2]}:
    _uv('pip', 'install', '--python', interpreter, '--no-deps', str(wheel), cwd=root)
interface = root / 'interface.json'
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover_installed
interface.write_bytes(package_interface.canonical_bytes(package_interface.build(discover_installed('reaudit_app:app'))))
config = RuntimeConfig(
    cozy_home=root/'home',
    credentials=Credentials(),
    child_base_env=tuple(
        (k, v) for k, v in os.environ.items()
        if not child_env.erased(k) and k != 'PYTHONPATH'
    ),
)
worker = Worker(
    config,
    WorkerOptions(
        root=root/'worker', devices='', accelerator_backend='none', python=sys.argv[2]
    ),
    InMemoryControlHost(),
)
try:
    executor = worker.supervision.spawn(imposed=worker.imposed(worker.lanes.envelope))
    assert executor.hello['executor_protocol_revision'] == 1, executor.hello
    reply = executor.call(
        Start(
            devices=worker.lanes.envelope.devices,
            application='reaudit_app:app',
            package_interface=str(interface),
        ),
        timeout=60,
    )
    assert reply['ok'], reply
    binding = Binding(application='reaudit_app:app', package_interface=str(interface))
    loaded = executor.call(
        Load(construction='model', devices=worker.lanes.envelope.devices, binding=binding),
        timeout=60,
    )
    assert loaded['ok'], loaded
    assert executor.call(Activate(construction='model'), timeout=60)['ok']
    prepared = executor.call(
        PrepareRequest(
            request_id='modeled', entrypoint='generate', construction='model', payload={'n': 41}
        ),
        timeout=60,
    )
    assert prepared['ok'], prepared
    spool = root / 'spool'
    spool.mkdir()
    invoked = executor.call(
        Invoke(
            request_id='modeled',
            entrypoint='generate',
            construction='model',
            spool=str(spool),
            deadline_s=None,
        ),
        timeout=60,
    )
    assert invoked['ok'] and invoked['outcome']['terminal'] == 'succeeded', invoked
    assert executor.call(Probe(), timeout=60)['ok']
    assert executor.call(Restore(), timeout=60)['ok']
    print(json.dumps({'parent':sys.executable,'child':sys.argv[2],'start':True,'probe':True,'restore':True}))
finally:
    worker.supervision.close()
"""
    parent, peer = (str(python), sys.executable) if reverse else (sys.executable, str(python))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [parent, "-c", script, str(tmp_path), peer],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
