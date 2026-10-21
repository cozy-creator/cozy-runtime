"""The OS seams under the executor's GPU path, on real calls: the CPUs a process may use, every
byte of a directory output and of a kernel store entry, a link out of a directory output, each
NVML query where the host has no NVIDIA driver, a kernel build where the OS cannot end its tree,
and an async call under the network fence, which still refuses loopback. Windows runs this file
on windows-2025 (`platforms.yaml`)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import msgspec
import pytest

from cozy_runtime._cpus import cpus
from cozy_runtime.author import App, Invocation, InvocationResult, Outcome, Outputs, Tree, attempt
from cozy_runtime.internal import accel, kernel_cache, kernel_compile

if sys.platform == "win32":
    import _winapi

#: What a text-mode descriptor rewrites: CRLF, LF and Ctrl-Z.
BYTES = b"a\r\nb\nc\x1ad\x00"

#: The executor's fence, armed as at its boot; then an async entrypoint (its loop's self-pipe is
#: a loopback pair on Windows) and non-blocking dials, each refused or admitted: the same pair
#: spelled out and its listener dialled over UDP, another process's 127.0.0.1 listener (port in
#: argv) plainly and after a UDP bind to its port, and a remote address after a free bind to it.
FENCED = """
import asyncio, json, socket, sys, time
from pathlib import Path
import msgspec
from cozy_runtime.author import App, Invocation, attempt
from cozy_runtime.internal.sandbox import FenceViolation, refuse_network

class Request(msgspec.Struct):
    pass

class Reply(msgspec.Struct):
    total: int

app = App()

@app.entrypoint(accelerator=False)
async def main(payload: Request) -> Reply:
    return Reply(await asyncio.get_running_loop().run_in_executor(None, sum, [1, 2]))

refuse_network("the seams test", allow_unix_seam=True)
deadline = time.monotonic() + 30
result, outcome, _ = attempt(app.get("main"), {}, Invocation("f", Path(sys.argv[1]), deadline))
print(outcome.terminal, outcome.message, result and result.result)

def dial(address, decoy=False, free=False, kind=socket.SOCK_STREAM):
    if decoy:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if free:
            udp.setsockopt(socket.IPPROTO_IP, 15, 1)  # Linux's IP_FREEBIND
        udp.bind(address)
    client = socket.socket(socket.AF_INET, kind)
    client.setblocking(False)
    try:
        client.connect(address)
    except FenceViolation as exc:
        return exc.fence
    except OSError:
        pass
    return "admitted"

service = ("127.0.0.1", int(sys.argv[2]))
with socket.create_server(("127.0.0.1", 0)) as listener:
    dials = {"own pair": dial(listener.getsockname())}
    dials["udp to own pair"] = dial(listener.getsockname(), kind=socket.SOCK_DGRAM)
dials["service"] = dial(service)
dials["udp decoy"] = dial(service, decoy=True)
if sys.platform == "linux":
    dials["free bind"] = dial(("192.0.2.1", 44444), decoy=True, free=True)
print(json.dumps(dials))
"""


class Request(msgspec.Struct):
    pass


class Result(msgspec.Struct):
    files: Tree


def test_a_process_may_use_at_least_one_cpu_and_never_more_than_the_host_has() -> None:
    assert 1 <= cpus() <= (os.cpu_count() or 1)


def _save_tree(source: Path, spool: Path) -> tuple[InvocationResult | None, Outcome]:
    app = App()

    @app.entrypoint(accelerator=False)
    def main(payload: Request, out: Outputs) -> Result:
        return Result(out.save_tree(source))

    deadline = time.monotonic() + 30
    result, outcome, _ = attempt(app.get("main"), {}, Invocation("seams", spool, deadline))
    return result, outcome


def test_a_directory_output_keeps_every_byte(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "raw.bin").write_bytes(BYTES)
    result, outcome = _save_tree(source, tmp_path)
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and isinstance(result.result, Result)
    assert (result.result.files.path / "nested" / "raw.bin").read_bytes() == BYTES


def test_a_directory_output_refuses_a_link_out_of_its_tree(tmp_path: Path) -> None:
    source, outside = tmp_path / "source", tmp_path / "outside"
    source.mkdir()
    outside.mkdir()
    (outside / "secret.bin").write_bytes(BYTES)
    if sys.platform == "win32":  # a junction: no privilege needed, and not a symlink
        _winapi.CreateJunction(str(outside), str(source / "escape"))
    else:
        (source / "escape").symlink_to(outside)
    result, outcome = _save_tree(source, tmp_path)
    assert result is None and outcome.code == "output_tree", outcome


def test_a_kernel_store_entry_is_published_whole_and_read_back_exactly(tmp_path: Path) -> None:
    store = kernel_cache.Store(tmp_path / "kernels")
    key = kernel_cache.Key.of("seams", {"bytes": len(BYTES)})
    store.put(key, BYTES, {})
    assert store.get(key) == BYTES
    with store.building(key):
        assert store.claim(key) is None


def test_a_host_without_the_nvidia_driver_answers_each_gpu_query_unknown() -> None:
    if accel._nvml_library() is not None:
        pytest.skip("this host has the NVIDIA driver")
    assert accel.driver_version() == ""
    assert accel.device_capability("0") == 0
    assert accel.nvml_index("GPU-00000000-0000-0000-0000-000000000000") == -1
    assert accel.compute_processes("0") is None
    assert accel.device_memory("0", "cuda").state == "unreadable"


@pytest.mark.skipif(hasattr(os, "killpg"), reason="this OS ends a build's tree by its group")
def test_a_kernel_build_is_absent_where_the_os_cannot_end_its_tree(tmp_path: Path) -> None:
    store = kernel_cache.Store(tmp_path / "kernels")
    job = kernel_compile.Job("python", kernel_cache.Key.of("seams", {}))
    assert kernel_compile.wait(store, job).state == "absent"


def test_an_async_call_runs_under_the_network_fence_which_still_refuses_loopback(
    tmp_path: Path,
) -> None:
    with socket.create_server(("127.0.0.1", 0)) as service:
        port = service.getsockname()[1]
        fenced = subprocess.run(
            [sys.executable, "-c", FENCED, str(tmp_path), str(port)],
            capture_output=True,
            text=True,
        )
    assert fenced.returncode == 0, fenced.stderr
    ran, dials = fenced.stdout.splitlines()
    assert ran == "succeeded  Reply(total=3)", fenced.stdout
    # Only where Python pairs loopback TCP (no AF_UNIX) is its own pair admitted.
    own = "admitted" if getattr(socket, "AF_UNIX", None) is None else "no_network"
    refused = {"udp to own pair": "no_network", "service": "no_network", "udp decoy": "no_network"}
    if sys.platform == "linux":
        refused["free bind"] = "no_network"
    assert json.loads(dials) == {"own pair": own, **refused}, fenced.stdout
