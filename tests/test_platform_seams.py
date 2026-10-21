"""The OS seams under the executor's GPU path, on real calls: the CPUs a process may use, every
byte of a directory output and of a kernel store entry, a link out of a directory output, each
NVML query where the host has no NVIDIA driver, and a kernel build where the OS cannot end its
tree. Windows runs this file on windows-2025 (`platforms.yaml`)."""

from __future__ import annotations

import os
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
