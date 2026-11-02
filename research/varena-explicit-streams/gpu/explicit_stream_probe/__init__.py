"""Real CUDA execution through ordinary Cozy; run only on the root-owned test GPU."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author import App, Context, ThreadPoolExecutor

app = App()


class Probe(msgspec.Struct, forbid_unknown_fields=True):
    gpu_uuid: str
    runtime_modules: dict[str, str]


class Report(msgspec.Struct):
    passed: bool
    detail: str


@app.entrypoint
def verify(payload: Probe, ctx: Context) -> Report:
    import torch
    from tensorfs.varena import torch as vt

    import cozy_runtime
    from cozy_runtime import graphs
    from cozy_runtime.graphs import domain
    from cozy_runtime.internal import accel, int8_linear, kernel_ready
    from cozy_runtime.internal.residency import _REQUEST

    request = _REQUEST.get()
    assert request is not None and request.id == ctx.request_id and not request.idle
    assert ctx.device.type == "cuda"
    index = ctx.device.index or 0
    current = torch.cuda.current_stream(index)
    assert current.cuda_stream not in (0, 1, 2)
    assert (
        str(torch.cuda.get_device_properties(index).uuid).removeprefix("GPU-").lower()
        == payload.gpu_uuid.removeprefix("GPU-").lower()
    )
    root = Path(cozy_runtime.__file__).resolve().parent
    changed = {
        "author/_activity.py",
        "graphs/domain.py",
        "internal/accel.py",
        "internal/executor.py",
        "internal/kernel_ready.py",
        "internal/parallel/mirror.py",
    }
    assert set(payload.runtime_modules) == changed
    for name, digest in payload.runtime_modules.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

    driver = accel._stream_driver()
    driver.cuStreamGetFlags.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    driver.cuStreamGetFlags.restype = ctypes.c_int

    def flags(stream: Any) -> int:
        found = ctypes.c_uint()
        assert driver.cuStreamGetFlags(stream.cuda_stream, ctypes.byref(found)) == 0
        assert found.value == 1
        return found.value

    def full(tensor: Any, expected: float) -> None:
        accel.drain(torch, "cuda")
        assert torch.equal(tensor.cpu(), torch.full(tensor.shape, expected))

    # Join the genuine startup provider before any capture/baseline; it uses a readiness thread.
    assert int8_linear.native(torch, index) is not None
    accel.drain(torch, "cuda")
    execution = {
        "handle": current.cuda_stream,
        "flags": flags(current),
        "provider": int8_linear.serving(index),
    }
    before = vt.stats(index)
    x = torch.ones((64, 64), device=current.device)
    weight = torch.full_like(x, 0.25)
    eager = torch.mm(x, weight)
    full(eager, 16.0)

    def probe(_: bool) -> tuple[int, Any]:
        return torch.cuda.current_stream(index).cuda_stream, torch.mm(x, weight)

    name = f"explicit-stream-proof-{ctx.request_id}"
    kernel_ready.prime(name, probe)
    opening, primed = kernel_ready.wait(name, probe)
    assert opening == current.cuda_stream
    full(primed, 16.0)
    # The proof opener must not retain its output in the process's ready-kernel cache.
    with kernel_ready._LOCK:
        kernel_ready._SLOTS.pop(name)
    with ThreadPoolExecutor(1) as workers:
        carried, threaded = workers.submit(probe, True).result()
    assert carried == current.cuda_stream
    full(threaded, 16.0)

    owner = domain.Domain(torch, request.memory)
    recording = None
    outputs: list[Any] = []
    escaped = None
    try:
        owner.initialize()
        owner.prime_blas()
        handles = [current.cuda_stream, owner.stream.cuda_stream, owner.control.cuda_stream]
        assert len(set(handles)) == 3
        stream_flags = [flags(owner.stream), flags(owner.control)]
        pool = domain.Pool(torch.cuda.graph_pool_handle())
        # A real matrix multiply, captured by Runtime's control/library fork and join.
        graph, why = graphs.capture(torch, lambda: outputs.append(torch.mm(x, weight)), pool, owner)
        assert graph is not None, why
        recording = graphs._Recording(torch, graph, pool, [x], outputs[0], None, (), owner)
        inputs = torch.full_like(x, 2.0)
        escaped = recording.output()
        with owner.using():
            recording([inputs], escaped)
        full(escaped, 32.0)
        # A deliberately selected foreign framework stream remains exactly that stream.
        other = torch.cuda.Stream(device=index)
        other.wait_stream(current)
        with torch.cuda.stream(other), ThreadPoolExecutor(1) as workers:
            carried_other = workers.submit(
                lambda: torch.cuda.current_stream(index).cuda_stream
            ).result()
            assert carried_other == other.cuda_stream
            with owner.using():
                recording([inputs], escaped)
        current.wait_stream(other)
        full(escaped, 32.0)
        recording.finish()
        recording.close()
        outputs.clear()
        graph = pool = recording = None
        owner.finish()
        assert owner.closed and owner.stream is None and owner.control is None
        # Escaped output is caller storage; domain destruction cannot invalidate its contents.
        full(escaped, 32.0)
        inputs = x = weight = eager = primed = threaded = escaped = None
        gc.collect()
        accel.drain(torch, "cuda")
        vt.trim(index)
        after = vt.stats(index)
        assert after["allocated_bytes"] == before["allocated_bytes"], (before, after)
        return Report(
            True,
            json.dumps(
                {
                    "scope": "real CUDA threads/capture/replay/cleanup; no whole-model claim",
                    "execution": execution,
                    "domain_flags": stream_flags,
                    "external_stream_preserved": carried_other,
                    "before": before,
                    "after": after,
                    "versions": {
                        n: importlib.metadata.version(n)
                        for n in ("torch", "tensorfs", "cozy-runtime")
                    },
                    "runtime_modules": payload.runtime_modules,
                },
                sort_keys=True,
            ),
        )
    finally:
        if recording is not None:
            recording.finish()
            recording.close()
        outputs.clear()
        owner.finish()
