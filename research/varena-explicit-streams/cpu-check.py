"""CPU thread/ordering contracts. No CUDA execution or numerical qualification."""

# ruff: noqa: E402 - immutable dependency overlay precedes product imports
from __future__ import annotations

import contextlib
import ctypes
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Reuse immutable dependencies; do not mutate any shared installation.
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(root / "src"),
    "/home/fidika/cozy_v2/outputs/xs138-p2-native-retention/metadata",
    "/home/fidika/cozy_v2/outputs/xs138-native-cache-candidate-a96c7c3e8/installed",
    "/home/fidika/cozy/.worktrees/cozy-creator/xs138-p0-torch/.task/torch-sdk/2.14.0+cu130-08187d9e/site-packages",
    "/home/fidika/cozy_v2/cozy-runtime/.venv/lib/python3.12/site-packages",
]
import torch

from cozy_runtime.author import _activity
from cozy_runtime.author._context import Device
from cozy_runtime.internal import accel, executor, kernel_ready

assert not torch.cuda.is_initialized()
assert accel._stream_driver.cache_info().currsize == 0

local = threading.local()
trace = []


class Stream:
    def __init__(self, pointer, device):
        self.cuda_stream, self.device = pointer, device

    def query(self):
        trace.append(("query", self.cuda_stream))
        return True


class Cuda:
    def is_initialized(self):
        return True

    def current_stream(self):
        return getattr(local, "stream", Stream(0, 0))

    def set_stream(self, stream):
        local.stream = stream

    def set_device(self, device):
        trace.append(("device", device))

    def current_device(self):
        return self.current_stream().device

    @contextlib.contextmanager
    def stream(self, stream):
        previous = self.current_stream()
        self.set_stream(stream)
        try:
            yield
        finally:
            self.set_stream(previous)

    @contextlib.contextmanager
    def device(self, device):
        yield

    def Event(self):
        class Event:
            def record(self, stream):
                trace.append(("record", stream.cuda_stream))

            def synchronize(self):
                trace.append(("event_done",))

        return Event()

    def synchronize(self):
        raise AssertionError("a completion boundary must not synchronize the device")

    def default_stream(self):
        raise AssertionError("default stream cannot prove owned-stream completion")

    ExternalStream = Stream


fake = SimpleNamespace(cuda=Cuda(), AcceleratorError=RuntimeError)
created, destroyed = [], []


def create(pointer, flags):
    created.append(flags)
    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 128
    return 0


def destroy(pointer):
    destroyed.append(pointer.value if isinstance(pointer, ctypes.c_void_p) else pointer)
    return 0


driver = SimpleNamespace(cuStreamCreate=create, cuStreamDestroy_v2=destroy)
with patch.object(accel, "_stream_driver", return_value=driver):
    owned = accel.nonblocking_stream(fake, 1)
    assert (owned.cuda_stream, owned.device, created) == (128, 1, [1])
    accel.destroy_stream(fake, owned)
    assert destroyed == [128]
    with patch.object(fake.cuda, "ExternalStream", side_effect=ValueError("wrapper refused")):
        try:
            accel.nonblocking_stream(fake, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("wrapper failure was hidden")
    assert destroyed == [128, 128]

host = executor.Executor.__new__(executor.Executor)
vars(host).update(
    host=host,
    _startup_reported=True,
    _device_ready=False,
    execution_stream=None,
    torch=fake,
    device_kind="cuda",
    rank=1,
    world=2,
    channel=None,
    group=None,
    poisoned="",
)
host._open_memory = lambda torch: trace.append(("memory", torch.cuda.current_stream().cuda_stream))
host._device = lambda: Device("cuda", 1)
old_worker, old_ready = _activity.WORKER_CONTEXT, kernel_ready.EXECUTION[0]
try:
    with (
        patch.object(accel, "initialized", return_value=False),
        patch.object(accel, "initialize", side_effect=lambda *_: trace.append(("initialize",))),
        patch.object(accel, "nonblocking_stream", return_value=owned),
        patch.object(
            accel,
            "first_launches",
            side_effect=lambda t, _: trace.append(("first", t.cuda.current_stream().cuda_stream)),
        ),
        patch.object(executor, "place"),
    ):
        assert host._initialize_device(fake) is None
    assert (
        trace.index(("initialize",))
        < trace.index(("device", 1))
        < trace.index(("memory", 128))
        < trace.index(("first", 128))
    )
    assert host.execution_stream is owned and host._device_ready

    # A command worker starts with thread-local default state; _reply selects the owner.
    host.handle = lambda command: {"ok": True, "stream": fake.cuda.current_stream().cuda_stream}
    answers = []

    def command_worker():
        assert fake.cuda.current_stream().cuda_stream == 0
        answers.append(executor._reply(host, {"cmd": "probe"}))
        assert fake.cuda.current_stream().cuda_stream == 0

    thread = threading.Thread(target=command_worker)
    thread.start()
    thread.join()
    assert answers == [{"ok": True, "stream": 128, "reply": "probe"}]

    # Readiness actually uses its background thread, not merely a captured Python context.
    ready = []
    kernel_ready.prime(
        "explicit-stream-cpu-proof",
        lambda _: ready.append(fake.cuda.current_stream().cuda_stream) or object(),
    )
    assert kernel_ready.wait("explicit-stream-cpu-proof", lambda _: None) is not None
    assert ready == [128]

    # Exported pools preserve the actual selected stream, not an unconditional owner alias.
    external = Stream(256, 1)
    with fake.cuda.stream(external), _activity.ThreadPoolExecutor(1) as pool:
        carried = pool.submit(lambda: fake.cuda.current_stream().cuda_stream).result()
    assert carried == 256 and fake.cuda.current_stream() is owned
    fake.is_grad_enabled = lambda: True
    fake.is_inference_mode_enabled = lambda: True
    fake.is_autocast_enabled = lambda _: False
    fake.inference_mode = fake.set_grad_enabled = contextlib.nullcontext
    group = SimpleNamespace(reserve=lambda _: None, unreserve=lambda _: None)
    placement = executor.mirror.GroupPlacement({"block": 1}, group, fake)
    remote = []
    with fake.cuda.stream(external):
        call = placement.carry(
            lambda: remote.append(fake.cuda.current_stream().cuda_stream), ("block",)
        )
    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    assert remote == [256]
    accel.drain(fake, "cuda")
    assert trace[-2:] == [("record", 128), ("event_done",)]
    assert host._quiescent() and trace[-1] == ("query", 128)
    host.memory = SimpleNamespace(close_cooperation=lambda: trace.append(("close",)))
    channel = SimpleNamespace(send=lambda _: trace.append(("shutdown",)))
    executor._shut(host, channel)
    assert trace[-4:] == [("record", 128), ("event_done",), ("close",), ("shutdown",)]
finally:
    _activity.WORKER_CONTEXT, kernel_ready.EXECUTION[0] = old_worker, old_ready

assert not torch.cuda.is_initialized()
assert accel._stream_driver.cache_info().currsize == 0
print(
    json.dumps(
        {
            "passed": True,
            "cuda_initialized": False,
            "runtime_stream_driver_loaded": False,
            "scope": "CPU boundary contracts only; CUDA calls simulated, no GPU qualification",
            "cases": [
                "nonblocking_creation",
                "failed_wrap_retirement",
                "initialization_order",
                "command_thread",
                "background_readiness",
                "actual_pool_stream",
                "event_completion",
                "owned_quiescence",
                "remote_call_stream",
                "shutdown_completion",
            ],
        }
    )
)
