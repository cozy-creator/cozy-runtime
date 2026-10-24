"""Prepared resident CUDA work. Python constructs once; the GPU owns loop continuation.

This deliberately accepts one deterministic, fixed-shape tensor step, not arbitrary Python
control flow. Randomness must be explicit tensor inputs. Serving never compiles this module.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cozy_runtime.author._context import Context


@dataclass(frozen=True, slots=True)
class Progress:
    completed_steps: int
    canceled: bool
    done: bool


def _native_error(error: Exception) -> None:
    import torch

    code = getattr(error, "error_code", None)
    if code == 2:
        raise torch.OutOfMemoryError(str(error)) from error
    if isinstance(code, int):
        fault = torch.AcceleratorError(str(error))
        fault.error_code = code
        raise fault from error
    raise error


class PreparedRepeat:
    """A one-shot GPU repeat with resident state, explicit timestep/noise inputs and owners.

    `step(index)` executes only during preparation. `index` is a one-element CUDA int64
    tensor; use device tensor indexing, never `.item()` or a Python branch. The step updates
    the declared state in place. All other inputs must remain immutable through completion.
    Capture retains every Tensor seen by ATen, the callback, graph pool, and explicit owners.
    Caller-provided weight bindings must already pin the complete resident dependency set.
    """

    def __init__(self, native: Any, state: tuple[Any, ...], device: int, steps: int) -> None:
        self._native, self.state, self.device, self.steps = native, state, device, steps
        self._submitted = False

    @classmethod
    def capture(
        cls,
        step: Callable[[Any], object],
        *,
        state: Sequence[Any],
        steps: int,
        owners: Sequence[object] = (),
    ) -> PreparedRepeat:
        import torch
        from tensorfs import varena
        from torch.utils._pytree import tree_leaves
        from torch.utils._python_dispatch import TorchDispatchMode

        try:
            from cozy_runtime._kernels import _execution
        except ImportError as error:
            raise RuntimeError("this Runtime wheel has no prebuilt GPU execution module") from error

        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 < steps <= 2**32 - 1:
            raise ValueError("repeat steps must be a positive uint32")
        tensors = tuple(state)
        if not tensors or any(not isinstance(t, torch.Tensor) or not t.is_cuda for t in tensors):
            raise ValueError("prepared repeat state must contain CUDA tensors")
        device = tensors[0].device.index
        if device is None or any(t.device.index != device for t in tensors):
            raise ValueError("prepared repeat uses one CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare a repeat outside graph capture")
        held: dict[int, Any] = {id(t): t for t in tensors}

        class CaptureOwners(TorchDispatchMode):
            def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
                if torch.Tag.nondeterministic_seeded in func.tags:
                    raise RuntimeError("GPU-driven repeats require explicit noise/RNG tensor inputs")
                for tensor in tree_leaves((args, kwargs or {})):
                    if isinstance(tensor, torch.Tensor):
                        held[id(tensor)] = tensor
                result = func(*args, **(kwargs or {}))
                for tensor in tree_leaves(result):
                    if isinstance(tensor, torch.Tensor):
                        held[id(tensor)] = tensor
                return result

        with torch.cuda.device(device), torch.no_grad():
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            index = torch.zeros(1, dtype=torch.int64, device=device)
            saved = tuple(t.clone() for t in tensors)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            try:
                with torch.cuda.stream(stream), CaptureOwners():
                    step(index)  # instantiate library workspaces before capture
                    for target, original in zip(tensors, saved, strict=True):
                        target.copy_(original)
                stream.synchronize()
                with varena.capture(device), torch.cuda.stream(stream), CaptureOwners():
                    graph.capture_begin(capture_error_mode="global")
                    try:
                        step(index)
                    finally:
                        graph.capture_end()
                stream.synchronize()
                for target, original in zip(tensors, saved, strict=True):
                    target.copy_(original)
                current.synchronize()
                # CUDAGraph owns capture allocations; its live template is cloned into a child
                # of the native conditional body. Its executable is never borrowed or replayed.
                native = _execution.ResidentRepeat(
                    graph.raw_cuda_graph(), index.data_ptr(), device, steps,
                    [graph, stream, index, step, *held.values(), *owners],
                )
            except _execution.NativeExecutionError as error:
                _native_error(error)
                raise AssertionError("unreachable")
            finally:
                # Restore state even when capture/eligibility fails; no prepared object escapes.
                for target, original in zip(tensors, saved, strict=True):
                    target.copy_(original)
                current.synchronize()
        return cls(native, tensors, device, steps)

    def submit(self, context: Context | None = None) -> Execution:
        import torch

        if self._submitted:
            raise RuntimeError("a prepared repeat can be submitted only once")
        if context is not None:
            context.raise_if_cancelled()
            if context.device.type != "cuda" or context.device.index not in (None, self.device):
                raise ValueError("prepared work does not belong to this request's CUDA device")
        self._submitted = True
        try:
            self._native.launch(torch.cuda.current_stream(self.device).cuda_stream)
        except Exception as error:
            _native_error(error)
        return Execution(self, context)


class Execution:
    """Observer/control handle. No observation or CPU loop is needed for GPU continuation."""

    def __init__(self, plan: PreparedRepeat, context: Context | None) -> None:
        self.plan, self.context = plan, context
        self._finished: Progress | None = None

    def poll(self) -> Progress:
        if self._finished is not None:
            return self._finished
        try:
            completed, canceled, done = self.plan._native.poll()
            if done:
                completed, canceled = self.plan._native.wait()
                self._finished = Progress(completed, canceled, True)
                return self._finished
            return Progress(completed, canceled, False)
        except Exception as error:
            _native_error(error)
            raise AssertionError("unreachable")

    def cancel(self) -> None:
        try:
            self.plan._native.cancel()
        except Exception as error:
            _native_error(error)

    def wait(self) -> tuple[Any, ...]:
        if self.context is None:
            try:
                completed, canceled = self.plan._native.wait()
            except Exception as error:
                _native_error(error)
                raise AssertionError("unreachable")
            self._finished = Progress(completed, canceled, True)
        else:
            while not self.poll().done:
                if self.context.cancelled:
                    self.cancel()
                time.sleep(0.01)
            self.context.raise_if_cancelled()
        return self.plan.state

    def close(self) -> None:
        try:
            self.plan._native.close()
        except Exception as error:
            _native_error(error)
