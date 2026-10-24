"""Prepared resident CUDA work. Python constructs once; the GPU owns loop continuation.

This deliberately accepts one deterministic, fixed-shape tensor step, not arbitrary Python
control flow. Randomness must be explicit tensor inputs. Serving never compiles this module.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from cozy_runtime.author._context import Context


@dataclass(frozen=True, slots=True)
class Progress:
    completed_steps: int
    canceled: bool
    done: bool


def _native_error(error: Exception) -> NoReturn:
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
    """A reusable GPU repeat with resident state, explicit timestep/noise inputs and owners.

    `step(index)` executes only during preparation. `index` is a one-element CUDA int64
    tensor; use device tensor indexing, never `.item()` or a Python branch. The step updates
    the declared state in place. All other inputs must remain immutable through completion.
    Capture retains external ATen dependencies, the graph pool, and explicit owners.
    Caller-provided weight bindings must already pin the complete resident dependency set.
    """

    def __init__(
        self,
        native: Any,
        state: tuple[Any, ...],
        device: int,
        steps: int,
        dependencies: tuple[Any, ...],
        bindings: tuple[Any, ...],
    ) -> None:
        self._native, self.state, self.device, self.steps = native, state, device, steps
        self._active: Execution | None = None
        self._submission = threading.Lock()
        self._dependencies, self._bindings = dependencies, bindings
        self._closed = False

    @staticmethod
    def _layout(tensor: Any) -> tuple[Any, ...]:
        storage = tensor.untyped_storage()
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
            tensor.storage_offset(),
            tensor.data_ptr(),
            storage.data_ptr(),
            storage.nbytes(),
        )

    def _validate(self) -> None:
        if self._closed:
            raise RuntimeError("prepared repeat is closed")
        for reference, layout in self._dependencies:
            tensor = reference()
            if tensor is None or self._layout(tensor) != layout:
                raise RuntimeError("prepared tensor storage or geometry changed")
        for reference, identity in self._bindings:
            binding = reference()
            if binding is None or not binding.valid() or binding.identity != identity:
                raise RuntimeError("prepared weight binding or fill identity changed")

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
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves

        from cozy_runtime.internal.residency import Binding

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
        held: dict[int, Any] = {id(t): t for t in tensors}

        class CaptureOwners(TorchDispatchMode):  # type: ignore[misc]  # Torch is optional in checks.
            def __init__(self, retain: bool) -> None:
                super().__init__()
                self.retain = retain
                self.produced: dict[int, Any] = {}

            def __torch_dispatch__(
                self, func: Any, types: Any, args: Any = (), kwargs: Any = None
            ) -> Any:
                if torch.Tag.nondeterministic_seeded in func.tags:
                    raise RuntimeError(
                        "GPU-driven repeats require explicit noise/RNG tensor inputs"
                    )
                for tensor in tree_leaves((args, kwargs or {})):
                    prior = self.produced.get(id(tensor))
                    if (
                        self.retain
                        and isinstance(tensor, torch.Tensor)
                        and (prior is None or prior() is not tensor)
                    ):
                        held[id(tensor)] = tensor
                result = func(*args, **(kwargs or {}))
                for tensor in tree_leaves(result):
                    if isinstance(tensor, torch.Tensor):
                        self.produced[id(tensor)] = weakref.ref(tensor)
                return result

        with torch.cuda.device(device), torch.no_grad():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare a repeat outside graph capture")
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            index = torch.zeros(1, dtype=torch.int64, device=device)
            saved = tuple(t.clone() for t in tensors)
            stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            try:
                with torch.cuda.stream(stream), CaptureOwners(False):
                    step(index)  # instantiate library workspaces before capture
                    for target, original in zip(tensors, saved, strict=True):
                        target.copy_(original)
                stream.synchronize()
                with varena.capture(device), torch.cuda.stream(stream), CaptureOwners(True):
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
                    graph.raw_cuda_graph(),
                    index.data_ptr(),
                    device,
                    steps,
                    [graph, stream, index, *held.values(),
                     *(tensor.untyped_storage() for tensor in held.values()), *owners],
                )
            except _execution.NativeExecutionError as error:
                _native_error(error)
            finally:
                # Restore state even when capture/eligibility fails; no prepared object escapes.
                for target, original in zip(tensors, saved, strict=True):
                    target.copy_(original)
                current.synchronize()
        dependencies = tuple((weakref.ref(t), cls._layout(t)) for t in held.values())
        bindings = tuple(
            (weakref.ref(owner), owner.identity) for owner in owners if isinstance(owner, Binding)
        )
        return cls(native, tensors, device, steps, dependencies, bindings)

    def close(self) -> None:
        with self._submission:
            if self._closed:
                return
            if self._active is not None:
                self._active.close()
            else:
                try:
                    self._native.close(0)
                except Exception as error:
                    _native_error(error)
                self._closed = True

    def submit(self, context: Context | None = None) -> Execution:
        import torch
        from tensorfs import varena

        if context is not None:
            context.raise_if_cancelled()
            if context.device.type != "cuda" or context.device.index not in (None, self.device):
                raise ValueError("prepared work does not belong to this request's CUDA device")
        with self._submission:
            self._validate()
            if self._active is not None and not self._active.poll().done:
                raise RuntimeError("a prepared repeat is already running")
            with torch.cuda.device(self.device):
                outputs = tuple(torch.empty_like(t) for t in self.state)
                readout = torch.cuda.Event()
            guard = varena.capture(self.device)
            guard.__enter__()
            try:
                epoch = self._native.launch(torch.cuda.current_stream(self.device).cuda_stream)
            except Exception as error:
                guard.__exit__(type(error), error, error.__traceback__)
                _native_error(error)
            self._active = Execution(self, context, guard, epoch, outputs, readout)
            return self._active


class Execution:
    """Observer/control handle. No observation or CPU loop is needed for GPU continuation."""

    def __init__(
        self,
        plan: PreparedRepeat,
        context: Context | None,
        guard: Any,
        epoch: int,
        outputs: tuple[Any, ...],
        readout: Any,
    ) -> None:
        self.plan, self.context = plan, context
        self._guard = guard
        self._finished: Progress | None = None
        self._epoch, self._outputs, self._readout = epoch, outputs, readout
        self._observation = threading.RLock()

    def _finish(self, completed: int, canceled: bool) -> Progress:
        import torch

        if self._finished is None:
            with torch.cuda.device(self.plan.device), torch.no_grad():
                for output, state in zip(self._outputs, self.plan.state, strict=True):
                    output.copy_(state)
                self._readout.record(torch.cuda.current_stream(self.plan.device))
                self._readout.synchronize()
            self._finished = Progress(completed, canceled, True)
            self._release_guard()
        return self._finished

    def _release_guard(self) -> None:
        if self._guard is not None:
            self._guard.__exit__(None, None, None)
            self._guard = None

    def poll(self) -> Progress:
        with self._observation:
            if self._finished is not None:
                return self._finished
            try:
                completed, canceled, done = self.plan._native.poll(self._epoch)
                if done:
                    completed, canceled = self.plan._native.wait(self._epoch)
                    return self._finish(completed, canceled)
                return Progress(completed, canceled, False)
            except Exception as error:
                _native_error(error)

    def cancel(self) -> None:
        try:
            self.plan._native.cancel(self._epoch)
        except Exception as error:
            _native_error(error)

    def wait(self) -> tuple[Any, ...]:
        if self.context is None:
            with self._observation:
                if self._finished is None:
                    try:
                        completed, canceled = self.plan._native.wait(self._epoch)
                    except Exception as error:
                        _native_error(error)
                    self._finish(completed, canceled)
        else:
            while not self.poll().done:
                if self.context.cancelled:
                    self.cancel()
                time.sleep(0.01)
            self.context.raise_if_cancelled()
        return self._outputs

    def close(self) -> None:
        try:
            self.plan._native.close(self._epoch)
            self.plan._closed = True
            self._release_guard()
        except Exception as error:
            _native_error(error)
