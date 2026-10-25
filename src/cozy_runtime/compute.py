"""Prepared resident CUDA work. Python constructs once; the GPU owns loop continuation.

This deliberately accepts one deterministic, fixed-shape tensor step, not arbitrary Python
control flow. Randomness must be explicit tensor inputs. Serving never compiles this module.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Sequence
from contextlib import ExitStack, suppress
from typing import TYPE_CHECKING, Any, NoReturn

from cozy_runtime.author._errors import CapabilityError, RuntimeFailure
from cozy_runtime.author._model import LANDED, PLACED
from cozy_runtime.author._prepared import Progress

if TYPE_CHECKING:
    from cozy_runtime.author._context import Context


def _zero_dropout_attention(func: Any, args: Any, kwargs: Any) -> bool:
    """Built-in forward SDPA is tagged seeded even when its dropout is a fixed zero."""
    schema = func._schema
    if schema.overload_name or schema.name not in (
        "aten::scaled_dot_product_attention",
        "aten::_scaled_dot_product_attention_math",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_cudnn_attention",
    ):
        return False
    # Flash, efficient/cuDNN and public/math schemas put dropout at different positions.
    for position, argument in enumerate(schema.arguments):
        if argument.name == "dropout_p":
            value = kwargs.get(
                "dropout_p", args[position] if position < len(args) else argument.default_value
            )
            return type(value) in (int, float) and value == 0
    return False


def _native_error(error: Exception) -> NoReturn:
    import torch

    if getattr(error, "eligibility_refusal", False):
        raise CapabilityError(str(error), code="prepared_graph_ineligible") from error

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
    Placed component roots or explicit weight bindings name the complete dependency set.
    """

    def __init__(
        self,
        native: Any,
        state: tuple[Any, ...],
        device: int,
        steps: int,
        dependencies: tuple[Any, ...],
        bindings: tuple[Any, ...],
        landed: int,
    ) -> None:
        self._native, self.state, self.device, self.steps = native, state, device, steps
        self._active: Execution | None = None
        self._submission = threading.RLock()
        self._dependencies, self._bindings = dependencies, bindings
        self._closed = False
        self._owned_bindings: tuple[Any, ...] = ()
        self._binding_cleanup = ExitStack()
        self._load: Any = None
        self._load_used = False
        self._landed = landed

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
        if self._landed != LANDED[0]:
            raise RuntimeError("prepared kernel implementations changed")
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
        resident_only: bool = False,
        load: Any = None,
    ) -> PreparedRepeat:
        """Capture a fixed body, optionally fed by one native raw LoadSlot.

        A load slot is initialized capture storage, not a cold-model compiler. Capture seals
        its unpublished Fill; one later submission reads its actual stored bytes before the
        native graph starts. Loaded submissions are one-shot and require wait/request join
        before output access. Pure resident repeats retain their existing reuse contract.

        Placed component roots are resolved to owned weight bindings. `resident_only=True`
        refuses missing weights before running the body; it never fills them for capture.
        Missing residency, unsupported hooks or changed model state raises CapabilityError
        with code `prepared_graph_ineligible`. A caller may continue eagerly after a capture
        refusal or a capture OOM, but must never retry after submit has begun.
        """
        import torch

        from cozy_runtime import graphs
        from cozy_runtime.internal.residency import Binding, StaleBinding

        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 < steps <= 2**32 - 1:
            raise ValueError("repeat steps must be a positive uint32")
        tensors = tuple(state)
        if not tensors or any(not isinstance(t, torch.Tensor) or not t.is_cuda for t in tensors):
            raise ValueError("prepared repeat state must contain CUDA tensors")
        device = tensors[0].device.index
        if device is None or any(t.device.index != device for t in tensors):
            raise ValueError("prepared repeat uses one CUDA device")
        if any(t.is_inference() for t in tensors):
            raise CapabilityError(
                "prepared mutable state must be allocated outside inference_mode",
                code="prepared_graph_ineligible",
            )
        if load is not None:
            from tensorfs import varena

            from cozy_runtime._kernels import _execution

            slot_type = getattr(varena, "LoadSlot", None)
            if (
                slot_type is None
                or not isinstance(load, slot_type)
                or not callable(getattr(_execution.ResidentRepeat, "continuation", None))
            ):
                raise CapabilityError(
                    "native loading requires a TensorFS LoadSlot",
                    code="prepared_load_unavailable",
                )
            owners = (*owners, load)
        with (
            torch.cuda.device(device),
            torch.inference_mode(False),
            ExitStack() as cleanup,
            ExitStack() as preparing,
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare a repeat outside graph capture")
            resolved: list[object] = []
            bindings: dict[int, Any] = {}
            models: dict[int, Any] = {}
            idle: set[int] = set()
            for owner in owners:
                placed = PLACED.get(owner) if isinstance(owner, torch.nn.Module) else None
                if isinstance(owner, Binding):
                    placed = PLACED.get(owner.model.module)
                if placed is not None:
                    mode, _ = placed[0].lora_choices((placed[1],))[placed[1]]
                    if mode == "per_step":
                        raise CapabilityError(
                            "prepared repeats require a fixed baked LoRA representation",
                            code="prepared_graph_ineligible",
                        )
                if isinstance(owner, Binding) or placed is not None:
                    binding = bindings.get(id(owner))
                    if binding is None:
                        if isinstance(owner, Binding):
                            if not owner.valid():
                                raise CapabilityError(
                                    "capture requires a valid weight binding",
                                    code="prepared_graph_ineligible",
                                )
                            binding = owner.model.retain(
                                tuple(b.path for b in owner.groups if b.path)
                            )
                        else:
                            assert placed is not None
                            binding = placed[0].retain(placed[1], ("",))
                        bindings[id(owner)] = binding
                        cleanup.callback(binding.close)
                        model = binding.model
                        if not model.memory.cuda or model.memory.device.index not in (None, device):
                            raise CapabilityError(
                                "prepared owners must use the state's CUDA device",
                                code="prepared_graph_ineligible",
                            )
                        models[id(model.module)] = model.module
                        if why := graphs._hooked(model.module):
                            raise CapabilityError(why, code="prepared_graph_ineligible")
                        if resident_only:
                            memory = model.memory
                            if id(memory) not in idle:
                                preparing.enter_context(
                                    memory.request(graphs.IDLE_REQUEST, idle=True)
                                )
                                idle.add(id(memory))
                            if not binding.resident():
                                raise CapabilityError(
                                    "capture requires the component's weights to be resident",
                                    code="prepared_graph_ineligible",
                                )
                        try:
                            preparing.enter_context(binding.prepare())
                        except StaleBinding as error:
                            raise CapabilityError(
                                str(error), code="prepared_graph_ineligible"
                            ) from error
                    resolved.append(binding)
                else:
                    resolved.append(owner)
                    if isinstance(owner, torch.nn.Module):
                        if why := graphs._hooked(owner):
                            raise CapabilityError(why, code="prepared_graph_ineligible")
                        models[id(owner)] = owner
            bypass = graphs.BYPASS.set(True)
            snapshots: list[Any] = []
            plan = None
            try:
                with torch.no_grad():
                    try:
                        snapshots = [graphs._State(torch, module) for module in models.values()]
                        plan = cls._capture_ready(step, state=tensors, steps=steps, owners=resolved)
                    finally:
                        changed = [
                            change for snapshot in snapshots if (change := snapshot.restore())
                        ]
                        if snapshots:
                            torch.cuda.current_stream().synchronize()
                    if changed:
                        raise CapabilityError(
                            "recording changed the model's state: " + ", ".join(changed),
                            code="prepared_graph_ineligible",
                        )
            except BaseException:
                if plan is not None:
                    plan.close()
                raise
            finally:
                graphs.BYPASS.reset(bypass)
            assert plan is not None
            if load is not None:
                try:
                    load.seal()
                except BaseException:
                    plan.close()
                    raise
                plan._load = load
            plan._owned_bindings = tuple(bindings.values())
            for binding in plan._owned_bindings:
                model = binding.model
                graphs.own_prepared(model.module, plan)
                if model.retire_graphs is None:
                    root = weakref.ref(model.module)

                    def retire(root: Any = root) -> None:
                        if (module := root()) is not None:
                            graphs.retire(torch, module)

                    model.retire_graphs = retire
            plan._binding_cleanup = cleanup.pop_all()
            return plan

    @classmethod
    def _capture_ready(
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

        tensors = tuple(state)
        device = tensors[0].device.index
        assert device is not None
        held: dict[int, Any] = {id(t): t for t in tensors}
        landed = LANDED[0]

        class CaptureOwners(TorchDispatchMode):  # type: ignore[misc]  # Torch is optional in checks.
            def __init__(self, retain: bool) -> None:
                super().__init__()
                self.retain = retain
                self.produced: dict[int, Any] = {}

            def __torch_dispatch__(
                self, func: Any, types: Any, args: Any = (), kwargs: Any = None
            ) -> Any:
                if torch.Tag.nondeterministic_seeded in func.tags and not _zero_dropout_attention(
                    func, args, kwargs or {}
                ):
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
            native = None
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
                    [
                        graph,
                        stream,
                        index,
                        *held.values(),
                        *(tensor.untyped_storage() for tensor in held.values()),
                        *owners,
                    ],
                )
            except _execution.NativeExecutionError as error:
                _native_error(error)
            finally:
                # Restore state even when capture/eligibility fails; no prepared object escapes.
                # A failed warm-up may have queued work before throwing on the host.
                current.wait_stream(stream)
                for target, original in zip(tensors, saved, strict=True):
                    target.copy_(original)
                current.synchronize()
                if native is None:
                    graph.reset()
                    held.clear()
            if landed != LANDED[0]:
                native.close_owner()
                held.clear()
                raise CapabilityError(
                    "kernel implementations changed during recording",
                    code="prepared_graph_ineligible",
                )
        dependencies = tuple((weakref.ref(t), cls._layout(t)) for t in held.values())
        bindings = tuple(
            (weakref.ref(owner), owner.identity) for owner in owners if isinstance(owner, Binding)
        )
        # The local mode class has a __class__ closure cycle. Native ownership has taken its
        # own references; do not leave GPU tensors retained until an unrelated Python GC pass.
        held.clear()
        return cls(native, tensors, device, steps, dependencies, bindings, landed)

    def _valid_definition(self) -> bool:
        try:
            self._validate()
            return True
        except RuntimeError:
            return False

    def _retire_if_idle(self) -> None:
        with self._submission:
            if self._closed:
                return
            if (
                self._active is not None
                and self._active._finished is None
                and not self._active.poll().done
            ):
                raise RuntimeError("cannot retire prepared work while its GPU execution is active")
            self.close()

    def _release_owned_bindings(self) -> None:
        from cozy_runtime import graphs

        bindings, self._owned_bindings = self._owned_bindings, ()
        for binding in bindings:
            graphs.forget_prepared(binding.model.module, self)
        self._binding_cleanup.close()

    def close(self) -> None:
        with self._submission:
            if self._closed:
                return
            settled_error: Exception | None = None
            if self._active is not None:
                active = self._active
                # A result waiter may hold _observation while waiting for completion.
                # Deliver cancellation first; it must never need that observer's lock.
                if active._finished is None:
                    active.cancel()
                try:
                    active._join()
                except Exception as error:
                    if active._finished is None:
                        raise  # No checked completion: preserve every graph/storage owner.
                    settled_error = error
            try:
                self._native.close_owner()
            except Exception as error:
                _native_error(error)
            self._closed = True
            self._release_owned_bindings()
            self._load = None
            if settled_error is not None:
                _native_error(settled_error)

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()

    def facts(self) -> dict[str, object]:
        """Facts from the actual native graph owner; a PreparedWork protocol is not evidence."""
        return dict(self._native.facts())

    def submit(self, context: Context | None = None) -> Execution:
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise CapabilityError(
                "prepared submission during CUDA capture is not supported",
                code="prepared_capture_active",
            )
        if context is not None:
            context.raise_if_cancelled()
            if context.device.type != "cuda" or context.device.index not in (None, self.device):
                raise ValueError("prepared work does not belong to this request's CUDA device")
        with self._submission:
            self._validate()
            if self._load is not None and self._load_used:
                raise RuntimeError("a prepared native load is one-shot; prepare a new slot")
            if self._active is not None and not self._active.poll().done:
                raise RuntimeError("a prepared repeat is already running")
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise CapabilityError(
                        "prepared submission during CUDA capture is not supported",
                        code="prepared_capture_active",
                    )
                outputs = tuple(torch.empty_like(t) for t in self.state)
                readout = torch.cuda.Event()
            if context is not None:
                context.raise_if_cancelled()
            claimed: list[Any] = []
            try:
                for binding in self._owned_bindings:
                    if binding.model not in claimed:
                        binding.model._claim()
                        claimed.append(binding.model)
                producer = torch.cuda.current_stream(self.device).cuda_stream
                job = None
                if self._load is None:
                    epoch = self._native.launch(producer)
                else:
                    # Arm on this OS thread before a TensorFS worker takes over. Its native
                    # continuation owns the recorded caller-ready event and execution epoch.
                    capsule, epoch = self._native.continuation(producer)
                    job = self._load.submit_native(capsule)
                    self._load_used = True
                self._active = Execution(self, context, epoch, outputs, readout, claimed, job)
                return self._active
            except BaseException as error:
                if isinstance(error, Exception) and getattr(error, "eligibility_refusal", False):
                    for model in reversed(claimed):
                        model._release()
                    _native_error(error)  # native preflight rejected before any launch/mutation
                # A signal or error after launch must not lose an in-flight native owner.
                try:
                    self._native.close_owner()
                    self._closed = True
                    self._load = None
                    for model in reversed(claimed):
                        model._release()
                    self._release_owned_bindings()
                except Exception as cleanup:
                    if getattr(cleanup, "error_code", None) not in (None, 2):
                        _native_error(cleanup)
                    raise RuntimeFailure(
                        "prepared submission completion is unproven; retire executor",
                        code="device_fault",
                    ) from cleanup
                if isinstance(error, Exception):
                    _native_error(error)
                raise


class Execution:
    """Observer/control handle. No observation or CPU loop is needed for GPU continuation."""

    def __init__(
        self,
        plan: PreparedRepeat,
        context: Context | None,
        epoch: int,
        outputs: tuple[Any, ...],
        readout: Any,
        models: list[Any],
        load_job: Any = None,
    ) -> None:
        self.plan, self.context = plan, context
        self._finished: Progress | None = None
        self._epoch, self._outputs, self._readout = epoch, outputs, readout
        self._observation = threading.RLock()
        self._models = models
        self._load_job = load_job
        self._failure: Exception | None = None

    def facts(self) -> dict[str, object]:
        """Native execution and, when present, its actual owned input read."""
        facts = self.plan.facts()
        if self._load_job is not None:
            facts["native_load"] = dict(self._load_job.facts())
        return facts

    def _failed_load(self, error: Exception) -> None:
        """A failed read may still prove cleanup; an exception alone is never that proof."""
        if self._load_job is None:
            return
        self._failure = error
        facts = self._load_job.facts()
        if facts.get("completion_proven", False):
            # No readout of failed/partial input. Public observers retain the original error;
            # close may now retire the graph and its bindings after this native certificate.
            self._finished = Progress(0, bool(facts.get("canceled", False)), True)
            models, self._models = self._models, []
            for model in reversed(models):
                model._release()

    def _finish(self, completed: int, canceled: bool) -> Progress:
        import torch

        if self._finished is None:
            with torch.cuda.device(self.plan.device), torch.no_grad():
                for output, state in zip(self._outputs, self.plan.state, strict=True):
                    output.copy_(state)
                self._readout.record(torch.cuda.current_stream(self.plan.device))
                self._readout.synchronize()
            self._finished = Progress(completed, canceled, True)
            models, self._models = self._models, []
            for model in reversed(models):
                model._release()
        return self._finished

    def poll(self) -> Progress:
        with self._observation:
            if self._failure is not None:
                _native_error(self._failure)
            if self._finished is not None:
                return self._finished
            try:
                completed, canceled, done = (
                    self.plan._native.poll(self._epoch)
                    if self._load_job is None
                    else self._load_job.poll()
                )
                if done:
                    completed, canceled = (
                        self.plan._native.wait(self._epoch)
                        if self._load_job is None
                        else self._load_job.wait()
                    )
                    return self._finish(completed, canceled)
                return Progress(completed, canceled, False)
            except Exception as error:
                self._failed_load(error)
                _native_error(error)

    def cancel(self) -> None:
        try:
            if self._load_job is None:
                self.plan._native.cancel(self._epoch)
            else:
                self._load_job.cancel()
        except Exception as error:
            _native_error(error)

    def _join(self) -> None:
        with self._observation:
            if self._finished is None:
                try:
                    completed, canceled = (
                        self.plan._native.wait(self._epoch)
                        if self._load_job is None
                        else self._load_job.wait()
                    )
                except Exception as error:
                    self._failed_load(error)
                    _native_error(error)
                self._finish(completed, canceled)

    def wait(self) -> tuple[Any, ...]:
        if self._failure is not None:
            _native_error(self._failure)
        if self.context is None:
            self._join()
        else:
            while not self.poll().done:
                if self.context.cancelled:
                    self.cancel()
                time.sleep(0.01)
            self.context.raise_if_cancelled()
        return self._outputs

    def close(self) -> None:
        with self.plan._submission:
            if self.plan._active is not self:
                raise RuntimeError("stale execution token")
            # Share the definition owner's join/readout path. In particular, don't race
            # another observer into two readouts or release the same model claim twice.
            self.plan.close()
