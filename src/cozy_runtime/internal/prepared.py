"""Native programs behind `author.denoise_loop` (`internal.loop` records them): recorded CUDA
graphs whose loop continuation and stage advancement run in native code, not Python.
"""

from __future__ import annotations

import hashlib
import threading
import time
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from typing import Any, NoReturn

import torch
from tensorfs import varena

from cozy_runtime import graphs
from cozy_runtime.author._context import Context
from cozy_runtime.author._errors import CapabilityError, RuntimeFailure
from cozy_runtime.author._model import LANDED, Building
from cozy_runtime.author._prepared import Progress
from cozy_runtime.graphs import domain
from cozy_runtime.internal.residency import _REQUEST

_execution: Any
try:
    import cozy_runtime._kernels._execution as _execution
except ModuleNotFoundError as missing:  # CPU and macOS wheels carry no native execution module
    if missing.name != "cozy_runtime._kernels._execution":
        raise
    _execution = None


def _zero_dropout_attention(func: Any, args: Any, kwargs: Any) -> bool:
    """Built-in forward SDPA is tagged seeded even when its dropout is a fixed zero."""
    schema = func._schema
    if schema.overload_name or schema.name not in (
        # The tagged selector chooses a backend; zero-dropout dispatch is deterministic.
        "aten::_fused_sdp_choice",
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
    if getattr(error, "load_unavailable", False):
        raise CapabilityError(str(error), code="prepared_load_unavailable") from error
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
    """One recorded graph, repeated natively `steps` times: a stage of a PreparedSequence.

    It retains its external ATen dependencies, the graph pool and its owners; each submission
    checks their layouts, the bindings' fill identities and the kernel epoch.
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
        self._capture_cache: dict[str, int] = {}
        self._load_ordering_requested = "checked"
        self._sequence_owner: Any = None
        self._load_ordering = "checked"
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

    def _invalid_reason(self) -> str | None:
        if self._closed:
            return "prepared repeat is closed"
        if self._landed != LANDED[0]:
            return "prepared kernel implementations changed"
        for reference, layout in self._dependencies:
            tensor = reference()
            if tensor is None or self._layout(tensor) != layout:
                return "prepared tensor storage or geometry changed"
        for reference, identity in self._bindings:
            binding = reference()
            if binding is None or not binding.valid() or binding.identity != identity:
                return "prepared weight binding or fill identity changed"
        return None

    def _validate(self) -> None:
        if reason := self._invalid_reason():
            raise RuntimeError(reason)

    def valid(self) -> bool:
        """Whether this definition's owners, layouts and implementations still match.

        Input values may change in place between completed submissions. This query neither
        prepares missing weights nor reserves a future submission; submit checks again.
        Unexpected storage/device errors propagate rather than becoming a cache miss.
        """
        with self._submission:
            return self._invalid_reason() is None

    def _retire_if_idle(self) -> None:
        sequence = self._sequence_owner() if self._sequence_owner is not None else None
        if sequence is not None and not sequence._closing:
            sequence._retire_if_idle()
            return
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
        bindings, self._owned_bindings = self._owned_bindings, ()
        for binding in bindings:
            graphs.forget_prepared(binding.model.module, self)
        self._binding_cleanup.close()

    def close(self) -> None:
        sequence = self._sequence_owner() if self._sequence_owner is not None else None
        if sequence is not None and not sequence._closing:
            sequence.close()
            return
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
        return {
            **self._native.facts(),
            "capture_cache": dict(self._capture_cache),
            "load_ordering_requested": self._load_ordering_requested,
            "load_ordering": self._load_ordering,
        }

class Execution:
    """Observer/control handle. No observation or CPU loop is needed for GPU continuation."""

    def __init__(
        self,
        plan: PreparedRepeat | PreparedSequence,
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
        self._load_request = _REQUEST.get()
        self._load_recorded = False
        self._failure: Exception | None = None
        self._queued_eligible = context is not None and load_job is None
        self._queued_offered: str | None = None

    def _offer_queued(self) -> None:
        # Scratch, outputs and resident weights are already retained. This optional
        # observer work never advances GPU execution or admits a streamed future workspace.
        if (
            self._queued_eligible
            and self._queued_offered is None
            and self._load_request is not None
        ):
            self._queued_offered = self._load_request.prefetch_queued()

    def facts(self) -> dict[str, object]:
        """Native execution and, when present, its actual owned input read."""
        facts = self.plan.facts()
        if self._load_job is not None:
            facts["native_load"] = dict(self._load_job.facts())
        return facts

    def completion_proven(self) -> bool:
        """Whether native work and readout/drain completed, independently of its result.

        A failed load sets _finished only after TensorFS certifies checked cleanup
        and this execution releases every model claim.
        poll() and wait() must still report that load's original error.
        """
        with self._observation:
            return self._finished is not None

    def _release_models(self) -> None:
        while self._models:
            self._models[-1]._release()
            self._models.pop()

    def _record_load(self, facts: Mapping[str, Any]) -> None:
        if self._load_recorded or not facts.get("done", False):
            return
        self._load_recorded = True
        if self._load_request is None or not facts.get("read_done", False):
            return
        for binding in self.plan._owned_bindings:
            if binding._load_slot is self.plan._load:
                for block, nbytes in zip(binding.groups, facts["input_read_bytes"], strict=True):
                    if nbytes:
                        # The native worker did this I/O. Record bytes once after completion;
                        # its separate native clocks are not Python read-wait measurements.
                        self._load_request.read(block, 0, int(nbytes))

    def _failed_load(self, error: Exception) -> None:
        """A native input/sequence job may prove cleanup; an exception is not that proof."""
        if self._load_job is None:
            return
        self._failure = error
        facts = self._load_job.facts()
        self._record_load(facts)
        if facts.get("completion_proven", False):
            # No readout of failed/partial input. Public observers retain the original error;
            # close may now retire the graph and its bindings after this native certificate.
            self._release_models()
            self._finished = Progress(0, bool(facts.get("canceled", False)), True)

    def _finish(self, completed: int, canceled: bool) -> Progress:
        if self._finished is None:
            if self._load_job is not None:
                self._record_load(self._load_job.facts())
            with torch.cuda.device(self.plan.device), torch.no_grad():
                for output, state in zip(self._outputs, self.plan.state, strict=True):
                    output.copy_(state)
                self._readout.record(torch.cuda.current_stream(self.plan.device))
                self._readout.synchronize()
            self._release_models()
            self._finished = Progress(completed, canceled, True)
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
            except Exception as error:
                self._failed_load(error)
                _native_error(error)
            # The input certificate covers producer/graph completion, not this readout.
            # Never reinterpret a copy, synchronization or model-release failure as a
            # clean input failure merely because the load worker finished successfully.
            if done:
                return self._finish(completed, canceled)
            # The one queued CPU preflight can finish after submit. Re-offer only while
            # native work remains active; latching an actual assignment bounds admission.
            self._offer_queued()
            return Progress(completed, canceled, False)

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


class PreparedSequence:
    """Own ordered prepared stages; one submission advances them entirely in native code.

    Stages are distinct definitions on one CUDA device. Their boundary state is ordinary
    persistent GPU storage: a later stage may read a previous stage's state at the same
    address. Output values are copied only after the whole native sequence finishes.
    Preparation is still explicit Python work. The native worker may perform host-required
    I/O and checked CUDA waits; sequential execution is not a single GPU graph launch.

    Constructing this object transfers definition ownership: close/idle retirement of any
    stage closes the whole sequence.

    `prefetch=False` omits speculative next-stage reads when other work has priority; True
    permits bounded lookahead only when actual capacity admits it.
    """

    @classmethod
    def record(cls) -> _SequenceCapture:
        """Record an ordered program in one private scratch pool, then `finish` ownership.
        Intermediate definitions cannot launch independently."""
        return _SequenceCapture()

    def __init__(
        self,
        stages: Sequence[PreparedRepeat],
        *,
        outputs: Sequence[Any] | None = None,
        repeat: tuple[int, int, int] | None = None,
        prefetch: bool = True,
    ) -> None:
        if not callable(getattr(_execution, "NativeSequence", None)):
            raise CapabilityError(
                "native sequence module is unavailable", code="prepared_graph_ineligible"
            )
        if type(prefetch) is not bool:
            raise TypeError("sequence prefetch policy must be a boolean")
        self.prefetch = prefetch
        self.stages = tuple(stages)
        if not self.stages or len(self.stages) > 65536:
            raise ValueError("a prepared sequence needs 1..65536 stages")
        if any(not isinstance(stage, PreparedRepeat) for stage in self.stages):
            raise TypeError("sequence stages must be PreparedRepeat definitions")
        if len({id(stage) for stage in self.stages}) != len(self.stages):
            raise ValueError("sequence definitions must be distinct; repeat inside a stage")
        if repeat is not None:
            if (
                not isinstance(repeat, tuple)
                or len(repeat) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in repeat)
            ):
                raise ValueError("repeat must be (first_stage, exclusive_end, positive_passes)")
            begin, end, passes = repeat
            if not 0 <= begin < end <= len(self.stages) or not 0 < passes <= 2**32 - 1:
                raise ValueError("invalid bounded repeat range")
            if (end - begin) * passes + len(self.stages) - (end - begin) > 2**32 - 1:
                raise ValueError("repeated stage count exceeds uint32")
            if passes > 1 and not getattr(
                _execution.ResidentRepeat, "supports_reusable_input", False
            ):
                raise CapabilityError(
                    "native stage rearm is unavailable", code="prepared_graph_ineligible"
                )
        self._repeat = repeat
        self.device = self.stages[0].device
        if any(stage.device != self.device for stage in self.stages):
            raise ValueError("a native sequence belongs to one CUDA device")
        self.state = tuple(self.stages[-1].state if outputs is None else outputs)
        declared = {id(tensor) for stage in self.stages for tensor in stage.state}
        if not self.state or any(
            not isinstance(t, torch.Tensor) or id(t) not in declared for t in self.state
        ):
            raise ValueError("sequence outputs must be declared persistent stage state")
        self.steps = sum(stage.steps for stage in self.stages)
        if repeat is not None:
            begin, end, passes = repeat
            self.steps += (passes - 1) * sum(stage.steps for stage in self.stages[begin:end])
        self._submission = threading.RLock()
        self._active: _SequenceExecution | None = None
        self._native: Any = None
        self._capture_domain: Any = None
        # A rejected constructor owns no definitions. In particular its destructor must
        # not close a stage already transferred to another live sequence.
        self._closed = True
        self._closing = False
        self._load = None
        self._owned_bindings: tuple[Any, ...] = ()
        with self._locked_stages():
            for stage_index, stage in enumerate(self.stages):
                stage._validate()
                if stage._sequence_owner is not None and stage._sequence_owner() is not None:
                    raise RuntimeError("a prepared stage already belongs to another sequence")
                if stage._active is not None and not stage._active.poll().done:
                    raise RuntimeError("a running definition cannot enter a new sequence")
                if stage._load is not None and not callable(
                    getattr(stage._load, "continuation", None)
                ):
                    raise CapabilityError(
                        "this input loader cannot defer its native stage",
                        code="prepared_load_unavailable",
                    )
                if (
                    self._repeated(stage_index)
                    and stage._load is not None
                    and not getattr(stage._load, "supports_reusable_input", False)
                ):
                    raise CapabilityError(
                        "this input loader cannot rearm a native stage",
                        code="prepared_load_unavailable",
                    )
            owner = weakref.ref(self)
            for stage in self.stages:
                stage._sequence_owner = owner
            self._closed = False

    def _repeated(self, index: int) -> bool:
        return (
            self._repeat is not None
            and self._repeat[2] > 1
            and self._repeat[0] <= index < self._repeat[1]
        )

    def _locked_stages(self) -> ExitStack:
        locks = ExitStack()
        try:
            for stage in sorted(self.stages, key=id):
                locks.enter_context(stage._submission)
            return locks
        except BaseException:
            locks.close()
            raise

    def valid(self) -> bool:
        with self._submission, self._locked_stages():
            return (
                not self._closed
                and (self._active is None or self._active._failure is None)
                and all(stage.valid() for stage in self.stages)
            )

    def facts(self) -> dict[str, object]:
        stages = [stage.facts() for stage in self.stages]
        return {
            "sequence": {} if self._native is None else self._native.facts(),
            "stages": stages,
            "host_graph_launches": sum(int(stage["host_graph_launches"]) for stage in stages),
            "native_stage_advancement": True,
        }

    def submit(self, context: Context | None = None) -> Execution:
        if not callable(getattr(_execution, "NativeSequence", None)):
            raise CapabilityError(
                "native sequence module is unavailable", code="prepared_graph_ineligible"
            )
        if torch.cuda.is_current_stream_capturing():
            raise CapabilityError(
                "sequence submission during capture is unsupported", code="prepared_capture_active"
            )
        if context is not None:
            context.raise_if_cancelled()
            if context.device.type != "cuda" or context.device.index not in (None, self.device):
                raise ValueError("prepared sequence does not belong to this request's CUDA device")
        with self._submission, self._locked_stages():
            if self._closed:
                raise RuntimeError("prepared sequence is closed")
            for stage in self.stages:
                stage._validate()
                if stage._active is not None and not stage._active.poll().done:
                    raise RuntimeError("a sequence stage is already running")
            with torch.cuda.device(self.device):
                outputs = tuple(torch.empty_like(value) for value in self.state)
                readout = torch.cuda.Event()
                producer = torch.cuda.current_stream(self.device).cuda_stream
            claimed: list[Any] = []
            capsules: list[Any] = []
            reads: list[tuple[int, Any, tuple[Any, ...]]] = []
            native = None
            execution = None
            phase_controllers: dict[Any, Any] = {}
            phase_definitions: dict[tuple[Any, tuple[str, ...]], tuple[str, tuple[Any, ...]]] = {}
            request = _REQUEST.get()
            phase_type = getattr(varena, "PhaseController", None)
            try:
                # Every program model is claimed before model-wide phase identities are
                # snapshotted. Unrelated declared siblings are not decorated or borrowed.
                for stage in self.stages:
                    for binding in stage._owned_bindings:
                        if binding.model not in claimed:
                            binding.model._claim()
                            claimed.append(binding.model)
                for stage_index, stage in enumerate(self.stages):
                    stage._validate()  # claims now fence concurrent model identity changes
                    if self._repeated(stage_index):
                        capsule, _ = stage._native.continuation(producer, reusable=True)
                    else:
                        capsule, _ = stage._native.continuation(producer)
                    if stage._load is not None:
                        capsule, observation = stage._load.continuation(
                            capsule, event_ordered=stage._load_ordering == "events"
                        )
                        reads.append((stage_index, observation, stage._owned_bindings))
                    if callable(phase_type) and request is not None and stage._owned_bindings:
                        memories = {binding.model.memory for binding in stage._owned_bindings}
                        if len(memories) == 1:
                            memory = next(iter(memories))
                            with memory.lock:
                                exclusive = (
                                    request.memory is memory
                                    and not request.idle
                                    and not memory._active_beside(request)
                                )
                                names = tuple(
                                    sorted(
                                        set().union(
                                            *(
                                                request.phase(binding.model.name)
                                                for binding in stage._owned_bindings
                                            )
                                        )
                                    )
                                )
                                selected = [
                                    model
                                    for model in claimed
                                    if model.memory is memory and model.name in names
                                ]
                                if exclusive and memory.native is not None:
                                    definition = phase_definitions.get((memory, names))
                                    if definition is None:
                                        regions = tuple(
                                            {
                                                id(region): region
                                                for model in selected
                                                for block in model.groups()
                                                for region in (block.region, block.final_region)
                                                if region is not None
                                            }.values()
                                        )
                                        # Model source/fill/generation identity belongs to this
                                        # submission. Actual readiness is inspected natively
                                        # at phase entry, never guessed from this snapshot.
                                        identity = repr(
                                            tuple(
                                                sorted(
                                                    (
                                                        model.name,
                                                        model.wanted_identity(),
                                                        model.generation,
                                                    )
                                                    for model in selected
                                                )
                                            )
                                        )
                                        # Only the compact digest crosses the capability and
                                        # progress boundaries; complete immutable identities
                                        # can contain large per-weight route tables.
                                        digest = hashlib.sha256(identity.encode()).hexdigest()
                                        label = "+".join(
                                            sorted(model.component for model in selected)
                                        )[:128]
                                        key = f"{label}:{digest}"
                                        definition = key, regions
                                        phase_definitions[(memory, names)] = definition
                            if exclusive and memory.native is not None:
                                controller = phase_controllers.get(memory)
                                if controller is None:
                                    controller = phase_type(memory.native)
                                    phase_controllers[memory] = controller
                                capsule = controller.wrap(capsule, *definition)
                    capsules.append(capsule)
                native = (
                    _execution.NativeSequence(capsules, self.device)
                    if self._repeat is None
                    else _execution.NativeSequence(capsules, self.device, self._repeat)
                )
                execution = _SequenceExecution(
                    self, context, outputs, readout, claimed, native, reads, phase_controllers
                )
                self._native, self._active = native, execution
                for stage in self.stages:
                    stage._active = execution
                # Every observer/capsule is strongly owned before optional Weak lookahead
                # links are installed. Admission allocates no source bytes and launches no work.
                execution._prepare_lookahead()
                # All Python/native owners exist before the sole native worker can start.
                native.start()
                execution._offer_queued()
                return execution
            except BaseException as error:
                if native is not None:
                    try:
                        native.close_owner()
                    except Exception as cleanup:
                        raise RuntimeFailure(
                            "native sequence submission completion is unproven; retire executor",
                            code="device_fault",
                        ) from cleanup
                else:
                    # No sequence worker exists. Each capsule's provider owns abandonment of
                    # its still-armed epoch; closing definitions checks all caller streams.
                    self._closing = True
                    try:
                        for stage in self.stages:
                            stage.close()
                        self._closed = True
                    except Exception as cleanup:
                        raise RuntimeFailure(
                            "native sequence preparation cleanup is unproven; retire executor",
                            code="device_fault",
                        ) from cleanup
                    finally:
                        self._closing = False
                if execution is not None:
                    execution._release_models()
                else:
                    while claimed:
                        claimed[-1]._release()
                        claimed.pop()
                if isinstance(error, Exception):
                    _native_error(error)
                raise

    def _retire_if_idle(self) -> None:
        with self._submission:
            if (
                self._active is not None
                and self._active._finished is None
                and not self._active.poll().done
            ):
                raise RuntimeError("cannot retire a sequence while GPU execution is active")
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        with self._submission, self._locked_stages():
            if self._closed:
                return
            error = None
            if self._active is not None:
                if self._active._finished is None:
                    self._active.cancel()
                try:
                    self._active._join()
                except Exception as failure:
                    if self._active._finished is None:
                        raise
                    error = failure
            if self._native is not None:
                try:
                    self._native.close_owner()
                except Exception as failure:
                    _native_error(failure)
            self._closing = True
            try:
                for stage in self.stages:
                    stage.close()
                if self._capture_domain is not None:
                    self._capture_domain.finish()
                    self._capture_domain = None
                self._closed = True
            finally:
                self._closing = False
            # The last execution refers back to this sequence: dropped here, its load
            # observations and phase controllers go when its request settles, not at a GC.
            self._active = self._native = None
            for stage in self.stages:
                stage._active = None
            if error is not None:
                _native_error(error)

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


class _SequenceCapture:
    """Preparation-only ownership; uses the existing sequence for every execution."""

    def __init__(self) -> None:
        self.stages: list[PreparedRepeat] = []
        self._domain: Any = None
        self._pool: Any = None
        self._areas: list[tuple[Any, int, int]] = []
        self._submission = threading.RLock()
        self._closing = False
        self._entered = self._finished = self._closed = self._capturing = False

    def __enter__(self) -> _SequenceCapture:
        if self._entered or self._closed:
            raise RuntimeError("sequence recording is one-shot")
        self._entered = True
        return self

    def __exit__(self, *_: Any) -> None:
        if not self._finished:
            self.close()

    def _external(self, tensor: Any) -> None:
        storage = tensor.untyped_storage()
        pointer, size = int(storage.data_ptr()), int(storage.nbytes())
        if size and any(
            tensor.device == device and pointer < end and start < pointer + size
            for device, start, end in self._areas
        ):
            raise CapabilityError(
                "sequence boundaries must not alias captured temporary storage",
                code="prepared_graph_ineligible",
            )

    def _prepare(self, state: tuple[Any, ...]) -> None:
        if not self._entered or self._closed or self._finished or not state:
            raise RuntimeError("capture inside an open sequence recording")
        for tensor in state:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                raise ValueError("sequence state must contain CUDA tensors")
            self._external(tensor)
        device = state[0].device.index
        if device is None or any(t.device.index != device for t in state):
            raise ValueError("sequence capture belongs to one device")
        if self._domain is None:
            self._domain = domain.Domain(torch, None)
            if self._domain.device != device:
                raise ValueError("sequence capture must use the current request device")
            try:
                self._domain.initialize()
                self._domain.prime_blas()
            except Building as error:
                raise CapabilityError(
                    f"capture waits for kernel readiness: {error}",
                    code="prepared_graph_ineligible",
                ) from error
            self._pool = domain.Pool(torch.cuda.graph_pool_handle())
        elif self._domain.device != device:
            raise ValueError("sequence captures cannot span devices")
        if why := self._domain.refusal():
            raise CapabilityError(why, code="prepared_graph_ineligible")

    def finish_parts(
        self, *, outputs: Sequence[Any], parts: Sequence[tuple[int, int, int]]
    ) -> tuple[PreparedSequence, ...]:
        """The recorded stages as sequences, one per (begin, end, passes) slice, each repeating
        its slice natively. They share this recording's pool and domain, which end with the
        last of them."""
        with self._submission, ExitStack() as locks:
            if not self._entered or self._closed or self._finished:
                raise RuntimeError("sequence recording is not open")
            for stage in sorted(self.stages, key=id):
                locks.enter_context(stage._submission)
            owner = weakref.ref(self)
            for stage in self.stages:
                if stage._sequence_owner is None or stage._sequence_owner() is not self:
                    raise RuntimeError("sequence recording lost a captured definition")
                stage._sequence_owner = None
            made: list[PreparedSequence] = []
            try:
                for begin, end, passes in parts:
                    made.append(
                        PreparedSequence(
                            self.stages[begin:end],
                            outputs=outputs,
                            repeat=(0, end - begin, passes) if passes > 1 else None,
                        )
                    )
            except BaseException:
                for sequence in made:  # transferred stages return to the recording
                    for stage in sequence.stages:
                        stage._sequence_owner = None
                    sequence._closed = True
                for stage in self.stages:
                    stage._sequence_owner = owner
                raise
            shared = _SharedDomain(self._domain, len(made))
            for sequence in made:
                sequence._capture_domain = shared
            self._domain = None
            self._finished = True
            return tuple(made)

    @contextmanager
    def capturing(self) -> Iterator[None]:
        """Stages are being recorded: residency cannot retire the recording meanwhile."""
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False

    def _retire_if_idle(self) -> None:
        if self._capturing:
            raise RuntimeError("cannot retire a sequence during stage preparation")
        self.close()

    def close(self) -> None:
        with self._submission:
            if self._closed or self._finished:
                return
            if self._capturing:
                raise RuntimeError("cannot close a sequence during stage preparation")
            self._closing = True
            try:
                for stage in self.stages:
                    stage.close()
                if self._domain is not None:
                    self._domain.finish()
                self._closed = True
            finally:
                self._closing = False

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


class _SharedDomain:
    """One recording's domain, finished when the last sequence made from it closes."""

    def __init__(self, domain: Any, users: int) -> None:
        self.domain, self.users = domain, users

    def finish(self) -> None:
        self.users -= 1
        if self.users == 0 and self.domain is not None:
            self.domain.finish()
            self.domain = None


class _SequenceExecution(Execution):
    def __init__(self, plan, context, outputs, readout, models, native, reads, phases=None):
        super().__init__(plan, context, 0, outputs, readout, models, native)
        self._stage_reads = reads
        # Only all-resident sequences have already retained every future stage workspace.
        # Source-backed stages keep their existing separate current-stage lookahead policy.
        self._queued_eligible = context is not None and not reads
        self._phase_controllers = phases or {}
        self._prefetch_cleanup = ExitStack()
        self._prefetch_admission: list[dict[str, object]] = []

    def _prepare_lookahead(self) -> None:
        if not self.plan.prefetch:
            self._prefetch_admission.append({"state": "disabled_by_author"})
            return
        request = self._load_request
        candidates: dict[Any, list[tuple[Any, ...]]] = {}
        for current, following in zip(self._stage_reads, self._stage_reads[1:], strict=False):
            index, job, bindings = current
            next_index, next_job, next_bindings = following
            # Resident stages between two reading stages run while the next one's bytes arrive.
            repeat = self.plan._repeat
            if repeat is not None and repeat[2] > 1 and next_index == repeat[1]:
                continue  # the suffix is next only on the final pass
            if (
                not bindings
                or not next_bindings
                or not getattr(job, "supports_native_lookahead", False)
                or not getattr(next_job, "supports_native_lookahead", False)
                or not callable(getattr(job, "link_next", None))
            ):
                continue
            memories = {binding.model.memory for binding in (*bindings, *next_bindings)}
            if len(memories) != 1:
                continue
            memory = next(iter(memories))
            regions = {
                id(region): region
                for binding in (*bindings, *next_bindings)
                for _, region in binding._stage_regions
            }
            current_bytes = sum(
                int(region.allocated_bytes)
                for binding in bindings
                for _, region in binding._stage_regions
            )
            next_bytes = sum(
                int(region.allocated_bytes)
                for binding in next_bindings
                for _, region in binding._stage_regions
            )
            # Native reports the same actual reservations. Refuse skew or hidden inputs;
            # do not use Python logical tensor sizes as allocation admission.
            if (
                not regions
                or int(job.reservation_bytes) != current_bytes
                or int(next_job.reservation_bytes) != next_bytes
            ):
                continue
            required = sum(int(region.allocated_bytes) for region in regions.values())
            missing = sum(
                int(region.allocated_bytes) for region in regions.values() if not region.resident
            )
            candidates.setdefault(memory, []).append(
                (index, next_index, job, next_job, required, missing, next_bytes)
            )
        for memory, pairs in candidates.items():
            required = max(pair[4] for pair in pairs)
            missing = max(pair[5] for pair in pairs)
            restore = memory._stage_prefetch_window(required, missing, request, self)
            if restore is None:
                self._prefetch_admission.append({"state": "capacity_skip", "pair_bytes": required})
                continue
            self._prefetch_cleanup.callback(restore)
            for index, next_index, job, next_job, pair_bytes, _, next_bytes in pairs:
                linked = bool(job.link_next(next_job, max_prefetch_bytes=next_bytes))
                self._prefetch_admission.append(
                    {
                        "from_stage": index,
                        "to_stage": next_index,
                        "pair_bytes": pair_bytes,
                        "state": "linked" if linked else "native_refusal",
                    }
                )

    def _release_models(self) -> None:
        # Base execution calls this only after checked native completion/readout. If the
        # drain is unproven, owners and this window lease remain retained for retirement.
        self._prefetch_cleanup.close()
        # Native phase entry changed preferences without a Python callback. After every
        # source/prefetch/consumer is checked, invalidate only our stale eager-policy cache;
        # the next ordinary phase recomputes it. Preserve another live request's policy.
        for memory in self._phase_controllers:
            with memory.lock:
                request = self._load_request
                if (
                    request is not None
                    and not memory._active_beside(request)
                    and memory._phase_request == request.id
                ):
                    memory._phase_names = frozenset()
                    memory._phase_request = None
        super()._release_models()

    def facts(self) -> dict[str, object]:
        facts = self.plan.facts()
        facts["sequence"] = dict(self._load_job.facts())
        facts["stage_prefetch_admission"] = list(self._prefetch_admission)
        facts["native_phases"] = [
            dict(controller.facts()) for controller in self._phase_controllers.values()
        ]
        facts["stage_inputs"] = [
            {"stage": index, **dict(job.facts())} for index, job, _ in self._stage_reads
        ]
        return facts

    def _record_load(self, facts: Mapping[str, Any]) -> None:
        if self._load_recorded or not facts.get("done", False):
            return
        for _, job, bindings in self._stage_reads:
            loaded = job.facts()
            if self._load_request is not None and (
                loaded.get("read_done", False) or getattr(job, "supports_native_lookahead", False)
            ):
                # Lookahead jobs expose complete padded ledgers after aggregate done,
                # including prefetched bytes when demand was canceled before starting.
                blocks = [block for binding in bindings for block in binding.groups]
                if blocks:
                    prefetched = loaded.get("prefetch_input_read_bytes", [0] * len(blocks))
                    for block, nbytes, speculative in zip(
                        blocks, loaded["input_read_bytes"], prefetched, strict=True
                    ):
                        nbytes, speculative = int(nbytes), int(speculative)
                        if not 0 <= speculative <= nbytes:
                            raise RuntimeError(
                                "native prefetch byte accounting exceeds total input reads"
                            )
                        if nbytes > speculative:
                            self._load_request.read(block, 0, nbytes - speculative)
                        if speculative:
                            self._load_request.read(block, 0, speculative, speculative=True)
        self._load_recorded = True
