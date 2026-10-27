"""`author.denoise_loop` in the Runtime: eager, one resident captured program, or streamed stages.

The first step always runs eagerly, for real: it warms kernels and brings the weights in through
ordinary residency, on the caller's stream (only recorded regions run on the recording's streams,
which end with the program). Then the loop runs resident when
every block of its components is still on the device, else as streamed stages cut at the
residency's own block boundaries, else eagerly. A refusal before submission drops that attempt
and the next mode continues from the next step; nothing runs twice. After submission nothing
falls back. A keyed program stays for its call site until a new key replaces it or residency
retires it under pressure; any other program closes before the loop returns.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
import weakref
from collections.abc import Callable, Hashable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from typing import Any

import torch
from tensorfs import varena
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from cozy_runtime import graphs
from cozy_runtime.author import _loop
from cozy_runtime.author._context import CURRENT, Context
from cozy_runtime.author._errors import CapabilityError, Fault, OutputError, fault
from cozy_runtime.author._model import _LOCAL_SCOPES, LANDED, PLACED, Building
from cozy_runtime.compute import (
    PreparedRepeat,
    PreparedSequence,
    _native_error,
    _SequenceCapture,
    _zero_dropout_attention,
)
from cozy_runtime.internal import kernel_ready
from cozy_runtime.internal.residency import (
    _CAPTURE_BLOCKS,
    _REQUEST,
    CUTTER,
    Binding,
    Model,
    ResidencyError,
)

_execution: Any
try:
    from cozy_runtime._kernels import _execution
except ModuleNotFoundError as missing:  # CPU and macOS wheels carry no native execution module
    if missing.name != "cozy_runtime._kernels._execution":
        raise
    _execution = None

#: The CUDA driver, for a recorded graph's node count (absent without an NVIDIA driver).
_DRIVER: Any
try:
    _DRIVER = ctypes.CDLL("libcuda.so.1")
except OSError:
    _DRIVER = None

#: Codes a preparation refusal carries: the loop continues in the next mode.
_REFUSED = (
    "prepared_graph_ineligible",
    "prepared_load_unavailable",
    "prepared_load_events_unavailable",
)
#: cudaError_t stream-capture codes (900-908): an operation the step runs cannot be recorded.
_CAPTURE_ERRORS = range(900, 909)
#: request id -> its loops, for the run record (`execution.loops`)
ROWS: dict[str, list[dict[str, Any]]] = {}


def _call(step: Callable[..., object], index: Any, tensors: tuple[Any, ...], phase: Any) -> None:
    if phase is _NONE:
        step(index, *tensors)
    else:
        step(index, *tensors, phase=phase)


_NONE = object()


def _refusal(error: BaseException) -> str | None:
    """The reason a preparation error lets the loop continue in another mode, else None."""
    if isinstance(error, CapabilityError):
        return str(error) if error.code in _REFUSED else None
    if isinstance(error, Building):
        return f"capture waits for kernel readiness: {error}"
    if isinstance(error, torch.OutOfMemoryError):
        return f"preparation exceeded capacity: {error}"
    if isinstance(error, ResidencyError):
        return f"residency refused preparation: {error}"
    code = getattr(error, "error_code", None)
    if isinstance(code, int) and code in _CAPTURE_ERRORS:
        return f"the step cannot be recorded: {error}"
    if isinstance(error, Exception) and fault(error) is Fault.OTHER:
        return f"recording failed: {type(error).__name__}: {error}"
    return None


@dataclass
class _Program:
    """A recorded native loop and the tensors its graphs read."""

    sequence: PreparedSequence
    mode: str
    tensors: tuple[Any, ...]
    counter: Any
    count: int
    ident: Hashable
    stages: int
    blocks: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: a kept program's cache handle: residency releases it under pressure while it is idle
    region: Any = None

    def valid(self) -> bool:
        return self.sequence.valid()

    def close(self) -> None:
        self.sequence.close()

    def release(self) -> None:
        if self.region is None:
            self.close()
        else:
            self.region.close()  # calls close unless a reclaim already did


class Loops:
    """The installed `author._loop.Loops`: one per executor process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kept: dict[Hashable, _Program] = {}

    def run(
        self,
        step: Callable[..., object],
        steps: range,
        state: tuple[Any, ...],
        carried: tuple[Any, ...],
        inputs: tuple[Any, ...],
        phases: tuple[Hashable, ...] | None,
        key: Hashable | None,
        on_step: Callable[[int], object] | None,
    ) -> None:
        ctx = CURRENT.get()
        row: dict[str, Any] = {
            "steps": len(steps),
            "mode": "eager",
            "native_steps": 0,
            "reused": False,
            "refusals": [],
        }
        request = _REQUEST.get()
        if request is not None:
            ROWS.setdefault(request.id, []).append(row)
        mutable = state + carried
        with torch.inference_mode(False), torch.no_grad(), _owned(mutable) as work:
            tensors = work + inputs
            models, why = _eligible(ctx, steps, tensors)
            if why:
                row["refusals"].append(why)
                _eager(step, steps, tensors, phases, steps, on_step, ctx)
                return
            assert ctx is not None
            site = (step.__code__, tuple(id(model) for model in models))
            ident = None if key is None else (key, _layouts(tensors), len(steps), phases)
            with self._lock:
                kept = self._kept.get(site)
                if kept is not None and (ident is None or kept.ident != ident or not kept.valid()):
                    del self._kept[site]
                    _close_idle(kept)
                    kept = None
            if kept is not None:
                with _taken(kept.lock, ctx), _pinned(kept) as pinned:
                    if pinned and kept.valid():
                        self._reuse(kept, step, steps, tensors, work, phases, on_step, ctx, row)
                        return
            program = None
            try:
                program = self._record(
                    step, steps, tensors, work, phases, models, key, ctx, row, ident
                )
                if on_step is not None:
                    on_step(steps[0])
                if program is None:
                    _eager(step, steps[1:], tensors, phases, steps, on_step, ctx)
                    return
                _submit(program, steps[1:], on_step, ctx, row)
                for value, slot in zip(work, program.tensors, strict=False):
                    if value is not slot:
                        value.copy_(slot)
            except BaseException:
                if program is not None:
                    program.close()
                raise
            if key is None:
                program.close()
            else:
                program.region = models[0].memory.region(
                    "denoise_loop", "program", release=program.close
                )
                with self._lock:
                    previous = self._kept.pop(site, None)
                    self._kept[site] = program
                if previous is not None:
                    _close_idle(previous)

    def _reuse(
        self,
        kept: _Program,
        step: Callable[..., object],
        steps: range,
        tensors: tuple[Any, ...],
        work: tuple[Any, ...],
        phases: tuple[Hashable, ...] | None,
        on_step: Callable[[int], object] | None,
        ctx: Context,
        row: dict[str, Any],
    ) -> None:
        """A kept program: this call's values into its tensors, its first step eagerly, the
        rest natively, and the state back out."""
        row.update(mode=kept.mode, reused=True, stages=kept.stages, blocks=kept.blocks)
        try:
            for slot, value in zip(kept.tensors, tensors, strict=True):
                slot.copy_(value)
            _eager(step, steps[:1], kept.tensors, phases, steps, on_step, ctx)
            _submit(kept, steps[1:], on_step, ctx, row)
            for value, slot in zip(work, kept.tensors, strict=False):
                value.copy_(slot)
        except BaseException:
            kept.close()
            raise

    def _record(
        self,
        step: Callable[..., object],
        steps: range,
        tensors: tuple[Any, ...],
        work: tuple[Any, ...],
        phases: tuple[Hashable, ...] | None,
        models: tuple[Model, ...],
        key: Hashable | None,
        ctx: Context,
        row: dict[str, Any],
        ident: Hashable,
    ) -> _Program | None:
        """Record the loop after its eager first step; None when every mode refused."""
        device = tensors[0].device
        if key is not None:  # the program owns copies; this call's values go in and out
            tensors = tuple(value.clone() for value in tensors)
        mutable = tensors[: len(work)]
        counter = torch.zeros(1, dtype=torch.int64, device=device)
        recording = PreparedSequence.record()
        recording.__enter__()
        try:
            try:
                recording._prepare((*mutable, counter))
            except BaseException as error:
                if (why := _refusal(error)) is None:
                    raise
                row["refusals"].append(f"recording: {why}")
                _eager(step, steps[:1], tensors, phases, steps, None, ctx)
                _back(work, tensors)
                return None
            # The caller's stream: residency records its consumers on the current stream, and
            # the recording's own streams end with the program.
            _eager(step, steps[:1], tensors, phases, steps, None, ctx)
            torch.cuda.empty_cache()
            rest = steps[1:]
            began = time.perf_counter()
            resident = all(not _unresident(model) for model in models)
            row["resident_after_first_step"] = resident
            attempts = (_record_resident, _record_streamed) if resident else (_record_streamed,)
            for attempt in attempts:
                ctx.raise_if_cancelled()
                try:
                    with recording.capturing():
                        mode, stages, blocks = attempt(
                            recording, step, rest, tensors, mutable, counter, phases, steps, models
                        )
                    repeat = None if mode == "resident" else (0, len(recording.stages), len(rest))
                    sequence = recording.finish(outputs=mutable, repeat=repeat)
                except BaseException as error:
                    if (why := _refusal(error)) is None:
                        raise
                    row["refusals"].append(f"{attempt.__name__.removeprefix('_record_')}: {why}")
                    _drop_stages(recording)
                    torch.cuda.empty_cache()
                    continue
                row.update(
                    mode=mode,
                    stages=stages,
                    blocks=blocks,
                    capture_ms=round((time.perf_counter() - began) * 1000, 3),
                )
                return _Program(sequence, mode, tensors, counter, len(rest), ident, stages, blocks)
            _back(work, tensors)  # the eager rest runs on the caller's tensors
            return None
        finally:
            if not recording._finished:
                recording.close()


def _back(work: tuple[Any, ...], tensors: tuple[Any, ...]) -> None:
    """The program's copies' values into the caller's tensors (a no-op for the same tensors)."""
    for value, slot in zip(work, tensors, strict=False):
        if value is not slot:
            value.copy_(slot)


@contextmanager
def _owned(mutable: tuple[Any, ...]) -> Iterator[tuple[Any, ...]]:
    """Ordinary tensors for the loop to advance; an inference-mode one gets a copy and its
    final values back."""
    work = tuple(t.clone() if t.is_inference() else t for t in mutable)
    yield work
    with torch.inference_mode():
        for original, value in zip(mutable, work, strict=True):
            if value is not original:
                original.copy_(value)


@contextmanager
def _taken(lock: threading.Lock, ctx: Context) -> Iterator[None]:
    while not lock.acquire(timeout=0.05):
        ctx.raise_if_cancelled()
    try:
        yield
    finally:
        lock.release()


def _close_idle(program: _Program) -> None:
    with program.lock:
        program.release()


@contextmanager
def _pinned(program: _Program) -> Iterator[bool]:
    """Whether the kept program is still there; held against reclaim while it runs."""
    try:
        pin = program.region.pin()
        pin.__enter__()
    except ResidencyError:  # closed, or a reclaim is releasing it now
        yield False
        return
    try:
        yield True
    finally:
        pin.__exit__(None, None, None)


def _layouts(tensors: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors)


def _eligible(
    ctx: Context | None, steps: range, tensors: tuple[Any, ...]
) -> tuple[tuple[Model, ...], str]:
    """The step's placed components, or why the loop runs eagerly."""
    if ctx is None:
        return (), "outside a request"
    if ctx.device.type != "cuda" or any(not t.is_cuda for t in tensors):
        return (), "not on CUDA"
    if _execution is None:
        return (), "this Runtime has no native execution module"
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        return (), "several ranks"
    if len(steps) < 2:
        return (), "one step"
    scopes = _LOCAL_SCOPES.get()
    if not scopes:
        return (), "outside a @uses_components scope"
    owner, _, components = scopes[-1]
    placement: Any = owner._cozy_residency
    models = tuple(
        placement.models[name]
        for name in components
        if placement is not None and name in placement.models
    )
    if not models:
        return (), "the scope's components are not placed on this device"
    return models, ""


def _eager(
    step: Callable[..., object],
    indices: range,
    tensors: tuple[Any, ...],
    phases: tuple[Hashable, ...] | None,
    steps: range,
    on_step: Callable[[int], object] | None,
    ctx: Context | None,
) -> None:
    if not len(indices):
        return
    index = torch.zeros(1, dtype=torch.int64, device=tensors[0].device)
    for i in indices:
        if ctx is not None:
            ctx.raise_if_cancelled()
        index.fill_(i)
        _call(step, index, tensors, _NONE if phases is None else phases[i - steps.start])
        if on_step is not None:
            on_step(i)


def _unresident(model: Model) -> str:
    binding = model.retain(("",))
    try:
        return binding.unresident()
    finally:
        binding.close()


def _drop_stages(recording: _SequenceCapture) -> None:
    """A refused attempt's stages go; the recording's domain and pool stay for the next."""
    with recording._submission:
        stages, recording.stages = recording.stages, []
        recording._areas.clear()
    for stage in stages:
        with stage._submission:
            stage._sequence_owner = None
        stage.close()


def _submit(
    program: _Program,
    rest: range,
    on_step: Callable[[int], object] | None,
    ctx: Context,
    row: dict[str, Any],
) -> None:
    """One native submission of the program's steps; no fallback once it is accepted."""
    with torch.no_grad():
        program.counter.fill_(rest.start)
    sequence = program.sequence
    execution: Any = (
        sequence.submit(ctx) if ctx._submit_prepared is None else ctx._submit_prepared(sequence)
    )
    streamed = program.mode == "streamed"
    reported = 0
    while True:
        if ctx.cancelled:
            execution.cancel()
        progress = execution.poll()
        done = (
            int(execution.facts()["sequence"]["completed_passes"])
            if streamed
            else int(progress.completed_steps)
        )
        if on_step is not None:
            for i in rest[reported:done]:
                on_step(i)
        reported = max(reported, done)
        if progress.done:
            break
        time.sleep(0.01)
    execution.wait()
    ctx.raise_if_cancelled()
    facts = execution.facts()
    done = int(facts["sequence"]["completed_passes"]) if streamed else int(progress.completed_steps)
    if progress.canceled or done != len(rest):
        raise OutputError(
            f"the native loop stopped after {done} of {len(rest)} steps", code="incomplete_denoise"
        )
    if on_step is not None:
        for i in rest[reported:]:
            on_step(i)
    row["native_steps"] = row.get("native_steps", 0) + done
    row["host_graph_launches"] = int(facts["host_graph_launches"])
    reads = [int(stage.get("read_bytes", 0)) for stage in facts.get("stage_inputs", ())]
    if reads:
        row["streamed_bytes"] = row.get("streamed_bytes", 0) + sum(reads)


# ------------------------------------------------------------------ recording


def _record_resident(
    recording: _SequenceCapture,
    step: Callable[..., object],
    rest: range,
    tensors: tuple[Any, ...],
    mutable: tuple[Any, ...],
    counter: Any,
    phases: tuple[Hashable, ...] | None,
    steps: range,
    models: tuple[Model, ...],
) -> tuple[str, int, int]:
    """One WHILE graph per run of equal phases, every component resident and held."""
    runs = _runs(rest, phases, steps)
    bindings: list[Binding] = []
    try:
        with ExitStack() as preparing:
            idle: set[int] = set()
            for model in models:
                _checked(model)
                binding = model.retain(("",))
                bindings.append(binding)
                if id(model.memory) not in idle:
                    preparing.enter_context(model.memory.request(graphs.IDLE_REQUEST, idle=True))
                    idle.add(id(model.memory))
                if why := binding.unresident():
                    raise CapabilityError(
                        f"capture requires the component's weights to be resident ({why})",
                        code="prepared_graph_ineligible",
                    )
                preparing.enter_context(binding.prepare())
            landed = LANDED[0]
            plans = []
            for _, count, phase in runs:

                def body(phase: Any = phase) -> None:
                    _call(step, counter, tensors, phase)
                    counter.add_(1)

                with _guarded(models, frozenset()):
                    graph = _Graph(recording, (*mutable, counter), count, tuple(bindings))
                    graph.record(body)
                plan = graph.plan(landed)
                _own(plan, tuple(bindings), recording)
                plans.append(plan)
    except BaseException:
        for binding in bindings:
            binding.close()
        raise
    for binding in bindings:  # the last stage closes the bindings every stage reads
        plans[-1]._binding_cleanup.callback(binding.close)
    return "resident", len(runs), 0


def _record_streamed(
    recording: _SequenceCapture,
    step: Callable[..., object],
    rest: range,
    tensors: tuple[Any, ...],
    mutable: tuple[Any, ...],
    counter: Any,
    phases: tuple[Hashable, ...] | None,
    steps: range,
    models: tuple[Model, ...],
) -> tuple[str, int, int]:
    """One pass cut into stages at the component's block boundaries, repeated natively."""
    runs = _runs(rest, phases, steps)
    if len(runs) != 1:
        raise CapabilityError(
            "streamed native runs one phase", code="prepared_graph_ineligible"
        )
    if len(models) != 1:
        raise CapabilityError(
            "streamed native runs one component", code="prepared_graph_ineligible"
        )
    (_, _, phase), (model,) = runs[0], models
    if not model.blocks:
        raise CapabilityError(
            f"{model.component} has no blocks to stream", code="prepared_graph_ineligible"
        )

    def body() -> None:
        _call(step, counter, tensors, phase)
        counter.add_(1)

    cutter = _Cutter(recording, model, (*mutable, counter))
    cutter.record(body)
    return "streamed", len(recording.stages), cutter.blocks


def _runs(
    rest: range, phases: tuple[Hashable, ...] | None, steps: range
) -> list[tuple[int, int, Any]]:
    """(first index, count, phase) of each run of equal phases in `rest`."""
    if phases is None:
        return [(rest.start, len(rest), _NONE)]
    runs: list[tuple[int, int, Any]] = []
    for i in rest:
        phase = phases[i - steps.start]
        if runs and runs[-1][2] == phase:
            first, count, _ = runs[-1]
            runs[-1] = (first, count + 1, phase)
        else:
            runs.append((i, 1, phase))
    return runs


def _checked(model: Model) -> None:
    placed = PLACED.get(model.module)
    if placed is not None and placed[0].lora_choices((placed[1],))[placed[1]][0] == "per_step":
        raise CapabilityError(
            "native loops need a fixed baked LoRA representation", code="prepared_graph_ineligible"
        )
    if why := graphs._hooked(model.module):
        raise CapabilityError(why, code="prepared_graph_ineligible")
    if not model.memory.cuda:
        raise CapabilityError("native loops need CUDA weights", code="prepared_graph_ineligible")


@contextmanager
def _guarded(models: tuple[Model, ...], immutable: frozenset[int]) -> Iterator[None]:
    """Recording leaves the modules' state as it found it, original operations recorded."""
    bypass = graphs.BYPASS.set(True)
    snapshots = [graphs._State(torch, m.module, immutable_buffers=immutable) for m in models]
    try:
        yield
    finally:
        changed = [change for snapshot in snapshots if (change := snapshot.restore())]
        graphs.BYPASS.reset(bypass)
    if changed:
        raise CapabilityError(
            "recording changed the model's state: " + ", ".join(changed),
            code="prepared_graph_ineligible",
        )


def _own(
    plan: PreparedRepeat,
    bindings: tuple[Binding, ...],
    recording: _SequenceCapture,
    cleanup: ExitStack | None = None,
) -> None:
    """The plan owns its bindings; residency retires it before releasing their weights."""
    plan._owned_bindings = bindings
    for binding in bindings:
        module = binding.model.module
        graphs.own_prepared(module, plan)
        if binding.model.retire_graphs is None:
            root = weakref.ref(module)

            def retire(root: Any = root) -> None:
                if (found := root()) is not None:
                    graphs.retire(torch, found)

            binding.model.retire_graphs = retire
    if cleanup is not None:
        plan._binding_cleanup = cleanup
    with recording._submission:
        plan._sequence_owner = weakref.ref(recording)
        recording.stages.append(plan)


class _Graph:
    """One CUDA graph recorded between begin() and end() into the recording's shared pool.

    Every tensor the recorded operations read that they did not produce is retained by the
    native owner, except the loaded inputs of a streamed stage and the scratch of the same pass,
    which the pool and the replay order keep valid.
    """

    def __init__(
        self,
        recording: _SequenceCapture,
        state: tuple[Any, ...],
        steps: int,
        bindings: tuple[Binding, ...],
        *,
        loaded: tuple[Any, ...] = (),
        scratch: list[tuple[Any, int, int]] | None = None,
        unbound: frozenset[int] = frozenset(),
        undeclared: tuple[tuple[Any, int, int], ...] = (),
    ) -> None:
        self.recording, self.state, self.steps, self.bindings = recording, state, steps, bindings
        self.device = state[0].device
        self.domain = recording._domain
        self.ranges = tuple(
            dict.fromkeys(
                (t.device, int(s.data_ptr()), int(s.data_ptr()) + s.nbytes())
                for t in loaded
                if (s := t.untyped_storage()).nbytes()
            )
        )
        self.loaded = bool(loaded)
        self.scratch = [] if scratch is None else scratch
        self.unbound, self.undeclared = unbound, undeclared
        self.held: dict[int, Any] = {id(t): t for t in state}
        self.produced: dict[int, Any] = {}
        self.generated: dict[int, tuple[Any, Any, Any, int, int]] = {}
        self.ops = 0
        self.graph: Any = None
        self.index: Any = None
        self._contexts = ExitStack()
        self._mode: Any = None
        self._caller: Any = None

    def _overlaps(self, tensor: Any, areas: Any) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        storage = tensor.untyped_storage()
        first, size = int(storage.data_ptr()), int(storage.nbytes())
        return bool(size) and any(
            tensor.device == device and first < end and begin < first + size
            for device, begin, end in areas
        )

    def begin(self) -> None:
        """Capture on the recording's library stream; only the recorded region runs there."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("a loop stage begins outside capture")
        stream, control = self.domain.stream, self.domain.control
        self._caller = torch.cuda.current_stream(self.device)
        self.index = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.held[id(self.index)] = self.index
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        context = CURRENT.get()
        pause = context._pause_stall_watch if context is not None else None
        try:
            self._contexts.enter_context(pause() if pause is not None else nullcontext())
            self._contexts.enter_context(kernel_ready.recording())
            self._contexts.enter_context(varena.capture(self.device.index))
            stream.wait_stream(self._caller)
            control.wait_stream(stream)
            torch.cuda.set_stream(control)
            try:
                self.graph.capture_begin(
                    pool=self.recording._pool.handle, capture_error_mode="global"
                )
            finally:
                torch.cuda.set_stream(stream)
        except BaseException:
            torch.cuda.set_stream(self._caller)
            self._contexts.close()
            self.graph = None
            raise
        self.domain.fork.record(control)
        stream.wait_event(self.domain.fork)
        self._mode = _Owners(self)
        self._mode.__enter__()

    def end(self) -> None:
        stream, control = self.domain.stream, self.domain.control
        if self.graph is None:
            return
        try:
            if self._mode is not None:
                mode, self._mode = self._mode, None
                mode.__exit__(None, None, None)
        finally:
            try:
                self.domain.join.record(stream)
                control.wait_event(self.domain.join)
                torch.cuda.set_stream(control)
                try:
                    self.graph.capture_end()
                finally:
                    torch.cuda.set_stream(self._caller)
            finally:
                self._contexts.close()
        control.synchronize()
        self.domain.graphs.add(self.graph)

    def abort(self) -> None:
        """After a failure inside the recorded region: end the capture, keep nothing."""
        with suppress(BaseException):
            self.end()
        self.reset()

    def nodes(self) -> int:
        """The recorded graph's node count: what it launches, whatever launched it."""
        count = ctypes.c_size_t(0)
        if _DRIVER.cuGraphGetNodes(
            ctypes.c_void_p(self.graph.raw_cuda_graph()), None, ctypes.byref(count)
        ):
            raise RuntimeError("cuGraphGetNodes failed")
        return int(count.value)

    def reset(self) -> None:
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
        self.held.clear()

    def record(self, body: Callable[[], object]) -> None:
        self.begin()
        try:
            body()
        except BaseException:
            self.abort()
            raise
        self.end()

    def plan(self, landed: int) -> PreparedRepeat:
        """The native owner of this graph, once recording ended."""
        areas = tuple(self.scratch) + tuple(
            (device, start, end) for _, _, device, start, end in self.generated.values()
        )
        held = {
            key: tensor
            for key, tensor in self.held.items()
            if not self._overlaps(tensor, areas)
            and not (self.loaded and self._overlaps(tensor, self.ranges))
        }
        owners = () if self.loaded else self.bindings
        try:
            native = _execution.ResidentRepeat(
                self.graph.raw_cuda_graph(),
                self.index.data_ptr(),
                self.device.index,
                self.steps,
                [
                    self.graph,
                    self.domain.stream,
                    self.index,
                    *held.values(),
                    *(tensor.untyped_storage() for tensor in held.values()),
                    *owners,
                    self.domain,
                    self.recording._pool,
                ],
            )
        except _execution.NativeExecutionError as error:
            self.reset()
            _native_error(error)
        except BaseException:
            self.reset()
            raise
        if landed != LANDED[0]:
            native.close_owner()
            self.reset()
            raise CapabilityError(
                "kernel implementations changed during recording", code="prepared_graph_ineligible"
            )
        dependencies = tuple((weakref.ref(t), PreparedRepeat._layout(t)) for t in held.values())
        identities = tuple((weakref.ref(b), b.identity) for b in self.bindings)
        self.held.clear()
        plan = PreparedRepeat(
            native, self.state, self.device.index, self.steps, dependencies, identities, landed
        )
        plan._capture_cache = {"recorded_ops": self.ops}
        return plan


class _Owners(TorchDispatchMode):  # type: ignore[misc]
    """What one recorded graph reads from outside itself, and the scratch it makes."""

    def __init__(self, graph: _Graph) -> None:
        super().__init__()
        self.g = graph

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        g, kwargs = self.g, kwargs or {}
        leaves = tree_leaves((args, kwargs))
        if g.unbound or g.undeclared:
            for tensor in leaves:
                if id(tensor) in g.unbound or g._overlaps(tensor, g.undeclared):
                    raise CapabilityError(
                        "a stage reads weights outside its block", code="prepared_graph_ineligible"
                    )
        if torch.Tag.nondeterministic_seeded in func.tags and not _zero_dropout_attention(
            func, args, kwargs
        ):
            raise CapabilityError(
                "native loops take noise as an input, never a random draw",
                code="prepared_graph_ineligible",
            )
        if g.ranges:
            for position, argument in enumerate(func._schema.arguments):
                if argument.alias_info is None or not argument.alias_info.is_write:
                    continue
                value = args[position] if position < len(args) else kwargs.get(argument.name)
                if any(g._overlaps(tensor, g.ranges) for tensor in tree_leaves(value)):
                    raise CapabilityError(
                        "the step writes loaded weight storage", code="prepared_graph_ineligible"
                    )
        inputs = set()
        for tensor in leaves:
            if isinstance(tensor, torch.Tensor):
                prior = g.produced.get(id(tensor))
                if prior is None or prior() is not tensor:
                    g.held[id(tensor)] = tensor
                inputs.add((tensor.device, tensor.untyped_storage().data_ptr()))
        result = func(*args, **kwargs)
        g.ops += 1
        for tensor in tree_leaves(result):
            if isinstance(tensor, torch.Tensor):
                g.produced[id(tensor)] = weakref.ref(tensor)
                storage = tensor.untyped_storage()
                pointer, size = int(storage.data_ptr()), int(storage.nbytes())
                if size and (tensor.device, pointer) not in inputs:
                    g.generated[id(tensor)] = (
                        weakref.ref(tensor),
                        weakref.ref(storage),
                        tensor.device,
                        pointer,
                        pointer + size,
                    )
        return result


class _Cutter:
    """One pass of the step recorded as stages cut at a component's block boundaries.

    The component's common weights are held for the program; each block is a stage that loads
    its own raw block at its own addresses. Scratch that crosses a cut stays in the shared pool:
    later stages read it at the address the earlier one wrote, as replay runs them in order. A
    block the cut cannot take cleanly (nested blocks, weights read across a cut, a derived
    layout) refuses the whole pass.
    """

    def __init__(self, recording: _SequenceCapture, model: Model, state: tuple[Any, ...]) -> None:
        self.recording, self.model, self.state = recording, model, state
        self.common: Binding | None = None
        self.graph: _Graph | None = None
        self.block: Any = None
        self.block_binding: Binding | None = None
        self.load: Any = None
        self.blocks = 0
        self.failed = False
        self.landed = LANDED[0]
        self.scratch: list[tuple[Any, int, int]] = []
        self.immutable: frozenset[int] = frozenset()

    def record(self, body: Callable[[], object]) -> None:
        _checked(self.model)
        model = self.model
        common = self.common = _binding(model, [model.common] if model.common else [])
        try:
            with ExitStack() as preparing:
                preparing.enter_context(common.prepare())
                self.immutable = common.immutable_buffer_ids()
                preparing.enter_context(_guarded((self.model,), self.immutable))
                token = CUTTER.set(self)
                preparing.callback(CUTTER.reset, token)
                self._open_common()
                try:
                    body()
                except BaseException:
                    self.failed = True
                    self._abort()
                    raise
                self._close()
            if self.blocks == 0:
                raise CapabilityError(
                    "the step entered none of the component's blocks",
                    code="prepared_graph_ineligible",
                )
            # Every common stage reads the one held common binding; the last one closes it.
            owners = [s for s in self.recording.stages if s._owned_bindings == (common,)]
            if owners:
                owners[-1]._binding_cleanup.callback(common.close)
                self.common = None
        finally:
            if self.common is not None:
                self.common.close()

    # Called by residency's block hooks while CUTTER is set.
    def enter(self, block: Any) -> None:
        if self.failed or block.model is not self.model:
            if block.model is not self.model and not self.failed:
                raise CapabilityError(
                    "the step runs another component's block", code="prepared_graph_ineligible"
                )
            return
        if self.block is not None:
            raise CapabilityError("nested blocks cannot be cut", code="prepared_graph_ineligible")
        self._close()
        self._open_block(block)

    def exit(self, block: Any) -> None:
        if self.failed or block.model is not self.model:
            return
        if sys.exc_info()[1] is not None:  # the forward failed: unwind, record nothing more
            self.failed = True
            self._abort()
            return
        if block is not self.block:
            raise CapabilityError(
                "a block cut closed out of order", code="prepared_graph_ineligible"
            )
        self._close()
        self._open_common()

    def _open_common(self) -> None:
        common = self.common
        assert common is not None
        self.graph = _Graph(
            self.recording,
            self.state,
            1,
            (common,),
            scratch=self.scratch,
            unbound=self.immutable - common.immutable_buffer_ids(selected=True),
            undeclared=common.undeclared_input_ranges(),
        )
        self._declared = _CAPTURE_BLOCKS.set(frozenset(id(b) for b in common.groups))
        self.graph.begin()

    def _open_block(self, block: Any) -> None:
        binding = _binding(self.model, [block])
        try:
            load = binding.prepare_load(stage=True)
        except BaseException:
            binding.close()
            raise
        if load is None:
            binding.close()
            raise CapabilityError(
                f"block {block.path} cannot be detached as a stage",
                code="prepared_graph_ineligible",
            )
        self.block, self.block_binding, self.load = block, binding, load
        self._preparing = ExitStack()
        try:
            self._preparing.enter_context(binding.prepare())
            self.graph = _Graph(
                self.recording,
                self.state,
                1,
                (binding,),
                loaded=tuple(binding.immutable_inputs()),
                scratch=self.scratch,
                unbound=self.immutable - binding.immutable_buffer_ids(selected=True),
                undeclared=binding.undeclared_input_ranges(),
            )
            self._declared = _CAPTURE_BLOCKS.set(frozenset(id(b) for b in binding.groups))
            self.graph.begin()
        except BaseException:
            self._preparing.close()
            self.block = self.block_binding = self.load = None
            binding.close()
            raise

    def _close(self) -> None:
        """End the current stage; keep it unless it recorded nothing."""
        graph, self.graph = self.graph, None
        if graph is None:
            return
        try:
            graph.end()
        finally:
            _CAPTURE_BLOCKS.reset(self._declared)
        self.scratch.extend(
            (device, start, end) for _, _, device, start, end in graph.generated.values()
        )
        block, binding, load = self.block, self.block_binding, self.load
        self.block = self.block_binding = self.load = None
        if block is None:  # a common stage
            if not graph.nodes():
                graph.reset()
                return
            assert self.common is not None
            plan = graph.plan(self.landed)
            _own(plan, (self.common,), self.recording)
            return
        assert binding is not None
        try:
            load.seal()
            self._preparing.close()  # parameter slots back before any capture-only view drops
            plan = graph.plan(self.landed)
            try:
                source = binding.detach_stage()
            except BaseException as error:
                plan.close()
                if isinstance(error, Exception) and not isinstance(error, ResidencyError):
                    _native_error(error)
                raise
        except BaseException:
            graph.reset()
            binding.close()
            raise
        events = bool(getattr(varena.LoadSlot, "supports_event_input", False)) and bool(
            getattr(_execution.ResidentRepeat, "supports_event_input", False)
        )
        plan._load = source
        plan._load_ordering_requested = "events"
        plan._load_ordering = "events" if events else "checked"
        cleanup = ExitStack()
        cleanup.callback(binding.close)
        _own(plan, (binding,), self.recording, cleanup)
        self.blocks += 1

    def _abort(self) -> None:
        graph, self.graph = self.graph, None
        if graph is not None:
            graph.abort()
            _CAPTURE_BLOCKS.reset(self._declared)
        if self.block_binding is not None:
            self._preparing.close()
            self.block_binding.close()
            self.block = self.block_binding = self.load = None


def _binding(model: Model, blocks: list[Any]) -> Binding:
    """A binding over exactly `blocks` (residency's retain always adds the common block)."""
    model._claim()
    try:
        model.memory._declare(model)
        return Binding(model, blocks)
    finally:
        model._release()


def install() -> None:
    _loop.install(Loops())
