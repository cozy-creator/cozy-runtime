"""Managed calls (`call`, `prefetch`); only the admitted attempt supplies their broker."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import re
import socket
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Generator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast, overload

import msgspec
from tensorfs.executor import (
    Answer,
    ChildCall,
    ChildCancel,
    ChildForget,
    ChildPoll,
    GpuRelease,
    ModelPrefetch,
    Request,
    SavedFile,
)

from cozy_runtime import _application_json as application_json
from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelChoice
from cozy_runtime.author._call_results import decode
from cozy_runtime.author._context import Context
from cozy_runtime.author._decode import MediaReads
from cozy_runtime.author._defaults import fit
from cozy_runtime.author._errors import CapabilityError, ConformanceError
from cozy_runtime.author._files import _UNBOUND, File, Files, References, Tree, asset_dec_hook
from cozy_runtime.author._model import WARNINGS, Model
from cozy_runtime.author._observations import Scalar
from cozy_runtime.author._services import (
    Attempt,
    Telemetry,
    _progress_scope,
    _ProgressScope,
    settle_frame,
)
from cozy_runtime.author._weights import Weights

_LOG = logging.getLogger(__name__)

R = TypeVar("R")
MAX_ACTIVE_CALLS = 32
MAX_CALL_BYTES = 48 * 1024
MAX_CALL_INDEX = (1 << 32) - 1

#: One request to the machine; blocks until its answer.
type Exchange = Callable[[Request], Answer]


class Nudges(Protocol):
    """The worker's nudges to an attempt: a byte each time one of its calls may have moved,
    b"" once the worker let go of its calls (a socket; a pipe on Windows)."""

    def recv(self, size: int, /) -> bytes: ...

    def close(self) -> None: ...


class ChildCallError(CapabilityError):
    """A child call's failure, as its RecordOwner states it, with the child's traceback."""

    def __init__(
        self,
        code: str,
        message: str,
        call_index: int,
        child_request_id: str = "",
        child_traceback: str = "",
    ):
        super().__init__(message, code=code)
        self.call_index = call_index
        self.child_request_id = child_request_id
        self.child_traceback = child_traceback


@overload
def call[R](  # type: ignore[explicit-any]  # any package function: call() names its arguments
    fn: Callable[..., Coroutine[object, object, R]], /, **arguments: object
) -> PendingCall[R]: ...


@overload
def call[R](fn: Callable[..., R], /, **arguments: object) -> PendingCall[R]: ...  # type: ignore[explicit-any]  # any package function: call() names its arguments


def call(fn: Callable[..., object], /, **arguments: object) -> PendingCall[object]:  # type: ignore[explicit-any]  # any package function: call() names its arguments
    """Run `fn` as a managed call: its own job on the machine, with its own models, memo and
    progress. Awaiting it returns `fn`'s result.

    `arguments` are `fn`'s own parameters as a direct call names them, less what is injected
    (Context, services): its request struct or fields, its `Files`, and a `ModelChoice` (or
    None, its default) per model argument. `fn` is a registered function of this package or a
    public one of a package this one depends on. A plain `fn(...)` is an ordinary Python call.
    """
    broker = _current.get()
    if broker is None:
        raise CapabilityError("a managed call has no admitted attempt", code="child_broker_absent")
    module, name = getattr(fn, "__module__", ""), getattr(fn, "__name__", "")
    return broker.reserve(module, name, arguments)


def prefetch[**P, R](
    callee: Callable[P, R],
    *,
    models: Mapping[str, ModelChoice | None] | None = None,
) -> None:
    """Prepare a known next call's models while current work runs.

    Runtime sends the hint to TensorD, which controls preparation and can ask a Runtime
    executor to construct/load a model on a GPU. The callable's inference is not invoked.
    Acknowledgment does not promise readiness; the later call resolves its own inputs.
    Explicit models select the same declared slots as an ordinary child call; omitted
    or None choices retain inherited bindings/defaults. Pass overrides to the actual
    call too: the machine may decline the hint.
    """
    broker = _current.get()
    if broker is None or broker.closed or broker.context is None:
        raise CapabilityError("model prefetch has no admitted attempt", code="child_broker_absent")
    broker.context.raise_if_cancelled()
    module, name = getattr(callee, "__module__", ""), getattr(callee, "__name__", "")
    if not callable(callee) or (module, name) not in broker.bindings:
        raise CapabilityError("prefetch needs an exact locked callable", code="child_undeclared")
    selected = {} if models is None else dict(models)
    if any(
        not isinstance(k, str) or (v is not None and not isinstance(v, ModelChoice))
        for k, v in selected.items()
    ):
        raise ConformanceError(
            "prefetch models require named ModelChoice values", code="child_arguments"
        )
    payload = canonical_json.encode(msgspec.to_builtins(selected))
    if len(payload) > MAX_CALL_BYTES:
        raise CapabilityError("model prefetch exceeds inline bound", code="child_payload_bound")
    answer = broker.exchange(ModelPrefetch(module=module, export=name, payload=payload.decode()))
    # The machine can decline an optional optimization; the call prepares on demand.
    if answer.code != "model_prefetch_unsupported":
        broker._check(answer, 0)


@dataclass(frozen=True, slots=True)
class _CallType:
    interface_digest: str
    module: str
    export: str
    request: type[msgspec.Struct]
    result: type[msgspec.Struct]
    #: Why the machine cannot serve this operation; "" when it can.
    unavailable: str = ""
    #: A callable bound from its own code: its model slots, which a caller passes beside its
    #: request and the call carries as `{payload, models}`.
    models: frozenset[str] | None = None
    #: The parameter whose struct is the request; empty: the request is its other arguments.
    request_parameter: str = ""


class PendingCall[R](Awaitable[R]):
    """Awaiting returns R unchanged."""

    def __init__(
        self,
        broker: _Broker,
        index: int,
        binding: _CallType,
        payload: bytes,
        ctx: Context,
        saved: tuple[SavedFile, ...] = (),
    ):
        self.broker, self.index, self.binding, self.payload = broker, index, binding, payload
        self.ctx, self.saved = ctx, saved
        self.request_id = ""
        self.task: asyncio.Task[R] | None = None
        self.progress_scope = _progress_scope.get()
        self.progress_sequence = 0

    def __await__(self) -> Generator[object, None, R]:
        if self.task is None:
            self.task = asyncio.create_task(self.broker.run(self))
        return self.task.__await__()


def _input_ref(value: object, request_id: str) -> str:
    """Forward only a verified input of this attempt; TensorD rechecks its custody."""
    if isinstance(value, (File, Tree)) and value._attempt not in (_UNBOUND, request_id):
        raise CapabilityError(
            f"file {value.ref!r} belongs to attempt {value._attempt!r}", code="foreign_asset"
        )
    if (
        isinstance(value, (File, Tree))
        and value.hydrated
        and value._attempt == request_id
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value.digest)
    ):
        if value._read_guard is not None:
            value._read_guard()
        return value.digest
    raise CapabilityError(
        "child media must be a file this attempt saved, or a verified input granted to it",
        code="child.asset_ungranted",
    )


class _Broker:
    """Memory-only per-attempt reservations. The RecordOwner owns every durable fact."""

    def __init__(
        self,
        request_id: str,
        bindings: Mapping[tuple[str, str], _CallType],
        exchange: Exchange,
        watch: Callable[[], Nudges],
    ) -> None:
        self.request_id, self.bindings, self.exchange = request_id, bindings, exchange
        #: Opens the worker's nudges, at the first wait on a call; None once opened.
        self.watch: Callable[[], Nudges] | None = watch
        self.nudges: Nudges | None = None
        self.nudged = 0
        self.nudge = asyncio.Event()
        self.calls: dict[int, PendingCall[object]] = {}
        self.next_index = 0
        self.closed = False
        self.activated: set[int] = set()
        self.context: Context | None = None
        self.grant_token = object()
        self.telemetry: Telemetry | None = None
        #: The attempt's media reader: received media read through it.
        self.media: MediaReads | None = None
        #: The attempt, whose saved files it may give its children, and those it has hashed.
        self.attempt: Attempt | None = None
        self.saved_sha256: dict[str, str] = {}

    def bind(
        self,
        ctx: Context,
        telemetry: Telemetry | None = None,
        media: MediaReads | None = None,
        attempt: Attempt | None = None,
    ) -> None:
        if self.closed or self.context is not None or ctx.request_id != self.request_id:
            raise CapabilityError("package broker escaped its attempt", code="escaped_handle")
        self.context = ctx
        self.telemetry = telemetry
        self.media = media
        self.attempt = attempt
        ctx._release_gpus = self.release_gpus

    def release_gpus(self) -> None:
        """`ctx.release_gpus()` of this attempt: its root's GPU lease ends at the worker."""
        if self.closed:
            raise CapabilityError("GPU release escaped its attempt", code="escaped_handle")
        self._check(self.exchange(GpuRelease()), 0)

    def _saved(self, value: object) -> SavedFile | None:
        """`value` as a file this attempt saved, whole in its spool for TensorD to check: a frame
        is encoded now. A child's input is named by SHA-256, which a saved file's digest isn't.
        Its child has it, so the result need not return it. `value` may be a labeled copy of the
        handle saved: its `ref` names the file, and only the handle saved hears of its frame."""
        attempt = self.attempt
        if attempt is None or not isinstance(value, File):
            return None
        saved = attempt.pending.get(value.ref)
        if saved is None:
            return None
        if value.ref in attempt.parts:
            raise CapabilityError(
                "a joined video has no one file to pass to a child; save it whole or pass its parts",
                code="child.asset_ungranted",
            )
        if value.ref in attempt.frames:
            settle_frame(attempt, value.ref)
        attempt.shown.add(value.ref)
        digest = self.saved_sha256.get(value.ref)
        if digest is None:
            assert saved._local is not None  # a saved file's bytes are in the spool
            with saved._local.open("rb") as blob:
                digest = f"sha256:{hashlib.file_digest(blob, 'sha256').hexdigest()}"
            self.saved_sha256[value.ref] = digest
        return SavedFile(asset_ref=value.ref, digest=digest, media_type=saved.media_type)

    def reserve(
        self,
        module: str,
        export: str,
        arguments: Mapping[str, object],
    ) -> PendingCall[object]:
        ctx = self.context
        if self.closed or ctx is None or ctx.request_id != self.request_id:
            raise CapabilityError("package call escaped its attempt", code="escaped_handle")
        ctx.raise_if_cancelled()
        binding = self.bindings.get((module, export))
        if binding is None:
            raise CapabilityError(
                f"{module}:{export} is not a registered function of this package or of a "
                "package it depends on",
                code="child_undeclared",
            )
        if binding.unavailable:
            raise CapabilityError(binding.unavailable, code="child_capability_unavailable")
        if len(self.calls) >= MAX_ACTIVE_CALLS:
            raise CapabilityError("active package call bound exceeded", code="child_fanout")
        if binding.models is not None:
            slots = binding.models
            arguments = {
                "payload": {k: v for k, v in arguments.items() if k not in slots},
                "models": {k: v for k, v in arguments.items() if k in slots},
            }
        struct = binding.request_parameter
        if self.next_index > MAX_CALL_INDEX:
            raise CapabilityError("package call index is exhausted", code="child_index_exhausted")
        saved: dict[str, SavedFile] = {}
        try:

            def asset_value(value: object) -> object:
                if isinstance(value, (Model, Weights)):
                    raise CapabilityError(
                        "a call's model is a ModelChoice; an injected Model belongs to this "
                        "attempt. Retain it with a zero-byte TensorFS derivation and pass "
                        "ModelChoice(ref=artifact)",
                        code="child.model_artifact_required",
                    )
                if isinstance(value, References):
                    return value.wire()
                if isinstance(value, Files):
                    return [
                        {"asset": item, "label": item.label, "fidelity": item.fidelity}
                        for item in value._items
                    ]
                file = self._saved(value)
                if file is not None:
                    saved[file.digest] = file
                    return file.digest
                return _input_ref(value, ctx.request_id)

            wire = msgspec.to_builtins(dict(arguments), enc_hook=asset_value)
            if struct:
                # The request struct's fields are the request; its Files argument joins them.
                given = dict(wire["payload"])
                fields = given.pop(struct, None)
                if not isinstance(fields, dict):
                    raise ConformanceError(
                        f"{module}:{export} takes its request as {struct}=",
                        code="child_arguments",
                    )
                wire["payload"] = {**fields, **given}
            plain, ignored, moved = fit(binding.request, wire)
            row: tuple[str, dict[str, Scalar]]
            for change in moved:
                row = (
                    "a package call passed media its callee has no use for; it was left out"
                    if change.kind == "ignored"
                    else "a package call passed a value over its callee's limit; it was clamped",
                    {
                        "callee": f"{module}:{export}",
                        "field": change.field,
                        "applied": str(change.applied),
                    },
                )
                if row not in WARNINGS:
                    WARNINGS.append(row)
            if ignored:
                # A field the installed callee does not declare is a warning, never a refusal.
                row = (
                    "a package call named fields its callee does not declare; they were dropped",
                    {"callee": f"{module}:{export}", "fields": ", ".join(ignored)},
                )
                if row not in WARNINGS:
                    WARNINGS.append(row)
                _LOG.warning("%s:%s does not declare %s; dropped", module, export, ignored)
            request = msgspec.convert(
                plain, type=binding.request, dec_hook=asset_dec_hook, strict=True
            )
            encoded = msgspec.to_builtins(
                request,
                enc_hook=lambda value: (
                    value.ref if isinstance(value, (File, Tree)) else asset_value(value)
                ),
            )
        except (TypeError, ValueError, msgspec.ValidationError) as exc:
            raise ConformanceError(
                "package call arguments do not match its interface", code="child_arguments"
            ) from exc
        payload = application_json.encode(encoded)
        if len(payload) > MAX_CALL_BYTES:
            raise CapabilityError("package call exceeds inline bound", code="child_payload_bound")
        saved_files = tuple(saved.values())
        call = PendingCall[object](self, self.next_index, binding, payload, ctx, saved_files)
        self.next_index += 1
        self.calls[call.index] = call
        return call

    async def run(self, call: PendingCall[R]) -> R:
        # All tasks in an ordinary gather become ready before activation. Flush
        # them by reserved index, regardless of scheduling/completion order.
        await asyncio.sleep(0)
        try:
            while True:
                call.ctx.raise_if_cancelled()
                if self.closed:
                    raise ChildCallError("escaped_handle", "parent attempt closed", call.index)
                for ready in tuple(self.calls.values()):
                    if ready.task is not None and ready.index not in self.activated:
                        answer = self.exchange(self._intent(ready))
                        self._observe(ready, answer)
                        self._check(answer, ready.index, call.ctx)
                        self.activated.add(ready.index)
                seen = self.nudged
                answer = self.exchange(ChildPoll(call_index=call.index))
                self._observe(call, answer)
                self._check(answer, call.index, call.ctx)
                if answer.state == "succeeded":
                    raw = answer.result.encode()
                    if len(raw) > MAX_CALL_BYTES:
                        raise ChildCallError(
                            "child_result_bound", "child result exceeds inline bound", call.index
                        )
                    try:
                        value = application_json.decode(raw)

                        def guard() -> None:
                            if self.closed:
                                raise CapabilityError(
                                    "child result escaped its attempt", code="escaped_handle"
                                )
                            call.ctx.raise_if_cancelled()

                        result = decode(
                            value,
                            call.binding.result,
                            answer.byte_grants,
                            request_id=self.request_id,
                            guard=guard,
                            grant_token=self.grant_token,
                            media=self.media,
                        )
                        return cast(R, result)
                    except (ValueError, TypeError, msgspec.ValidationError) as exc:
                        raise ChildCallError(
                            "child_result_invalid",
                            "child result differs from its interface",
                            call.index,
                        ) from exc
                await self._changed(seen, call.ctx.deadline, call.index)
        except asyncio.CancelledError:
            if call.index in self.activated and not self.closed:
                self.exchange(ChildCancel(call_index=call.index))
            raise
        finally:
            try:
                if call.index in self.activated and not self.closed:
                    self.exchange(ChildForget(call_index=call.index))
            finally:
                self.activated.discard(call.index)
                self.calls.pop(call.index, None)

    async def _changed(self, seen: int, deadline: float, index: int) -> None:
        """Until the worker nudges this attempt after `seen`: a call moved or it was cancelled.
        Its nudges are read on a thread of their own, from the first wait on."""
        if self.watch is not None:
            watch, self.watch = self.watch, None
            self.nudges = watch()
            loop = asyncio.get_running_loop()
            threading.Thread(
                target=self._read, args=(self.nudges, loop), name="call-nudges", daemon=True
            ).start()
            return  # a change before they were opened nudged no one: look once more
        if self.nudged != seen:
            return
        if self.nudges is None:
            raise ChildCallError(
                "child_events_closed", "the machine let go of this attempt's calls", index
            )
        remaining = deadline - time.monotonic()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self.nudge.wait(), None if math.isinf(remaining) else max(remaining, 0)
            )

    def _read(self, nudges: Nudges, loop: asyncio.AbstractEventLoop) -> None:
        with contextlib.suppress(RuntimeError):  # the attempt's loop closed first
            with contextlib.suppress(OSError):
                while nudges.recv(4096):
                    loop.call_soon_threadsafe(self._nudged, False)
            loop.call_soon_threadsafe(self._nudged, True)

    def _nudged(self, closed: bool) -> None:
        if closed:
            self._unwatch()
        self.nudged += 1
        nudge, self.nudge = self.nudge, asyncio.Event()
        nudge.set()

    def _unwatch(self) -> None:
        nudges, self.nudges = self.nudges, None
        if nudges is not None:
            if isinstance(nudges, socket.socket):
                with contextlib.suppress(OSError):
                    nudges.shutdown(socket.SHUT_RDWR)  # its reader's recv returns
            nudges.close()

    @staticmethod
    def _observe[T](call: PendingCall[T], answer: Answer) -> None:
        child_id = answer.child_request_id
        if child_id:
            if len(child_id) > 256 or call.request_id not in ("", child_id):
                raise ChildCallError(
                    "child_identity_changed", "child request identity changed", call.index
                )
            call.request_id = child_id
        progress = answer.progress
        if progress is not None and progress.sequence > call.progress_sequence:
            call.progress_sequence = progress.sequence
            scope = call.progress_scope
            if scope is None and call.broker.telemetry is not None:
                scope = _ProgressScope(call.broker.telemetry, call.binding.export)
            if scope is not None:
                scope.telemetry._child_progress(progress.payload, scope)  # type: ignore[arg-type]  # a child's progress is a JSON object

    @staticmethod
    def _intent[T](call: PendingCall[T]) -> ChildCall:
        return ChildCall(
            call_index=call.index,
            module=call.binding.module,
            export=call.binding.export,
            payload=call.payload.decode(),
            progress_label=call.progress_scope.name if call.progress_scope else "",
            saved=call.saved,
        )

    @staticmethod
    def _check(answer: Answer, index: int, ctx: Context | None = None) -> None:
        if not answer.ok:
            if ctx is not None:
                # A child ended by this attempt's own cancel is that cancel, not its failure.
                ctx.raise_if_cancelled()
            raise ChildCallError(
                answer.code,
                answer.detail,
                index,
                answer.child_request_id,
                answer.traceback,
            )

    def close(self) -> None:
        self.closed = True
        self._unwatch()

    def finish(self) -> None:
        self.close()
        missing = list(self.calls)
        if missing:
            raise ChildCallError(
                "unawaited_child_call", "reserved package call was never awaited", missing[0]
            )


_current: ContextVar[_Broker | None] = ContextVar("cozy_package_call_attempt", default=None)
