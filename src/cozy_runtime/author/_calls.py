"""Typed package calls; only the admitted attempt supplies their hidden broker."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
import re
import socket
import time
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Concatenate, ParamSpec, TypeVar, cast, overload

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._assets import Asset, Assets, Tree, asset_dec_hook
from cozy_runtime.author._capture import ActivationCapture, ExecutionObservation
from cozy_runtime.author._context import Context
from cozy_runtime.author._errors import CapabilityError, ConformanceError
from cozy_runtime.author._executor_requests import (
    MAX_CALL_INDEX,
    Answer,
    CallState,
    ChildCall,
    ChildCancel,
    ChildForget,
    ChildPoll,
    Exchange,
    GpuRelease,
    ModelPrefetch,
)
from cozy_runtime.author._memo import MemoDependency
from cozy_runtime.author._memo import declare as declare_memo
from cozy_runtime.author._model_defaults import DefaultLadder, model_defaults
from cozy_runtime.author._services import Telemetry, _progress_scope, _ProgressScope

_LOG = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
MAX_ACTIVE_CALLS = 32
MAX_CALL_BYTES = 48 * 1024
#: Only a worker that cannot nudge its parent (Runtime before child_events) is polled.
_UNNOTIFIED_POLL_SECONDS = 0.05


class ChildCallError(CapabilityError):
    """A bounded RecordOwner verdict, never an implementation traceback."""

    def __init__(self, code: str, message: str, call_index: int, child_request_id: str = ""):
        super().__init__(message, code=code)
        self.call_index = call_index
        self.child_request_id = child_request_id


@dataclass(frozen=True, slots=True)
class _Export:
    implementation: Callable[..., object]
    module: str
    name: str
    memoize: bool = False
    defaults: Mapping[str, DefaultLadder] = field(default_factory=dict)


@overload
def invocable[**P, R](
    fn: Callable[Concatenate[Context, P], Awaitable[R]],
    /,
    *,
    memoize: bool = False,
    memo_version: str | None = None,
    memo_dependencies: tuple[MemoDependency, ...] = (),
    defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
) -> Callable[P, PendingCall[R]]: ...


@overload
def invocable[**P, R](
    *,
    memoize: bool = False,
    memo_version: str | None = None,
    memo_dependencies: tuple[MemoDependency, ...] = (),
    defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
) -> Callable[[Callable[Concatenate[Context, P], Awaitable[R]]], Callable[P, PendingCall[R]]]: ...


def invocable[**P, R](
    fn: Callable[Concatenate[Context, P], Awaitable[R]] | None = None,
    /,
    *,
    memoize: bool = False,
    memo_version: str | None = None,
    memo_dependencies: tuple[MemoDependency, ...] = (),
    defaults: Mapping[str, Sequence[Mapping[str, str | int]]] | None = None,
) -> (
    Callable[P, PendingCall[R]]
    | Callable[[Callable[Concatenate[Context, P], Awaitable[R]]], Callable[P, PendingCall[R]]]
):
    """Mark an asynchronous implementation; explicitly register it on the package App.

    Calling this exported name always reserves a managed call. App registration keeps
    the implementation privately for ordinary admitted dispatch; importing a module
    never registers it or resolves a package/environment.

    At the call site, a Model parameter takes a retained ModelArtifact from conversion
    or derivation. The injected Model object itself belongs to its current attempt;
    it is not a portable artifact receipt and cannot be forwarded to another call.

    Memoized operations automatically bind their Python implementation. Declare
    helper callables or Python modules in ``memo_dependencies`` when their logic
    affects the result; source changes invalidate reuse without a manual version
    bump. Optional ``memo_version`` salts behavior outside those sources, such as
    native algorithms or data tables. Unavailable source disables reuse, not execution.
    """
    if type(memoize) is not bool:
        raise ConformanceError(
            "invocable memoization declaration must be a boolean", code="invocable_memoize"
        )
    if fn is None:
        return lambda target: invocable(
            target,
            memoize=memoize,
            memo_version=memo_version,
            memo_dependencies=memo_dependencies,
            defaults=defaults,
        )
    if not inspect.iscoroutinefunction(fn) or "<locals>" in fn.__qualname__:
        raise ConformanceError(
            "an invocable must be a module-level async function", code="invocable_signature"
        )
    declare_memo(fn, memoize=memoize, version=memo_version, dependencies=memo_dependencies)
    export = _Export(fn, fn.__module__, fn.__name__, memoize, model_defaults(defaults))

    def proxy(*args: P.args, **kwargs: P.kwargs) -> PendingCall[R]:
        if args:
            raise ConformanceError(
                "an invocable takes only named operation arguments",
                code="invocable_signature",
            )
        broker = _current.get()
        if broker is None:
            raise CapabilityError(
                "package call has no admitted attempt", code="child_broker_absent"
            )
        arguments = dict(kwargs)
        capture = arguments.pop("capture", None)
        if capture is not None and not isinstance(capture, ActivationCapture):
            raise ConformanceError("capture needs ActivationCapture", code="capture_option")
        call = broker.reserve(export.module, export.name, arguments)
        call.capture = capture
        return cast(PendingCall[R], call)

    proxy.__name__ = fn.__name__
    proxy.__qualname__ = fn.__qualname__
    proxy.__module__ = fn.__module__
    proxy.__doc__ = fn.__doc__
    proxy.__annotations__ = dict(fn.__annotations__)
    # No __wrapped__ / raw / local bypass is exposed by the public callable.
    metadata = cast(Any, proxy)
    signature = inspect.signature(fn)
    parameters = list(signature.parameters.values())
    metadata.__signature__ = signature.replace(parameters=parameters[1:])
    if parameters:
        proxy.__annotations__.pop(parameters[0].name, None)
    metadata._cozy_export = export
    return proxy


def _export(fn: Callable[..., object]) -> _Export | None:
    value = getattr(fn, "_cozy_export", None)
    return value if isinstance(value, _Export) else None


def prefetch(
    callee: Callable[..., object],
    *,
    models: Mapping[str, ModelArtifact | None] | None = None,
) -> None:
    """Prepare a known next call's models while current work runs.

    This optional hint downloads and checks only the selected models, without
    invoking the callable or reserving GPU memory. Omitted models use the same
    captured defaults as a later ordinary call. No future media inputs are needed.
    Preparation failures are observed when those models are actually invoked.
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
        not isinstance(k, str) or (v is not None and not isinstance(v, ModelArtifact))
        for k, v in selected.items()
    ):
        raise ConformanceError(
            "prefetch models require named ModelArtifact values", code="child_arguments"
        )
    payload = canonical_json.encode(msgspec.to_builtins(selected))
    if len(payload) > MAX_CALL_BYTES:
        raise CapabilityError("model prefetch exceeds inline bound", code="child_payload_bound")
    answer = broker.exchange(
        ModelPrefetch(module=module, export=name, payload=payload.decode()), Answer
    )
    # An older/nonlocal worker can decline an optional optimization. Normal calls
    # retain their ordinary demand preparation path.
    if answer.code not in (
        "model_prefetch_unsupported",
        "weights_host_unavailable",
        "unknown_durable_request",
    ):
        broker._check(answer, 0)


@dataclass(frozen=True, slots=True)
class _CallType:
    interface_digest: str
    module: str
    export: str
    request: type[msgspec.Struct]
    result: type[msgspec.Struct]
    python_result: type[msgspec.Struct] | None = None
    #: Request fields the serving Runtime accepts; ``None`` when it did not say.
    served_fields: frozenset[str] | None = None
    #: Why the serving Runtime cannot serve this operation; "" when it can.
    unavailable: str = ""


RUNTIME_UPDATE = (
    "this package's SDK is newer than this Runtime; update the rental's Runtime "
    "(`cozy rental update <name>`) or pin an older SDK"
)


def _served(binding: _CallType, request: msgspec.Struct, value: dict[str, Any]) -> dict[str, Any]:
    """Omit an option the serving Runtime predates while it keeps its default; else refuse."""
    served = binding.served_fields
    if served is None:
        return value
    unsupported = []
    for field_info in msgspec.structs.fields(binding.request):
        name = field_info.encode_name
        if name in served:
            continue
        default = (
            field_info.default_factory()
            if field_info.default_factory is not msgspec.NODEFAULT
            else field_info.default
        )
        if default is not msgspec.NODEFAULT and getattr(request, field_info.name) == default:
            value.pop(name, None)
        else:
            unsupported.append(name)
    if unsupported:
        raise CapabilityError(
            f"{binding.module}.{binding.export} does not support {', '.join(unsupported)} "
            f"here: {RUNTIME_UPDATE}",
            code="child_capability_unavailable",
        )
    return value


class PendingCall[R](Awaitable[R]):
    """Awaiting returns R unchanged; observation is available after the call settles."""

    def __init__(
        self, broker: _Broker, index: int, binding: _CallType, payload: bytes, ctx: Context
    ):
        self.broker, self.index, self.binding, self.payload = broker, index, binding, payload
        self.ctx = ctx
        self.request_id = ""
        self.task: asyncio.Task[R] | None = None
        self.capture: ActivationCapture | None = None
        self.observation: ExecutionObservation | None = None
        self.progress_scope = _progress_scope.get()
        self.progress_sequence = 0

    def __await__(self) -> Generator[Any, None, R]:
        from cozy_runtime.author._activity import observe_await

        observe_await()
        if self.task is None:
            self.task = asyncio.create_task(self.broker.run(self))
        return self.task.__await__()


def _input_ref(value: object, request_id: str) -> str:
    """Forward only a verified input of this attempt; Creator rechecks its custody."""
    if (
        isinstance(value, (Asset, Tree))
        and value.hydrated
        and value._attempt == request_id
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value.digest)
    ):
        if value._read_guard is not None:
            value._read_guard()
        return value.digest
    raise CapabilityError(
        "child media must be a verified input granted to this parent attempt",
        code="child.asset_ungranted",
    )


class _Broker:
    """Memory-only per-attempt reservations. The RecordOwner owns every durable fact."""

    def __init__(
        self,
        request_id: str,
        bindings: Mapping[tuple[str, str], _CallType],
        exchange: Exchange,
        watch: Callable[[], int | None] | None = None,
    ) -> None:
        self.request_id, self.bindings, self.exchange = request_id, dict(bindings), exchange
        #: Opens the worker's nudge socket once; None when no peer can nudge.
        self.watch = watch
        self.nudges: socket.socket | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.nudged = 0
        self.nudge = asyncio.Event()
        self.calls: dict[int, PendingCall[Any]] = {}
        self.next_index = 0
        self.closed = False
        self.activated: set[int] = set()
        self.context: Context | None = None
        self.grant_token = object()
        self.telemetry: Telemetry | None = None

    def bind(self, ctx: Context, telemetry: Telemetry | None = None) -> None:
        if self.closed or self.context is not None or ctx.request_id != self.request_id:
            raise CapabilityError("package broker escaped its attempt", code="escaped_handle")
        self.context = ctx
        self.telemetry = telemetry
        ctx._release_gpus = self.release_gpus

    def release_gpus(self) -> None:
        """`ctx.release_gpus()` of this attempt: its root's GPU lease ends at the worker."""
        if self.closed:
            raise CapabilityError("GPU release escaped its attempt", code="escaped_handle")
        answer = self.exchange(GpuRelease(), Answer)
        # A worker that predates the release declines it; the lease then ends with the root.
        if answer.code not in (
            "weights_host_unavailable",
            "unknown_durable_request",
            "no_durable_exchange",
        ):
            self._check(answer, 0)

    def reserve(
        self,
        module: str,
        export: str,
        arguments: Mapping[str, object],
    ) -> PendingCall[Any]:
        ctx = self.context
        if self.closed or ctx is None or ctx.request_id != self.request_id:
            raise CapabilityError("package call escaped its attempt", code="escaped_handle")
        ctx.raise_if_cancelled()
        binding = self.bindings.get((module, export))
        if binding is None:
            raise CapabilityError(
                "export is not an exact locked dependency", code="child_undeclared"
            )
        if binding.unavailable:
            raise CapabilityError(binding.unavailable, code="child_capability_unavailable")
        if len(self.calls) >= MAX_ACTIVE_CALLS:
            raise CapabilityError("active package call bound exceeded", code="child_fanout")
        if self.next_index > MAX_CALL_INDEX:
            raise CapabilityError("package call index is exhausted", code="child_index_exhausted")
        try:

            def asset_value(value: object) -> object:
                from cozy_runtime.author._model import Model

                if isinstance(value, Model):
                    raise CapabilityError(
                        "managed model calls require a retained ModelArtifact from conversion "
                        "or derivation; an injected Model belongs to this attempt. Retain it "
                        "with a zero-byte TensorFS derivation before passing it to a child",
                        code="child.model_artifact_required",
                    )
                if isinstance(value, Assets):
                    return [
                        {"asset": item, "label": item.label, "fidelity": item.fidelity}
                        for item in value._items
                    ]
                return _input_ref(value, ctx.request_id)

            plain = msgspec.to_builtins(dict(arguments), enc_hook=asset_value)
            request = msgspec.convert(
                plain, type=binding.request, dec_hook=asset_dec_hook, strict=True
            )
            encoded = msgspec.to_builtins(
                request,
                enc_hook=lambda value: (
                    value.ref if isinstance(value, (Asset, Tree)) else asset_value(value)
                ),
            )
        except (TypeError, ValueError, msgspec.ValidationError) as exc:
            raise ConformanceError(
                "package call arguments do not match its interface", code="child_arguments"
            ) from exc
        payload = canonical_json.encode(_served(binding, request, encoded))
        if len(payload) > MAX_CALL_BYTES:
            raise CapabilityError("package call exceeds inline bound", code="child_payload_bound")
        call = PendingCall[Any](self, self.next_index, binding, payload, ctx)
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
                        answer = self.exchange(self._intent(ready), CallState)
                        self._observe(ready, answer)
                        self._check(answer, ready.index, answer.child_request_id)
                        self.activated.add(ready.index)
                seen = self.nudged
                answer = self.exchange(ChildPoll(call_index=call.index), CallState)
                self._observe(call, answer)
                self._check(answer, call.index, answer.child_request_id)
                if answer.state == "succeeded":
                    raw = answer.result.encode()
                    if len(raw) > MAX_CALL_BYTES:
                        raise ChildCallError(
                            "child_result_bound", "child result exceeds inline bound", call.index
                        )
                    try:
                        value = canonical_json.decode(raw)
                        from cozy_runtime.author._call_results import decode

                        def guard() -> None:
                            if self.closed:
                                raise CapabilityError(
                                    "child result escaped its attempt", code="escaped_handle"
                                )
                            call.ctx.raise_if_cancelled()

                        result, call.observation = decode(
                            value,
                            call.binding.result,
                            list(answer.byte_grants),
                            request_id=self.request_id,
                            guard=guard,
                            observation=call.observation,
                            grant_token=self.grant_token,
                            python_type=call.binding.python_result,
                        )
                        return cast(R, result)
                    except (ValueError, TypeError, msgspec.ValidationError) as exc:
                        raise ChildCallError(
                            "child_result_invalid",
                            "child result differs from its interface",
                            call.index,
                        ) from exc
                await self._changed(seen, call.ctx.deadline)
        except asyncio.CancelledError:
            if call.index in self.activated and not self.closed:
                self.exchange(ChildCancel(call_index=call.index), CallState)
            raise
        finally:
            try:
                if call.index in self.activated and not self.closed:
                    self.exchange(ChildForget(call_index=call.index), Answer)
            finally:
                self.activated.discard(call.index)
                self.calls.pop(call.index, None)

    async def _changed(self, seen: int, deadline: float) -> None:
        """Until the worker nudges this attempt after `seen`: a call moved or it was cancelled."""
        if self.watch is not None:
            watch, self.watch = self.watch, None
            descriptor = watch()
            if descriptor is not None:
                self.nudges = socket.socket(fileno=descriptor)
                self.nudges.setblocking(False)
                self.loop = asyncio.get_running_loop()
                self.loop.add_reader(self.nudges.fileno(), self._drain)
                return  # a change before the socket existed nudged no one: poll once more
        if self.nudges is None:
            await asyncio.sleep(_UNNOTIFIED_POLL_SECONDS)
            return
        if self.nudged == seen:
            remaining = deadline - time.monotonic()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self.nudge.wait(), None if math.isinf(remaining) else max(remaining, 0)
                )

    def _drain(self) -> None:
        assert self.nudges is not None
        try:
            while self.nudges.recv(4096):
                pass
            self._unwatch()  # the worker closed the attempt's socket: poll from now on
        except BlockingIOError:
            pass
        except OSError:
            self._unwatch()
        self.nudged += 1
        nudge, self.nudge = self.nudge, asyncio.Event()
        nudge.set()

    def _unwatch(self) -> None:
        if self.nudges is not None:
            if self.loop is not None and not self.loop.is_closed():
                self.loop.remove_reader(self.nudges.fileno())
            self.nudges.close()
            self.nudges = None

    @staticmethod
    def _observe(call: PendingCall[Any], answer: CallState) -> None:
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
                scope = _ProgressScope(call.broker.telemetry, call.binding.export, None)
            if scope is not None:
                scope.telemetry._child_progress(progress.payload, scope)
        if answer.observation is None:
            return
        try:
            call.observation = msgspec.convert(
                answer.observation,
                type=ExecutionObservation,
                dec_hook=asset_dec_hook,
                strict=True,
            )
        except (ValueError, TypeError, msgspec.ValidationError) as exc:
            # An observation is evidence about the child, never its result.
            _LOG.warning("child call %d: unreadable execution observation: %s", call.index, exc)

    @staticmethod
    def _intent(call: PendingCall[Any]) -> ChildCall:
        return ChildCall(
            call_index=call.index,
            module=call.binding.module,
            export=call.binding.export,
            payload=call.payload.decode(),
            progress_label=call.progress_scope.name if call.progress_scope else "",
            capture=call.capture,
        )

    @staticmethod
    def _check(answer: Answer, index: int, child_request_id: str = "") -> None:
        if not answer.ok:
            raise ChildCallError(answer.code, answer.detail[:1024], index, child_request_id)

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


def _invoke_proxy(
    module: str,
    export: str,
    arguments: Mapping[str, object],
    *,
    capture: ActivationCapture | None = None,
) -> PendingCall[Any]:
    broker = _current.get()
    if broker is None:
        raise CapabilityError("package call has no admitted attempt", code="child_broker_absent")
    if capture is not None and not isinstance(capture, ActivationCapture):
        raise ConformanceError("capture needs ActivationCapture", code="capture_option")
    call = broker.reserve(module, export, arguments)
    call.capture = capture
    return call
