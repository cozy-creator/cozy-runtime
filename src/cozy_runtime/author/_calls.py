"""Typed package calls; only the admitted attempt supplies their hidden broker."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Concatenate, ParamSpec, TypeVar, cast, overload

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._assets import Asset, Assets, Tree, asset_dec_hook
from cozy_runtime.author._capture import ActivationCapture, ExecutionObservation
from cozy_runtime.author._context import Context
from cozy_runtime.author._errors import CapabilityError, ConformanceError
from cozy_runtime.author._model_defaults import DefaultLadder, model_defaults
from cozy_runtime.author._services import Telemetry, _progress_scope, _ProgressScope

P = ParamSpec("P")
R = TypeVar("R")
MAX_ACTIVE_CALLS = 32
MAX_CALL_INDEX = (1 << 32) - 1
MAX_CALL_BYTES = 48 * 1024


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
    defaults: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
) -> Callable[P, PendingCall[R]]: ...


@overload
def invocable[**P, R](
    *, memoize: bool = False, defaults: Mapping[str, Sequence[Mapping[str, str]]] | None = None
) -> Callable[[Callable[Concatenate[Context, P], Awaitable[R]]], Callable[P, PendingCall[R]]]: ...


def invocable[**P, R](
    fn: Callable[Concatenate[Context, P], Awaitable[R]] | None = None,
    /,
    *,
    memoize: bool = False,
    defaults: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
) -> (
    Callable[P, PendingCall[R]]
    | Callable[[Callable[Concatenate[Context, P], Awaitable[R]]], Callable[P, PendingCall[R]]]
):
    """Mark an asynchronous implementation; explicitly register it on the package App.

    Calling this exported name always reserves a managed call. App registration keeps
    the implementation privately for ordinary admitted dispatch; importing a module
    never registers it or resolves a package/environment.
    """
    if type(memoize) is not bool:
        raise ConformanceError(
            "invocable memoization declaration must be a boolean", code="invocable_memoize"
        )
    if fn is None:
        return lambda target: invocable(target, memoize=memoize, defaults=defaults)
    if not inspect.iscoroutinefunction(fn) or "<locals>" in fn.__qualname__:
        raise ConformanceError(
            "an invocable must be a module-level async function", code="invocable_signature"
        )
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
        call = broker.reserve("", export.module, export.name, arguments)
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


@dataclass(frozen=True, slots=True)
class _CallType:
    interface_digest: str
    module: str
    export: str
    request: type[msgspec.Struct]
    result: type[msgspec.Struct]


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
        bindings: Mapping[tuple[str, str, str], _CallType],
        exchange: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.request_id, self.bindings, self.exchange = request_id, dict(bindings), exchange
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

    def reserve(
        self,
        interface_digest: str,
        module: str,
        export: str,
        arguments: Mapping[str, object],
    ) -> PendingCall[Any]:
        ctx = self.context
        if self.closed or ctx is None or ctx.request_id != self.request_id:
            raise CapabilityError("package call escaped its attempt", code="escaped_handle")
        ctx.raise_if_cancelled()
        binding = self.bindings.get((interface_digest, module, export))
        if binding is None:
            raise CapabilityError(
                "export is not an exact locked dependency", code="child_undeclared"
            )
        if len(self.calls) >= MAX_ACTIVE_CALLS:
            raise CapabilityError("active package call bound exceeded", code="child_fanout")
        if self.next_index > MAX_CALL_INDEX:
            raise CapabilityError("package call index is exhausted", code="child_index_exhausted")
        try:

            def asset_value(value: object) -> object:
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
            payload = canonical_json.encode(
                msgspec.to_builtins(
                    request,
                    enc_hook=lambda value: (
                        value.ref if isinstance(value, (Asset, Tree)) else asset_value(value)
                    ),
                )
            )
        except (TypeError, ValueError, msgspec.ValidationError) as exc:
            raise ConformanceError(
                "package call arguments do not match its interface", code="child_arguments"
            ) from exc
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
                        answer = self.exchange("child_call", self._intent(ready))
                        self._observe(ready, answer)
                        self._check(answer, ready.index)
                        self.activated.add(ready.index)
                answer = self.exchange("child_poll", {"call_index": call.index})
                self._observe(call, answer)
                self._check(answer, call.index)
                if answer.get("state") == "succeeded":
                    raw = str(answer.get("result", "")).encode()
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
                            answer.get("byte_grants", []),
                            request_id=self.request_id,
                            guard=guard,
                            observation=call.observation,
                            grant_token=self.grant_token,
                        )
                        return cast(R, result)
                    except (ValueError, TypeError, msgspec.ValidationError) as exc:
                        raise ChildCallError(
                            "child_result_invalid",
                            "child result differs from its interface",
                            call.index,
                        ) from exc
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            if call.index in self.activated and not self.closed:
                self.exchange("child_cancel", {"call_index": call.index})
            raise
        finally:
            try:
                if call.index in self.activated and not self.closed:
                    self.exchange("child_forget", {"call_index": call.index})
            finally:
                self.activated.discard(call.index)
                self.calls.pop(call.index, None)

    @staticmethod
    def _observe(call: PendingCall[Any], answer: Mapping[str, Any]) -> None:
        child_id = answer.get("child_request_id")
        if child_id:
            if (
                not isinstance(child_id, str)
                or len(child_id) > 256
                or call.request_id not in ("", child_id)
            ):
                raise ChildCallError(
                    "child_identity_changed", "child request identity changed", call.index
                )
            call.request_id = child_id
        progress = answer.get("progress")
        if (
            isinstance(progress, dict)
            and type(progress.get("sequence")) is int
            and progress["sequence"] > call.progress_sequence
            and isinstance(progress.get("payload"), dict)
        ):
            call.progress_sequence = progress["sequence"]
            scope = call.progress_scope
            if scope is None and call.broker.telemetry is not None:
                scope = _ProgressScope(call.broker.telemetry, call.binding.export, None)
            if scope is not None:
                scope.telemetry._child_progress(progress["payload"], scope)
        if answer.get("observation") is None:
            return
        try:
            call.observation = msgspec.convert(
                answer["observation"],
                type=ExecutionObservation,
                dec_hook=asset_dec_hook,
                strict=True,
            )
        except (ValueError, TypeError, msgspec.ValidationError) as exc:
            raise ChildCallError(
                "child_observation_invalid", "child observation is invalid", call.index
            ) from exc

    @staticmethod
    def _intent(call: PendingCall[Any]) -> dict[str, Any]:
        return {
            "call_index": call.index,
            "interface_digest": call.binding.interface_digest,
            "module": call.binding.module,
            "export": call.binding.export,
            "payload": call.payload.decode(),
            **({"capture": call.capture.document()} if call.capture is not None else {}),
        }

    @staticmethod
    def _check(answer: Mapping[str, Any], index: int) -> None:
        if not answer.get("ok"):
            raise ChildCallError(
                str(answer.get("code", "child_unavailable")),
                str(answer.get("detail", "package call unavailable"))[:1024],
                index,
                str(answer.get("child_request_id", "")),
            )

    def finish(self) -> None:
        self.closed = True
        missing = list(self.calls)
        if missing:
            raise ChildCallError(
                "unawaited_child_call", "reserved package call was never awaited", missing[0]
            )


_current: ContextVar[_Broker | None] = ContextVar("cozy_package_call_attempt", default=None)


def _invoke_proxy(
    interface_digest: str,
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
    call = broker.reserve(interface_digest, module, export, arguments)
    call.capture = capture
    return call
