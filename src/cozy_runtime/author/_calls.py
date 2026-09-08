"""Typed package calls; only the admitted attempt supplies their hidden broker."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Concatenate, ParamSpec, TypeVar, cast, overload

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._assets import Asset, Assets, Tree, asset_dec_hook
from cozy_runtime.author._context import Context
from cozy_runtime.author._errors import CapabilityError, ConformanceError

P = ParamSpec("P")
R = TypeVar("R")
MAX_DIRECT_CALLS = 32
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


@overload
def invocable[**P, R](
    fn: Callable[Concatenate[Context, P], Awaitable[R]], /, *, memoize: bool = False
) -> Callable[P, Awaitable[R]]: ...


@overload
def invocable[**P, R](
    *, memoize: bool = False
) -> Callable[[Callable[Concatenate[Context, P], Awaitable[R]]], Callable[P, Awaitable[R]]]: ...


def invocable[**P, R](
    fn: Callable[Concatenate[Context, P], Awaitable[R]] | None = None,
    /,
    *,
    memoize: bool = False,
) -> (
    Callable[P, Awaitable[R]]
    | Callable[[Callable[Concatenate[Context, P], Awaitable[R]]], Callable[P, Awaitable[R]]]
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
        return lambda target: invocable(target, memoize=memoize)
    if not inspect.iscoroutinefunction(fn) or "<locals>" in fn.__qualname__:
        raise ConformanceError(
            "an invocable must be a module-level async function", code="invocable_signature"
        )
    export = _Export(fn, fn.__module__, fn.__name__, memoize)

    def proxy(*args: P.args, **kwargs: P.kwargs) -> Awaitable[R]:
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
        return cast(Awaitable[R], broker.reserve("", export.module, export.name, kwargs))

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


class _Call[R](Awaitable[R]):
    def __init__(
        self, broker: _Broker, index: int, binding: _CallType, payload: bytes, ctx: Context
    ):
        self.broker, self.index, self.binding, self.payload = broker, index, binding, payload
        self.ctx = ctx
        self.task: asyncio.Task[R] | None = None

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
        if isinstance(value, Asset) and value._read_guard is not None:
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
        self.calls: list[_Call[Any]] = []
        self.closed = False
        self.activated: set[int] = set()
        self.context: Context | None = None

    def bind(self, ctx: Context) -> None:
        if self.closed or self.context is not None or ctx.request_id != self.request_id:
            raise CapabilityError("package broker escaped its attempt", code="escaped_handle")
        self.context = ctx

    def reserve(
        self,
        interface_digest: str,
        module: str,
        export: str,
        arguments: Mapping[str, object],
    ) -> _Call[Any]:
        ctx = self.context
        if self.closed or ctx is None or ctx.request_id != self.request_id:
            raise CapabilityError("package call escaped its attempt", code="escaped_handle")
        ctx.raise_if_cancelled()
        binding = self.bindings.get((interface_digest, module, export))
        if binding is None:
            raise CapabilityError(
                "export is not an exact locked dependency", code="child_undeclared"
            )
        if len(self.calls) >= MAX_DIRECT_CALLS:
            raise CapabilityError("direct package call bound exceeded", code="child_fanout")
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
        call = _Call[Any](self, len(self.calls), binding, payload, ctx)
        self.calls.append(call)
        return call

    async def run(self, call: _Call[R]) -> R:
        # All tasks in an ordinary gather become ready before activation. Flush
        # them by reserved index, regardless of scheduling/completion order.
        await asyncio.sleep(0)
        try:
            while True:
                call.ctx.raise_if_cancelled()
                if self.closed:
                    raise ChildCallError("escaped_handle", "parent attempt closed", call.index)
                for ready in self.calls:
                    if ready.task is not None and ready.index not in self.activated:
                        answer = self.exchange("child_call", self._intent(ready))
                        self._check(answer, ready.index)
                        self.activated.add(ready.index)
                answer = self.exchange("child_poll", {"call_index": call.index})
                self._check(answer, call.index)
                if answer.get("state") == "succeeded":
                    raw = str(answer.get("result", "")).encode()
                    if len(raw) > MAX_CALL_BYTES:
                        raise ChildCallError(
                            "child_result_bound", "child result exceeds inline bound", call.index
                        )
                    try:
                        value = canonical_json.decode(raw)
                        return cast(
                            R, msgspec.convert(value, type=call.binding.result, strict=True)
                        )
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

    @staticmethod
    def _intent(call: _Call[Any]) -> dict[str, Any]:
        return {
            "call_index": call.index,
            "interface_digest": call.binding.interface_digest,
            "module": call.binding.module,
            "export": call.binding.export,
            "payload": call.payload.decode(),
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
        missing = [call.index for call in self.calls if call.task is None]
        if missing:
            raise ChildCallError(
                "unawaited_child_call", "reserved package call was never awaited", missing[0]
            )


_current: ContextVar[_Broker | None] = ContextVar("cozy_package_call_attempt", default=None)


def _invoke_proxy(
    interface_digest: str, module: str, export: str, arguments: Mapping[str, object]
) -> Awaitable[Any]:
    broker = _current.get()
    if broker is None:
        raise CapabilityError("package call has no admitted attempt", code="child_broker_absent")
    return broker.reserve(interface_digest, module, export, arguments)
