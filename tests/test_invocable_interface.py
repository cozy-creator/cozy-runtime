from __future__ import annotations

import asyncio
import hashlib
import inspect
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import msgspec
import pytest

from conftest import mypy_cache
from cozy_runtime.author import (
    App,
    CapabilityError,
    ConformanceError,
    Context,
    Invocation,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.internal import canonical, interface_wheel, package_interface
from cozy_runtime.internal.discovery import Discovered
from durable_seam import wire


class Detail(msgspec.Struct, frozen=True):
    length: int


class Result(msgspec.Struct, frozen=True):
    text: str
    detail: Detail


class Empty(msgspec.Struct):
    pass


@invocable(memoize=True)
async def transform(ctx: Context, *, text: str, count: int = 2) -> Result:
    ctx.raise_if_cancelled()
    return Result(text * count, Detail(len(text) * count))


async def invalid_positional(ctx: Context, text: str) -> Result:
    return Result(text, Detail(len(text)))


async def invalid_bytes(ctx: Context, *, data: bytes) -> Result:
    return Result(str(data), Detail(len(data)))


async def invalid_default(ctx: Context, *, values: list[int] = []) -> Result:  # noqa: B006
    return Result(str(values), Detail(len(values)))


def described() -> tuple[App, bytes]:
    app = App()
    app.job(transform)
    found = Discovered(
        app, __name__ + ":app", Path(__file__).parent, sys.modules[__name__], describe(app), {}
    )
    return app, package_interface.canonical_bytes(package_interface.build(found))


def test_invocable_dispatch_uses_private_body_and_export_requires_owner(tmp_path: Path) -> None:
    app, raw = described()
    assert not inspect.iscoroutinefunction(transform)
    assert list(inspect.signature(transform).parameters) == ["text", "count"]
    for forbidden in ("raw", "local", "__wrapped__"):
        assert not hasattr(transform, forbidden)
    with pytest.raises(CapabilityError, match="child_broker_absent"):
        transform(text="hello")
    result, outcome, _ = attempt(
        app.get("transform"), {"text": "hi"}, Invocation("request", tmp_path, time.monotonic() + 5)
    )
    assert outcome.terminal == "succeeded"
    assert result is not None and result.result == Result("hihi", Detail(4))
    descriptor = package_interface.read_bytes(raw)["jobs"][0]
    assert descriptor["invocable"]["defaults"] == {"request/count": 2}
    assert descriptor["invocable"]["memoize"] is True
    assert descriptor["invocable"]["type_names"]["result/detail"] == "Detail"


def test_legacy_memo_declaration_refuses_instead_of_silently_disabling_cache() -> None:
    with pytest.raises(TypeError, match="reusable"):
        cast(Any, invocable)(reusable=True)
    _, raw = described()
    document = canonical.parse_canonical(raw)
    declaration = document["jobs"][0]["invocable"]
    declaration["reusable"] = declaration.pop("memoize")
    with pytest.raises(ConformanceError, match="missing required field `memoize`"):
        package_interface.read_bytes(canonical.write(document))


def test_additive_invocable_metadata_from_another_runtime_still_generates() -> None:
    _, raw = described()
    document = canonical.parse_canonical(raw)
    declaration = document["jobs"][0]["invocable"]
    declaration["future_hint"] = {"added_by": "a newer Runtime"}
    declaration["type_names"]["request/removed"] = "Removed"
    declaration["defaults"]["request/removed"] = 3
    declaration["capabilities"] = ["network-read", "gpu", "gpu"]
    evolved = canonical.write(document)
    package_interface.read_bytes(evolved)

    def generated(interface: bytes) -> dict[str, bytes]:
        digest = hashlib.sha256(interface).hexdigest().encode()
        return {
            name: body.replace(digest, b"<interface>")
            for name, body in interface_wheel.generate(interface).items()
        }

    assert generated(evolved) == generated(raw)


@pytest.mark.parametrize("fn", [invalid_positional, invalid_bytes, invalid_default])
def test_unportable_invocable_signature_refuses_before_execution(
    fn: Callable[..., Awaitable[Result]],
) -> None:
    app = App()
    app.job(invocable(fn))
    with pytest.raises(ConformanceError):
        describe(app)


def test_generated_interface_is_deterministic_typed_and_imports_without_implementation(
    tmp_path: Path,
) -> None:
    _, raw = described()
    files = interface_wheel.generate(raw)
    assert files == interface_wheel.generate(raw)
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        if name.endswith(".py"):
            compile(body, name, "exec")
    # A fresh interpreter imports only generated code. It cannot reach the actual
    # transformation body, and calling the proxy outside admission refuses.
    check = """
import importlib,inspect,sys,time,typing
sys.path.insert(0,sys.argv[1])
m=importlib.import_module(sys.argv[2])
assert m.__file__.startswith(sys.argv[1])
assert not inspect.iscoroutinefunction(m.transform)
assert typing.get_type_hints(m.transform)['return'].__args__[0] is m.Result
assert typing.get_type_hints(m.Result)['detail'] is m.Detail
from cozy_runtime.author import CapabilityError
assert list(inspect.signature(m.transform).parameters)==['text','count']
try:m.transform(text='hello')
except CapabilityError as e:assert e.code=='child_broker_absent'
else:raise AssertionError('generated proxy executed without its RecordOwner')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", check, str(tmp_path), __name__],
        check=True,
        capture_output=True,
        text=True,
    )
    caller = tmp_path / "caller.py"
    caller.write_text(
        f"from {__name__} import transform,Result\n"
        "async def main()->Result:\n"
        "    return await transform(text='hello',count=3)\n"
    )
    checked = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--cache-dir", str(mypy_cache()), str(caller)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    caller.write_text(
        f"from {__name__} import transform,Result\n"
        "async def main()->Result:\n"
        "    return await transform(text='hello',count='wrong')\n"
    )
    refused = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--cache-dir", str(mypy_cache()), str(caller)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "arg-type" in refused.stdout


def test_ordinary_main_reserves_self_call_with_implicit_attempt_lifetime(tmp_path: Path) -> None:
    from cozy_runtime.author._calls import ChildCallError, _Broker, _CallType

    declared, _ = described()
    operation = next(surface for surface in describe(declared) if surface.name == "transform")
    assert isinstance(operation.payload_type, type) and issubclass(
        operation.payload_type, msgspec.Struct
    )
    held: list[Awaitable[Result]] = []

    def main() -> None:
        held.append(transform(text="hello"))

    app = App()

    @app.job
    async def wrapper(ctx: Context, payload: Empty) -> Result:
        main()
        return Result("parent", Detail(0))

    def no_exchange(kind: str, body: dict[str, object]) -> dict[str, object]:
        raise AssertionError("an unawaited reservation must not reach the RecordOwner")

    broker = _Broker(
        "parent",
        {
            (__name__, "transform"): _CallType(
                "sha256:" + "13" * 32,
                __name__,
                "transform",
                operation.payload_type,
                Result,
            )
        },
        wire(no_exchange),
    )
    _, outcome, _ = attempt(
        app.get("wrapper"),
        {},
        Invocation("parent", tmp_path, time.monotonic() + 60, calls=broker),
    )
    assert outcome.code == "unawaited_child_call", outcome
    assert broker.context is not None and broker.context.request_id == "parent"
    assert broker.calls[0].ctx is broker.context
    assert len(held) == 1 and broker.closed

    async def escaped() -> Result:
        return await held[0]

    with pytest.raises(ChildCallError, match="escaped_handle"):
        asyncio.run(escaped())
    with pytest.raises(CapabilityError, match="child_broker_absent"):
        transform(text="outside")


def test_implicit_operation_call_observes_actual_parent_cancellation(tmp_path: Path) -> None:
    from cozy_runtime.author._calls import _Broker

    app = App()
    cancelled = False

    def main() -> None:
        nonlocal cancelled
        cancelled = True
        transform(text="canceled")

    @app.job
    def wrapper(ctx: Context, payload: Empty) -> Result:
        main()
        raise AssertionError("canceled attempt reserved another child")

    def no_exchange(kind: str, body: dict[str, object]) -> dict[str, object]:
        raise AssertionError("canceled attempt cannot reach the RecordOwner")

    _, outcome, _ = attempt(
        app.get("wrapper"),
        {},
        Invocation(
            "parent",
            tmp_path,
            time.monotonic() + 60,
            cancel=lambda: cancelled,
            calls=_Broker("parent", {}, wire(no_exchange)),
        ),
    )
    assert outcome.terminal == "canceled", outcome
