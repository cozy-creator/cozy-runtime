from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import msgspec
import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import (
    App,
    CapabilityError,
    Context,
    Invocation,
    Loader,
    Model,
    ModelArtifact,
    ObjectRef,
    Weights,
    WeightsOutput,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author.fakes import fake_context
from cozy_runtime.cli import describe as describe_cli
from cozy_runtime.cli.io import Options
from cozy_runtime.internal import interface_wheel, package_interface
from cozy_runtime.internal.discovery import Discovered
from durable_seam import wire
from native_weights import NativeExecution


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("Weights[Source] is never constructed")


class Result(msgspec.Struct):
    manifest: str


@invocable(memoize=True)
async def consume(ctx: Context, *, source: Weights[Source]) -> Result:
    ctx.raise_if_cancelled()
    return Result(source.checkpoint_ref)


@invocable(memoize=True)
async def echo(ctx: Context, *, value: ModelArtifact) -> ModelArtifact:
    return value


def test_injected_model_refuses_as_model_before_reserving_a_child() -> None:
    app = App()
    app.entrypoint(consume, accelerator=False)
    surface = describe(app)[0]
    assert isinstance(surface.payload_type, type) and issubclass(
        surface.payload_type, msgspec.Struct
    )
    assert isinstance(surface.result_type, type) and issubclass(surface.result_type, msgspec.Struct)
    key = ("sha256:" + "21" * 32, __name__, "consume")
    exchanges: list[tuple[str, dict[str, Any]]] = []

    def exchange(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        exchanges.append((kind, payload))
        return {}

    broker = _Broker(
        "parent",
        {key[1:]: _CallType(*key, surface.payload_type, surface.result_type)},
        wire(exchange),
    )
    broker.bind(fake_context(request_id="parent"))
    with pytest.raises(CapabilityError, match="retained ModelArtifact") as error:
        broker.reserve(*key[1:], {"source": Source.for_test()})
    assert error.value.code == "child.model_artifact_required"
    assert not broker.calls and broker.next_index == 0 and not exchanges

    artifact = ModelArtifact(
        "producer", "result", ObjectRef("sha256:" + "17" * 32, 512), "sha256:" + "18" * 32
    )
    call = broker.reserve(*key[1:], {"source": artifact})
    assert call.index == 0 and json.loads(call.payload)["source"] == msgspec.to_builtins(artifact)


@invocable
async def produce(ctx: Context) -> ModelArtifact:
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    with ctx.output("result").open(
        Derivation(
            sources={},
            targets={
                "model": Target(
                    add={"weight": Tensor("f32", (512,), plain, {"value": Part("f32", (512,))})}
                )
            },
            configs={},
            order=(("model", "weight"),),
        )
    ) as writer:
        writer.add_part("model", "weight", "value", b"\x31" * 2048)
        return ctx.adopt_model(writer.commit())


def test_author_native_creation_call_returns_its_model_artifact(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    host = NativeExecution(store, tmp_path, "request", {}, {"result": 2048})
    app = App()
    app.entrypoint(produce, weights=(WeightsOutput("result", 2048),), accelerator=False)
    describe(app)
    result, outcome, _ = attempt(
        app.get("produce"),
        {},
        Invocation(
            "request",
            tmp_path / "spool",
            time.monotonic() + 30,
            tensorfs_output=host.client.open_output,
            tensorfs_adopt=host.client.adopt_model,
        ),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and isinstance(result.result, ModelArtifact)
    assert result.result.producer_request_id == "request"
    assert store.manifest(result.result.manifest.digest)


def test_model_proxy_is_an_exact_artifact_and_dispatch_checks_bound_model(tmp_path: Path) -> None:
    app = App()
    app.entrypoint(consume, accelerator=False)
    app.entrypoint(echo, accelerator=False)
    found = Discovered(app, __name__ + ":app", tmp_path, sys.modules[__name__], describe(app), {})
    raw = package_interface.canonical_bytes(package_interface.build(found))
    metadata = package_interface.read_bytes(raw)
    consuming = next(row for row in metadata["entrypoints"] if row["name"] == "consume")
    # The raw argument is a model slot of the call's `models`, not a request field.
    assert consuming["invocable"]["parameters"] == [] and consuming["request"]["fields"] == []
    assert consuming["models"][0]["weights"] is True
    assert next(row for row in metadata["entrypoints"] if row["name"] == "echo")["result"] == {
        "input": "model"
    }
    interface_path = tmp_path / "package-interface.json"
    interface_path.write_bytes(raw)
    assert (
        describe_cli.run(
            None, Options(json=True, package_interface=str(interface_path))
        ).canonical_json
        == raw
    )
    assert (
        describe_cli.run("echo", Options(json=True, package_interface=str(interface_path))).document
        is not None
    )
    for name, body in interface_wheel.generate(raw).items():
        if name.endswith(".py"):
            compile(body, name, "exec")
    artifact = ModelArtifact(
        "producer", "result", ObjectRef("sha256:" + "17" * 32, 512), "sha256:" + "18" * 32
    )
    invocation = Invocation(
        "consumer",
        tmp_path / "valid",
        time.monotonic() + 5,
        sources={"source": Weights(artifact.manifest.digest)},
    )
    result, outcome, _ = attempt(
        app.get("consume"), {"source": msgspec.to_builtins(artifact)}, invocation
    )
    assert (
        outcome.terminal == "succeeded"
        and result is not None
        and result.result.manifest == artifact.manifest.digest
    )
    changed = msgspec.to_builtins(artifact)
    changed["manifest"]["digest"] = "sha256:" + "19" * 32
    _, refused, _ = attempt(app.get("consume"), {"source": changed}, invocation)
    assert refused.code == "model_artifact_binding"
    # Existing top-level fixed Model bindings remain valid without a derived handle.
    result, outcome, _ = attempt(app.get("consume"), {}, invocation)
    assert outcome.terminal == "succeeded" and result is not None
