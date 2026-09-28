"""The public native facades reserve real admitted calls before any byte work."""

import time
from pathlib import Path

import msgspec

from cozy_runtime.author import App, Context, Invocation, ObjectRef, SourceArtifact, attempt
from cozy_runtime.author._calls import _Broker
from cozy_runtime.author.sources import convert_cozytensors, download_civitai, download_huggingface
from cozy_runtime.internal import source_interfaces
from durable_seam import wire


class Empty(msgspec.Struct):
    pass


def test_public_source_facades_use_the_admitted_broker(tmp_path: Path) -> None:
    app = App()
    source = SourceArtifact(
        "source-owner", "source", ObjectRef("sha256:" + "11" * 32, 100), "sha256:" + "22" * 32
    )

    @app.job
    async def script(ctx: Context, payload: Empty) -> Empty:
        download_huggingface("example/model", revision="a" * 40)
        download_civitai(123)
        convert_cozytensors(source, profiles=("reviewed/model/1",))
        return Empty()

    def exchange(*_: object) -> dict[str, object]:
        raise AssertionError("unawaited source operations must not execute")

    broker = _Broker(
        "parent",
        {
            (binding.module, binding.export): binding
            for binding in source_interfaces.bindings().values()
        },
        wire(exchange),
    )
    _, outcome, _ = attempt(
        app.get("script"), {}, Invocation("parent", tmp_path, time.monotonic() + 10, calls=broker)
    )
    assert outcome.code == "unawaited_child_call"
    assert [(call.index, call.binding.export) for call in broker.calls.values()] == [
        (0, "download_huggingface"),
        (1, "download_civitai"),
        (2, "convert_cozytensors"),
    ]
    assert all(call.binding.module == source_interfaces.MODULE for call in broker.calls.values())
