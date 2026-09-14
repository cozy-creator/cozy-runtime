from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest

from cozy_runtime.author import (
    ConformanceError,
    Invocation,
    ModelArtifact,
    attempt,
    describe,
    script_app,
)
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._script_types import _ScriptSource
from native_weights import NativeExecution
from test_weights_resume_native import _committed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("context", [False, True])
def test_plain_main_executes_only_inside_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, context: bool
) -> None:
    module = f"client_script_{int(asynchronous)}_{int(context)}"
    imported, ran = tmp_path / "imported", tmp_path / "ran"
    code = (
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('remote import')\n"
        f"{'async ' if asynchronous else ''}def main({'ctx' if context else ''}):\n"
    )
    if context:
        code += "    ctx.raise_if_cancelled()\n    ctx.log('preparing')\n"
        code += "    ctx.progress(0.5, stage='prepare')\n"
    code += f"    Path({str(ran)!r}).write_text('completed')\n"
    (tmp_path / (module + ".py")).write_text(code)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    app = script_app(module)
    describe(app)
    assert not imported.exists() and not ran.exists()
    result, outcome, _ = attempt(
        app.get("main"), {}, Invocation("script", tmp_path / "out", time.monotonic() + 5)
    )
    assert outcome.terminal == "succeeded" and result is not None, outcome
    assert imported.read_text() == "remote import"
    assert ran.read_text() == "completed"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("parameter", ["ctx: Context", "*, ctx: 'ExecutionContext'"])
def test_plain_main_receives_declared_execution_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, parameter: str
) -> None:
    module = f"typed_context_script_{int(asynchronous)}_{int(parameter.startswith('*'))}"
    imported = tmp_path / "imported"
    deadline = time.monotonic() + 5
    (tmp_path / (module + ".py")).write_text(
        "from pathlib import Path\n"
        "from cozy_runtime.author import Context, Context as ExecutionContext\n"
        f"Path({str(imported)!r}).touch()\n"
        f"{'async ' if asynchronous else ''}def main({parameter}) -> str:\n"
        "    assert isinstance(ctx, Context)\n"
        "    assert ctx.request_id == 'typed-script'\n"
        f"    assert ctx.deadline == {deadline!r}\n"
        "    ctx.raise_if_cancelled()\n"
        "    return str(ctx.device)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    app = script_app(module)
    describe(app)
    assert not imported.exists()
    result, outcome, _ = attempt(
        app.get("main"), {}, Invocation("typed-script", tmp_path / "out", deadline)
    )
    assert outcome.terminal == "succeeded" and result is not None, outcome
    assert imported.exists()
    assert result.result.value == "cpu"


@pytest.mark.parametrize(
    "body",
    [
        "def other(): pass",
        "def main(value): pass",
        "def main(*args): pass",
        "def main():\n    raise ValueError('quality gate')\n    yield 1",
        "async def main():\n    raise ValueError('quality gate')\n    yield 1",
        "def main():\n    yield from [1]",
    ],
)
def test_script_entrypoint_shape_refuses_without_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    (tmp_path / "bad_script.py").write_text("raise RuntimeError('must not import')\n" + body)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    with pytest.raises(ConformanceError):
        script_app("bad_script")


def test_nested_generator_does_not_change_main_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "nested_generator_script.py").write_text(
        "def main():\n    def values():\n        yield 7\n    assert list(values()) == [7]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    app = script_app("nested_generator_script")
    _, outcome, _ = attempt(
        app.get("main"), {}, Invocation("nested", tmp_path / "out", time.monotonic() + 5)
    )
    assert outcome.terminal == "succeeded"


def test_returned_generator_is_not_reported_as_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "returned_generator_script.py").write_text(
        "def work():\n"
        "    raise ValueError('quality gate')\n"
        "    yield 7\n"
        "def main():\n"
        "    return work()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    app = script_app("returned_generator_script")
    _, outcome, _ = attempt(
        app.get("main"), {}, Invocation("generator", tmp_path / "out", time.monotonic() + 5)
    )
    assert outcome.terminal == "failed"


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [(None, "succeeded"), (0, "succeeded"), (3, "failed"), ("quality gate", "failed")],
)
def test_script_exit_is_an_author_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: object, expected: str
) -> None:
    module = f"exit_script_{expected}_{type(exit_code).__name__}"
    (tmp_path / (module + ".py")).write_text(f"def main():\n    raise SystemExit({exit_code!r})\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    app = script_app(module)
    _, outcome, _ = attempt(
        app.get("main"), {}, Invocation("exit", tmp_path / "out", time.monotonic() + 5)
    )
    assert outcome.terminal == expected


def test_script_observes_cancellation_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "side_effect"
    (tmp_path / "canceled_script.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef main(): pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    app = script_app("canceled_script")
    _, outcome, _ = attempt(
        app.get("main"),
        {},
        Invocation("canceled", tmp_path / "out", time.monotonic() + 5, cancel=lambda: True),
    )
    assert outcome.terminal == "canceled"
    assert not marker.exists()


@pytest.mark.parametrize("with_context", [False, True])
def test_plain_main_uses_real_native_weights_and_returns_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_context: bool
) -> None:
    store, _, _, source_receipt = _committed(tmp_path)
    source = source_receipt.artifact.manifest.digest
    module = "native_inheritance_script_" + str(with_context)
    (tmp_path / (module + ".py")).write_text(
        "# /// script\n# [tool.cozy.weights]\n# checkpoint = 2048\n# ///\n"
        "from cozy_runtime.author import Context, Model, ModelArtifact, Telemetry\n"
        "from tensorfs.derived import Derivation, Target\n"
        f"def main({'ctx: Context, ' if with_context else ''}*, "
        "source: Model[object], tel: Telemetry) -> ModelArtifact:\n"
        '    tel.log("inheriting")\n'
        "    with ctx.tensorfs_source(source) as capability:\n"
        "        view = capability.inspect()\n"
        "    assert sum(len(rows) for rows in view.components.values()) == 1\n"
        '    with ctx.output("checkpoint").open(Derivation({"source":view.source},\n'
        '        {"model":Target(source="source",source_component="model")},\n'
        '        {}, (("model","weight"),))) as transaction:\n'
        "        tel.progress(1.0)\n"
        "        return ctx.adopt_model(transaction.commit())\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    if not with_context:
        with pytest.raises(ConformanceError):
            script_app(module)
        return
    app = script_app(module)
    describe(app)
    host = NativeExecution(
        store, tmp_path, "script-weights", {"source": source_receipt.artifact}, {"checkpoint": 2048}
    )
    result, outcome, _ = attempt(
        app.get("main"),
        {},
        Invocation(
            "script-weights",
            tmp_path / "out",
            time.monotonic() + 10,
            models={"source": _derive_model(_ScriptSource, source)},
            tensorfs_source=host.client.source,
            tensorfs_output=host.client.open_output,
            tensorfs_adopt=host.client.adopt_model,
        ),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    assert isinstance(result.result.value, ModelArtifact)
    assert result.result.value.manifest == source_receipt.artifact.manifest
    assert result.result.value.producer_request_id == "script-weights"
    assert result.result.value.tensorfs_receipt_digest != source_receipt.tensorfs_receipt_digest


def test_plain_main_saves_and_returns_real_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = "image_script"
    (tmp_path / (module + ".py")).write_text(
        "from cozy_runtime.author import ImageAsset, ImageFrame, Outputs, Telemetry\n"
        "async def main(*, out: Outputs, tel: Telemetry) -> ImageAsset:\n"
        '    tel.log("saving")\n'
        '    return out.save_image(ImageFrame(2, 2, b"\\xff\\x00\\x00" * 4), format="png")\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    app = script_app(module)
    describe(app)
    result, outcome, _ = attempt(
        app.get("main"), {}, Invocation("script-image", tmp_path / "out", time.monotonic() + 10)
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    assert result.result.value.media_type == "image/png"
    assert len(result.outputs) == 1


@pytest.mark.parametrize(
    "declaration",
    [
        "def main(*, source): pass",
        "def main(*, source: unknown()): pass",
        "def main(*, source: Model = None): pass",
        "def main(*, source: int): pass",
        "def main(ctx: int): pass",
        "def main(ctx: unknown()): pass",
        "# /// script\n# [tool.cozy.weights]\n# result = -1\n# ///\ndef main(): pass",
        '# /// script\n# [tool.cozy.models]\n# missing = "a/b"\n# ///\ndef main(): pass',
    ],
)
def test_script_injected_contract_refuses_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declaration: str
) -> None:
    (tmp_path / "bad_capability_script.py").write_text(
        'from cozy_runtime.author import Model\nraise RuntimeError("not executed")\n' + declaration
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ConformanceError, match="script"):
        script_app("bad_capability_script")
