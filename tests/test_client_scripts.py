from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest

from cozy_runtime.author import ConformanceError, Invocation, attempt, describe, script_app


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
