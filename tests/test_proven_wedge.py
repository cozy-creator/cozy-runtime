"""A wedged operator child is ended on the measurement; a working one runs to its answer."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, child_env, liveness, native_wheel, spawn
from test_end_to_end import NO_EXECUTOR

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

SAMPLE = 0.1


@pytest.fixture(autouse=True)
def fine_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", SAMPLE)


def floor() -> float:
    return liveness.noise_floor()


def _still(verdict: str) -> float:
    """The stillness a wedge verdict reports it measured before ending the subject."""
    found = re.search(r"no measurable progress for ([0-9.]+) s", verdict)
    assert found, verdict
    return float(found.group(1))


def _base_env() -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, v) for k, v in os.environ.items() if not child_env.erased(k)))


def _operator(code: str) -> dict[str, object]:
    """One real fixture run: `exec(code)` in the runner, under its rlimits, answering null."""
    return native_wheel.qualify_operator(
        python=Path(sys.executable),
        fixture="builtins:exec",
        expected_result_digest=canonical.digest(None),
        environment=RUNNER_ENV,
        request=code,
    )


def test_an_operator_child_that_keeps_working_runs_to_its_answer() -> None:
    """Steady work far past the noise floor is not a wedge: the fixture runs to exit 0."""
    seconds = 4 * floor()
    started = time.monotonic()
    result = _operator(
        "import hashlib, time\n"
        "digest = b'x'\n"
        f"until = time.monotonic() + {seconds}\n"
        "while time.monotonic() < until:\n"
        "    digest = hashlib.sha256(digest).digest()\n"
    )
    elapsed = time.monotonic() - started
    assert result["result_digest"] == canonical.digest(None)
    assert elapsed >= seconds, f"{elapsed:.1f}s: the fixture did not run its course"


def test_an_operator_child_that_stops_dead_is_ended_on_the_measurement() -> None:
    """`signal.pause()` burns nothing forever; the child is killed with its group, by name."""
    started = time.monotonic()
    with pytest.raises(native_wheel.NativeWheelRefusal) as caught:
        _operator("import signal\nsignal.pause()\n")
    elapsed = time.monotonic() - started
    assert caught.value.code == "package_environment_native_operator_wedged"
    assert "no measurable progress for" in caught.value.detail
    assert elapsed > floor(), f"{elapsed:.1f}s: ended before the noise floor"
    # The measured stillness, not wall time: interpreter startup under load is not stillness.
    assert _still(caught.value.detail) < 6 * floor(), "the wait was a clock"


RUNNER_ENV = spawn.seal(_base_env(), {})
