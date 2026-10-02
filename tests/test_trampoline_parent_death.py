"""Real Linux credential/exec/parent-death boundaries, including the privileged caveat."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.exits import Exit

FIXTURE = Path(__file__).parent / "testdata/trampoline_parent_death.py"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux PDEATHSIG contract")
CREDENTIALS = [
    pytest.param(-1, -1, id="default-credentials"),
    pytest.param(os.geteuid(), os.getegid(), id="same-credentials"),
    pytest.param(
        65532,
        65532,
        id="foreign-credentials",
        marks=pytest.mark.skipif(os.geteuid() != 0, reason="requires privileged fixture"),
    ),
]


@pytest.mark.parametrize(("uid", "gid"), CREDENTIALS)
def test_guard_is_armed_after_credentials_and_exec(uid: int, gid: int) -> None:
    result = subprocess.run(
        [sys.executable, str(FIXTURE), "report", str(uid), str(gid)],
        capture_output=True,
        text=True,
        check=True,
    )
    facts = json.loads(result.stdout)
    assert facts["guard"] == signal.SIGKILL
    assert facts["uid"] == (uid if uid >= 0 else os.geteuid())
    assert facts["gid"] == (gid if gid >= 0 and os.geteuid() == 0 else os.getegid())
    assert facts["no_new_privs"] == 1


@pytest.mark.parametrize(("uid", "gid"), CREDENTIALS)
def test_actual_creator_death_kills_the_executor(uid: int, gid: int) -> None:
    with subprocess.Popen(
        [sys.executable, str(FIXTURE), "death", str(uid), str(gid)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as fixture:
        assert fixture.stdout is not None
        assert fixture.stdin is not None
        released = False
        try:
            facts = json.loads(fixture.stdout.readline())
            assert facts["guard"] == signal.SIGKILL
            fixture.stdin.write("release")
            fixture.stdin.flush()
            released = True
        finally:
            if not released:
                fixture.stdin.write("cleanup")
                fixture.stdin.flush()
            fixture.stdin.close()
        terminal = json.loads(fixture.stdout.readline())
        assert terminal["returncode"] == -signal.SIGKILL
        assert fixture.wait() == 0


@pytest.mark.skipif(os.geteuid() != 0, reason="requires privileged store-owner fixture")
def test_native_foreign_store_keeps_the_guard_or_records_known_gap() -> None:
    result = subprocess.run(
        [sys.executable, str(FIXTURE), "filesystem_identity"],
        capture_output=True,
        text=True,
        check=True,
    )
    facts = json.loads(result.stdout)
    assert facts["before"] == signal.SIGKILL and facts["uid"] == 0
    if facts["after"] == 0:
        pytest.xfail("native foreign Store-owner fscredentials clear PDEATHSIG after trampoline")
    assert facts["after"] == signal.SIGKILL


def test_wrong_expected_parent_refuses_before_exec() -> None:
    result = subprocess.run(
        [sys.executable, str(FIXTURE), "orphan"], capture_output=True, text=True, check=False
    )
    assert result.returncode == Exit.structural
    assert "expected parent" in result.stderr
    assert result.stdout == ""
