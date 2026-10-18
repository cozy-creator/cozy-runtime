"""A process that arms parent death is gone once its parent is killed, on every POSIX
platform: Linux's PR_SET_PDEATHSIG, darwin's watching thread."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX parent death")

CHILD = """
import os, subprocess, sys
watcher = subprocess.Popen(
    [sys.executable, "-c", "import sys, time\\nfrom cozy_runtime.internal import proctree\\n"
     "proctree.arm_parent_death()\\nprint('armed', flush=True)\\ntime.sleep(600)"],
    stdout=subprocess.PIPE, text=True,
)
assert watcher.stdout.readline().strip() == "armed"
print(watcher.pid, flush=True)
time.sleep(600)
"""


def test_a_process_armed_against_parent_death_dies_with_its_parent() -> None:
    parent = subprocess.Popen(
        [sys.executable, "-c", "import time\n" + CHILD], stdout=subprocess.PIPE, text=True
    )
    assert parent.stdout is not None
    watcher = int(parent.stdout.readline())
    parent.kill()
    parent.wait()
    # A hang bound for a watcher that never dies; the product waits without one.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(watcher, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(watcher, signal.SIGKILL)
    pytest.fail("the armed process outlived its parent")
