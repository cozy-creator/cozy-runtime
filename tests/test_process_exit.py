"""Kernel exit observation stays exact and non-reaping above select's FD_SETSIZE."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from cozy_runtime.internal import proctree


@pytest.mark.skipif(sys.platform != "linux", reason="Runtime process observation uses pidfds")
@pytest.mark.parametrize("minimum_fd", [0, 2048])
def test_exit_observer_waits_without_reaping(minimum_fd: int) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read(); sys.exit(23)"],
        stdin=subprocess.PIPE,
    )
    observer = proctree.observe_process_exit(proctree.process_identity(child.pid))
    try:
        if minimum_fd:
            # Duplicate the real pidfd above the limit without opening thousands of files.
            descriptor = fcntl.fcntl(observer.descriptor, fcntl.F_DUPFD_CLOEXEC, minimum_fd)
            observer.close()
            observer = proctree.ProcessExitObserver(observer.process, descriptor)
        assert observer.descriptor >= minimum_fd
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(observer.wait)
            try:
                with pytest.raises(TimeoutError):
                    waiting.result(timeout=0.1)
            finally:
                assert child.stdin is not None
                child.stdin.close()
            waiting.result(timeout=5)

        # No Popen.poll/wait until this proof: both would reap the process themselves.
        status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        assert status is not None
        assert status.si_pid == child.pid
        assert status.si_code == os.CLD_EXITED
        assert status.si_status == 23
        assert proctree.process_identity(child.pid) == observer.process
        # An already-exited, unreaped process must also remain immediately observable.
        observer.wait()
        assert child.wait(timeout=5) == 23
    finally:
        observer.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Runtime process observation uses pidfds")
@pytest.mark.parametrize("sig", [signal.SIGBUS, None])
def test_exit_peek_reads_how_a_child_ended_without_reaping(sig: signal.Signals | None) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read(); sys.exit(23)"],
        stdin=subprocess.PIPE,
    )
    process = proctree.process_identity(child.pid)
    observer = proctree.observe_process_exit(process)
    ended = -sig if sig else 23
    try:
        assert not proctree.dying(process)
        assert proctree.peek_child_exit(process) is None, "still running"
        assert child.stdin is not None
        if sig:
            os.kill(child.pid, sig)
        else:
            child.stdin.close()
        observer.wait()
        assert proctree.dying(process)
        status = proctree.peek_child_exit(process)
        assert status is not None and os.waitstatus_to_exitcode(status) == ended
        # Still the same unreaped zombie: the peek repeats, and the one real reap answers.
        assert proctree.process_state(process) == "Z"
        assert proctree.peek_child_exit(process) == status
        assert child.wait(timeout=5) == ended
        assert proctree.peek_child_exit(process) is None
    finally:
        observer.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
