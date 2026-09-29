"""Two real control streams cannot deadlock on reciprocal idle reclamation."""

import os
import socket
import threading
from typing import Any

import pytest

from cozy_runtime.author._executor_requests import Answer, HostRoom, Request, encode
from cozy_runtime.internal import proctree
from cozy_runtime.internal.executor_commands import Probe, ReclaimHost
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker.child import Executor


def endpoint() -> tuple[Executor, Channel]:
    ours, theirs = socket.socketpair()
    peer = Channel(theirs)
    child = Executor(
        epoch=1,
        process=proctree.process_identity(os.getpid()),
        scope=proctree.ProcessTreeScope("test-host-cache", os.getuid()),
        channel=Channel(ours),
        seal_digest="",
    )
    return child, peer


def test_reciprocal_midcall_host_requests_never_wait_for_other_stream() -> None:
    a, pa = endpoint()
    b, pb = endpoint()
    barrier = threading.Barrier(2)
    answers: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def peer(channel: Channel) -> None:
        try:
            command = channel.recv()
            assert command is not None and command["cmd"] == "probe"
            channel.send(
                {"event": "request", "seq": 1, **encode(HostRoom(reservation="copy", nbytes=1))}
            )
            answer = channel.recv()
            assert answer is not None and answer["ok"]
            channel.send({"reply": "probe", "ok": True})
        except BaseException as error:
            errors.append(error)

    def call(ours: Executor, other: Executor) -> None:
        def room(_: Request) -> Answer:
            barrier.wait(5)
            answer = other.call_if_idle(ReclaimHost(nbytes=1), eligible=lambda: True)
            answers.append(answer)
            barrier.wait(5)  # Both refusals happen while both originating calls remain held.
            return Answer(ok=True)

        try:
            assert ours.call(Probe(), on_request=room)["ok"]
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=peer, args=(pa,)),
        threading.Thread(target=peer, args=(pb,)),
        threading.Thread(target=call, args=(a, b)),
        threading.Thread(target=call, args=(b, a)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        assert not any(thread.is_alive() for thread in threads)
        assert not errors
        assert len(answers) == 2 and all(answer["code"] == "executor_busy" for answer in answers)
    finally:
        for channel in (a.channel, b.channel, pa, pb):
            channel.close()


def test_idle_call_rechecks_eligibility_after_stream_acquisition_without_sending() -> None:
    child, peer = endpoint()
    try:
        answer = child.call_if_idle(ReclaimHost(nbytes=1), eligible=lambda: False)
        assert answer["code"] == "executor_busy" and not child._calls.locked()
        # A refused call must not leave an unmatched command on the peer stream.
        seen: list[str] = []

        def respond() -> None:
            frame = peer.recv()
            assert frame is not None
            seen.append(frame["cmd"])
            peer.send({"reply": frame["cmd"], "ok": True})

        thread = threading.Thread(target=respond)
        thread.start()
        observed: list[tuple[bool, bool]] = []
        assert child.call_if_idle(
            Probe(),
            eligible=lambda: True,
            on_reply=lambda reply: observed.append((reply["ok"], child._calls.locked())),
        )["ok"]
        assert observed == [(True, True)]
        thread.join(5)
        assert seen == ["probe"]
    finally:
        child.channel.close()
        peer.close()


@pytest.mark.parametrize("mode", ["exit", "stalled"])
def test_idle_reclaim_ends_on_child_exit_or_proven_stalled_progress(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    import sys
    import time

    from cozy_runtime.internal import liveness
    from cozy_runtime.internal.worker.child import ExecutorGone

    # Increase observation resolution; the normal measured-progress rule and its
    # sample-derived noise floor remain the production implementation.
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", 0.05)
    ours, theirs = socket.socketpair()
    program = """
import os, signal, socket, sys
from cozy_runtime.internal.seam import Channel
channel = Channel(socket.socket(fileno=int(sys.argv[1])))
print('ready', flush=True)
if sys.argv[2] == 'stalled':
    os.kill(os.getpid(), signal.SIGSTOP)
channel.recv()
sys.exit(3)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(theirs.fileno()), mode],
        pass_fds=(theirs.fileno(),),
        stdout=subprocess.PIPE,
        text=True,
    )
    theirs.close()
    executor = Executor(
        epoch=1,
        process=proctree.process_identity(process.pid),
        scope=proctree.ProcessTreeScope("test-idle-reclaim-exit", os.getuid()),
        channel=Channel(ours),
        seal_digest="",
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        started = time.monotonic()
        with pytest.raises(ExecutorGone) as caught:
            executor.call_if_idle(ReclaimHost(nbytes=1), eligible=lambda: True)
        assert not executor._calls.locked()
        process.wait(timeout=5)
        if mode == "stalled":
            assert "wedged during 'reclaim_host'" in str(caught.value)
            assert "no measurable progress" in str(caught.value)
            assert time.monotonic() - started > liveness.noise_floor()
        else:
            assert process.returncode == 3 and not executor.wedged
    finally:
        executor.channel.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
