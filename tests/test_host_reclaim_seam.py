"""Two real control streams cannot deadlock on reciprocal idle reclamation."""

import os
import socket
import threading

from cozy_runtime.author._executor_requests import Answer, HostRoom, encode
from cozy_runtime.internal import proctree
from cozy_runtime.internal.executor_commands import Probe, ReclaimHost
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker.child import Executor


def endpoint():
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


def test_reciprocal_midcall_host_requests_never_wait_for_other_stream():
    a, pa = endpoint()
    b, pb = endpoint()
    barrier = threading.Barrier(2)
    answers = []
    errors = []

    def peer(channel):
        try:
            assert channel.recv()["cmd"] == "probe"
            channel.send(
                {"event": "request", "seq": 1, **encode(HostRoom(reservation="copy", nbytes=1))}
            )
            assert channel.recv()["ok"]
            channel.send({"reply": "probe", "ok": True})
        except BaseException as error:
            errors.append(error)

    def call(ours, other):
        def room(_):
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


def test_idle_call_rechecks_eligibility_after_stream_acquisition_without_sending():
    child, peer = endpoint()
    try:
        answer = child.call_if_idle(ReclaimHost(nbytes=1), eligible=lambda: False)
        assert answer["code"] == "executor_busy" and not child._calls.locked()
        # A refused call must not leave an unmatched command on the peer stream.
        seen = []

        def respond():
            frame = peer.recv()
            seen.append(frame["cmd"])
            peer.send({"reply": frame["cmd"], "ok": True})

        thread = threading.Thread(target=respond)
        thread.start()
        observed = []
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
