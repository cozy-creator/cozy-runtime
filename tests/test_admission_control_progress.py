"""A real control-seam peer blocks preparation while Worker control changes its fences.

The peer never imports a model, invokes an entrypoint, or opens an accelerator context.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cozy_runtime.internal import proctree
from cozy_runtime.internal.executor_commands import Vacate
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker.child import Executor, LoadedConstruction
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import DeclaredBinding, PreparedModel
from cozy_runtime.internal.worker.session import Placement, Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_lazy_model_admission import BINDING, INSTALLATION, offer

_PEER = """
import socket, sys
from cozy_runtime.internal.seam import Channel
channel = Channel(socket.socket(fileno=int(sys.argv[1])))
gate = socket.socket(fileno=int(sys.argv[2]))
while (command := channel.recv()) is not None:
    assert command['cmd'] in ('prepare_request', 'vacate'), command
    if command['cmd'] == sys.argv[3]:
        gate.sendall(b'p')
        assert gate.recv(1) == b'c'
    channel.send({'reply': command['cmd'], 'ok': True})
"""


@pytest.mark.parametrize("change", ["unchanged", "cancel", "binding", "executor"])
@pytest.mark.parametrize("phase", ["prepare_request", "vacate", "lifecycle"])
def test_control_remains_responsive_and_commit_rechecks_prepared_fences(
    tmp_path: Path, change: str, phase: str
) -> None:
    worker = Worker(
        _config(tmp_path / "home"), WorkerOptions(root=tmp_path / "worker"), InMemoryControlHost()
    )
    worker.placement = Placement(
        pb.Placement(placement_id="model", installation_id=INSTALLATION, bindings_digest=b"s" * 32),
        b"p" * 32,
    )
    declared = DeclaredBinding(
        entrypoint_binding_digest=BINDING,
        entrypoint="generate",
        model_class="fixture:Model",
        model_binding_path="model",
        model_parameter_name="model",
        release="sha256:" + "d" * 64,
        installation_id=INSTALLATION,
    )
    worker.bindings[BINDING] = declared
    ours, theirs = socket.socketpair()
    gate, peer_gate = socket.socketpair()
    gate.settimeout(5)
    peer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _PEER,
            str(theirs.fileno()),
            str(peer_gate.fileno()),
            "prepare_request" if phase == "lifecycle" else phase,
        ],
        pass_fds=(theirs.fileno(), peer_gate.fileno()),
    )
    theirs.close()
    peer_gate.close()
    executor = Executor(
        epoch=1,
        process=proctree.process_identity(peer.pid),
        scope=proctree.ProcessTreeScope("admission-control-proof", os.getuid()),
        channel=Channel(ours),
        seal_digest="",
        ready_bindings={BINDING},
    )
    executor.loaded[declared.construction_key()] = LoadedConstruction(
        declared.construction_key(), PreparedModel(), frozenset({"model"}), frozenset({BINDING})
    )
    worker.supervision.current = executor
    attempt = worker.engine.offer(offer("blocked-prepare"))
    worker._bind_attempt_slot(attempt)
    worker.engine.hold(attempt)
    worker.engine.stage(attempt)
    assert worker.engine.claim_staged(attempt)
    results: list[bool] = []
    failures: list[BaseException] = []

    def enter() -> None:
        try:
            with worker.lanes.by_id[attempt.lane_id].device:
                if phase != "vacate":
                    results.append(worker.enter(attempt))
                else:
                    slot = worker._bind_attempt_slot(attempt)

                    def reclaim() -> None:
                        executor.call(Vacate(), timeout=None)

                    try:
                        results.append(
                            worker.engine.enter(
                                attempt,
                                before_plan=reclaim,
                                commit=lambda finish: worker._commit_admission(
                                    attempt, slot, finish
                                ),
                            )
                        )
                    except Exception as exc:
                        from cozy_runtime.internal.worker.attempts import AttemptRefusal

                        if not isinstance(exc, AttemptRefusal):
                            raise
                        results.append(worker._admission_failed(attempt, exc))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=enter)
    thread.start()
    try:
        assert gate.recv(1) == b"p"  # The actual executor command is blocked on its peer.
        hold = (
            worker.supervision.hold(executor) if phase == "lifecycle" else contextlib.nullcontext()
        )
        with hold:
            if phase == "lifecycle":
                gate.sendall(b"c")
                deadline = time.monotonic() + 5
                while True:
                    frame = sys._current_frames().get(thread.ident or 0)
                    while frame is not None:
                        if (
                            frame.f_code.co_name == "hold"
                            and frame.f_locals.get("self") is worker.supervision
                        ):
                            break
                        frame = frame.f_back
                    if frame is not None:
                        break
                    assert time.monotonic() < deadline, "commit never reached lifecycle wait"
                    threading.Event().wait(0.001)
            assert worker.control_lock.acquire(timeout=1), "preparation blocked Worker control"
            try:
                if change == "cancel":
                    worker.cancel(
                        pb.CancelAttempt(
                            request_id=attempt.request_id,
                            attempt_ordinal=attempt.attempt,
                            invocation_spec_digest=attempt.digest,
                            reason=pb.CANCEL_REASON_CLIENT,
                        )
                    )
                elif change == "binding":
                    worker.bindings.clear()
                elif change == "executor":
                    worker.supervision.current = None
            finally:
                worker.control_lock.release()
        if phase != "lifecycle":
            gate.sendall(b"c")
        thread.join(5)
        assert not thread.is_alive() and not failures
        assert results == [change == "unchanged"]
        assert len(worker.records.of_kind("accepted")) == 1
        assert len(worker.records.of_kind("entered")) == (1 if change == "unchanged" else 0)
        assert len(worker.records.of_kind("outcome")) == (0 if change == "unchanged" else 1)
        if change == "cancel":
            assert attempt.canceling == "client"
    finally:
        if thread.is_alive():
            gate.sendall(b"c")
            thread.join(5)
        worker.supervision.current = None  # The test, not production supervision, owns this peer.
        executor.channel.close()
        gate.close()
        peer.wait(timeout=5)
        worker.shutdown()
