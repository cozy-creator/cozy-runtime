"""Back-to-back attempts on one warm executor hand off without a probe round trip and without
a durable commit on the executor's path.

Real Worker, real durable journal, a real package environment and a real CPU executor serving
marco-polo's weightless `marco`, submitted as Creator submits a direct serving root. Nothing
on the path is replaced; the executor's commands and the journal's commits are only counted.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import accel, package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor_commands import Invoke, PrepareRequest
from cozy_runtime.internal.worker import child, workspace
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace_executions import TERMINAL
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_end_to_end import NO_EXECUTOR
from test_gpu_scheduler import MARCO, VIRTUAL, GiB, Machine, driverless

pytestmark = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

OWNER = "owner"
INSTALLATION = "release-handoff-fixture"
DIGEST = "sha256:" + hashlib.sha256(b"handoff-binding").hexdigest()
RUNS = 12
#: A card no host has: nothing sealed to it reaches a device.
CARD = VIRTUAL.split(",")[0]


def _install(root: Path) -> None:
    """A package environment as the worker records one: a venv reaching this interpreter's
    packages and marco-polo."""
    venv = root / "installations" / INSTALLATION / "venv"
    subprocess.run(
        [sys._base_executable, "-m", "venv", "--without-pip", str(venv)],  # type: ignore[attr-defined]
        check=True,
    )
    (site,) = (venv / "lib").glob("python*/site-packages")
    (site / "handoff_fixture.pth").write_text(
        f"import site; site.addsitedir({sysconfig.get_paths()['purelib']!r})\n{MARCO}\n"
    )
    (venv.parent / "installation.json").write_text(
        json.dumps({"python": str(venv / "bin" / "python"), "package": "t/m", "release": "1"})
    )


class Sent(NamedTuple):
    """One command a worker thread sent an executor, and when it was answered."""

    command: str
    thread: int
    sent: float
    answered: float


class Pod:
    def __init__(self, root: Path) -> None:
        self.worker = Worker(
            replace(_config(root / "home"), record_owner_public_key=signed_claims.PUBLIC_KEY),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                tensorfs_root=root / "store",
                install_root=root / "install",
                artifact_cache=root / "artifacts",
                devices="",
                threads=1,
                accelerator_backend="none",
            ),
            InMemoryControlHost(),
        )
        assert self.worker.accept_claim(signed_claims.claim(OWNER), lambda frame: None)[0] >= 1
        assert self.worker.executions is not None
        self.executions = self.worker.executions
        _install(root / "install")
        self.placement: dict[str, Any] = {
            "placement_id": "tmpl-handoff",
            "installation_id": INSTALLATION,
            "package_interface": base64.b64encode(
                package_interface.canonical_bytes(package_interface.build(discover(MARCO)))
            ).decode(),
            "bindings_digest": DIGEST,
            "package": {"package": "t/m", "release": "1"},
            "entrypoints": [{"name": "marco", "entrypoint_binding_digest": DIGEST}],
        }
        prepared = self.worker._placement_from_entry(
            documents.from_body(self.placement, pb.Placement), b""
        )
        self.worker.prepared_installations[prepared.prepared_key] = prepared

    def serve(self, name: str) -> None:
        """A direct serving root of `marco` with its payload."""
        raw_set = canonical_json.encode(
            {"format": "cozy.worker.v1.PlacementSet/1", "placements": [self.placement]}
        )
        state = pb.DesiredWorkerState(
            wire_minor=WIRE_MINOR,
            posture=pb.POSTURE_ACCEPTING,
            placement_set=pb.DesiredPlacementSet(
                placement_set_digest=hashlib.sha256(raw_set).digest(),
                placement_set_canonical_bytes=raw_set,
            ),
        )
        payload = canonical_json.encode({"message": "marco"})
        payload_digest = documents.spell(documents.digest_of(payload))
        spec, spec_digest = documents.identity(
            pb.InvocationSpec(
                installation_id=INSTALLATION,
                payload_digest=payload_digest,
                inputs=[
                    pb.InputBinding(
                        input_id="payload",
                        digest=payload_digest,
                        length=len(payload),
                        kind_mime="application/json",
                    )
                ],
                serving=pb.ServingInvocationSpec(
                    entrypoint_binding_digest=DIGEST,
                    attempt_binding_id=DIGEST,
                    bindings_digest=DIGEST,
                ),
            )
        )
        offer = pb.AttemptOffer(
            request_id=name,
            attempt_ordinal=1,
            placement_id=self.placement["placement_id"],
            invocation_spec_canonical_bytes=spec,
            invocation_spec_digest=spec_digest,
            grant=pb.DeliveryGrant(
                invocation_spec_digest=spec_digest,
                inputs=[
                    pb.InputAccess(
                        input_id="payload",
                        url="data:application/json;base64," + base64.b64encode(payload).decode(),
                    )
                ],
            ),
        )
        preparation = {
            "installations": {},
            "state": base64.b64encode(state.SerializeToString()).decode(),
        }
        self.executions.submit(
            OWNER,
            name,
            hashlib.sha256(name.encode()).digest(),
            offer,
            expected_execution_workspace_id=self.executions.workspace_id,
            worker_boot=self.worker.fence.worker_boot_id,
            preparation=canonical_json.encode(preparation),
            worker_id=self.worker.options.worker_id,
        )
        self.worker.execution(OWNER, name)

    def settle(self, name: str) -> str:
        deadline = time.monotonic() + 300  # a hang bound on a loaded box, not a budget
        while (state := self.executions.status(OWNER, name).state) not in TERMINAL:
            assert time.monotonic() < deadline, f"{name} never settled"
            time.sleep(0.01)
        return state

    def events(self, name: str) -> list[tuple[int, int, str, dict[str, Any]]]:
        return [
            (event.sequence, event.at_ms, event.kind, canonical_json.decode(event.body))
            for event in self.executions.events(OWNER, name).events
        ]


class Commits:
    """(thread, ms) of every journal commit made at `synchronous=FULL`: the ones a caller
    waits on an fsync for."""

    def __init__(self) -> None:
        self.full = True
        self.durable: list[tuple[int, float]] = []

    def __call__(self, statement: str) -> None:
        # The journal's connection is used under its mutex, so statements arrive in order.
        if statement.startswith("PRAGMA synchronous="):
            self.full = statement.endswith("FULL")
        elif statement == "COMMIT" and self.full:
            self.durable.append((threading.get_ident(), time.perf_counter() * 1000))

    def on(self, thread: int, start: float, end: float) -> int:
        return sum(1 for t, at in self.durable if t == thread and start <= at <= end)

    def by_workers(self, since: int) -> int:
        """Commits after the `since`-th made by any thread but the test's own submissions."""
        return sum(1 for t, _ in self.durable[since:] if t != threading.get_ident())


def _commits(monkeypatch: pytest.MonkeyPatch) -> Commits:
    commits = Commits()
    connect = workspace._Handle.connect

    def traced(self: workspace._Handle) -> workspace.Journal:
        db = connect(self)
        db.set_trace_callback(commits)
        return db

    monkeypatch.setattr(workspace._Handle, "connect", traced)
    return commits


@pytest.fixture
def pod(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Pod, list[Sent], Commits]]:
    driverless(monkeypatch)
    sent: list[Sent] = []
    call = child.Executor.call

    def counted(self: child.Executor, command: Any, *args: Any, **kwargs: Any) -> Any:
        at = time.perf_counter() * 1000
        try:
            return call(self, command, *args, **kwargs)
        finally:
            sent.append(
                Sent(type(command).__name__, threading.get_ident(), at, time.perf_counter() * 1000)
            )

    monkeypatch.setattr(child.Executor, "call", counted)
    commits = _commits(monkeypatch)
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-handoff.", dir="/tmp") as raw:
        made = Pod(Path(raw))
        try:
            yield made, sent, commits
        finally:
            made.worker.shutdown()


def test_back_to_back_attempts_hand_off_in_one_exchange_each(
    pod: tuple[Pod, list[Sent], Commits],
) -> None:
    """Twelve attempts queued on one warm executor: each is one prepare and one invoke, the
    reply carries the after-state the ledger closes from, and the worker commits only the
    dispatch mark and the terminal. Prints the medians the PR quotes."""
    machine, sent, commits = pod
    machine.serve("warmup")
    assert machine.settle("warmup") == "succeeded"
    names = [f"run-{n:02d}" for n in range(RUNS)]
    started, committed = len(sent), len(commits.durable)
    for name in names:
        machine.serve(name)
    assert all(machine.settle(name) == "succeeded" for name in names)
    commands = sorted(sent[started:], key=lambda row: row.sent)
    invokes = [row for row in commands if row.command == Invoke.__name__]
    prepares = [row for row in commands if row.command == PrepareRequest.__name__]
    handoffs = list(itertools.pairwise(invokes))
    idle = [b.sent - a.answered for a, b in handoffs]
    to_prepare = [b.sent - a.answered for a, b in zip(invokes, prepares[1:], strict=False)]
    # What the next attempt's lane thread waits on an fsync for between the previous reply
    # and its own invoke: the dispatch mark, which must be durable before the executor runs.
    on_path = [commits.on(b.thread, a.answered, b.sent) for a, b in handoffs]
    per_attempt = commits.by_workers(committed) / RUNS
    print(
        json.dumps(
            {
                "reply_to_next_invoke_ms": round(statistics.median(idle), 2),
                "reply_to_next_prepare_ms": round(statistics.median(to_prepare), 2),
                "synchronous_commits_per_handoff": round(statistics.mean(on_path), 2),
                "synchronous_commits_per_attempt": per_attempt,
                "commands": [row.command for row in commands[:3]],
            }
        )
    )
    assert [row.command for row in commands] == [PrepareRequest.__name__, Invoke.__name__] * RUNS
    assert on_path == [1] * (RUNS - 1), on_path
    # The dispatch mark and the terminal; the rest rides the next journal access.
    assert per_attempt == 2, per_attempt
    for name in names:
        events = machine.events(name)
        kinds = [kind for _, _, kind, _ in events]
        logs = [body["payload"].get("name") for _, _, kind, body in events if kind == "log"]
        assert kinds.index("executor") < kinds.index("running"), kinds
        assert logs.index("executor invoke") < logs.index("executor release"), logs
        assert [at for _, at, _, _ in events] == sorted(at for _, at, _, _ in events), events


def test_a_released_gpu_reaches_the_next_call_without_a_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One card no host has, and a root's twelve serving calls queued on it. Each is
    granted the card, refuses at activation (its template names no installed package) and
    releases it. The scheduler's pass journals its events and the next call's phase as one
    deferred batch, so the next call is woken and dispatched without waiting on a commit.
    Prints the median the PR quotes."""
    driverless(monkeypatch)
    measured = accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)
    monkeypatch.setattr(accel, "device_memory", lambda entry, kind: measured)
    commits = _commits(monkeypatch)
    released: dict[str, tuple[float, int]] = {}
    dispatched: dict[str, tuple[float, int]] = {}
    dispatch = Worker.dispatch_machine

    def timed(worker: Worker, unit: Any, offered: pb.AttemptOffer, *args: Any) -> Any:
        dispatched[offered.request_id] = time.perf_counter() * 1000, threading.get_ident()
        return dispatch(worker, unit, offered, *args)

    monkeypatch.setattr(Worker, "dispatch_machine", timed)
    with tempfile.TemporaryDirectory(prefix="cz-handoff.", dir="/tmp") as raw:
        machine = Machine(Path(raw), CARD, "boot-one")
        try:
            release = machine.worker.stages.release

            def timed_release(key: str, **facts: Any) -> None:
                """The first release of a call's grant, and the commits its pass made."""
                at, before = time.perf_counter() * 1000, len(commits.durable)
                release(key, **facts)
                mine = sum(1 for t, _ in commits.durable[before:] if t == threading.get_ident())
                released.setdefault(key.rpartition("#")[0], (at, mine))

            machine.worker.stages.release = timed_release  # type: ignore[method-assign]
            machine.root("A")
            committed = len(commits.durable)
            calls = [machine.child("A", "qwen") for _ in range(RUNS)]
            bound = time.monotonic() + 300  # a hang bound on a loaded box, not a budget
            while not all(machine.executions.status(OWNER, c).state in TERMINAL for c in calls):
                assert time.monotonic() < bound
                time.sleep(0.01)
            per_call = commits.by_workers(committed) / RUNS
            grants = [body["key"] for _, kind, body in machine.journal("A") if kind == "gpu.grant"]
        finally:
            machine.close()
    order = sorted(calls, key=lambda call: dispatched[call][0])
    handoffs = list(itertools.pairwise(order))
    gaps = [dispatched[b][0] - released[a][0] for a, b in handoffs]
    # The releasing pass, then the woken call up to its dispatch: nothing waits on an fsync.
    on_path = [
        released[a][1] + commits.on(dispatched[b][1], released[a][0], dispatched[b][0])
        for a, b in handoffs
    ]
    print(
        json.dumps(
            {
                "release_to_next_dispatch_ms": round(statistics.median(gaps), 2),
                "synchronous_commits_per_handoff": round(statistics.mean(on_path), 2),
                "synchronous_commits_per_call": round(per_call, 2),
            }
        )
    )
    assert grants == [f"{call}#1" for call in order], grants
    assert on_path == [0] * (RUNS - 1), on_path
