"""A rented upload asks the record owner before computing, and tells it what it computed.

Real Worker, delegated executor, native upload child and TensorFS store, driven through the
machine-execution RPCs a rented run uses. The owner is a Creator observer: one wait read,
then reads from its own cursor, answering each `memo.lookup` it reads before its next read.
Stand-ins: the Hugging Face origin and Tensorhub's publication routes (upload_standins).
The script waits at a gate until the observer is attached, so no case races its reads.
"""

from __future__ import annotations

import base64
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import grpc
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR
from test_machine_partial_work import Machine, machine
from upload_standins import Hub, HuggingFace

FIXTURE = Path(__file__).parent / "testdata/native_source"
DESTINATION = "example/uploaded"
SCRIPT = """import asyncio
from pathlib import Path
import msgspec
from cozy_runtime.author import App, Context
from cozy_runtime.author.sources import upload_huggingface
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    checkpoint: str

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    while not Path(GATE).exists():
        await asyncio.sleep(0.02)
    ref = await upload_huggingface('example/model', revision='REVISION', destination='DESTINATION',
        profiles=('fixture/first/1',), carriers=('provider/first.safetensors',))
    return Result(ref.checkpoint)
""".replace("REVISION", "a" * 40).replace("DESTINATION", DESTINATION)
MEMBERS = {"provider/first.safetensors": (FIXTURE / "first.safetensors").read_bytes()}

Answer = Callable[[dict[str, Any], int], pb.MachineMemoAnswer | None]
pytestmark = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


class Counted(dict[str, bytes]):
    """The origin's members; every body or header request reads one."""

    reads = 0

    def __getitem__(self, key: str) -> bytes:
        self.reads += 1
        return super().__getitem__(key)


class Owner(threading.Thread):
    """Creator's observer of one execution. `answer` decides each lookup; None reads past.

    `hold` stops reading at the first lookup. `resume`, when given, keeps the observer
    between its reads after the first until it is set."""

    def __init__(
        self,
        pod: Machine,
        request: str,
        answer: Answer,
        *,
        hold: bool = False,
        resume: threading.Event | None = None,
    ):
        super().__init__(daemon=True)
        self.pod, self.answer, self.hold, self.resume = pod, answer, hold, resume
        self.query = query(pod, request)
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.refusals: list[str] = []
        self.attached, self.asked = threading.Event(), threading.Event()
        self.start()

    def run(self) -> None:
        client, cursor = self.pod.client, 0
        while True:
            client.ListMachineExecutionEvents(
                pb.MachineExecutionEventsQuery(
                    execution=self.query, after=cursor, limit=1, wait=True
                )
            )
            page = client.ListMachineExecutionEvents(
                pb.MachineExecutionEventsQuery(execution=self.query, after=cursor)
            )
            for event in page.events:
                body = canonical_json.decode(event.body_canonical_bytes)
                self.events.append((event.kind, body))
                if event.kind != "memo.lookup":
                    continue
                self.asked.set()
                if self.hold:
                    return
                if (answer := self.answer(body, event.sequence)) is not None:
                    self.control(answer)
            cursor = page.next_after
            if not self.attached.is_set():
                self.attached.set()
                if self.resume is not None:
                    self.resume.wait()
            state = client.GetMachineExecution(self.query).state
            if not page.events and state in ("succeeded", "failed", "canceled"):
                return

    def control(self, answer: pb.MachineMemoAnswer) -> None:
        try:
            self.pod.client.ControlMachineExecution(
                pb.MachineExecutionControl(
                    execution=self.query,
                    command_id=f"memo-{answer.lookup_sequence}",
                    action=pb.MACHINE_EXECUTION_ACTION_ANSWER_MEMO,
                    memo=answer,
                )
            )
        except grpc.RpcError as exc:
            assert exc.code() == grpc.StatusCode.FAILED_PRECONDITION, exc
            self.refusals.append(dict(exc.trailing_metadata() or ())["cozy-error-code"])

    def kinds(self) -> list[str]:
        self.join(60)
        assert not self.is_alive()
        return [kind for kind, _ in self.events if kind.startswith("memo.")]


def query(pod: Machine, request: str) -> pb.MachineExecutionQuery:
    assert pod.worker.executions is not None
    return pb.MachineExecutionQuery(
        claim=pod.claim,
        request_id=request,
        expected_execution_workspace_id=pod.worker.executions.workspace_id,
    )


def recorded(result: bytes) -> Callable[[dict[str, Any], int], pb.MachineMemoAnswer]:
    """The owner's index: the checkpoint another machine recorded for this computation."""
    return lambda body, sequence: pb.MachineMemoAnswer(
        lookup_sequence=sequence,
        computation_digest=documents.raw(body["computation_digest"]),
        result_canonical_bytes=result,
    )


@contextmanager
def pod(monkeypatch: pytest.MonkeyPatch, gate: Path, origin: str, hub: Hub) -> Iterator[Machine]:
    with machine(monkeypatch, SCRIPT.replace("GATE", repr(str(gate)))) as rented:
        calls = rented.worker.source_calls
        assert calls is not None
        calls.endpoints = {"huggingface": origin}
        calls.native_registry = (FIXTURE / "registry.json").read_bytes()
        calls.publication = lambda *_: hub.client()
        rented.start(())
        yield rented


def observed(
    pod: Machine, gate: Path, request: str, answer: Answer, owner_memo: bool = True, **how: Any
) -> Owner:
    """Submit behind the gate; open it once the owner's observer is attached."""
    gate.unlink(missing_ok=True)
    pod.submit(request, owner_memo=owner_memo)
    owner = Owner(pod, request, answer, **how)
    assert owner.attached.wait(120)
    gate.touch()
    return owner


def succeeded(pod: Machine, request: str) -> str:
    body: Any = pod.outcome(request)
    assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
    return str(
        canonical_json.decode(base64.b64decode(body["result"]["inline_result"]))["checkpoint"]
    )


def journal(pod: Machine, request: str) -> list[Any]:
    assert pod.worker.executions is not None
    return list(pod.worker.executions.events("owner", request).events)


def memo_kinds(pod: Machine, request: str) -> list[str]:
    return [e.kind for e in journal(pod, request) if e.kind.startswith("memo.")]


def test_a_fresh_pod_takes_the_owners_recorded_checkpoint_without_moving_a_byte(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate, files, hub = tmp_path / "gate", Counted(MEMBERS), Hub()
    with HuggingFace(files).running() as origin, hub.running():
        with pod(monkeypatch, gate, origin, hub) as first:
            # Without owner_memo it announces what it computed and never asks.
            owner = observed(first, gate, "computed", lambda *_: None, owner_memo=False)
            checkpoint = succeeded(first, "computed")
            assert owner.kinds() == ["memo.record"]
            (record,) = [body for kind, body in owner.events if kind == "memo.record"]
            (call,) = first.rows("SELECT result,computation_digest FROM native_calls")
            assert record == {
                "call_index": 0,
                "operation": "upload_huggingface",
                "computation_digest": documents.spell(call["computation_digest"]),  # type: ignore[arg-type]
                "result": canonical_json.decode(call["result"]),  # type: ignore[arg-type]
            }
            assert record["result"]["checkpoint"] == checkpoint
            assert record["result"]["destination"] == DESTINATION
        result = canonical_json.encode(record["result"])
        uploads, publications = sum(hub.uploads.values()), len(hub.publications)

        with pod(monkeypatch, gate, origin, hub) as fresh:
            asked_at: list[int] = []

            def answer(body: dict[str, Any], sequence: int) -> pb.MachineMemoAnswer:
                asked_at.append(files.reads)
                assert body == {
                    "call_index": 0,
                    "operation": "upload_huggingface",
                    "computation_digest": record["computation_digest"],
                }
                return recorded(result)(body, sequence)

            owner = observed(fresh, gate, "reused", answer)
            assert succeeded(fresh, "reused") == checkpoint
            assert owner.kinds() == ["memo.lookup"] and not owner.refusals
            # Zero source requests after the lookup, and nothing published again.
            assert asked_at == [files.reads]
            assert (sum(hub.uploads.values()), len(hub.publications)) == (uploads, publications)
            (call,) = fresh.rows("SELECT state,result,computation_digest FROM native_calls")
            assert call["state"] == "complete" and call["result"] == result
            assert documents.spell(call["computation_digest"]) == record["computation_digest"]  # type: ignore[arg-type]

            # A lost reply's repeat replays; any other answer finds no outstanding lookup.
            (sequence,) = [e.sequence for e in journal(fresh, "reused") if e.kind == "memo.lookup"]
            owner.control(recorded(result)(record, sequence))
            owner.control(recorded(b"")(record, sequence))
            assert owner.refusals == ["memo_lookup_not_outstanding"]


def test_an_unanswered_refused_or_unobserved_lookup_computes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate, hub = tmp_path / "gate", Hub()
    gate.touch()
    with (
        HuggingFace(MEMBERS).running() as origin,
        hub.running(),
        pod(monkeypatch, gate, origin, hub) as rented,
    ):
        submission = rented.submit("seed")
        succeeded(rented, "seed")
        (record,) = [
            canonical_json.decode(e.body)
            for e in journal(rented, "seed")
            if e.kind == "memo.record"
        ]
        result = record["result"]
        elsewhere = canonical_json.encode({**result, "destination": "example/other"})

        def mistaken(body: dict[str, Any], sequence: int) -> pb.MachineMemoAnswer:
            answer = recorded(canonical_json.encode(result))(body, sequence)
            answer.computation_digest = bytes(32)
            return answer

        cases: dict[str, tuple[Answer, list[str]]] = {
            "empty": (recorded(b""), []),
            "passed": (lambda *_: None, []),
            "elsewhere": (recorded(elsewhere), ["memo_result_invalid"]),
            "mistaken": (mistaken, ["memo_digest_mismatch"]),
        }
        for request, (answer, refusals) in cases.items():
            owner = observed(rented, gate, request, answer)
            succeeded(rented, request)
            assert owner.kinds() == ["memo.lookup", "memo.record"], request
            assert owner.refusals == refusals, request

        # Never read: no lookup at all. A resubmission keeps each execution's owner_memo.
        rented.submit("unobserved", owner_memo=True)
        succeeded(rented, "unobserved")
        assert memo_kinds(rented, "unobserved") == ["memo.record"]
        rented.client.SubmitMachineExecution(submission)
        flags = {
            row["request"]: row["owner_memo"]
            for row in rented.rows("SELECT request,owner_memo FROM executions")
        }
        assert flags == {"seed": 0, **dict.fromkeys([*cases, "unobserved"], 1)}

        # An observer that left (its wait read ended with it gone) is not asked.
        gate.unlink()
        rented.submit("departed", owner_memo=True)
        after = 0
        while True:
            try:
                after = rented.client.ListMachineExecutionEvents(
                    pb.MachineExecutionEventsQuery(
                        execution=query(rented, "departed"), after=after, wait=True
                    ),
                    timeout=0.5,
                ).next_after
            except grpc.RpcError as exc:
                assert exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED, exc
                break
        memo = rented.worker.owner_memo
        assert memo is not None
        rented.wait("the departure", lambda: ("owner", "departed") not in memo.observed, 30)
        gate.touch()
        succeeded(rented, "departed")
        assert memo_kinds(rented, "departed") == ["memo.record"]

        # Between its reads an observer is still attached: the lookup waits for its next read.
        resume = threading.Event()
        owner = observed(rented, gate, "between", recorded(elsewhere), resume=resume)
        rented.wait("the lookup", lambda: memo_kinds(rented, "between") == ["memo.lookup"], 120)
        resume.set()
        succeeded(rented, "between")
        assert owner.kinds() == ["memo.lookup", "memo.record"]
        assert owner.refusals == ["memo_result_invalid"]

        # A parked lookup waits for its owner, however long; canceling the run unparks it.
        owner = observed(rented, gate, "canceled", lambda *_: None, hold=True)
        assert owner.asked.wait(120)
        for _ in range(25):
            assert rented.state("canceled") == "running"
            threading.Event().wait(0.02)
        rented.cancel("canceled")
        assert rented.outcome("canceled")["status"] == pb.OUTCOME_STATUS_CANCELED
        assert memo_kinds(rented, "canceled") == ["memo.lookup"]
