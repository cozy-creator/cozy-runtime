#!/usr/bin/env python3
"""Every descriptor-declared Context/service fact, exercised against a REAL invocation.

Not a test suite: each arm below runs the production invocation kernel over a real endpoint
and reports the value the runtime actually WROTE, observed from inside author code. The
point is the writer, not the reader.

The v1 lesson this exists for, in v1's own words (pgw#1584): `ctx.boot_warmup`'s property,
its docstring and its reader all shipped, but `grep -rn "boot_warmup=True" src/` returned
nothing — so every `if ctx.boot_warmup:` arm in the fleet was unreachable code and every
endpoint that had written one had written it for a flag that was permanently False. It was
found by a human running grep. This script is that grep, mechanized and tied to the
descriptor: COVERAGE IS ENFORCED — a fact the descriptor declares with no exercise here is a
refusal, so the fact list and the exercise list cannot drift apart.

    uv run python scripts/conformance.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import msgspec  # noqa: E402

from cozy_runtime.author import (  # noqa: E402
    App,
    Budget,
    BudgetFacts,
    Cancelled,
    CheckpointReceipt,
    Checkpoints,
    CheckpointSave,
    Context,
    Device,
    FileAsset,
    Invocation,
    Outputs,
    Scratch,
    Telemetry,
    attempt,
)
from cozy_runtime.internal import descriptor as desc  # noqa: E402
from cozy_runtime.internal.discovery import discover  # noqa: E402

Observation = str
probe = App()


class In(msgspec.Struct):
    pass


class Facts(msgspec.Struct):
    request_id: str
    deadline: float
    cancelled: bool
    boot_warmup: bool
    device_spelling: str
    raise_if_cancelled: str


@probe.entrypoint
def report(ctx: Context, payload: In) -> Facts:
    """Read every Context member the descriptor declares, from inside author code."""
    del payload
    try:
        ctx.raise_if_cancelled()
        cancel_spelling = "returned"
    except Cancelled as exc:
        cancel_spelling = f"raised {exc.code}"
    return Facts(
        request_id=ctx.request_id,
        deadline=ctx.deadline,
        cancelled=ctx.cancelled,
        boot_warmup=ctx.boot_warmup,
        device_spelling=str(ctx.device),
        raise_if_cancelled=cancel_spelling,
    )


class Saved(msgspec.Struct):
    file: FileAsset


@probe.entrypoint
def uses_services(payload: In, out: Outputs, tel: Telemetry) -> Saved:
    """Exercise the members of the services this signature requests."""
    del payload
    tel.progress(0.5, stage="probing")
    tel.log("operator diagnostic", level="info")
    tel.metric("probe", 1.0, unit="count")
    tel.training_metric("loss", 0.5, step=0)
    with tel.stage("bracket"):
        pass
    tel.step_callback(1, stage="probing")(0)
    out.temporary_file(suffix=".raw").write_bytes(b"spool")
    return Saved(file=out.save_bytes(b"conformance", media_type="text/plain"))


class JobFacts(msgspec.Struct):
    checkpoint_dir: str
    mktemp: str
    materialize_blob: str
    receipt: str
    replayed: bool
    rate: float
    cap: float
    spent_is_positive: bool


@probe.job
def uses_job_services(payload: In, scratch: Scratch, ckpt: Checkpoints, budget: Budget) -> JobFacts:
    """Every member of the three JOB services, called from inside author code (cr-009)."""
    del payload
    receipt = ckpt.save("state", b"durable", operation_key="op/1")
    replay = ckpt.save("state", b"durable", operation_key="op/1")
    return JobFacts(
        checkpoint_dir=scratch.checkpoint_dir(key="resume").name,
        mktemp=scratch.mktemp(suffix=".raw").suffix,
        materialize_blob=str(scratch.materialize_blob(b"bytes", name="blob.bin").name),
        receipt=receipt.receipt_id,
        replayed=replay.replayed,
        rate=budget.gpu_rate_usd_per_hour,
        cap=budget.cap_usd,
        spent_is_positive=budget.spent_usd() > 0.0,
    )


def _run(spool: Path, **invocation: object) -> tuple[Facts, float]:
    inv = Invocation(spool=spool, deadline=time.monotonic() + 30.0, **invocation)  # type: ignore[arg-type]
    envelope, outcome, _ = attempt(probe.get("report"), {}, inv)
    assert envelope is not None, outcome
    assert isinstance(envelope.result, Facts)
    return envelope.result, inv.deadline


# Each entry: the descriptor-declared fact -> an arm that PROVES a writer put a distinct
# value there. An arm returns the observation; a missing writer makes it raise.
def _context_arms(spool: Path) -> Mapping[str, Callable[[], Observation]]:
    def request_id() -> Observation:
        facts, _ = _run(spool, request_id="attempt-conformance-7")
        assert facts.request_id == "attempt-conformance-7", facts.request_id
        return f"handler read {facts.request_id!r} — the value the runtime supplied"

    def deadline() -> Observation:
        facts, expected = _run(spool, request_id="deadline")
        assert facts.deadline == expected, (facts.deadline, expected)
        assert facts.deadline > time.monotonic(), "a live attempt's deadline is in the future"
        return f"handler read the supplied monotonic instant ({facts.deadline:.3f})"

    def cancelled() -> Observation:
        live, _ = _run(spool, request_id="live")
        dead, _ = _run(spool, request_id="dead", cancel=lambda: True)
        assert live.cancelled is False and dead.cancelled is True, (live, dead)
        return "False under a live attempt, True under a cancelling one — the cancel hook writes"

    def raise_if_cancelled() -> Observation:
        live, _ = _run(spool, request_id="live")
        dead, _ = _run(spool, request_id="dead", cancel=lambda: True)
        assert live.raise_if_cancelled == "returned", live.raise_if_cancelled
        assert dead.raise_if_cancelled == "raised cancelled", dead.raise_if_cancelled
        return "returns on a live attempt, raises Cancelled on a cancelled one"

    def boot_warmup() -> Observation:
        cold, _ = _run(spool, request_id="cold")
        warm, _ = _run(spool, request_id="warm", boot_warmup=True)
        assert cold.boot_warmup is False and warm.boot_warmup is True, (cold, warm)
        return "False on an ordinary attempt, True on the warm pass — THE v1 drift arm"

    def device() -> Observation:
        facts, _ = _run(spool, request_id="dev", device=Device("cuda", 3))
        assert facts.device_spelling == "cuda:3", facts.device_spelling
        return f"handler read {facts.device_spelling!r} — the runtime's device, torch spelling"

    return {
        "request_id": request_id,
        "deadline": deadline,
        "cancelled": cancelled,
        "raise_if_cancelled": raise_if_cancelled,
        "boot_warmup": boot_warmup,
        "device": device,
    }


def _service_arms(spool: Path) -> Mapping[str, Callable[[], Observation]]:
    def save() -> Observation:
        inv = Invocation(request_id="svc", spool=spool, deadline=time.monotonic() + 30.0)
        envelope, outcome, _ = attempt(probe.get("uses_services"), {}, inv)
        assert envelope is not None, outcome
        assert len(envelope.outputs) == 1, envelope.outputs
        row = envelope.outputs[0]
        return (
            f"save_bytes + temporary_file: 1 output row, {row['size_bytes']} bytes, "
            f"{row['digest'][:19]}…"
        )

    def telemetry() -> Observation:
        inv = Invocation(request_id="tel", spool=spool, deadline=time.monotonic() + 30.0)
        envelope, outcome, record = attempt(probe.get("uses_services"), {}, inv)
        assert envelope is not None, outcome
        kinds = sorted({event.kind for event in record.ring})
        assert len(record.ring), "Telemetry recorded nothing — its writer is gone"
        caps = record.ring.caps()
        return (
            f"{caps['admitted']} observations across kinds {kinds}; ring holds "
            f"{caps['kept']}/{caps['max_events']} at {caps['held_bytes']} B"
        )

    def _job_facts() -> JobFacts:
        saved: list[CheckpointSave] = []

        def sink(save: CheckpointSave) -> CheckpointReceipt:
            held = next((s for s in saved if s.operation_key == save.operation_key), None)
            if held is None:
                saved.append(save)
            return CheckpointReceipt(
                receipt_id=f"ckpt-{len(saved)}",
                operation_key=save.operation_key,
                logical_key=save.logical_key,
                content_digest=save.content_digest,
                length=save.length,
                replayed=held is not None,
            )

        inv = Invocation(
            request_id="job",
            spool=spool,
            deadline=time.monotonic() + 30.0,
            scratch=spool / "scratch",
            checkpoints=sink,
            budget=BudgetFacts(
                gpu_rate_usd_per_hour=2.5,
                gpu_count=2,
                cap_usd=100.0,
                started_monotonic=time.monotonic() - 3600.0,
            ),
        )
        (spool / "scratch").mkdir(parents=True, exist_ok=True)
        envelope, outcome, _ = attempt(probe.get("uses_job_services"), {}, inv)
        assert envelope is not None, outcome
        assert isinstance(envelope.result, JobFacts)
        return envelope.result

    def scratch() -> Observation:
        facts = _job_facts()
        return (
            f"checkpoint_dir(key=) -> {facts.checkpoint_dir!r}, mktemp -> {facts.mktemp!r}, "
            f"materialize_blob -> {facts.materialize_blob!r}"
        )

    def checkpoints() -> Observation:
        facts = _job_facts()
        assert facts.replayed, "the same identity did not replay its receipt"
        return f"save -> {facts.receipt}, and the same identity REPLAYED it"

    def budget() -> Observation:
        facts = _job_facts()
        assert facts.spent_is_positive, "an hour on 2 devices at 2.5 USD/h spent nothing"
        return (
            f"gpu_rate_usd_per_hour {facts.rate}, cap_usd {facts.cap}, spent_usd is positive "
            "after an hour on 2 devices — all from the typed config channel"
        )

    return {
        "save": save,
        "telemetry": telemetry,
        "scratch": scratch,
        "checkpoints": checkpoints,
        "budget": budget,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-conformance-") as tmp:
        return _run_all(Path(tmp))


def _run_all(spool: Path) -> int:
    found = discover(ROOT / "examples")
    body = desc.build(found)

    declared_context = {str(f["name"]) for f in body["context_facts"]}
    declared_services = set(body["services"])
    context_arms = _context_arms(spool)
    service_arms = _service_arms(spool)

    print("=== conformance: every descriptor-declared Context fact")
    missing = sorted(declared_context - set(context_arms))
    if missing:
        print(
            f"NO EXERCISE for descriptor-declared Context fact(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    failures = 0
    for fact in sorted(declared_context):
        try:
            print(f"  ctx.{fact:20} {context_arms[fact]()}")
        except Exception as exc:
            failures += 1
            print(f"  ctx.{fact:20} NO WRITER — {type(exc).__name__}: {exc}", file=sys.stderr)

    print("\n=== conformance: the service capabilities this probe declares")
    for capability in sorted(declared_services):
        if capability not in service_arms:
            # Honest gap, not a silent pass: cr-009 (job services) and cr-012 (egress) own
            # the bodies these capabilities name, so there is nothing to exercise yet.
            members = body["services"][capability]["members"]
            print(f"  {capability:14} declared, {len(members)} member(s), no body yet — SKIPPED")
            continue
        try:
            print(f"  {capability:14} {service_arms[capability]()}")
        except Exception as exc:
            failures += 1
            print(f"  {capability:14} NO WRITER — {type(exc).__name__}: {exc}", file=sys.stderr)

    exercised = len(declared_context) + len(service_arms)
    print(f"\n{exercised - failures}/{exercised} declared facts have a live writer")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
