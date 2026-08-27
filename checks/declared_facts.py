#!/usr/bin/env python3
"""Every public Context/service fact, exercised against a REAL invocation.

Not a test suite: each arm below runs the production invocation kernel over a real endpoint
and reports the value the runtime actually WROTE, observed from inside author code. The
point is the writer, not the reader.

The v1 lesson this exists for, in v1's own words (pgw#1584): `ctx.boot_warmup`'s property,
its docstring and its reader all shipped, but `grep -rn "boot_warmup=True" src/` returned
nothing — so every `if ctx.boot_warmup:` arm in the fleet was unreachable code and every
endpoint that had written one had written it for a flag that was permanently False. It was
found by a human running grep. This script is that grep, mechanized and tied to the
SDK: COVERAGE IS ENFORCED — a public fact with no exercise here is a refusal, so Runtime's
author contract and the exercise list cannot drift apart.

    uv run python checks/declared_facts.py
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import msgspec  # noqa: E402

from cozy_runtime.author import (  # noqa: E402
    App,
    ArtifactCommit,
    ArtifactOutput,
    ArtifactSink,
    ArtifactSource,
    AssetBound,
    Budget,
    BudgetFacts,
    Cancelled,
    CheckpointDeclaration,
    Checkpoints,
    CheckpointSave,
    Context,
    CopyComponent,
    Device,
    FileAsset,
    ImageAsset,
    ImageFrame,
    Invocation,
    MediaDecoder,
    Outputs,
    Scratch,
    Telemetry,
    Tree,
    attempt,
)
from cozy_runtime.author._artifact_sink import (  # noqa: E402
    ArtifactBytes,
    _ArtifactRequest,
    _ArtifactWriterBackend,
)
from cozy_runtime.author._assets import GrantedInput  # noqa: E402
from cozy_runtime.author._services import SERVICES, Attempt  # noqa: E402
from cozy_runtime.internal import schema  # noqa: E402

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


class MediaIn(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=1024, max_decoded_bytes=3),
    ]


class MediaFact(msgspec.Struct):
    service: str


class TaggedImage(msgspec.Struct, tag="image", tag_field="type"):
    image: str


class TaggedVideo(msgspec.Struct, tag="video", tag_field="type"):
    video: str


class UntaggedReference(msgspec.Struct):
    file: str


class IntegerTagged(msgspec.Struct, tag=1, tag_field="type"):
    number: int


TaggedReference = TaggedImage | TaggedVideo


def _tagged_union_conformance() -> str:
    rendered = schema.render(list[TaggedReference])
    assert isinstance(rendered, dict) and isinstance(rendered.get("list"), dict), rendered
    union = rendered["list"]
    assert union["tag_field"] == "type", union
    members = union["union"]
    assert isinstance(members, list) and len(members) == 2, members
    tags = {member["tag"]: member for member in members}
    assert set(tags) == {"image", "video"}, tags
    for tag, member in tags.items():
        assert member["tag_field"] == "type", member
        discriminator = member["fields"][0]
        assert discriminator == {
            "name": "type",
            "type": {"literal": [tag]},
            "wire": "required",
        }, discriminator

    decoder = msgspec.json.Decoder(list[TaggedReference])
    decoded = decoder.decode(b'[{"type":"image","image":"ref"}]')
    assert decoded == [TaggedImage("ref")], decoded
    for invalid in (
        b'[{"image":"ref"}]',
        b'[{"type":"audio","image":"ref"}]',
        b'[{"type":"video","image":"ref"}]',
    ):
        try:
            decoder.decode(invalid)
        except msgspec.ValidationError:
            continue
        raise AssertionError(f"tagged-union decoder accepted {invalid!r}")

    assert schema.render(UntaggedReference | None) == {
        "union": [
            "null",
            {
                "fields": [{"name": "file", "type": "str", "wire": "required"}],
            },
        ]
    }
    single = schema.render(TaggedImage)
    assert single["fields"][0]["wire"] == "optional", single
    scalar_union = schema.render(TaggedImage | str)
    tagged_member = next(member for member in scalar_union["union"] if isinstance(member, dict))
    assert tagged_member["fields"][0]["wire"] == "optional", tagged_member
    try:
        schema.render(TaggedImage | UntaggedReference)
    except Exception as exc:
        assert getattr(exc, "code", "") == "tagged_union_incoherent", exc
    else:
        raise AssertionError("mixed tagged/untagged union rendered")
    try:
        schema.render(TaggedImage | IntegerTagged)
    except Exception as exc:
        assert getattr(exc, "code", "") == "tagged_union_incoherent", exc
    else:
        raise AssertionError("mixed string/integer tags rendered")
    return "type=image|video published and missing/unknown/mismatched tags refused"


@probe.entrypoint
def uses_media_decode(payload: MediaIn, decoder: MediaDecoder) -> MediaFact:
    """Prove the declared service is constructed; the PyAV members have their live driver."""
    del payload
    return MediaFact(type(decoder).__name__)


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
    declaration: str
    replayed: bool
    save_refusal: str
    rate: float
    cap: float
    spent_is_positive: bool


class ArtifactFact(msgspec.Struct):
    output_slot: str
    snapshot_digest: str


@probe.job(artifacts=(ArtifactOutput("checkpoint", 1),))
def uses_artifact_sink(payload: In, artifacts: ArtifactSink) -> ArtifactFact:
    del payload
    with artifacts.open(
        "checkpoint",
        sources={"source": Tree("granted-source", digest="sha256:" + "a" * 64)},
        targets={"copied": CopyComponent("source", "weights")},
        configs={},
    ) as writer:
        committed = writer.commit()
    return ArtifactFact(committed.output_slot, committed.snapshot_digest)


@probe.job
def uses_job_services(payload: In, scratch: Scratch, ckpt: Checkpoints, budget: Budget) -> JobFacts:
    """Every member of the three JOB services, called from inside author code (cr-009).

    `save()` is one of them and it is HARD-DISABLED (#553a): its refusal is the member's
    conformance fact, exactly as a returned value is for the others.
    """
    del payload
    receipt = ckpt.declare("state", b"durable", operation_key="op/1")
    replay = ckpt.declare("state", b"durable", operation_key="op/1")
    # No fake-green line to guard: `save()` is typed `NoReturn`, so the branch where it
    # returns a receipt does not exist for a detector to catch (#553a/#563f).
    try:
        ckpt.save("state", b"durable", operation_key="op/2")
    except Exception as exc:
        refusal = f"{type(exc).__name__}:{getattr(exc, 'code', '')}"
    return JobFacts(
        checkpoint_dir=scratch.checkpoint_dir(key="resume").name,
        mktemp=scratch.mktemp(suffix=".raw").suffix,
        materialize_blob=str(scratch.materialize_blob(b"bytes", name="blob.bin").name),
        declaration=receipt.declaration_id,
        replayed=replay.replayed,
        save_refusal=refusal,
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

    def media_decode() -> Observation:
        source = Attempt("media-source", spool / "media-source")
        source.spool.mkdir(parents=True, exist_ok=True)
        image = Outputs(source).save_image(ImageFrame(1, 1, b"\x10\x20\x30"))
        assert image._local is not None
        grant = GrantedInput(
            "image", image._local, image.media_type, image.digest, image.size_bytes
        )
        inv = Invocation(
            request_id="media",
            spool=spool / "media",
            deadline=time.monotonic() + 30,
            assets={"image": grant},
        )
        envelope, outcome, _ = attempt(probe.get("uses_media_decode"), {"image": "ref"}, inv)
        assert envelope is not None, outcome
        assert envelope.result == MediaFact("MediaDecoder"), envelope.result
        return "handler received the attempt-bound MediaDecoder declared by its signature"

    def _job_facts() -> JobFacts:
        saved: list[CheckpointSave] = []

        def sink(save: CheckpointSave) -> CheckpointDeclaration:
            held = next((s for s in saved if s.operation_key == save.operation_key), None)
            if held is None:
                saved.append(save)
            return CheckpointDeclaration(
                declaration_id=f"ckpt-{len(saved)}",
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
        assert facts.replayed, "the same identity did not replay its declaration"
        assert facts.save_refusal == "CheckpointSinkUnbuilt:checkpoint_sink_unbuilt", (
            "Checkpoints.save() did not refuse: " + facts.save_refusal
        )
        return (
            f"declare -> {facts.declaration}, and the same identity REPLAYED it; "
            f"save() refuses {facts.save_refusal} (#553a — no ArtifactSink stores the bytes)"
        )

    def budget() -> Observation:
        facts = _job_facts()
        assert facts.spent_is_positive, "an hour on 2 devices at 2.5 USD/h spent nothing"
        return (
            f"gpu_rate_usd_per_hour {facts.rate}, cap_usd {facts.cap}, spent_usd is positive "
            "after an hour on 2 devices — all from the typed config channel"
        )

    def artifact_sink() -> Observation:
        declared = probe.get("uses_artifact_sink").surface.artifact_outputs
        assert declared == (ArtifactOutput("checkpoint", 1),)

        class Backend(_ArtifactWriterBackend):
            sources: tuple[ArtifactSource, ...] = ()
            replayed = False

            def source_read_into(
                self,
                source: str,
                component: str,
                key: str,
                role: str,
                offset: int,
                into: object,
            ) -> None:
                del source, component, key, role, offset, into
                raise AssertionError("the copy-only declaration reads no source payload")

            def add_part(self, component: str, key: str, role: str, data: ArtifactBytes) -> None:
                del component, key, role, data
                raise AssertionError("the copy-only declaration adds no part")

            def add_config(self, name: str, data: ArtifactBytes) -> None:
                del name, data
                raise AssertionError("the copy-only declaration adds no config")

            def commit(self) -> ArtifactCommit:
                return ArtifactCommit("checkpoint", "sha256:" + "b" * 64, "sha256:" + "c" * 64)

            def fence(self) -> bool:
                return False

        def open_writer(_request: _ArtifactRequest) -> Backend:
            return Backend()

        inv = Invocation(
            request_id="artifact",
            spool=spool,
            deadline=time.monotonic() + 30.0,
            artifact_sink=open_writer,
        )
        envelope, outcome, _ = attempt(probe.get("uses_artifact_sink"), {}, inv)
        assert envelope is not None, outcome
        assert envelope.result == ArtifactFact("checkpoint", "sha256:" + "b" * 64)
        return (
            "descriptor declares checkpoint/1B and handler received one attempt-bound sink "
            "with only Runtime-authored commit identities"
        )

    return {
        "save": save,
        "telemetry": telemetry,
        "media_decode": media_decode,
        "scratch": scratch,
        "checkpoints": checkpoints,
        "budget": budget,
        "artifact_sink": artifact_sink,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-conformance-") as tmp:
        return _run_all(Path(tmp))


def _run_all(spool: Path) -> int:
    declared_context = {
        name
        for name in [*Context.__dataclass_fields__, *dir(Context)]
        if not name.startswith("_") and not inspect.isclass(getattr(Context, name, None))
    }
    declared_services = set(SERVICES)
    context_arms = _context_arms(spool)
    service_arms = _service_arms(spool)

    print("=== conformance: tagged-union descriptor truth")
    try:
        print("  tagged_reference    " + _tagged_union_conformance())
    except Exception as exc:
        print(f"  tagged_reference    REFUSED — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n=== conformance: every public Context fact")
    missing = sorted(declared_context - set(context_arms))
    if missing:
        print(
            f"NO EXERCISE for public Context fact(s): {', '.join(missing)}",
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
            members = [
                name
                for name, value in vars(SERVICES[capability]).items()
                if not name.startswith("_") and callable(value)
            ]
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
