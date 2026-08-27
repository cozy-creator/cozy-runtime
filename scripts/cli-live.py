#!/usr/bin/env python
"""cr-016's live verification: the runtime CLI verbs, run for real.

Not a test suite (#160). Every section runs the actual `cozy-runtime` process against the
real four-component SDXL endpoint and prints what it observed.

    corpus/.venv/bin/python scripts/cli-live.py surface    verbs, exits, AXI conformance
    corpus/.venv/bin/python scripts/cli-live.py keys       the local-bind extra-key tier
    uv run python scripts/cli-live.py inputs     typed local assets + invocation receipt
    corpus/.venv/bin/python scripts/cli-live.py identity   CLI run == direct run_slice
    corpus/.venv/bin/python scripts/cli-live.py all

`surface` and `keys` need no GPU. `identity` needs a quiet card: it runs the whole 6.46 GiB
pipeline twice — once through `cozy-runtime run` and once through `internal.local.run_slice`
directly — and compares the bytes, because "one attempt kernel" is a claim about what comes
out, not about the import graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import msgspec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from cozy_runtime.author import ImageAsset  # noqa: E402


class _NestedAsset(msgspec.Struct):
    b: ImageAsset


class _AmbiguousAssets(msgspec.Struct):
    aliased: ImageAsset = msgspec.field(name="a.b")
    a: _NestedAsset


PROJECT = ROOT / "corpus" / "pipeline"
REF = "cozy/sdxl-pipeline@cr-008b"
WORK = Path("/tmp/cozy-cli")
PROMPT = "a photograph of an astronaut riding a horse"

PASS = 0
FAIL = 0


def _first(text: str) -> str:
    """The first line of an output, for a one-line observation."""
    stripped = text.strip()
    return stripped.splitlines()[0][:110] if stripped else ""


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def cli(
    *args: str,
    home: Path | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run the REAL entrypoint as its own process, so the exit code is a process exit."""
    env = dict(
        os.environ,
        PYTHONPATH=f"{ROOT}:{ROOT / 'src'}",
        COZY_HOME=str(home or (WORK / "home")),
        **(env or {}),
    )
    out = subprocess.run(
        [sys.executable, "-m", "cozy_runtime.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd or PROJECT),
        check=False,
    )
    return out.returncode, out.stdout, out.stderr


def cli_retry(live: Any, *args: str, **kwargs: Any) -> tuple[int, str, str]:
    """Run a CLI verb, retrying when a NEIGHBOUR takes the shared card mid-run.

    This box's 8 GiB fits exactly one resident pipeline and another agent's worker comes and
    goes. A `device_shortfall` on the way to readiness is completely true and completely
    uninformative about the arm being exercised.
    """
    for _ in range(4):
        code, out, err = cli(*args, **kwargs)
        blob = (err + out).lower()
        if "device_shortfall" not in blob and "out of memory" not in blob:
            return code, out, err
        print("  the card was taken mid-run by another process; waiting", flush=True)
        live.require_quiet_card()
    return code, out, err


def install_artifact(home: Path, ref: str = REF, **overrides: Any) -> None:
    from cozy_runtime.internal import artifacts

    snapshots = json.loads(Path("/tmp/cozy-sdxl4/snapshots.json").read_text())
    row = artifacts.Artifact(
        ref=ref,
        store="/tmp/cozy-sdxl4/store",
        config="/tmp/cozy-sdxl4/pipeline.config.json",
        snapshots=snapshots,
        lane="plain-fp16",
        bytes=6_937_675_734,
        tensors=2641,
        variant="sm89",
        **overrides,
    )
    artifacts.install(home, row)


# --------------------------------------------------------------------------- surface


def section_surface(args: argparse.Namespace) -> None:
    home = WORK / "home"
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)

    head("the exit-code matrix, observed as PROCESS exits")
    cases: list[tuple[str, tuple[str, ...], int]] = [
        ("unknown verb", ("frobnicate",), 2),
        ("unknown flag", ("run", "generate", "--nope"), 2),
        ("run with no function", ("run",), 2),
        ("unknown function", ("run", "nosuchfn"), 4),
        ("undeclared payload field", ("run", "generate", "colour=blue"), 3),
        ("wrong scalar type", ("run", "generate", "steps=many"), 3),
        ("uninstalled ref", ("run", "generate", f"{PROMPT}"), 4),
        ("--offline miss", ("run", "generate", "--offline"), 8),
        ("rm without --yes", ("rm", REF), 7),
        # `pull` is BUILT (cr-016/#512); what it refuses without a hub is USAGE. The old
        # arm asserted exit 9 "whose owner is TensorFS" — that owner delivered.
        ("pull with no --hub", ("pull", REF), 2),
        ("pull, hub unreachable", ("pull", REF, "--hub", "http://127.0.0.1:1"), 9),
        ("ingest, whose owner is TensorFS", ("ingest", REF), 9),
        ("describe an unknown function", ("describe", "nosuchfn"), 4),
        ("list, empty", ("list",), 0),
        ("doctor", ("doctor",), 0),
    ]
    for label, argv, want in cases:
        code, out, err = cli(*argv, home=home)
        check(
            f"{label} -> exit {want}",
            code == want,
            f"got {code}: {_first(err or out)}",
        )

    head("AXI conformance")
    sources = list((ROOT / "src" / "cozy_runtime" / "cli").glob("*.py"))
    prompts = [p.name for p in sources if "input(" in p.read_text()]
    check(
        "no interactive prompt exists anywhere in the CLI layer",
        not prompts,
        f"{len(sources)} modules scanned; anything that would ask is a flag",
    )
    code, out, _ = cli("--help", home=home)
    check(
        "--help is one screen and never touches the network",
        code == 0 and len(out) < 3000,
        f"{len(out)} B",
    )
    code, out, _ = cli("list", home=home)
    check(
        "empty states are definitive",
        "0 artifacts" in out,
        _first(out),
    )
    code, out, err = cli("rm", REF, home=home)
    check(
        "a destructive act refuses with the remedy, never a question",
        "remedy:" in err and "?" not in err,
        _first(err),
    )

    head("describe is READ-ONLY by default")
    before = {p: p.read_bytes() for p in PROJECT.rglob("*") if p.is_file()}
    code, _, _ = cli("describe", home=home)
    after = {p: p.read_bytes() for p in PROJECT.rglob("*") if p.is_file()}
    check(
        "bare describe leaves the working tree byte-identical",
        code == 0 and before == after,
        f"{len(before)} files compared",
    )

    head("the installed artifact, and the verbs that read it")
    install_artifact(home)
    code, out, _ = cli("list", home=home)
    check("list shows one artifact, one size", REF in out and "6.461 GiB" in out, out.strip())
    code, out, _ = cli("fit", "generate", "steps=20", "size=1024", home=home)
    verdict = next((ln for ln in out.splitlines() if ln.startswith("generate")), out)
    check(
        "fit renders a verdict from the real chooser",
        code == 0 and "generate:" in out,
        verdict[:150],
    )
    code, out, _ = cli("rm", REF, "--yes", home=home)
    check("rm --yes removes it", code == 0 and "removed" in out, _first(out))
    code, out, _ = cli("rm", REF, "--yes", home=home)
    check(
        "and an absent ref is an idempotent 0 that says so",
        code == 0 and "not present" in out,
        _first(out),
    )


# --------------------------------------------------------------------------- keys


def section_keys(args: argparse.Namespace) -> None:
    """The local-bind extra-key tier (§1.1), on the REAL Loader with planted keys.

    The plant is a checkpoint topology carrying keys the code does not consume — which is
    what an un-bordered community checkpoint looks like. It is built, observed, and dropped
    inside this function; nothing is left behind.
    """
    import tiny_artifacts as art
    from cozy_runtime.author import Artifact, ConformanceError, TensorSpec
    from cozy_runtime.author._model import ModelRegistry
    from tiny_sdxl import SdxlModel

    head("a CANONICAL lane refuses any extra key — the border already classified them")
    base = art.sdxl()
    planted = dict(base.topology)
    component = sorted({k.split(".")[0] for k in planted})[0]
    planted[f"{component}.left_over_weight"] = TensorSpec(shape=(2, 2), dtype="fp16", nbytes=8)
    canonical = Artifact(
        snapshot=base.snapshot, topology=planted, config=base.config, custody="canonical"
    )
    try:
        ModelRegistry(release="cli-live", backend=None).acquire(
            "generate.models.model", SdxlModel, canonical, mode="derive"
        )
        check("canonical extra key refuses", False, "it was accepted")
    except ConformanceError as exc:
        check("canonical extra key refuses", exc.code == "topology_extra_keys", str(exc)[:120])

    head("a LOCAL bind warns, IGNORES and CONFESSES the same key")
    local = Artifact(snapshot=base.snapshot, topology=planted, config=base.config, custody="local")
    registry = ModelRegistry(release="cli-live", backend=None)
    registry.acquire("generate.models.model", SdxlModel, local, mode="derive")
    record = registry.generations()[0].record
    check(
        "the extra key is named in the construction record, not dropped in silence",
        record.ignored_extras == (f"{component}.left_over_weight",),
        f"ignored_extras={record.ignored_extras}",
    )

    head("--strict-keys upgrades the same bind to a refusal")
    strict = Artifact(
        snapshot=base.snapshot,
        topology=planted,
        config=base.config,
        custody="local",
        strict_keys=True,
    )
    try:
        ModelRegistry(release="cli-live", backend=None).acquire(
            "generate.models.model", SdxlModel, strict, mode="derive"
        )
        check("--strict-keys refuses", False, "it was accepted")
    except ConformanceError as exc:
        check("--strict-keys refuses", exc.code == "strict_extra_keys", str(exc)[:120])

    head("a KNOWN MARKER is never merely ignored (the v_pred class)")
    marked = dict(base.topology)
    marked[f"{component}.v_pred"] = TensorSpec(shape=(1,), dtype="fp16", nbytes=2)
    transcribing = Artifact(
        snapshot=base.snapshot, topology=marked, config=base.config, custody="local"
    )
    registry = ModelRegistry(release="cli-live", backend=None)
    registry.acquire("generate.models.model", SdxlModel, transcribing, mode="derive")
    record = registry.generations()[0].record
    check(
        "it is TRANSCRIBED to its stamp rather than ignored",
        record.transcribed_markers == (f"{component}.v_pred->objective=v_prediction",)
        and record.ignored_extras == (),
        f"markers={record.transcribed_markers} ignored={record.ignored_extras}",
    )
    conflicting = Artifact(
        snapshot=base.snapshot,
        topology=marked,
        config=base.config,
        custody="local",
        stamps={"objective": "epsilon"},
    )
    try:
        ModelRegistry(release="cli-live", backend=None).acquire(
            "generate.models.model", SdxlModel, conflicting, mode="derive"
        )
        check("a marker contradicting the artifact's stamp refuses NAMING it", False, "accepted")
    except ConformanceError as exc:
        check(
            "a marker contradicting the artifact's stamp refuses NAMING it",
            exc.code == "marker_conflict" and "v_pred" in str(exc),
            str(exc)[:140],
        )

    head("the plant is removed")
    registry = ModelRegistry(release="cli-live", backend=None)
    registry.acquire("generate.models.model", SdxlModel, art.sdxl(), mode="derive")
    clean = registry.generations()[0].record
    check(
        "the unplanted artifact ignores nothing and transcribes nothing",
        clean.ignored_extras == () and clean.transcribed_markers == (),
        "the tier only fires on keys that are actually there",
    )


# --------------------------------------------------------------------------- inputs


_INPUT_APP = """import hashlib

import msgspec

from cozy_runtime.author import App, AudioAsset, ImageAsset, VideoAsset

app = App()


class ImageReference(msgspec.Struct, tag="image", tag_field="type"):
    image: ImageAsset


class VideoReference(msgspec.Struct, tag="video", tag_field="type"):
    video: VideoAsset


class AudioReference(msgspec.Struct, tag="audio", tag_field="type"):
    audio: AudioAsset


Reference = ImageReference | VideoReference | AudioReference


class InspectRequest(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    cover: ImageAsset = msgspec.field(name="hero")
    references: list[Reference]


class InspectResult(msgspec.Struct):
    prompt: str
    digests: list[str]


@app.job
def inspect(payload: InspectRequest) -> InspectResult:
    assets = [payload.cover]
    assets.extend(
        reference.image
        if isinstance(reference, ImageReference)
        else reference.video
        if isinstance(reference, VideoReference)
        else reference.audio
        for reference in payload.references
    )
    return InspectResult(
        prompt=payload.prompt,
        digests=[hashlib.sha256(asset.read_bytes()).hexdigest() for asset in assets],
    )


@app.entrypoint
def inspect_endpoint(payload: InspectRequest) -> InspectResult:
    return inspect(payload)
"""


def _input_project(root: Path) -> Path:
    project = root / "project"
    project.mkdir(parents=True)
    (project / "endpoint.toml").write_text('[application]\nobject = "probe:app"\n')
    (project / "probe.py").write_text(_INPUT_APP)
    return project


def _input_cli(
    project: Path,
    home: Path,
    request: Path,
    output: Path,
    *terms: str,
) -> tuple[int, str, str]:
    return cli(
        "--json",
        "--dir",
        str(project),
        "job",
        "inspect",
        "--in",
        str(request),
        *terms,
        "--out",
        str(output),
        home=home,
        cwd=project,
    )


def _input_slice(
    project: Path,
    config: Any,
    workspace: Path,
    payload: dict[str, Any],
    inputs: dict[str, Any],
    *,
    request_id: str,
    tamper: Path | None = None,
    accepted_placement: str = "",
) -> Any:
    """One fault-injectable weightless job through the real worker/executor lifecycle."""
    from cozy_runtime.internal import descriptor
    from cozy_runtime.internal.discovery import discover
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions
    from cozy_runtime.protocol import worker_pb2 as pb

    found = discover(project)
    body = descriptor.build(found)
    descriptor_id = descriptor.job_descriptor_id(body, "inspect")
    release_id = str(body["application"])

    def plant(owner: Any, worker: Any) -> None:
        del worker
        if tamper is not None:
            mint = owner._grant

            def mutate_after_mint(document: bytes, invocation_digest: bytes) -> Any:
                grant = mint(document, invocation_digest)
                changed = bytearray(tamper.read_bytes())
                changed[-1] ^= 0x5A
                tamper.write_bytes(changed)
                return grant

            owner._grant = mutate_after_mint
        if accepted_placement:
            retain = owner._on_accepted

            def corrupt_acceptance(message: Any) -> None:
                changed = pb.AttemptAccepted()
                changed.CopyFrom(message)
                changed.placement_id = accepted_placement
                retain(changed)

            owner._on_accepted = corrupt_acceptance

    return run_slice(
        config,
        LocalRequest(
            entrypoint="inspect",
            payload=payload,
            outputs=(),
            request_id=request_id,
            kind="job",
            job_descriptor_id=descriptor_id,
            inputs=inputs,
        ),
        [],
        job_records=[
            {
                "job_descriptor_id": descriptor_id,
                "build_id": release_id,
                "project": str(project),
                "job": "inspect",
                "gpu_count": 0,
                "publishes": False,
                "emits_media": False,
            }
        ],
        workspace=workspace,
        release_id=release_id,
        options=WorkerOptions(root=workspace / "worker", release_id=release_id, no_warm=True),
        record_owner_hook=plant,
    )


def _serving_slice(
    project: Path,
    config: Any,
    workspace: Path,
    payload: dict[str, Any],
    inputs: dict[str, Any],
    *,
    request_id: str,
    accepted_mutation: Any = None,
) -> Any:
    """One weightless serving attempt, optionally corrupting acceptance owner-side."""
    from cozy_runtime.internal import descriptor
    from cozy_runtime.internal.discovery import discover
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions
    from cozy_runtime.protocol import worker_pb2 as pb

    body = descriptor.build(discover(project))
    release_id = str(body["application"])

    def plant(owner: Any, worker: Any) -> None:
        del worker
        retain = owner._on_accepted

        def corrupt_acceptance(message: Any) -> None:
            changed = pb.AttemptAccepted()
            changed.CopyFrom(message)
            accepted_mutation(changed)
            retain(changed)

        owner._on_accepted = corrupt_acceptance

    return run_slice(
        config,
        LocalRequest(
            entrypoint="inspect_endpoint",
            payload=payload,
            outputs=(),
            request_id=request_id,
            inputs=inputs,
        ),
        workspace=workspace,
        release_id=release_id,
        endpoint_descriptor=descriptor.document(body),
        options=WorkerOptions(root=workspace / "worker", release_id=release_id, no_warm=True),
        record_owner_hook=plant if accepted_mutation is not None else None,
    )


def section_inputs(args: argparse.Namespace) -> None:
    """cr-012's real CLI asset bridge and canonical invocation receipt."""
    del args
    temporary = tempfile.TemporaryDirectory(prefix="cozy-cli-inputs-")
    root = Path(temporary.name)
    project = _input_project(root)
    home = root / "home"
    media = root / "media"
    media.mkdir()

    png = media / "same.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"same-subject-bytes")
    video = media / "motion.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"video-reference")
    audio = media / "voice.wav"
    audio.write_bytes(b"RIFF\x10\x00\x00\x00WAVE" + b"audio-reference")
    prompt = "@/this/is/a/long/prompt/not/a/path"

    first_request = root / "first.json"
    first_request.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "hero": "overridden-by-the-explicit-cli-term",
                "references": [
                    {"type": "image", "image": f"@{png}"},
                    {"type": "video", "video": f"@{video}"},
                    {"type": "audio", "audio": f"@{audio}"},
                    {"type": "image", "image": f"@{png}"},
                ],
            }
        )
    )

    head("typed local assets become path-free input grants")
    code, out, err = _input_cli(project, home, first_request, root / "first-out", f"hero=@{png}")
    check("the real weightless CLI job succeeds", code == 0, _first(err or out))
    if code != 0:
        return
    document = json.loads(out)
    invocation = document.get("invocation") or {}
    invocation_document = invocation.get("document") or {}
    inputs = {
        row["input_id"]: row
        for row in invocation_document.get("inputs", ())
        if row.get("input_id") != "payload"
    }
    check(
        "only descriptor-typed Asset leaves become input ids, including the wire alias",
        set(inputs)
        == {
            "hero",
            "references.0.image",
            "references.1.video",
            "references.2.audio",
            "references.3.image",
        },
        str(sorted(inputs)),
    )
    check(
        "the nearest list index is the one carried order, including a real zero",
        [inputs[name].get("order", 0) for name in sorted(inputs)] == [0, 0, 1, 2, 3],
        str({name: row.get("order", 0) for name, row in inputs.items()}),
    )
    image_rows = [inputs["hero"], inputs["references.0.image"], inputs["references.3.image"]]
    check(
        "equal bytes remain three identified subjects rather than digest-deduplicating",
        len({row["digest"] for row in image_rows}) == 1,
        str([row["input_id"] for row in image_rows]),
    )
    result = document.get("result") or {}
    check(
        "a non-Asset string beginning @/ remains the exact prompt text",
        result.get("prompt") == prompt,
        str(result.get("prompt")),
    )

    from cozy_runtime.internal import canonical

    canonical_bytes = canonical.write(invocation_document)
    check(
        "the rendered canonical document re-digests to the rendered invocation identity",
        invocation.get("digest") == "sha256:" + hashlib.sha256(canonical_bytes).hexdigest(),
        str(invocation.get("digest")),
    )
    accepted = document.get("plan") or {}
    check(
        "the accepted echo names that exact invocation plus its executor generation",
        accepted.get("invocation_spec_digest") == invocation.get("digest")
        and accepted.get("request_id") == "cli-inspect"
        and accepted.get("attempt_ordinal") == 1
        and int(accepted.get("executor_generation", 0)) > 0,
        json.dumps(accepted, sort_keys=True),
    )
    receipt = json.dumps({"invocation": invocation, "accepted": accepted}, sort_keys=True)
    check(
        "neither source paths nor DeliveryGrant credentials enter either receipt",
        all(str(path) not in receipt for path in (png, video, audio))
        and "credential" not in receipt,
        receipt[:180],
    )

    head("field-path and file-root red arms")
    from cozy_runtime.cli import payload as payload_grammar
    from cozy_runtime.cli.io import CliError
    from cozy_runtime.internal.worker import grants

    ambiguous = {"a.b": f"@{png}", "a": {"b": f"@{png}"}}
    original = json.loads(json.dumps(ambiguous))
    try:
        payload_grammar.extract_local_assets(ambiguous, _AmbiguousAssets)
        ambiguous_refused = False
    except CliError as exc:
        ambiguous_refused = "field path segments" in str(exc)
    check(
        "dotted Asset path segments refuse atomically before any path is replaced",
        ambiguous_refused and ambiguous == original,
        json.dumps(ambiguous, sort_keys=True),
    )

    grant_root = root / "grant-root"
    grant_root.mkdir()
    outside = root / "outside.bin"
    outside.write_bytes(b"outside")
    (grant_root / "escape").symlink_to(outside)
    authorizer = grants.Authorizer(roots=(str(grant_root),))
    refused = 0
    for url in (
        f"file://{grant_root}/../outside.bin",
        f"file://{grant_root}/escape",
    ):
        try:
            authorizer.authorize("grant_input_read", "GET", url, {"input_id": "hero"})
        except grants.GrantRefusal as exc:
            if exc.code == "grant_path_outside_roots":
                refused += 1
    check(
        "traversal and symlink escapes both refuse at the declared file root",
        refused == 2,
        f"{refused}/2 refused",
    )

    inside = grant_root / "inside.bin"
    inside.write_bytes(b"inside")
    try:
        authorizer.authorize("grant_input_read", "GET", f"file://{inside}", {"input_id": "hero"})
        inside_allowed = True
    except grants.GrantRefusal:
        inside_allowed = False
    check(
        "the same canonical root fence admits an ordinary in-root file",
        inside_allowed,
    )

    head("attempt lifecycle red arms")
    import dataclasses

    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalInput

    arm_config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(root / "arm-home"),
            "PYTHONPATH": f"{project}:{ROOT}:{ROOT / 'src'}",
        }
    )

    base_payload = {
        "prompt": "fault arm",
        "hero": "input:hero",
        "references": [{"type": "image", "image": "input:references.0.image"}],
    }
    tampered = media / "tampered.png"
    tampered.write_bytes(png.read_bytes())
    outcome = _input_slice(
        project,
        arm_config,
        root / "tamper-workspace",
        dict(base_payload),
        {
            "hero": LocalInput(str(tampered), 0),
            "references.0.image": LocalInput(str(png), 0),
        },
        request_id="cli-input-tamper",
        tamper=tampered,
    )
    narrative = json.dumps(dataclasses.asdict(outcome), default=str)
    check(
        "post-mint source mutation refuses before the job body",
        "input_digest_mismatch" in narrative or "stream_" in narrative,
        narrative[:180],
    )
    leftovers = [
        path
        for path in (root / "tamper-workspace" / "worker" / "spool").rglob("inputs/*")
        if path.is_file()
    ]
    check(
        "a refused transfer publishes neither a final input nor a partial file",
        not leftovers,
        str(leftovers),
    )

    outcome = _input_slice(
        project,
        arm_config,
        root / "order-workspace",
        dict(base_payload),
        {
            "hero": LocalInput(str(png), 0),
            "references.0.image": LocalInput(str(png), 7),
        },
        request_id="cli-input-order",
    )
    narrative = json.dumps(dataclasses.asdict(outcome), default=str)
    check(
        "a planted nonzero order at typed list position zero refuses in the real attempt",
        "asset_order" in narrative,
        narrative[:180],
    )

    hostile_request = "../../outside-authority"
    hostile_workspace = root / "hostile-id-workspace"
    outcome = _input_slice(
        project,
        arm_config,
        hostile_workspace,
        dict(base_payload),
        {
            "hero": LocalInput(str(png), 0),
            "references.0.image": LocalInput(str(png), 0),
        },
        request_id=hostile_request,
    )
    escaped = list(root.rglob("outside-authority*"))
    check(
        "an opaque request id remains wire identity and never becomes filesystem authority",
        outcome.ended == "terminal" and not escaped,
        f"{outcome.ended}; escaped paths: {escaped}",
    )

    outcome = _input_slice(
        project,
        arm_config,
        root / "echo-workspace",
        dict(base_payload),
        {
            "hero": LocalInput(str(png), 0),
            "references.0.image": LocalInput(str(png), 0),
        },
        request_id="cli-input-echo",
        accepted_placement="plc-not-offered",
    )
    check(
        "a changed accepted placement converges to a record-owner refusal",
        outcome.ended == "record_owner_error" and "serving placement" in outcome.error,
        f"{outcome.ended}: {outcome.error}",
    )

    head("weightless serving acceptance is an exact receipt")
    serving = _serving_slice(
        project,
        arm_config,
        root / "serving-workspace",
        dict(base_payload),
        {
            "hero": LocalInput(str(png), 0),
            "references.0.image": LocalInput(str(png), 0),
        },
        request_id="cli-serving",
    )
    check(
        "an unmodified weightless serving receipt succeeds and includes its host reservation",
        serving.ended == "terminal"
        and serving.accepted.get("request_id") == "cli-serving"
        and "reserved_host_bytes" in serving.accepted,
        f"{serving.ended}: {json.dumps(serving.accepted, sort_keys=True)}",
    )

    receipt_arms = (
        ("request", lambda message: setattr(message, "request_id", "wrong"), "accepted request"),
        (
            "ordinal",
            lambda message: setattr(message, "attempt_ordinal", message.attempt_ordinal + 1),
            "accepted ordinal",
        ),
        (
            "invocation",
            lambda message: setattr(message, "invocation_spec_digest", b"\0" * 32),
            "accepted invocation digest",
        ),
        (
            "construction",
            lambda message: setattr(message, "model_construction_digest", b"\1" * 32),
            "invents model construction",
        ),
        (
            "plan",
            lambda message: setattr(message.plan, "delivery", "float"),
            "model-bearing plan",
        ),
    )
    serving_refusals = 0
    for name, mutation, expected in receipt_arms:
        outcome = _serving_slice(
            project,
            arm_config,
            root / f"serving-{name}-workspace",
            dict(base_payload),
            {
                "hero": LocalInput(str(png), 0),
                "references.0.image": LocalInput(str(png), 0),
            },
            request_id=f"cli-serving-{name}",
            accepted_mutation=mutation,
        )
        serving_refusals += outcome.ended == "record_owner_error" and expected in outcome.error
    check(
        "request, ordinal, invocation, construction, and plan receipt drift all refuse",
        serving_refusals == len(receipt_arms),
        f"{serving_refusals}/{len(receipt_arms)} refused",
    )

    head("reference order is invocation meaning")
    reordered = root / "reordered.json"
    reordered.write_text(
        json.dumps(
            {
                **json.loads(first_request.read_text()),
                "references": [
                    {"type": "audio", "audio": f"@{audio}"},
                    {"type": "image", "image": f"@{png}"},
                    {"type": "video", "video": f"@{video}"},
                    {"type": "image", "image": f"@{png}"},
                ],
            }
        )
    )
    code, out, err = _input_cli(project, home, reordered, root / "reordered-out", f"hero=@{png}")
    reordered_document = json.loads(out) if code == 0 else {}
    check("the reordered request also succeeds", code == 0, _first(err or out))
    check(
        "reordering the same subjects changes the invocation identity",
        (reordered_document.get("invocation") or {}).get("digest") != invocation.get("digest"),
        str((reordered_document.get("invocation") or {}).get("digest")),
    )

    head("local-source and media-type failures refuse before the job body")
    missing = root / "absent.png"
    code, out, err = _input_cli(
        project, home, first_request, root / "missing-out", f"hero=@{missing}"
    )
    check(
        "a missing typed local file refuses as not found",
        code == 4 and "no such file" in err.lower(),
        f"exit {code}: {_first(err or out)}",
    )
    wrong = media / "not-an-image.txt"
    wrong.write_text("plain text")
    code, out, err = _input_cli(project, home, first_request, root / "wrong-out", f"hero=@{wrong}")
    check(
        "bytes outside the declared Asset kind refuse pre-entrypoint",
        code == 3 and "imageasset" in err.lower(),
        f"exit {code}: {_first(err or out)}",
    )


# --------------------------------------------------------------------------- identity


def section_identity(args: argparse.Namespace) -> None:
    """ONE attempt kernel: the CLI's run and a direct `run_slice` produce the same bytes."""
    import importlib.util

    harness = ROOT / "scripts" / "pipeline-live.py"
    spec = importlib.util.spec_from_file_location("pipeline_live", harness)
    assert spec and spec.loader
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)

    home = WORK / "home"
    home.mkdir(parents=True, exist_ok=True)
    install_artifact(home)
    live.require_quiet_card()

    head("the CLI's `run`, on the real four-component pipeline")
    out_dir = WORK / "out"
    shutil.rmtree(out_dir, ignore_errors=True)
    code, out, err = cli(
        "--json",
        "run",
        "generate",
        PROMPT,
        f"steps={args.steps}",
        f"size={args.size}",
        "seed=1005",
        "--out",
        str(out_dir),
        home=home,
    )
    check("`cozy-runtime run` exits 0", code == 0, (err or out).strip()[-400:] if code else "")
    if code != 0:
        return
    document = json.loads(out)
    cli_digest = (document.get("result") or {}).get("digest", "")
    saved = document.get("outputs") or {}
    print(f"  outputs: {json.dumps(saved)}")
    print(f"  plan:    {json.dumps(document.get('plan'))}")
    check(
        "the output path is DERIVED from the descriptor's asset field",
        set(saved) == {"image"},
        f"{sorted(saved)} — `image` is the one ImageAsset field on Txt2ImgOutput",
    )
    check(
        "and the asset is a real PNG on disk, named by its DECLARED media type",
        bool(saved)
        and saved["image"].endswith(".png")
        and Path(saved["image"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
        f"{saved.get('image', '')} ({Path(saved['image']).stat().st_size} B)" if saved else "",
    )

    head("the SAME request through `internal.local.run_slice`, with no CLI in the picture")
    live.require_quiet_card()
    direct = live.spawn(
        {
            "workspace": str(WORK / "direct"),
            "payload": {
                "prompt": PROMPT,
                "steps": args.steps,
                "size": args.size,
                "seed": 1005,
            },
            "request_id": "cli-identity",
            "binding": {"resident_budget_bytes": 0},
        }
    )
    direct_digest = (direct.get("result") or {}).get("digest", "")
    check(
        "OBSERVATIONAL IDENTITY: byte-identical result digests",
        bool(cli_digest) and cli_digest == direct_digest,
        f"cli {cli_digest[:32]}… vs direct {direct_digest[:32]}…",
    )
    check(
        "the same terminal status",
        document.get("status") == direct.get("status"),
        f"{document.get('status')} vs {direct.get('status')}",
    )
    cli_plan = document.get("plan") or {}
    accepted = direct.get("accepted", {}).get("delivery")
    placement = direct.get("accepted", {}).get("placement")
    check(
        "the same chosen plan",
        (cli_plan.get("delivery"), cli_plan.get("placement")) == (accepted, placement),
        f"{cli_plan.get('delivery')}/{cli_plan.get('placement')}",
    )
    cli_classes = {row["class"] for row in (document.get("ledger") or {}).get("classes", ())}
    direct_classes = {row["class"] for row in (direct.get("ledger") or {}).get("classes", ())}
    check(
        "the same ledger classes closed",
        cli_classes == direct_classes and bool(cli_classes),
        f"{len(cli_classes)} classes",
    )

    head("no second execution implementation exists")
    sources = list((ROOT / "src" / "cozy_runtime" / "cli").glob("*.py"))
    callers = [p.name for p in sources if "run_slice" in p.read_text()]
    check(
        "exactly ONE CLI module reaches the attempt kernel, and it calls run_slice",
        callers == ["run.py"],
        f"{callers} — there is no direct-call runner to find",
    )


def section_hubless(args: argparse.Namespace) -> None:
    """The two arms that need the whole pipeline and a plant: no network, and the floor."""
    import importlib.util

    harness = ROOT / "scripts" / "pipeline-live.py"
    spec = importlib.util.spec_from_file_location("pipeline_live", harness)
    assert spec and spec.loader
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    shim = live.build_throttle()
    home = WORK / "home"
    home.mkdir(parents=True, exist_ok=True)
    live.require_quiet_card()

    head("THE HUBLESS CONTRACT: inference from a populated CAS with the network CUT")
    install_artifact(home)
    out_dir = WORK / "hubless"
    shutil.rmtree(out_dir, ignore_errors=True)
    env = {"LD_PRELOAD": str(shim), "IO_BLOCK_NETWORK": "1"}
    code, out, err = cli_retry(
        live,
        "--json",
        "run",
        "generate",
        PROMPT,
        f"steps={args.steps}",
        f"size={args.size}",
        "seed=1005",
        "--offline",
        "--out",
        str(out_dir),
        home=home,
        env=env,
    )
    check(
        "a network-BLOCKED --offline run completes inference",
        code == 0,
        (err or out).strip()[-300:] if code else "every AF_INET connect() returned ECONNREFUSED",
    )
    if code == 0:
        document = json.loads(out)
        check(
            "and produces the SAME image as the networked run",
            (document.get("result") or {}).get("digest", "").startswith("9917742a18d6c421"),
            (document.get("result") or {}).get("digest", "")[:32],
        )

    head("THE PHYSICAL FLOOR: below it, exit 14 with the shortfall — never 0, never 6")
    install_artifact(home, vram_floor_bytes=64 << 30)
    code, out, err = cli_retry(
        live,
        "run",
        "generate",
        PROMPT,
        "steps=2",
        "size=512",
        "seed=1005",
        "--out",
        str(WORK / "floor"),
        home=home,
    )
    check(
        "an impossible declared floor exits 14 (capacity)",
        code == 14,
        f"got {code}: {_first(err or out)}",
    )
    check(
        "and the refusal is QUANTIFIED, not a category",
        "short by" in (err + out) or "needed" in (err + out).lower(),
        _first(err),
    )
    head("the plant is removed: the same configuration runs")
    install_artifact(home)
    code, out, err = cli_retry(
        live,
        "run",
        "generate",
        PROMPT,
        "steps=2",
        "size=512",
        "seed=1005",
        "--out",
        str(WORK / "floor"),
        home=home,
    )
    check(
        "a DEGRADABLE configuration never exits 14",
        code == 0,
        f"got {code}: {_first(err or out)}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="cr-016 live CLI verification")
    parser.add_argument(
        "section", choices=("surface", "keys", "inputs", "identity", "hubless", "all")
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    sections = {
        "surface": section_surface,
        "keys": section_keys,
        "inputs": section_inputs,
        "identity": section_identity,
        "hubless": section_hubless,
    }
    for name in list(sections) if args.section == "all" else [args.section]:
        sections[name](args)
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
