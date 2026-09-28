"""`cozy-runtime run` and `cozy-runtime job` — the git-clone contract's load-bearing verbs.

One-shot UX over an EPHEMERAL LOCAL WORKER and the same attempt kernel `serve` uses.
Weightless entrypoints and jobs remain local. Model-bound entrypoints use the explicit
DevelopmentPackage lane: the existing package.toml resolver selects a local TensorFS Repository
release and the worker independently re-derives that exact selection before STAGED. No flat
binding record or Tensorhub hardware-evidence claim is minted.

What this module DOES NOT do is decide anything the runtime decides. The plan, the ladder,
the confession and the shortfall all arrive on the outcome; this renders them.
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author import ActivationCapture, ConformanceError, InvalidRequest
from cozy_runtime.cli import payload as payload_grammar
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import bindings as binding_table
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.local import (
    LocalRequest,
    SliceOutcome,
    development_placement_documents,
    run_slice,
)
from cozy_runtime.internal.worker.session import WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

VALUE_FLAGS = (
    "--model",
    "--seed",
    "--out",
    "--timeout",
    "--in",
    "--input",
    "--org",
    "--tensorfs-root",
    "--capture",
)
BOOL_FLAGS = ("--stream",)

#: THE EXECUTION-PATH OVERRIDE's reserved run keys (cr-125). `kernel.` is a dotted namespace
#: claimed BEFORE the payload grammar sees a term, exactly as `--model` claims the weights
#: off the owner's ladder, so `kernel.attention` can never land on a package's own field. One
#: axis: attention. A GEMM or fusion pin is not here because no measurement has asked for one.
KERNEL_AXES = ("attention",)

GIB = 1 << 30
MIB = 1 << 20

#: How a terminal status becomes a process exit. The runtime already decided what happened;
#: this is a table, never a second judgement.
TERMINAL_EXITS: Mapping[str, Exit] = {
    "OUTCOME_STATUS_SUCCEEDED": Exit.ok,
    "OUTCOME_STATUS_FAILED": Exit.failed,
    "OUTCOME_STATUS_CANCELED": Exit.canceled,
    "OUTCOME_STATUS_ABANDONED": Exit.failed,
}

#: A refusal terminal carries a CAUSE, and the cause is what picks the code — a refusal is
#: not one condition. `capacity` is the quantified below-floor exit (14), which is never
#: exit 0 and never the structural 6.
CAUSE_EXITS: Mapping[int, Exit] = {
    pb.CauseCode.CAUSE_CODE_INVALID_REQUEST: Exit.validation,
    pb.CauseCode.CAUSE_CODE_UNSUPPORTED_INPUT: Exit.structural,
    pb.CauseCode.CAUSE_CODE_CONSTRAINT_INFEASIBLE: Exit.capacity,
    pb.CauseCode.CAUSE_CODE_ARTIFACT_UNFETCHABLE: Exit.offline_miss,
    pb.CauseCode.CAUSE_CODE_CAPABILITY_UNAVAILABLE: Exit.structural,
    pb.CauseCode.CAUSE_CODE_DEADLINE_EXPIRED: Exit.deadline,
    pb.CauseCode.CAUSE_CODE_LOCAL_SAFETY: Exit.validation,
    pb.CauseCode.CAUSE_CODE_CLIENT_CANCEL: Exit.canceled,
}


class Refusal(CliError):
    """A typed refusal that already carries its exit code from the runtime's own verdict."""


def split_flags(args: Sequence[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Separate payload terms from the verb's flags. `--model` is REPEATABLE.

    A `kernel.<axis>=<value>` term is CLAIMED here and returned under its own key, so the
    reserved namespace is resolved before any payload field is; an axis this runtime does
    not have refuses rather than becoming a field named `kernel.gemm`.
    """
    terms: list[str] = []
    flags: dict[str, list[str]] = {}
    rest = list(args)
    while rest:
        item = rest.pop(0)
        name, eq, inline = item.partition("=")
        if name in VALUE_FLAGS:
            if not eq:
                if not rest:
                    raise CliError(
                        "usage", f"{name} needs a value", f"pass {name} <value>", Exit.usage
                    )
                inline = rest.pop(0)
            flags.setdefault(name, []).append(inline)
        elif item in BOOL_FLAGS:
            flags.setdefault(item, []).append("")
        elif name.startswith("kernel."):
            axis = name.removeprefix("kernel.")
            if axis not in KERNEL_AXES or not eq or not inline:
                raise CliError(
                    "usage",
                    f"{item!r} is not an execution-path override",
                    "the reserved namespace is "
                    + ", ".join(f"kernel.{a}=<name>" for a in KERNEL_AXES),
                    Exit.usage,
                )
            flags.setdefault(name, []).append(inline)
        elif item.startswith("--"):
            raise CliError(
                "usage",
                f"unknown flag {item!r}",
                "run `cozy-runtime run --help` for this verb's flags",
                Exit.usage,
            )
        else:
            terms.append(item)
    return terms, flags


def _overrides(values: Sequence[str], slots: Sequence[Any]) -> dict[str, str]:
    """`--model [binding-path=]ref`. The bare form is legal only with ONE model slot."""
    out: dict[str, str] = {}
    for value in values:
        path, eq, ref = value.partition("=")
        if not eq:
            if len(slots) != 1:
                raise CliError(
                    "usage",
                    f"--model {value!r} is bare, but this function declares {len(slots)} "
                    "model binding paths",
                    "qualify it: --model "
                    + (slots[0].path if slots else "<binding-path>")
                    + "=<ref>",
                    Exit.usage,
                )
            out[slots[0].path] = value
            continue
        if path not in {s.path for s in slots}:
            raise CliError(
                "usage",
                f"--model {path}=… names no binding path on this function",
                f"declared: {', '.join(sorted(s.path for s in slots)) or 'none'}",
                Exit.usage,
            )
        out[path] = ref
    return out


def run(
    verb: str,
    args: list[str],
    opts: Options,
    config: RuntimeConfig,
    *,
    emit_line: Any = print,
) -> Result:
    """One `run`/`job` invocation, end to end. Raises `Refusal` with the mapped exit code."""
    terms, flags = split_flags(args)
    if not terms:
        raise CliError(
            "usage",
            f"{verb} needs a function name",
            f"cozy-runtime {verb} <function> [<primary>] [key=value …]",
            Exit.usage,
            next=("cozy-runtime describe — the functions this release registers",),
        )
    function, terms = terms[0], terms[1:]
    project = Path(opts.dir).expanduser().resolve()

    try:
        found = discover(project)
        # The worker re-derives this document from source and refuses any difference, so
        # both sides use its one builder; the import above only supplies Python types.
        body = static_interface.build(project)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "a package's surface must be valid before it can run (§1.0)",
            Exit.validation,
        ) from exc

    kind = "job" if verb == "job" else "entrypoint"
    surface = next((s for s in found.surfaces if s.name == function), None)
    if surface is None or surface.kind != kind or surface.hidden:
        known = sorted(s.name for s in found.surfaces if s.kind == kind and not s.hidden)
        raise CliError(
            "not_found",
            f"no registered {kind} named {function!r}",
            f"this release registers: {', '.join(known) or f'0 {kind}s'}",
            Exit.not_found,
            next=("cozy-runtime describe — the full list",),
        )

    slots = surface.model_bindings
    if slots and kind == "job":
        raise CliError(
            "structural",
            f"{function} declares {len(slots)} derive-only Model param(s), but the local CLI "
            "has not resolved source/profile outputs into exact Manifest rows",
            "publish or prepare the source Manifests first; the worker accepts only exact "
            "model:<parameter> bindings and never guesses a local model or loads one",
            Exit.structural,
        )
    capture = None
    if flags.get("--capture"):
        from cozy_runtime.probe._capture import validate_components

        try:
            capture = msgspec.json.decode(flags["--capture"][-1], type=ActivationCapture)
            validate_components(capture, [slot.model_class for slot in slots])
            if kind != "entrypoint":
                raise InvalidRequest("capture is an entrypoint option", code="capture_mode")
        except (ValueError, msgspec.DecodeError, InvalidRequest) as exc:
            raise CliError(
                "capture",
                str(exc),
                "pass --capture as an ActivationCapture JSON object",
                Exit.validation,
            ) from exc
    overrides = _overrides(flags.get("--model") or [], slots)
    tensorfs_root: Path | None = None
    if slots:
        values = flags.get("--tensorfs-root") or []
        if len(values) != 1:
            raise CliError(
                "usage",
                "model-bound local execution requires one --tensorfs-root",
                "pass the explicit local TensorFS Store root",
                Exit.usage,
            )
        tensorfs_root = Path(values[0]).expanduser().resolve()

    doc = _doc_for(body, function, kind)
    request_schema = doc["request"]
    document = payload_grammar.parse(terms, request_schema, infile=(flags.get("--in") or [""])[0])
    seed = (flags.get("--seed") or [""])[0]
    if seed and "seed" in payload_grammar.fields(request_schema):
        document["seed"] = int(seed)
    inputs = payload_grammar.extract_local_assets(document, surface.payload_type)
    outputs = payload_grammar.asset_paths(doc["result"])
    warnings: list[str] = []

    job_records: list[dict[str, Any]] = []
    trees: dict[str, str] = {}
    descriptor_id = ""
    if kind == "job":
        descriptor_id = package_interface.job_descriptor_id(body, function)
        trees = _input_trees(flags.get("--input") or [])

    out_dir = Path((flags.get("--out") or ["outputs"])[0]).expanduser().resolve()
    workspace = out_dir / ".run"
    if workspace.exists():
        from cozy_runtime.internal.worker.byte_inputs import remove_checkout

        remove_checkout(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    package_installation_id = "local-" + uuid.uuid4().hex
    if kind == "job":
        interface_path = workspace / "metadata" / "package-interface.json"
        interface_path.parent.mkdir()
        interface_path.write_bytes(package_interface.canonical_bytes(body))
        job_records.append(
            {
                "job_descriptor_id": descriptor_id,
                "installation_id": package_installation_id,
                "application": str(body["application"]),
                "package_interface": str(interface_path),
                "python": sys.executable,
                "job": function,
                "publishes": bool(surface.publishes),
                "emits_media": bool(surface.emits_media),
                "gpu_rate_micro_usd_per_hour": 0,
                "cap_micro_usd": 0,
                "reclaim_on_terminal": True,
            }
        )

    request = LocalRequest(
        entrypoint=function,
        payload=document,
        capture=capture,
        outputs=outputs,
        request_id=f"cli-{function}-{uuid.uuid4().hex}",
        timeout_ms=_timeout((flags.get("--timeout") or [""])[0]),
        kind=kind if kind == "job" else "entrypoint",
        job_descriptor_id=descriptor_id,
        trees=trees,
        inputs=inputs,
        org=(flags.get("--org") or ["local"])[0],
    )
    # The worker narrates itself to stdout, which is the right place for a long-lived
    # `serve` process and the wrong place for a verb whose stdout IS a result document. The
    # narration is not suppressed — it is a real progress channel — it just goes where
    # progress goes, so `--json` emits exactly the document and nothing else.
    # A local source checkout is the installed release for this one process. The device
    # executor is exec-first and imports the interface's application in a fresh interpreter,
    # so it must inherit the same explicit project root that discovery used above. Remote
    # serving imports its wheel from the provisioned environment and never enters this path.
    with contextlib.chdir(project), contextlib.redirect_stdout(sys.stderr):
        outcome = run_slice(
            # This worker and its executable plans belong to this one output
            # workspace. Another CLI invocation may use a different interpreter
            # or metadata path for the same source revision.
            replace(config, cozy_home=workspace / "home"),
            request,
            job_records=job_records,
            workspace=workspace,
            package_installation_id=package_installation_id,
            package_interface_document=body if kind == "entrypoint" else None,
            project=project,
            development_tensorfs_root=tensorfs_root,
            development_binding_overrides=overrides,
            options=WorkerOptions(
                root=workspace / "worker",
                release_id=package_installation_id,
                # THE EXECUTION-PATH OVERRIDE (cr-125), and the only entrance it has: this
                # ephemeral worker. `serve` takes no such flag, so a serving deployment
                # cannot be pinned and keeps choosing the fastest kernel by itself.
                attention_pin=(flags.get("kernel.attention") or [""])[-1],
            ),
        )
    return _render(outcome, out_dir, warnings, opts, verb=verb, stream="--stream" in flags)


def _input_trees(values: Sequence[str]) -> dict[str, str]:
    """`--input <ref>=<dir>` — one MATERIALIZED input tree grant per typed Tree field.

    Locally the RecordOwner grants a directory it can already read; on the network path
    tfs-003 materializes the snapshot and th-004 mints the same grant. Either way, the job's
    typed field carries the ref and the GRANT carries the capability.
    """
    out: dict[str, str] = {}
    for value in values:
        ref, eq, path = value.partition("=")
        if not eq or not ref:
            raise CliError(
                "usage",
                f"--input {value!r} is not <ref>=<directory>",
                "grant one materialized input tree per typed Tree field: "
                "--input org/model@release=/path/to/tree",
                Exit.usage,
            )
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise CliError(
                "not_found",
                f"--input {ref}: {resolved} is not a directory",
                "an input tree is a MATERIALIZED directory; a job never opens a store by name",
                Exit.not_found,
            )
        out[ref] = str(resolved)
    return out


def _doc_for(body: Mapping[str, Json], function: str, kind: str) -> Mapping[str, Json]:
    entries = body["jobs" if kind == "job" else "entrypoints"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        if entry["name"] == function:
            return entry
    raise AssertionError(f"{function} was discovered but is not in the package interface")


def _timeout(spelled: str) -> int:
    if not spelled:
        return 0
    units = {"ms": 1, "s": 1000, "m": 60_000, "h": 3_600_000}
    for suffix, scale in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if spelled.endswith(suffix):
            try:
                return int(float(spelled[: -len(suffix)]) * scale)
            except ValueError:
                break
    raise CliError(
        "usage",
        f"--timeout {spelled!r} is not a duration",
        "durations are <n>ms | <n>s | <n>m | <n>h",
        Exit.usage,
    )


def resolved_bindings(opts: Options) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every declared slot's RESOLVED SELECTION, from the one resolver (cl-010's debt 7).

    It prints what `package.toml` selects: the ref, lane and rule. Downloaded model state is
    owned by TensorFS and the worker placement, not by a Runtime-local index.

    It resolves and reports; it constructs nothing, touches no device and loads no weights.
    """
    project = Path(opts.dir).expanduser().resolve()
    try:
        if opts.package_interface:
            body = package_interface.read_bytes(
                Path(opts.package_interface).expanduser().resolve().read_bytes(),
                "package bindings interface",
            )
        else:
            body = static_interface.build(
                project,
                environment_python=Path(opts.environment_python)
                if opts.environment_python
                else None,
            )
        table = binding_table.read_table(project / "package.toml")
        slots = _interface_slots(body)
        optional = frozenset(slot.path for slot in slots if slot.job)
    except (OSError, ConformanceError) as exc:
        raise CliError(
            "validation",
            str(exc),
            "supply package.toml and the exact package source or interface",
            Exit.validation,
        ) from exc
    visible = list(body.get("entrypoints", []))
    if not slots:
        plans = (
            list(
                development_placement_documents(
                    body,
                    package="local/development-package",
                    release="0.0.0",
                    installation_id="local-" + uuid.uuid4().hex,
                ).subjects()
            )
            if visible and not opts.package_interface
            else []
        )
        return [], plans
    try:
        # ONE resolve over the WHOLE declared slot set — jobs' included — which is what a
        # serving deployment does: the selection is checked against every slot the release
        # declares, so a binding for a slot that does not exist refuses here rather than
        # being silently unused by whichever entrypoint happened to be asked for. A job's
        # slot is declared but its source binds per invocation, so it may stay unbound.
        resolved = binding_table.resolve(slots, table, optional=optional)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "code states capability, bindings state selection — name the slot in package.toml",
            Exit.validation,
        ) from exc
    records: list[dict[str, Any]] = []
    for slot in sorted(slots, key=lambda s: s.path):
        chosen = resolved.get(slot.path)
        if chosen is None:  # an unbound job slot: declared, selected per invocation
            continue
        records.append(
            {
                "model_binding_path": slot.path,
                "model_parameter_name": slot.param,
                "model_class": slot.class_key,
                "ref": chosen.ref,
                "lane": chosen.lane or "",
                "source": chosen.source,
            }
        )
    return records, []


@dataclass(frozen=True, slots=True)
class _InterfaceSlot:
    path: str
    param: str
    class_key: str
    job: bool


def _interface_slots(body: Mapping[str, Any]) -> list[_InterfaceSlot]:
    """Recover the binding vocabulary already frozen into PackageInterface/1.

    Jobs' model slots are declared exactly like entrypoints' — the jobs array is read
    wherever the entrypoints array is — a job's slot just binds per invocation.
    """
    out: dict[str, _InterfaceSlot] = {}
    for collection, job in (("entrypoints", False), ("jobs", True)):
        rows = body.get(collection, [])
        if not isinstance(rows, list):
            raise ConformanceError(
                f"package interface {collection} are malformed", code="package_interface"
            )
        for callable_ in rows:
            if not isinstance(callable_, dict):
                raise ConformanceError(
                    f"package interface {collection} entry is malformed",
                    code="package_interface",
                )
            models = callable_.get("models", [])
            if not isinstance(models, list):
                raise ConformanceError(
                    "package interface model slots are malformed", code="package_interface"
                )
            for model in models:
                if not isinstance(model, dict):
                    raise ConformanceError(
                        "package interface model slot is malformed", code="package_interface"
                    )
                path, class_key = model.get("path"), model.get("class")
                if not isinstance(path, str) or not isinstance(class_key, str) or not class_key:
                    raise ConformanceError(
                        "package interface model slot identity is malformed",
                        code="package_interface",
                    )
                prefix = f"{callable_.get('name', '')}.models."
                if not path.startswith(prefix) or not path.removeprefix(prefix):
                    raise ConformanceError(
                        f"package interface model slot {path!r} is malformed",
                        code="package_interface",
                    )
                slot = _InterfaceSlot(path, path.removeprefix(prefix), class_key, job)
                if path in out and out[path] != slot:
                    raise ConformanceError(
                        f"package interface model slot {path!r} conflicts",
                        code="package_interface",
                    )
                out[path] = slot
    return [out[path] for path in sorted(out)]


def _render(
    outcome: SliceOutcome,
    out_dir: Path,
    warnings: list[str],
    opts: Options,
    *,
    verb: str,
    stream: bool,
) -> Result:
    """The outcome as rows, or the typed refusal it actually is."""
    if outcome.ended == "unbound_entrypoint":
        raise CliError(
            "not_found",
            outcome.error,
            "the installed artifact and the requested function must name the same binding; "
            "`cozy-runtime describe` lists what this release registers",
            Exit.not_found,
            next=("cozy-runtime describe",),
        )
    if outcome.ended != "terminal":
        raise CliError(
            "unavailable",
            f"the run ended {outcome.ended}: {outcome.error or 'no terminal was produced'}",
            "the worker could not become ready; `cozy-runtime doctor` reports this host's "
            "device, driver and CAS facts",
            Exit.unavailable,
            next=("cozy-runtime doctor",),
        )
    status = outcome.status
    terminal = outcome.terminal
    code = _exit_for(status, terminal.cause)

    saved = _save(outcome, out_dir)
    rows: list[tuple[str, str]] = []
    for name, path in sorted(saved.items()):
        stat = Path(path).stat()
        rows.append((name, f"{path}  {stat.st_size} B"))
    if not saved:
        rows.append(("outputs", "0 assets"))
    if outcome.accepted.lane_id:
        rows.append(("lane", outcome.accepted.lane_id))
    # The bind's CONFESSION, rendered where the person who ran it will see it. The runtime
    # already decided to ignore those keys; this is the "loudly" half of "loudly ignored".
    facts = _bind_facts(outcome)
    if ignored := _split(facts.get("ignored_extra_keys")):
        warnings.append(
            f"{len(ignored)} extra checkpoint key(s) the code does not consume were IGNORED "
            f"on a local bind: {', '.join(str(k) for k in ignored[:6])}"
            + ("…" if len(ignored) > 6 else "")
        )
        rows.append(("ignored_extra_keys", ", ".join(str(k) for k in ignored)))
    if markers := _split(facts.get("transcribed_markers")):
        rows.append(("transcribed_markers", ", ".join(str(m) for m in markers)))
    rows.append(("runtime", f"{outcome.timings.get('wall_ms', 0) / 1000:.2f} s"))
    if peak := terminal.metrics.peak_device_memory_bytes:
        rows.append(("peak_vram", f"{peak / MIB:.0f} MiB"))
    rows.extend(
        (
            "adjusted",
            f"{row.field}: requested {row.requested} -> applied {row.applied} ({row.reason})",
        )
        for row in terminal.result.adjustments
    )
    # The protocol messages are rendered as their documents here, at the output boundary only.
    document: dict[str, Any] = {
        "status": status,
        "result": outcome.result,
        "observation": documents.body(terminal.observation)
        if terminal.HasField("observation")
        else None,
        "outputs": saved,
        "invocation": {
            "digest": outcome.invocation_digest,
            "document": documents.document(outcome.invocation),
        },
        "accepted": documents.body(outcome.accepted),
        "ledger": outcome.ledger,
        "timings": outcome.timings,
        "warnings": warnings,
    }
    # THE PUBLICATION, rendered. The runtime mints a typed receipt per publication and
    # journals it before the terminal names it, and this verb dropped all three of the
    # facts that carry it — a producer could not read its own publication from the CLI that
    # made it (job-001). The scope and the durable checkpoints ride with it: they are what
    # a producer promotes out of, and what it resumes from.
    if outcome.publication:
        document["publication"] = outcome.publication
        rows.append(
            (
                "publication",
                f"{len(outcome.publication.get('outputs') or [])} output(s), "
                f"{outcome.publication.get('total_bytes', 0)} B under "
                f"{outcome.publication.get('grant_id') or 'no scope'}",
            )
        )
    if outcome.publication_scope:
        document["publication_scope"] = outcome.publication_scope
        rows.append(("publication_scope", outcome.publication_scope))
    if outcome.checkpoints:
        document["checkpoints"] = msgspec.to_builtins(outcome.checkpoints)
        rows.append(("checkpoints", f"{len(outcome.checkpoints)} durable receipt(s)"))
    if code != Exit.ok:
        raise Refusal(
            _name_for(code),
            terminal.safe_message or f"the attempt ended {status}",
            terminal.cause.detail or "the terminal's own remedy is rendered verbatim",
            code,
        )
    result = Result(
        rows=tuple(rows),
        document=document if opts.json else None,
        next=(f"{verb} again with --json for the full typed result",),
    )
    for warning in warnings:
        print(f"warning: {warning}")
    if stream:
        print(json.dumps({"event": "terminal", "status": status}))
    return result


def _split(value: Any) -> list[str]:
    """The boot record carries scalars, so the confession travels comma-joined."""
    if not value:
        return []
    return [part for part in str(value).split(", ") if part]


def _bind_facts(outcome: SliceOutcome) -> dict[str, Any]:
    """The `load` boot step's own record — where the bind's confession is recorded."""
    posture = (outcome.triage or {}).get("posture") or {}
    for step in posture.get("boot_steps", ()):
        if isinstance(step, dict) and step.get("step") == "load":
            return dict(step)
    return {}


def _name_for(code: Exit) -> str:
    return code.name


def _exit_for(status: str, cause: pb.OutcomeCause) -> Exit:
    if status == "OUTCOME_STATUS_REFUSED":
        return CAUSE_EXITS.get(cause.code, Exit.validation)
    return TERMINAL_EXITS.get(status, Exit.failed)


def _save(outcome: SliceOutcome, out_dir: Path) -> dict[str, str]:
    """Move the granted outputs into `--out`, named by their package-interface FIELD PATH.

    The EXTENSION comes from the asset's own declared media type, not from the granted
    destination's filename: a RecordOwner grants an opaque location and the worker writes
    bytes to it, so the grant path knows nothing about what the encoder produced. The
    author's returned asset does.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, path in sorted(outcome.outputs.items()):
        source = Path(path)
        label = name.replace(".", "_")
        if not label or Path(label).name != label:
            raise CliError(
                "output_path",
                "output ID is not a local filename",
                "use named result fields",
                Exit.failed,
            )
        target = out_dir / f"{label}{_extension(outcome, name)}"
        if source.is_dir():
            from cozy_runtime.internal.worker.byte_inputs import remove_checkout

            remove_checkout(target)
            shutil.copytree(source, target)
        elif source.is_file():
            shutil.copyfile(source, target)
        else:
            continue
        saved[name] = str(target)
    return saved


def _extension(outcome: SliceOutcome, field: str) -> str:
    """`.png` from `image/png`. An unknown or absent media type stays extensionless."""
    node: object = outcome.result
    for part in field.split("."):
        if not isinstance(node, dict):
            return ""
        node = node.get(part)
    media = node.get("media_type", "") if isinstance(node, dict) else ""
    return mimetypes.guess_extension(str(media)) or "" if media else ""
