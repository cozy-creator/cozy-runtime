"""`cozy-runtime run` and `cozy-runtime job` — the git-clone contract's load-bearing verbs.

One-shot UX over an EPHEMERAL LOCAL WORKER and the same attempt kernel `serve` uses:
`internal.local.run_slice` IS the body, and this module is the LOCAL COORDINATOR adapter
around it (cozy-runtime-cli.md §Local authority) — it resolves the ref, mints the binding
plan record, derives the granted output field paths from the descriptor, and maps the
terminal to an exit code. There is no second execution implementation and no in-process
direct-call runner: the observational-identity arm proves a CLI run and a direct
`run_slice` call produce the same result bytes, the same manifest and the same lifecycle.

What this module DOES NOT do is decide anything the runtime decides. The plan, the ladder,
the confession and the shortfall all arrive on the outcome; this renders them.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli import payload as payload_grammar
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import artifacts
from cozy_runtime.internal import bindings as binding_table
from cozy_runtime.internal import descriptor as desc
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.local import LocalRequest, SliceOutcome, run_slice
from cozy_runtime.internal.worker.session import WorkerOptions

VALUE_FLAGS = (
    "--model",
    "--lane",
    "--adapter",
    "--seed",
    "--out",
    "--timeout",
    "--in",
    "--input",
    "--org",
)
BOOL_FLAGS = ("--offline", "--stream", "--strict-keys")

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
CAUSE_EXITS: Mapping[str, Exit] = {
    "CAUSE_CODE_INVALID_REQUEST": Exit.validation,
    "CAUSE_CODE_UNSUPPORTED_INPUT": Exit.structural,
    "CAUSE_CODE_CONSTRAINT_INFEASIBLE": Exit.capacity,
    "CAUSE_CODE_ARTIFACT_UNFETCHABLE": Exit.offline_miss,
    "CAUSE_CODE_CAPABILITY_UNAVAILABLE": Exit.structural,
    "CAUSE_CODE_DEADLINE_EXPIRED": Exit.deadline,
    "CAUSE_CODE_LOCAL_SAFETY": Exit.validation,
    "CAUSE_CODE_CLIENT_CANCEL": Exit.canceled,
}


class Refusal(CliError):
    """A typed refusal that already carries its exit code from the runtime's own verdict."""


def split_flags(args: Sequence[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Separate payload terms from the verb's flags. `--model` is REPEATABLE."""
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
        known = {s.path for s in slots} | {s.class_key for s in slots}
        if path not in known:
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
        body = desc.build(found)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "an endpoint's surface must be valid before it can run (§1.0)",
            Exit.validation,
        ) from exc

    kind = "job" if verb == "job" else "entrypoint"
    surface = next((s for s in found.surfaces if s.name == function), None)
    if surface is None or surface.kind != kind:
        known = sorted(s.name for s in found.surfaces if s.kind == kind)
        raise CliError(
            "not_found",
            f"no registered {kind} named {function!r}",
            f"this release registers: {', '.join(known) or f'0 {kind}s'}",
            Exit.not_found,
            next=("cozy-runtime describe — the full list",),
        )

    doc = _doc_for(body, function, kind)
    request_schema = doc["request"]
    document = payload_grammar.parse(
        terms, request_schema, infile=(flags.get("--in") or [""])[0], project=project
    )
    seed = (flags.get("--seed") or [""])[0]
    if seed and "seed" in payload_grammar.fields(request_schema):
        document["seed"] = int(seed)
    outputs = payload_grammar.asset_paths(doc["result"])

    slots = surface.model_bindings
    record: dict[str, Any] | None = None
    warnings: list[str] = []
    if slots and kind == "job":
        raise CliError(
            "structural",
            f"{function} declares {len(slots)} read-only Model param(s); this build's local "
            "job lane runs the torch-free job attempt only",
            "a job's read-only Model params construct for THAT attempt through the ordinary "
            "Loader path (cr-002b) and are never retained — wiring that construction into the "
            "job lane is the next increment; today, run model-bound work as an entrypoint",
            Exit.structural,
        )
    if slots:
        record, warn = _binding_record(
            config, found, surface, function, flags, offline="--offline" in flags
        )
        warnings.extend(warn)

    job_records: list[dict[str, Any]] = []
    trees: dict[str, str] = {}
    descriptor_id = ""
    if kind == "job":
        descriptor_id = desc.job_descriptor_id(body, function)
        trees = _input_trees(flags.get("--input") or [])
        job_records.append(
            {
                "job_descriptor_id": descriptor_id,
                "build_id": release_id_of(body),
                "project": str(found.project),
                "job": function,
                "gpu_count": int((surface.resources.gpu_count or 0) if surface.resources else 0),
                "publishes": bool(surface.publishes),
                "emits_media": bool(surface.emits_media),
            }
        )

    out_dir = Path((flags.get("--out") or ["outputs"])[0]).expanduser().resolve()
    workspace = out_dir / ".run"
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    request = LocalRequest(
        entrypoint=function,
        payload=document,
        outputs=outputs,
        request_id=f"cli-{function}",
        timeout_ms=_timeout((flags.get("--timeout") or [""])[0]),
        kind=kind if kind == "job" else "entrypoint",
        job_descriptor_id=descriptor_id,
        trees=trees,
        org=(flags.get("--org") or ["local"])[0],
    )
    release_id = str(body["application"])
    # The worker narrates itself to stdout, which is the right place for a long-lived
    # `serve` process and the wrong place for a verb whose stdout IS a result document. The
    # narration is not suppressed — it is a real progress channel — it just goes where
    # progress goes, so `--json` emits exactly the document and nothing else.
    with contextlib.redirect_stdout(sys.stderr):
        outcome = run_slice(
            config,
            request,
            [record] if record else [],
            job_records=job_records,
            workspace=workspace,
            release_id=release_id,
            options=WorkerOptions(root=workspace / "worker", release_id=release_id, no_warm=True),
        )
    return _render(outcome, out_dir, warnings, opts, verb=verb, stream="--stream" in flags)


def release_id_of(body: Mapping[str, Json]) -> str:
    """The build this job runs from. Locally that is the configured application export."""
    return str(body["application"])


def _input_trees(values: Sequence[str]) -> dict[str, str]:
    """`--input <ref>=<dir>` — one MATERIALIZED input tree grant per typed Tree field.

    Locally the coordinator grants a directory it can already read; on the network path
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
                "--input org/repo@release=/path/to/tree",
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
    raise AssertionError(f"{function} was discovered but is not in the descriptor")


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


def resolved_bindings(config: RuntimeConfig, opts: Options) -> list[dict[str, Any]]:
    """Every declared slot's RESOLVED SELECTION, from the one resolver (cl-010's debt 7).

    It prints what `endpoint.toml` SELECTS — the ref, the lane, and which rule won — plus
    whatever the local CAS has for it. Deliberately NOT the full binding-plan record: an
    uninstalled artifact still has a selection, and a verb that refused `offline_miss` here
    would be useless for the question it exists to answer ("what does this project bind?").
    That question is the whole of the second reader cozy-creator grew, and answering it is
    what deletes `internal/launch/bindings.go` (the deletion is cl-003's).

    It resolves and reports; it constructs nothing, touches no device and loads no weights.
    """
    project = Path(opts.dir).expanduser().resolve()
    try:
        found = discover(project)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "an endpoint's surface must be valid before its bindings can resolve (§1.0)",
            Exit.validation,
        ) from exc
    slots = [
        slot
        for surface in found.surfaces
        if surface.kind == "entrypoint"
        for slot in surface.model_bindings
    ]
    if not slots:
        return []
    try:
        # ONE resolve over the WHOLE declared slot set, which is what a serving deployment
        # does: the selection is checked against every slot the release declares, so a
        # binding for a slot that does not exist refuses here rather than being silently
        # unused by whichever entrypoint happened to be asked for.
        resolved = binding_table.resolve(slots, found.bindings)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "code states capability, bindings state selection — name the slot in endpoint.toml",
            Exit.validation,
        ) from exc
    records: list[dict[str, Any]] = []
    for slot in sorted(slots, key=lambda s: s.path):
        chosen = resolved[slot.path]
        installed = artifacts.find(config.cozy_home, chosen.ref)
        records.append(
            {
                "model_binding_path": slot.path,
                "model_parameter_name": slot.param,
                "model_class": slot.class_key,
                "ref": chosen.ref,
                "lane": chosen.lane or "",
                "source": chosen.source,
                "components": list(installed.components) if installed else [],
                "store": str(config.cozy_home) if installed else "",
                "snapshots": dict(installed.snapshots) if installed else {},
                "custody": installed.custody if installed else "",
                "stamps": (
                    {k: list(v) for k, v in sorted(installed.stamps.items())} if installed else {}
                ),
                "installed": installed is not None,
            }
        )
    return records


def _binding_record(
    config: RuntimeConfig,
    found: Any,
    surface: Any,
    function: str,
    flags: Mapping[str, list[str]],
    *,
    offline: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve this call's ONE model slot into the binding-plan record the worker reads."""
    slots = surface.model_bindings
    if len(slots) != 1:
        raise CliError(
            "structural",
            f"{function} declares {len(slots)} model binding paths; this build's local "
            "coordinator mints ONE binding-plan record per attempt",
            "multi-slot local runs land with the multi-binding attempt record (cr-008b's "
            "construction-identity coalescing); the network coordinator already ships many",
            Exit.structural,
        )
    slot = slots[0]
    try:
        resolved = binding_table.resolve(
            slots,
            found.bindings,
            overrides=_overrides(flags.get("--model") or [], slots),
        )
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "code states capability, bindings state selection — name the slot in "
            "endpoint.toml or pass --model <binding-path>=<ref>",
            Exit.validation,
        ) from exc
    chosen = resolved[slot.path]
    installed = artifacts.find(config.cozy_home, chosen.ref)
    if installed is None:
        if offline:
            raise CliError(
                "offline_miss",
                f"{chosen.ref} is not in the local CAS and --offline forbids fetching it",
                f"run `cozy-runtime pull {chosen.ref}` once with a network, then rerun offline",
                Exit.offline_miss,
                next=(f"cozy-runtime pull {chosen.ref}",),
            )
        raise CliError(
            "not_found",
            f"{chosen.ref} is not installed locally",
            f"run `cozy-runtime pull {chosen.ref}` to fetch it into the local CAS",
            Exit.not_found,
            next=(f"cozy-runtime pull {chosen.ref}", "cozy-runtime list"),
        )

    warnings: list[str] = []
    strict = "--strict-keys" in flags
    if installed.custody == "local" and not strict:
        warnings.append(
            f"{chosen.ref} is a LOCAL bind: extra keys the code does not consume are ignored "
            "and confessed on the attempt record — pass --strict-keys to refuse instead"
        )

    # THE RECORD NAMES THE REF, NOT WHAT THIS HOST RESOLVED IT TO (#567e). The index lookup
    # above stays because it is what makes a miss refuse HERE, before a worker is spawned,
    # with the pull that fixes it named — but its answer is not copied into the document.
    # The worker resolves the ref against its own index, which on a local run is this same
    # index reached by the same reader, so the local answer is unchanged by construction.
    return {
        "entrypoint_binding_plan_id": _plan_id(found.application, function),
        "project": str(found.project),
        "model_class": slot.class_key,
        "model_binding_path": slot.path,
        "model_parameter_name": slot.param,
        "artifact_ref": chosen.ref,
        # The CLASS's declared component names — a code fact from the release both machines
        # hold, never the artifact's component list. `resolve` intersects them.
        "components": sorted({c for names in slot.components.values() for c in names}),
        "release": chosen.ref,
        "host_bytes": 2 * GIB,
        "pinned_bytes": 256 * MIB,
        "entrypoint": function,
        "model_construction_digest": "",
        "resident_budget_bytes": 0,
        "tenancy": "local",
        "strict_keys": strict,
    }, warnings


def _plan_id(application: str, entrypoint: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(f"ebp/{application}/{entrypoint}".encode()).hexdigest()


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
    cause = (terminal.get("cause") or {}) if isinstance(terminal, dict) else {}
    code = _exit_for(status, cause)

    saved = _save(outcome, out_dir)
    rows: list[tuple[str, str]] = []
    for name, path in sorted(saved.items()):
        stat = Path(path).stat()
        rows.append((name, f"{path}  {stat.st_size} B"))
    if not saved:
        rows.append(("outputs", "0 assets"))
    accepted = outcome.accepted
    if accepted:
        rows.append(
            (
                "plan",
                f"{accepted.get('delivery', '?')}/{accepted.get('placement', '?')} "
                f"({accepted.get('compute_dtype', '?')})",
            )
        )
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
    metrics = terminal.get("metrics", {}) if isinstance(terminal, dict) else {}
    rows.append(("runtime", f"{outcome.timings.get('wall_ms', 0) / 1000:.2f} s"))
    if peak := int(metrics.get("peak_vram_bytes", 0)):
        rows.append(("peak_vram", f"{peak / MIB:.0f} MiB"))
    for row in _adjustments(terminal):
        rows.append(("adjusted", row))
    document: dict[str, Any] = {
        "status": status,
        "result": outcome.result,
        "outputs": saved,
        "plan": accepted,
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
        document["checkpoints"] = outcome.checkpoints
        rows.append(("checkpoints", f"{len(outcome.checkpoints)} durable receipt(s)"))
    if code != Exit.ok:
        message = str(terminal.get("safe_message", "")) if isinstance(terminal, dict) else ""
        detail = str(cause.get("detail", "")) if isinstance(cause, dict) else ""
        raise Refusal(
            _name_for(code),
            message or f"the attempt ended {status}",
            detail or "the terminal's own remedy is rendered verbatim",
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
    """The `load` boot step's own record — where the bind's confession is journaled."""
    posture = (outcome.triage or {}).get("posture") or {}
    for step in posture.get("boot_steps", ()):
        if isinstance(step, dict) and step.get("step") == "load":
            return dict(step)
    return {}


def _adjustments(terminal: Mapping[str, Any]) -> list[str]:
    """The adjustments envelope, verbatim — the runtime clamped it, this only prints it."""
    envelope = terminal.get("result") or {}
    rows = envelope.get("adjustments") or () if isinstance(envelope, dict) else ()
    return [
        f"{row.get('field', '?')}: requested {row.get('requested', '?')} -> applied "
        f"{row.get('applied', '?')} ({row.get('reason', '')})"
        for row in rows
        if isinstance(row, dict)
    ]


def _name_for(code: Exit) -> str:
    return code.name


def _exit_for(status: str, cause: Mapping[str, Any]) -> Exit:
    if status == "OUTCOME_STATUS_REFUSED":
        return CAUSE_EXITS.get(_cause_name(cause.get("code")), Exit.validation)
    return TERMINAL_EXITS.get(status, Exit.failed)


def _cause_name(code: Any) -> str:
    """The cause's spelling. A canonical document carries the ENUM'S NUMBER, not its name —
    which is right for the wire and was silently wrong here: keying the exit table on names
    matched nothing, so every refusal came out as exit 3 including the capacity one."""
    from cozy_runtime.protocol import worker_pb2 as pb

    if isinstance(code, str):
        return code
    if isinstance(code, int):
        try:
            return str(pb.CauseCode.Name(code))
        except ValueError:
            return ""
    return ""


def _save(outcome: SliceOutcome, out_dir: Path) -> dict[str, str]:
    """Move the granted outputs into `--out`, named by their descriptor FIELD PATH.

    The EXTENSION comes from the asset's own declared media type, not from the granted
    destination's filename: a coordinator grants an opaque location and the worker writes
    bytes to it, so the grant path knows nothing about what the encoder produced. The
    author's returned asset does.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, path in sorted(outcome.outputs.items()):
        source = Path(path)
        if not source.is_file():
            continue
        target = out_dir / f"{name.replace('.', '_')}{_extension(outcome, name)}"
        shutil.copyfile(source, target)
        saved[name] = str(target)
    return saved


def _extension(outcome: SliceOutcome, field: str) -> str:
    """`.png` from `image/png`. An unknown or absent media type stays extensionless."""
    import mimetypes

    node: Any = outcome.result
    for part in field.split("."):
        if not isinstance(node, dict):
            return ""
        node = node.get(part)
    media = node.get("media_type", "") if isinstance(node, dict) else ""
    return mimetypes.guess_extension(str(media)) or "" if media else ""
