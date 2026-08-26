"""`cozy-runtime fit` — this host's verdicts, from the SAME arithmetic `run` uses.

One row per function and lane: `fits(<delivery>, <placement>)` · `fits-degraded(<plan>)` ·
`structurally-incompatible` · `capacity(<shortfall>)`. WITH a payload the verdict is EXACT —
the real `PlanChooser.choose` walk for that request's shape against a ledger holding this
host's MEASURED device facts. Without one it prints structural/range information, labeled.

The point of this verb is that it re-derives nothing: `choose` is the one function that
prices a rung, so `fit` and `run` cannot disagree. What `fit` does differently is refuse to
execute — it takes no I/O, loads no weights, and never touches the device.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli import artifacts
from cozy_runtime.cli import payload as payload_grammar
from cozy_runtime.cli import run as run_verb
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import descriptor as desc
from cozy_runtime.internal import hostfacts
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    PlanChooser,
    PlanRefusal,
    PreparedModel,
    PreparedRequest,
)

MIB = 1 << 20


def run(args: list[str], opts: Options, config: RuntimeConfig) -> Result:
    terms, flags = run_verb.split_flags(args)
    project = Path(opts.dir).expanduser().resolve()
    try:
        found = discover(project)
        body = desc.build(found)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "an endpoint's surface must be valid to be fitted",
            Exit.validation,
        ) from exc

    facts = hostfacts.measure()
    wanted = terms[0] if terms else ""
    surfaces = [
        s for s in found.surfaces if s.kind == "entrypoint" and (not wanted or s.name == wanted)
    ]
    if wanted and not surfaces:
        raise CliError(
            "not_found",
            f"no registered entrypoint named {wanted!r}",
            f"this release registers: "
            f"{', '.join(sorted(s.name for s in found.surfaces)) or '0 functions'}",
            Exit.not_found,
        )

    rows: list[tuple[str, str]] = [
        ("device", facts.gpu_name or facts.state("gpu_name")),
        (
            "allocatable",
            f"{_free_bytes(facts) / MIB:.0f} MiB measured"
            if _free_bytes(facts) >= 0
            else "unreadable",
        ),
    ]
    verdicts: list[dict[str, Any]] = []
    for surface in surfaces:
        doc = run_verb._doc_for(body, surface.name, "entrypoint")
        exact = bool(terms[1:]) or bool(flags.get("--in"))
        document = (
            payload_grammar.parse(
                terms[1:], doc["request"], infile=(flags.get("--in") or [""])[0], project=project
            )
            if exact
            else {}
        )
        verdict = _verdict(config, found, surface, document, facts, exact=exact)
        verdicts.append(verdict)
        rows.append((surface.name, verdict["rendered"]))
    return Result(
        rows=tuple(rows),
        # `HostFacts` is a SLOTTED dataclass and has no `__dict__`, so reading one made
        # every `--json` invocation exit 1 `internal` (found by cl-010 driving the
        # product CLI, which fell back to the row rendering to work around it).
        # `dataclasses.asdict` is the reader that works for both shapes.
        document={"host": dataclasses.asdict(facts), "verdicts": verdicts} if opts.json else None,
        lines=(
            ()
            if any(v["exact"] for v in verdicts)
            else ("note: no payload given — structural/range information only, not an exact fit",)
        ),
        next=("cozy-runtime fit <function> <primary> key=value — the EXACT verdict",),
    )


def _free_bytes(facts: hostfacts.HostFacts) -> int:
    """Free device bytes right now, read without a CUDA context."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return int(out.splitlines()[0]) * MIB
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return -1


def _verdict(
    config: RuntimeConfig,
    found: Any,
    surface: Any,
    document: dict[str, Any],
    facts: hostfacts.HostFacts,
    *,
    exact: bool,
) -> dict[str, Any]:
    slots = surface.model_bindings
    if not slots:
        return {
            "function": surface.name,
            "exact": exact,
            "rendered": "fits(native, all_resident) — weightless, no model binding",
            "verdict": "fits",
        }
    try:
        record, _ = run_verb._binding_record(
            config, found, surface, surface.name, {}, offline=False
        )
    except CliError as exc:
        return {
            "function": surface.name,
            "exact": False,
            "rendered": f"unbound — {exc.message}",
            "verdict": "unbound",
        }
    installed = artifacts.find(config.cozy_home, str(record["release"]))
    assert installed is not None
    free = _free_bytes(facts)
    ledger = Ledger(host_budget=record["host_bytes"] * 4, pinned_budget=record["pinned_bytes"])
    ledger.device_free = free
    ledger.device_total = facts.vram_total_bytes
    share = installed.bytes // max(len(installed.components), 1)
    ledger.resident = dict.fromkeys(installed.components, share)
    declared = DeclaredBinding.read(dict(record))
    chooser = PlanChooser(ledger)
    # `fit` prices a HYPOTHETICAL: there is no endpoint process to prepare the request, so
    # the shape axes are normalized from the document by the one normalizer and the record
    # says out loud that nothing resolved it (its digests are empty).
    request = PreparedRequest.unresolved(surface.name, _axes(surface, document))
    try:
        # NOTHING PREPARED THIS: `fit` estimates before any executor exists, so the
        # resolved half is empty and the estimate says so through the plan document.
        chosen = chooser.choose(declared, PreparedModel(), request)
    except PlanRefusal as exc:
        return {
            "function": surface.name,
            "exact": exact,
            "verdict": "capacity",
            "rendered": f"capacity({exc.detail})",
            "shortfall": exc.shortfall,
        }
    why = chosen.decision_explanation
    degraded = "DEGRADED" in why or "short by" in why
    label = "fits-degraded" if degraded else "fits"
    return {
        "function": surface.name,
        "exact": exact,
        "verdict": label,
        "rendered": f"{label}({chosen.delivery}, {chosen.placement}) · {chosen.compute_dtype} · "
        f"{chosen.decision_explanation[:160]}",
        "plan": chosen.document(),
    }


def _axes(surface: Any, document: Mapping[str, Any]) -> Mapping[str, int]:
    """The normalized shape axes of a document, through `author._demand` and nothing else.

    A document `fit` was handed may be partial — it is an estimate over a request nobody has
    sent — so a decode that does not converge yields NO axes rather than invented ones.
    """
    import msgspec

    from cozy_runtime.author._demand import normalize

    try:
        return normalize(msgspec.convert(dict(document), type=surface.payload_type)).values
    except (msgspec.ValidationError, TypeError):
        return {}
