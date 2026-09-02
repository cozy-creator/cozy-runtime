"""`cozy-runtime hostfacts` — local host facts and weightless structural verdicts.

Model-bound capacity requires the exact selected plan closure. This local verb does not
mint or approximate that closure, so those rows remain explicitly unbound. Whether code
FITS a checkpoint is `tensorfs.fit`'s answer (model-code-fit §3), asked by the worker at
construction and by `cozy-runtime derive` + `tfs fit` by hand; nothing here fits anything.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli import payload as payload_grammar
from cozy_runtime.cli import run as run_verb
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import descriptor as desc
from cozy_runtime.internal import hostfacts
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit

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
            "a package's surface must be valid to be fitted",
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
        if exact:
            payload_grammar.parse(terms[1:], doc["request"], infile=(flags.get("--in") or [""])[0])
        verdict = _verdict(surface, exact=exact)
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
        next=("cozy-runtime hostfacts <function> <primary> key=value — the EXACT verdict",),
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
    surface: Any,
    *,
    exact: bool,
) -> dict[str, Any]:
    slots = surface.model_bindings
    if not slots:
        return {
            "function": surface.name,
            "exact": exact,
            "rendered": "runnable(native, all_resident) — weightless, no model binding",
            "verdict": "runnable",
        }
    return {
        "function": surface.name,
        "exact": False,
        "rendered": "unbound — capacity requires the selected plan closure",
        "verdict": "unbound",
    }
