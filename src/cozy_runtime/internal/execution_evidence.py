"""The per-rank execution record: which process on which GPU ran an attempt, when, with what.

ONE shape, shared by the attempt's triage `measurements.execution.ranks`, the attention
observation and the GPU scheduler's enter/exit events. Fields are added, never changed:

    {"rank": int, "pid": int, "ordinal": int, "uuid": str, "arch": str,
     "start_us": int, "end_us": int,
     "attention": {"requested": str, "observed": str, "impl": str, "kernels"?: [kernel]}}
    kernel = {"kernel": str, "state": str, "line": str, "served": bool,
              "progress"?: float, "compile_ms"?: float, "detail"?: str}

`ordinal` is the host CUDA ordinal (-1 without one), `uuid` the NVML `GPU-…` identity and
`arch` its `sm_XY` ("" off CUDA). `start_us`/`end_us` are wall-clock unix MICROseconds bounding
the rank's work for the attempt, 0 for a rank that did none: nanoseconds since the epoch exceed
the ±2^53 integer range canonical JSON (and every JSON reader) admits. `attention.requested` is
the request's pin ("" = automatic), `observed` what the rank's processors held, in the pin
grammar, and `impl` the kernel implementation under that backend when one was observed.
`kernels` is every compiled kernel this rank knows of: `state` ready, compiling (with its
`progress`), failed or absent, whether it `served` this attempt, and how long its compile took.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict


class Boot(TypedDict):
    """`Applied.document()`, the boot record as it crosses the executor seam (cr-124). A
    group's `merge` keys `artifacts` by model parameter, then kernel."""

    kernels: dict[str, int]
    hosts: dict[str, dict[str, int]]
    offered: list[str]
    pin: str
    artifacts: NotRequired[dict[str, object]]
    skipped: NotRequired[dict[str, str]]
    line: str


class RankAttention(TypedDict):
    """A rank's `attention` (the shape above)."""

    requested: str
    observed: str
    impl: str
    kernels: NotRequired[list[dict[str, object]]]


class RankEvidence(TypedDict):
    """One rank's record (the shape above). Mappings, not structs: the executor extends them
    by key and they land in the attempt's measurements as JSON."""

    rank: int
    pid: int
    ordinal: int
    uuid: str
    arch: str
    start_us: int
    end_us: int
    attention: RankAttention


def identity(torch: Any, cuda: bool, local: int, sealed: Sequence[str]) -> dict[str, Any]:
    """This process's `pid`, host `ordinal`, `uuid` and `arch` for device `local` of its seal."""
    if not cuda:
        return {"pid": os.getpid(), "ordinal": -1, "uuid": "", "arch": ""}
    entry = sealed[local] if local < len(sealed) else str(local)
    try:
        properties = torch.cuda.get_device_properties(local)
        uuid, arch = f"GPU-{properties.uuid}", f"sm_{properties.major}{properties.minor}"
    except Exception:  # identity is observation: it never fails the attempt
        uuid, arch = "", ""
    ordinal = int(entry) if entry.isdigit() else -1
    return {"pid": os.getpid(), "ordinal": ordinal, "uuid": uuid, "arch": arch}


def attention(
    requested: str = "",
    observed: str = "",
    impl: str = "",
    kernels: Sequence[Mapping[str, Any]] = (),
) -> RankAttention:
    row = RankAttention(requested=requested, observed=observed, impl=impl)
    if kernels:
        row["kernels"] = [dict(kernel) for kernel in kernels]
    return row


def observed(hosts: Mapping[str, Mapping[str, int]]) -> str:
    """Site backends in the pin grammar: one name when every site agrees, else per component."""
    by_component: dict[str, set[str]] = {}
    for component, kernels in hosts.items():
        names = by_component.setdefault(component.rsplit("/", 1)[-1], set())
        names.update(name for name, sites in kernels.items() if sites)
    every: set[str] = set()
    for names in by_component.values():
        every |= names
    if len(every) <= 1:
        return next(iter(every), "")
    return ",".join(
        f"{component}={'+'.join(sorted(names))}"
        for component, names in sorted(by_component.items())
        if names
    )


def now_us() -> int:
    return time.time_ns() // 1000


def record(
    rank: int,
    who: Mapping[str, Any],
    start_us: int,
    end_us: int,
    seen: RankAttention | None = None,
) -> RankEvidence:
    return {
        "rank": rank,
        "pid": int(who.get("pid", 0)),
        "ordinal": int(who.get("ordinal", -1)),
        "uuid": str(who.get("uuid", "")),
        "arch": str(who.get("arch", "")),
        "start_us": start_us,
        "end_us": end_us,
        "attention": (seen or attention()).copy(),
    }


def widen(held: dict[int, RankEvidence], row: RankEvidence) -> None:
    """Fold one mirrored call's record into its rank's: earliest start, latest end."""
    current = held.get(row["rank"])
    if current is None:
        held[row["rank"]] = row.copy()
        return
    current["start_us"] = min(current["start_us"], row["start_us"])
    current["end_us"] = max(current["end_us"], row["end_us"])
