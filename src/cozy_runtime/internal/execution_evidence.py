"""The per-rank execution record: which process on which GPU ran an attempt, when, with what.

ONE shape, shared by the attempt's triage `measurements.execution.ranks`, the attention
observation and the GPU scheduler's enter/exit events. Fields are added, never changed:

    {"rank": int, "pid": int, "ordinal": int, "uuid": str, "arch": str,
     "start_us": int, "end_us": int,
     "attention": {"requested": str, "observed": str, "impl": str,
                   "probes"?: [probe], "skipped"?: [probe]}}
    probe = {"kernel": str, "status": str, "source": str, "shape": [int],
             "rel_l2_by_rows": {group: float}, "budget": float, "budget_state": str}

`ordinal` is the host CUDA ordinal (-1 without one), `uuid` the NVML `GPU-…` identity and
`arch` its `sm_XY` ("" off CUDA). `start_us`/`end_us` are wall-clock unix MICROseconds bounding
the rank's work for the attempt, 0 for a rank that did none: nanoseconds since the epoch exceed
the ±2^53 integer range canonical JSON (and every JSON reader) admits. `attention.requested` is
the request's pin ("" = automatic), `observed` what the rank's processors held, in the pin
grammar, and `impl` the kernel implementation under that backend when one was observed.
`probes` is the rank's numeric probe of each kernel it served, `skipped` each kernel its probe
refused (`status` the refusal code); `source` is when it was measured (warm, prepare, request)
and `budget_state` whether `budget` was calibrated on this architecture.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any


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
    probed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {"requested": requested, "observed": observed, "impl": impl, **(probed or {})}


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
    seen: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "pid": int(who.get("pid", 0)),
        "ordinal": int(who.get("ordinal", -1)),
        "uuid": str(who.get("uuid", "")),
        "arch": str(who.get("arch", "")),
        "start_us": start_us,
        "end_us": end_us,
        "attention": dict(seen or attention()),
    }


def widen(held: dict[int, dict[str, Any]], row: Mapping[str, Any]) -> None:
    """Fold one mirrored call's record into its rank's: earliest start, latest end."""
    rank = int(row["rank"])
    current = held.get(rank)
    if current is None:
        held[rank] = dict(row)
        return
    current["start_us"] = min(current["start_us"], int(row["start_us"]))
    current["end_us"] = max(current["end_us"], int(row["end_us"]))
