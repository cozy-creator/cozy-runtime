"""The per-GPU execution record: which process on which GPU ran an attempt, when, with what.

The executor reports one record per process of its group, in process order, as `gpus`:

    {"gpu": int, "pid": int, "uuid": str, "arch": str, "start_us": int, "end_us": int,
     "attention": {"requested": str, "observed": str, "impl": str, "kernels"?: [kernel],
                   "segments"?: [segment]}}
    kernel = {"kernel": str, "state": str, "line": str, "served": bool,
              "progress"?: float, "compile_ms"?: float, "detail"?: str}
    segment = {"stage": str, "step": int, "at_us": int, "observed": str, "dense": str}

`gpu` is the number nvidia-smi shows (NVML's index for the card's UUID, else the seal's
`CUDA_VISIBLE_DEVICES` entry; -1 off CUDA), `uuid` the NVML `GPU-…` identity and `arch` its
`sm_XY` ("" off CUDA). `start_us`/`end_us` are wall-clock unix MICROseconds bounding the
GPU's work for the attempt, 0 for one that did none: nanoseconds since the epoch exceed the
±2^53 integer range canonical JSON (and every JSON reader) admits. `attention.requested` is
the request's pin ("" = automatic), `observed` what the process's processors held, in the
pin grammar, and `impl` the kernel implementation under that backend when one was observed.
`kernels` is every compiled kernel this GPU knows of: `state` ready, compiling (with its
`progress`), failed or absent, whether it `served` this attempt, and how long its compile
took. `segments` is present when the run took a kernel that finished compiling mid-run: each
selection from the step of its stage it first served (`stage` "" is the run's start), in the
pin grammar, with Sol's dense reference (`dense`, "" without Sol). `observed` and `kernels`
then cover every selection the run served.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict

from cozy_runtime.internal import accel


class RuntimeGPU(TypedDict):
    device_index: int
    device_name: str
    device_uuid: str
    driver_version: str
    memory_bytes: int
    pci_bus_id: str


class AttentionFallback(TypedDict):
    component: str
    preferred: str
    selected: str
    reason: str


class Boot(TypedDict):
    """`Applied.document()`, the boot record as it crosses the executor seam (cr-124). A
    group's `merge` keys `artifacts` by model parameter, then kernel."""

    kernels: dict[str, int]
    hosts: dict[str, dict[str, int]]
    offered: list[str]
    pin: str
    artifacts: NotRequired[dict[str, object]]
    skipped: NotRequired[dict[str, str]]
    fallbacks: NotRequired[list[AttentionFallback]]
    line: str


class RankAttention(TypedDict):
    """A process's `attention` (the shape above)."""

    requested: str
    observed: str
    impl: str
    kernels: NotRequired[list[dict[str, object]]]
    segments: NotRequired[list[dict[str, object]]]
    fallbacks: NotRequired[list[AttentionFallback]]


class GpuEvidence(TypedDict):
    """One process's record (the shape above). Mappings, not structs: the executor extends
    them by key and they land in the attempt's measurements as JSON."""

    gpu: int
    pid: int
    uuid: str
    arch: str
    start_us: int
    end_us: int
    attention: RankAttention


def identity(torch: Any, cuda: bool, local: int, sealed: Sequence[str]) -> dict[str, Any]:
    """This process's `pid`, `gpu`, `uuid` and `arch` for device `local` of its seal."""
    if not cuda:
        return {"pid": os.getpid(), "gpu": -1, "uuid": "", "arch": ""}
    entry = sealed[local] if local < len(sealed) else str(local)
    try:
        properties = torch.cuda.get_device_properties(local)
        uuid, arch = f"GPU-{properties.uuid}", f"sm_{properties.major}{properties.minor}"
    except Exception:  # identity is observation: it never fails the attempt
        uuid, arch = "", ""
    gpu = accel.nvml_index(uuid)
    if gpu < 0:
        gpu = int(entry) if entry.isdigit() else -1
    return {"pid": os.getpid(), "gpu": gpu, "uuid": uuid, "arch": arch}


def attention(
    requested: str = "",
    observed: str = "",
    impl: str = "",
    kernels: Sequence[Mapping[str, Any]] = (),
    segments: Sequence[Mapping[str, Any]] = (),
    fallbacks: Sequence[AttentionFallback] = (),
) -> RankAttention:
    row = RankAttention(requested=requested, observed=observed, impl=impl)
    if kernels:
        row["kernels"] = [dict(kernel) for kernel in kernels]
    if segments:
        row["segments"] = [dict(segment) for segment in segments]
    if fallbacks:
        row["fallbacks"] = list(fallbacks)
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
    who: Mapping[str, Any],
    start_us: int,
    end_us: int,
    seen: RankAttention | None = None,
) -> GpuEvidence:
    return {
        "gpu": int(who.get("gpu", -1)),
        "pid": int(who.get("pid", 0)),
        "uuid": str(who.get("uuid", "")),
        "arch": str(who.get("arch", "")),
        "start_us": start_us,
        "end_us": end_us,
        "attention": (seen or attention()).copy(),
    }


def widen(held: dict[int, GpuEvidence], process: int, row: GpuEvidence) -> None:
    """Fold one mirrored call's record into group process `process`'s: earliest start, latest
    end, and the attention it served through its latest call."""
    current = held.get(process)
    if current is None:
        held[process] = row.copy()
        return
    current["start_us"] = min(current["start_us"], row["start_us"])
    current["end_us"] = max(current["end_us"], row["end_us"])
    current["attention"] = row["attention"]


def gpu_name(devices: Sequence[str], rank: int) -> str:
    """How a person names the GPU group process `rank` drives: `GPU 3`, nvidia-smi's number,
    from the seal's `CUDA_VISIBLE_DEVICES` entries (process r drives entry r). A `GPU-…`
    UUID entry is its own name."""
    entry = devices[rank] if 0 <= rank < len(devices) else str(rank)
    return f"GPU {entry}" if entry.isdigit() else entry


def gpu_names(devices: Sequence[str]) -> str:
    """Several seal entries as a person reads them: `GPU 2`, `GPUs 0-3`, `GPUs 1, 3`."""
    names = [gpu_name(devices, rank) for rank in range(len(devices))]
    if len(names) == 1:
        return names[0]
    numbers = [int(entry) for entry in devices if entry.isdigit()]
    if len(numbers) == len(devices) > 2 and numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"GPUs {numbers[0]}-{numbers[-1]}"
    return "GPUs " + ", ".join(name.removeprefix("GPU ") for name in names)


class GpuIdentity(TypedDict):
    gpu: int
    uuid: str


def identify(entry: str, inventory: Sequence[RuntimeGPU] = ()) -> GpuIdentity:
    """One envelope entry (an index or a `GPU-…` UUID) as `{"gpu": n, "uuid": u}`: the number
    nvidia-smi shows (-1 when unknown) and the NVML identity ("" when unknown)."""
    for row in inventory:
        if entry in (str(row["device_index"]), row["device_uuid"]):
            return {"gpu": row["device_index"], "uuid": row["device_uuid"]}
    if entry.isdigit():
        return {"gpu": int(entry), "uuid": ""}
    return {"gpu": -1, "uuid": entry}
