#!/usr/bin/env python3
"""Kernel formation: every native kernel Runtime builds on a machine compiles for every card it
serves, with the worker image's own toolchain and no GPU. The release gate
(`.github/workflows/kernel-formation.yaml`).

A kernel here is the exact job a machine or executor submits for a card of that capability,
built by the compile manager into a scratch store: the int8/fp8 op and every attention artifact
a candidate admits (`kernel_sources`), the Python kernel tree and Sol's card objects
(`attention_sol`), and the Triton sets an executor builds at start (`triton_kernels`: nvfp4, the
LM decode and the fused DiT layouts below). Upstream's `setup.py` asks torch which card it
builds for; `formation/sitecustomize.py` answers with the job's card. Nothing else stands in for
a device: the compilers produce the real binaries.

`--ledger` records each formed key. A key digests every build input (sources, recipe,
toolchain, target), so a recorded key is recalled, never compiled again. A failure names the
kernel, the card and the compiler's error, and the run exits 1.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from triton import compile as triton_compile
from triton.backends.compiler import GPUTarget

from cozy_runtime import fusion, lm
from cozy_runtime.author.kernels import TritonKernels
from cozy_runtime.internal import (
    arch,
    attention,
    attention_sol,
    config,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    machine_kernels,
    nvfp4_linear,
    triton_kernels,
)

#: Every card the family rules serve, by compute capability.
TARGETS = (80, 86, 89, 90, 100, 103, 120, 121)
#: SM90's Sol object is per length layout (`attention_sol.variant`): one whole block, and H3's
#: 15 s length (a partial last block).
SOL_SM90_TOKENS = (64, 109_104)


def _rows(block: Any, arguments: Mapping[str, Any]) -> fusion.Modulation:
    raise AssertionError("formation never runs a block")


def _dit(
    width: int,
    head_dim: int,
    norms: tuple[str, ...],
    norm: Literal["layer", "rms"],
    rope: fusion.Rope | None,
    gate: Literal["identity", "tanh"] = "identity",
    shift: bool = False,
) -> TritonKernels:
    """A fused DiT layout's kernel set, declared as a package declares it."""
    mlp = fusion.GatedMLP("mlp", gate="gate", down="down")
    block = fusion.DiTBlock(
        object, width, head_dim, _rows, norms, norm, "attn", (), mlp, rope, gate=gate, shift=shift
    )
    return block.kernels


#: Klein's layout (parallel, shifted LayerNorm, fp32 interleaved RoPE) and Qwen Image's
#: (sequential, tanh gates), then every other value of each constexpr: RMSNorm with weights,
#: half-split partial RoPE, no RoPE.
LAYOUTS = (
    _dit(3072, 128, ("norm",), "layer", fusion.Rope("r", True, fp32=True), shift=True),
    _dit(4096, 128, ("n1", "n2"), "layer", fusion.Rope("r", True), gate="tanh"),
    _dit(2048, 64, ("n1", "n2"), "rms", fusion.Rope("r", False, 32), "tanh", shift=True),
    _dit(1536, 64, ("norm",), "rms", None),
)
TRITON = (nvfp4_linear.KERNELS, lm.KERNELS, *LAYOUTS)


def kernels(sm: int) -> Iterator[tuple[str, Callable[[], kernel_compile.Job]]]:
    """What a card of capability `sm` builds, by name: its 8-bit op, the attention artifacts its
    candidates admit, and the Triton sets."""
    row = arch.row(sm)
    yield machine_kernels.INT8, functools.partial(
        kernel_sources.job, machine_kernels.INT8, sm, family=row.family
    )
    artifacts = {
        c.artifact
        for c in attention.BY_NAME.values()
        if c.artifact not in ("", kernel_sources.PYTHON) and not attention._off_row(c, row)
    }
    for artifact in sorted(artifacts):
        yield artifact, functools.partial(kernel_sources.job, artifact, sm)
    for found in TRITON:
        yield found.name, functools.partial(triton_kernels.job, found, sm)


def sol(store: kernel_cache.Store, sm: int) -> list[kernel_compile.Job]:
    """Sol's objects for the card, as the machine starts them, and SM90's length layouts."""
    found = machine_kernels.sol_objects(store, {sm: ""})
    if sm == 90:
        found += [attention_sol.job(tokens, (9, 0)) for tokens in SOL_SM90_TOKENS]
    return found


def _name(job: kernel_compile.Job) -> str:
    blocks = job.spec.get("blocks")
    return (
        f"{job.key.kernel}[blocks={blocks},full={job.spec['full']}]" if blocks else job.key.kernel
    )


def _entries(sets: Mapping[str, TritonKernels], name: str, sm: int) -> str:
    """Which entries of a failed Triton set do not compile, each with its error."""
    rows = []
    for entry in sets[name].entries:
        try:
            triton_compile(
                triton_kernels._source(entry),
                target=GPUTarget("cuda", sm, triton_kernels.WARP_SIZE),
                options={"num_warps": entry.num_warps},
            )
        except Exception as exc:
            rows.append(f"{entry.key}: {type(exc).__name__}: {exc}")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", default=",".join(map(str, TARGETS)))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=None)
    options = parser.parse_args()
    targets = [int(sm) for sm in options.targets.split(",")]
    # Builders inherit this path: upstream's setup.py imports the card's stand-in from it.
    sys.path[0] = str(Path(__file__).resolve().parent / "formation")
    config.hide_gpus()
    store = kernel_cache.Store((options.store or Path(tempfile.mkdtemp())).resolve())
    kernel_cache.configure(store)
    ledger: Path = options.ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    formed: set[str] = set()
    for line in ledger.read_text().splitlines() if ledger.exists() else ():
        try:
            formed.add(json.loads(line)["key"])
        except (ValueError, KeyError, TypeError):
            continue
    sets = {found.name: found for found in TRITON}
    failed: list[str] = []
    counts = {"formed": 0, "recalled": 0}
    pending: list[tuple[str, int, kernel_compile.Job]] = []
    serial: list[tuple[str, int, kernel_compile.Job]] = []
    seen: set[str] = set()

    def fail(name: str, sm: int, detail: str) -> None:
        failed.append(f"{name} sm{sm}")
        print(f"FAILED   sm{sm:<4} {name}\n  " + detail.strip().replace("\n", "\n  "), flush=True)
        note = detail.strip()[:2000].replace("%", "%25").replace("\r", "").replace("\n", "%0A")
        print(f"::error title=kernel formation: {name} sm{sm}::{note}", flush=True)

    def end(name: str, sm: int, job: kernel_compile.Job) -> None:
        state = kernel_compile.wait(store, job)
        if state.state == "ready":
            counts["formed"] += 1
            print(f"formed   sm{sm:<4} {name}  {state.ms / 1000:.0f} s", flush=True)
            if job.key.digest not in formed:
                formed.add(job.key.digest)
                with ledger.open("a") as log:
                    log.write(json.dumps({"key": job.key.digest, "kernel": name, "sm": sm}) + "\n")
            return
        found = kernel_compile._failure(job.key) or {}
        detail = f"{found.get('code', state.state)}: {found.get('detail', state.detail)}"
        fail(name, sm, f"{_entries(sets, name, sm)}\n{detail}" if name in sets else detail)

    def start(name: str, sm: int, job: kernel_compile.Job) -> None:
        if job.key.digest in seen:  # another card's: the same artifact
            return
        seen.add(job.key.digest)
        if job.key.digest in formed:
            counts["recalled"] += 1
            print(f"recalled sm{sm:<4} {name}", flush=True)
        elif job.kind == "setup":
            # Their compiler jobs are admitted by the machine's compile ledger: start them all.
            kernel_compile.submit(store, job)
            pending.append((name, sm, job))
        else:
            # Triton and Sol builders take no admission: one at a time, while those compile.
            serial.append((name, sm, job))

    for found in TRITON:
        if not found.entries:
            fail(found.name, 0, "the set has no entries: Triton does not import")
    # The Python kernel tree first, and always built: Sol's keys hash the sources it holds.
    try:
        end(kernel_sources.PYTHON, 0, kernel_sources.job(kernel_sources.PYTHON, 0))
    except kernel_sources.Absent as exc:
        fail(kernel_sources.PYTHON, 0, f"Absent: {exc}")
    for sm in targets:
        for name, make in kernels(sm):
            try:
                start(name, sm, make())
            except Exception as exc:
                fail(name, sm, f"{type(exc).__name__}: {exc}")
        try:
            objects = sol(store, sm)
        except Exception as exc:
            fail(attention_sol.KERNEL, sm, f"{type(exc).__name__}: {exc}")
            continue
        for job in objects:
            start(_name(job), sm, job)
    for name, sm, job in serial + pending:
        end(name, sm, job)
    print(
        f"{counts['formed']} formed, {counts['recalled']} recalled, {len(failed)} failed"
        f" for sm{','.join(map(str, targets))}" + (f": {', '.join(failed)}" if failed else ""),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
