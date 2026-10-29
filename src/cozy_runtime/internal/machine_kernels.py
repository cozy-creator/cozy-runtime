"""The machine's kernels, compiled at machine start on the cards present while weights download:
first the links between its GPUs (`parallel.links`), which a group reads, then each card row's
attention cells (`boot`) and the LoRA bake set (`lora_bake.KERNELS`), so a new generation's first
LoRA request bakes, and the Runtime's own Triton sets the row runs (`triton_sets`), so no request
compiles them. The Runtime wheel's prebuilt kernels (the 8-bit op, FA3) need no build; packages'
sets build in their import-only parents.

The machine launches `python -m cozy_runtime.internal.machine_kernels` once per installed
generation, under that generation's own seal: its interpreter and torch, every card of the
machine, and the kernel store and scratch scope its executors use. Each key here is therefore
the key an executor computes, by the same functions. For each distinct card architecture it
starts every artifact of `boot` at once, through the compile manager, niced:
the machine's compile ledger admits their compiler jobs within its CPUs and memory
(`kernel_compile.compiler_job`); Sol's card objects start once its sources are in. A request
finds each one ready or building and, without an override, never waits for one: it serves the
best kernel ready (`attention`), and no executor builds a key twice. This process holds no
device memory (the link probe is its child). It prints one JSON line for the links, one per
artifact as it ends and a last one, and exits nonzero when one failed.
"""

from __future__ import annotations

import dataclasses
import json
import site
import sys
import time
from collections.abc import Mapping
from importlib import import_module

from cozy_runtime.author.kernels import TritonKernels
from cozy_runtime.internal import (
    accel,
    arch,
    attention,
    attention_sol,
    config,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    lora_bake,
    nvfp4_linear,
    prebuilt,
    svdq_linear,
    triton_kernels,
)
from cozy_runtime.internal.parallel import links


def cards(devices: str) -> dict[int, str]:
    """Each distinct card architecture among the sealed devices, and the first card of it."""
    entries = [entry.strip() for entry in devices.split(",") if entry.strip()]
    return {sm: entry for entry in reversed(entries) if (sm := accel.device_capability(entry))}


def boot(row: arch.Row) -> list[attention.Candidate]:
    """What a card builds ahead of any request: the first kernel the row runs at each level of
    its cells, dense and sparse, and the dense kernels those compute with. Every installed
    package's declared levels are among them; a later kernel builds when a site reaches it."""
    found: list[attention.Candidate] = []
    for cells in (attention.DENSE, attention.SPARSE):
        for names in cells.get(row.family, {}).values():
            for candidate in (attention.BY_NAME[n] for n in names):
                if not attention._off_row(candidate, row):
                    found += [candidate, *(attention.BY_NAME[n] for n in candidate.dense)]
                    break
    return found


def triton_sets(row: arch.Row) -> list[TritonKernels]:
    """The Runtime's own Triton sets this row runs: its boot kernels' (`Candidate.kernels`, named
    as `attention` opens them: Sage3's fixed means, Sage2's fixed K mean, the E4M3 quantizers) and
    the encoded linears' (nvfp4 on every row, SVDQuant on FP4 rows). None without Triton."""
    if triton_kernels.TRITON_ABSENT:
        return []
    found = [import_module(c.kernels).KERNELS for c in boot(row) if c.kernels]
    found += [nvfp4_linear.KERNELS, *([svdq_linear.KERNELS] if row.fp4 else [])]
    return list({kernels.name: kernels for kernels in found if kernels.entries}.values())


def _job(artifact: str, sm: int, device: str) -> kernel_compile.Job | None:
    try:
        job = kernel_sources.job(artifact, sm)
    except kernel_sources.Absent:
        return None
    # Upstream's setup.py builds for the cards it sees: this one.
    return dataclasses.replace(job, env={"CUDA_VISIBLE_DEVICES": device})


def jobs(present: Mapping[int, str]) -> list[kernel_compile.Job]:
    """Each card architecture's attention (`boot`), once per key, none the wheel serves; its
    LoRA bake set; and its Runtime Triton sets (`triton_sets`)."""
    found: dict[str, kernel_compile.Job] = {}
    for sm, device in sorted(present.items()):
        for candidate in boot(arch.row(sm)):
            if candidate.artifact and not prebuilt.serves(candidate.name, sm):
                job = _job(candidate.artifact, sm, device)
                if job is not None:
                    found.setdefault(job.key.digest, job)
        if lora_bake.KERNELS.entries and not triton_kernels.TRITON_ABSENT:
            bake = triton_kernels.job(lora_bake.KERNELS, sm)
            found.setdefault(bake.key.digest, bake)
        for kernels in triton_sets(arch.row(sm)):
            job = triton_kernels.job(kernels, sm)
            found.setdefault(job.key.digest, job)
    return list(found.values())


def sol_objects(store: kernel_cache.Store, present: Mapping[int, str]) -> list[kernel_compile.Job]:
    """Sol's objects for these cards, once its sources are in the store: their keys hash them."""
    try:
        tree = store.entry(kernel_sources.job(kernel_sources.PYTHON, 0).key)
    except kernel_sources.Absent:
        return []
    if tree is None:
        return []
    site.addsitedir(str(tree / kernel_cache.SITE))
    found: list[kernel_compile.Job] = []
    for sm in sorted(present):
        if attention_sol.route(divmod(sm, 10)) in attention_sol.ROUTES:
            found += attention_sol.card_jobs(store, divmod(sm, 10))
    return found


def main() -> int:
    started = time.perf_counter()
    environment = config.inherited_environment()
    store = kernel_cache.from_sealed(environment)
    devices = config.parse_visible_devices(environment.get("CUDA_VISIBLE_DEVICES")) or ()
    present = cards(",".join(devices))
    if store is not None and len(devices) > 1 and all(map(accel.device_capability, devices)):
        began = time.perf_counter()
        measured = links.measured(links.directory(environment), devices)
        took = round((time.perf_counter() - began) * 1000)
        route = {**measured.nccl(), "copy_engines": measured.copy_engines, "ms": took}
        print(json.dumps({"links": measured.why(), **route}), flush=True)
    # Builders are handed the card they build for; nothing in this process opens one.
    config.hide_gpus()
    ended: list[str] = []

    def done(job: kernel_compile.Job, state: kernel_compile.State) -> None:
        ended.append(state.state)
        print(json.dumps({**state.document(), "key": job.key.digest[:16]}), flush=True)

    def progress(state: kernel_compile.State) -> None:
        print(json.dumps(state.document()), flush=True)

    if store is not None and present:
        kernel_cache.configure(store)
        kernel_compile.waiting = progress
        built = jobs(present)
        for job in built:
            kernel_compile.submit(store, job)
        sources = [job for job in built if job.key.kernel == kernel_sources.PYTHON]
        for job in sources:
            done(job, kernel_compile.wait(store, job))
        try:
            objects = sol_objects(store, present)
        except Exception as exc:  # its DSL does not import here; an executor says the same
            objects = []
            print(json.dumps({"kernel": attention_sol.KERNEL, "state": "absent"}), flush=True)
            print(f"sol objects: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        for job in objects:
            kernel_compile.submit(store, job)
        for job in [*(job for job in built if job not in sources), *objects]:
            done(job, kernel_compile.wait(store, job))
    last = {
        "done": True,
        "cards": sorted(present),
        "driver": accel.driver_version(),
        "store": str(store.own) if store is not None else "",
        "kernels": len(ended),
        "failed": sum(state != "ready" for state in ended),
        "ms": round((time.perf_counter() - started) * 1000),
    }
    print(json.dumps(last), flush=True)
    return 1 if last["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
