"""The machine's kernels, compiled at machine start on the cards present, best first, while
weights download; first, the links between its GPUs (`parallel.links`), which a group reads.

The machine launches `python -m cozy_runtime.internal.machine_kernels` once per installed
generation, under that generation's own seal: its interpreter and torch, every card of the
machine, and the kernel store and scratch scope its executors use. Each key here is therefore
the key an executor computes, by the same functions. For each distinct card architecture it
starts every artifact of `LADDER` the card admits at once, through the compile manager, niced:
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

from cozy_runtime.internal import (
    accel,
    attention,
    attention_sol,
    config,
    kernel_cache,
    kernel_compile,
    kernel_sources,
)
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.parallel import links

#: The best attention first, as the installed models rank it: H3's DiT ladder (Sol,
#: SageAttention2, FA3 BF16; SDPA needs no compile). Models without a preference use the
#: ranked table, whose only compiled kernel is FA3.
LADDER: tuple[str, ...] = ("sol-attn", "sageattention", "flash-attn3")


def cards(devices: str) -> dict[int, str]:
    """Each distinct card architecture among the sealed devices, and the first card of it."""
    entries = [entry.strip() for entry in devices.split(",") if entry.strip()]
    return {sm: entry for entry in reversed(entries) if (sm := accel.device_capability(entry))}


def ladder(present: Mapping[int, str]) -> list[kernel_compile.Job]:
    """The ladder artifacts each card architecture admits, in ladder order, once per key."""
    jobs: dict[str, kernel_compile.Job] = {}
    for sm, device in sorted(present.items()):
        facts = DeviceFacts("cuda", "", sm, "", "")
        # The approved Blackwell recipe needs BOTH binaries before selection. The
        # registry's capability predicate is shared with the model chooser; other
        # cards keep their existing boot ladder and compile optional pins on demand.
        hybrid = attention.BY_NAME["kitchen-sol-fp4-lowmem"]
        names = (
            LADDER
            if attention._device_admits(hybrid, facts)
            else ("sageattention3-fp4-global-lowmem", "kitchen-int8", *LADDER)
        )
        for name in names:
            candidate = attention.BY_NAME[name]
            if attention._device_admits(candidate, facts):
                continue
            try:
                job = kernel_sources.job(candidate.artifact, sm)
            except kernel_sources.Absent:
                continue
            # Upstream's setup.py builds for the cards it sees: this one.
            job = dataclasses.replace(job, env={"CUDA_VISIBLE_DEVICES": device})
            jobs.setdefault(job.key.digest, job)
    return list(jobs.values())


def sol_objects(store: kernel_cache.Store, present: Mapping[int, str]) -> list[kernel_compile.Job]:
    """Sol's objects for these cards, once its sources are in the store: their keys hash them."""
    try:
        tree = store.entry(kernel_sources.job(kernel_sources.PYTHON, 0).key)
    except kernel_sources.Absent:
        return []
    if tree is None:
        return []
    site.addsitedir(str(tree / kernel_cache.SITE))
    jobs: list[kernel_compile.Job] = []
    for sm in sorted(present):
        if divmod(sm, 10) in attention_sol.ROUTES:
            jobs += attention_sol.card_jobs(store, divmod(sm, 10))
    return jobs


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
        jobs = ladder(present)
        for job in jobs:
            kernel_compile.submit(store, job)
        sources = [job for job in jobs if job.key.kernel == kernel_sources.PYTHON]
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
        for job in [*(job for job in jobs if job not in sources), *objects]:
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
