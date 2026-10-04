"""The machine's kernels, compiled at machine start on the cards present, best first, while
weights download.

The machine launches `python -m cozy_runtime.internal.machine_kernels` once per installed
generation, under that generation's own seal: its interpreter and torch, every card of the
machine, and the kernel store and scratch scope its executors use. Each key here is therefore
the key an executor computes, by the same functions. For each distinct card architecture it
compiles the artifacts of `LADDER` the card admits, in its order, through the compile manager
(`kernel_compile.in_order`): one wide build at a time, as wide as the machine measures, niced;
then Sol's card objects. An executor finds each one ready, or building and claimed, and waits
for the one it needs (`kernel_compile.wait`): it builds none twice and serves no other kernel
in its place. This process holds no device memory. It prints one JSON line per artifact as it
ends and a last one, and exits nonzero when one failed.
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
        for name in LADDER:
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
    present = cards(environment.get("CUDA_VISIBLE_DEVICES", ""))
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
        kernel_compile.machine_slot(store.own, create=True)
        kernel_compile.in_order(store, ladder(present), done)
        try:
            objects = sol_objects(store, present)
        except Exception as exc:  # its DSL does not import here; an executor says the same
            objects = []
            print(json.dumps({"kernel": attention_sol.KERNEL, "state": "absent"}), flush=True)
            print(f"sol objects: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        for job in objects:
            kernel_compile.submit(store, job)
        for job in objects:
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
