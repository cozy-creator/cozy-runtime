"""The machine's kernels, compiled at worker boot in the CUDA worker image from its pinned
sources, then served to a newer executor from the store and to an older one from the image
kernel site. Run by `tests/test_kernel_boot.py` inside one container; each mode prints one JSON
line.

- `machine`: the worker's boot (`machine_kernels.boot`), as root like a pod's worker: H3's
  ladder the card admits, best first, each build as wide as the container measures.
- `executor-warm`: a newer executor under its own uid while the machine compiles: H3's warm
  and a construction see the machine's builds as compiling and start none of their own.
- `executor-serve`: the same executor later: Sol and SageAttention2 from the worker's
  namespace, Sol's card object and the fused glue its own; Sol matches upstream's reference.
- `unpublish`, `publish`: take the machine's trees out of the image kernel site and put them
  back, as the machine does when a build lands (`MachineKernels._done`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

from cozy_runtime.author import AttentionContext
from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.internal import (
    accel,
    attention,
    attention_sol,
    fusion,
    jit_cache,
    kernel_cache,
    kernel_compile,
    kernel_site,
    kernel_sources,
    machine_kernels,
)
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.models.minimax_h3.model import DIT_ATTENTION, H3Model

TOKENS = 4096
EXECUTOR = 64001


def main() -> None:
    mode, root = sys.argv[1], Path(sys.argv[2])
    sm = accel.device_capability("0")
    out: dict[str, Any] = {"sm": sm}
    if mode == "machine":
        machine(out, root)
    elif mode in ("publish", "unpublish"):
        site(mode, root)
    else:
        store = kernel_cache.Store(
            kernel_cache.namespace(root, os.geteuid()), kernel_cache.namespace(root, 0)
        )
        kernel_cache.configure(store)
        device = DeviceFacts("cuda", "", sm, "", "")
        context = AttentionContext("ref2va_dit", "", "cuda", "", sm, "bfloat16", 128, 1, (), "", "")
        ladder = H3Model.choose_attention(object.__new__(H3Model), context)
        assert ladder is not None
        out["h3"] = list(ladder)
        module = WanAttention(dim=1024, heads=8, dim_head=128, processor=WanAttnProcessor())
        if mode == "executor-warm":
            executor_warm(out, device, {"ref2va_dit": module.to(torch.bfloat16)}, ladder)
        else:
            roots = {"ref2va_dit": module.to("cuda", torch.bfloat16)}
            executor_serve(out, device, roots, ladder, store)
        out["own_builders"] = sorted(
            lock.parent.name
            for lock in store.own.glob("*/*.lock")
            if lock.stem in kernel_compile._STARTED
        )
    print(json.dumps(out))


def machine(out: dict[str, Any], root: Path) -> None:
    jit_cache.machine(root, uid=EXECUTOR, gid=EXECUTOR)  # as the worker prepares an executor
    started = time.monotonic()
    booted = machine_kernels.boot(root, "0", dict(os.environ))
    assert booted is not None and booted.thread is not None
    out["ladder"] = list(machine_kernels.LADDER)
    out["h3_ladder"] = [name for name in DIT_ATTENTION if name != "sdpa"]
    out["site"] = booted.site
    out["width"] = kernel_compile.width(kernel_sources.RECIPES["sageattention"].unit_bytes)
    out["order"] = [job.key.kernel for _, job in booted.jobs]
    (root.parent / "machine.started").write_text(booted.line())
    ready: dict[str, float] = {}
    while True:
        alive = booted.thread.is_alive()
        for _, job in booted.jobs:
            if kernel_compile.status(booted.store, job).state == "ready":
                ready.setdefault(job.key.kernel, round(time.monotonic() - started, 1))
        if not alive:
            break
        time.sleep(1)
    out["compiles"] = {
        job.key.kernel: {
            **kernel_compile.status(booted.store, job).document(),
            "ready_after_boot_s": ready.get(job.key.kernel),
            **_producer(booted.store, job),
        }
        for _, job in booted.jobs
    }
    out["published"] = booted.published
    out["manifest"] = json.loads((kernel_site.ROOT / kernel_site.MANIFEST).read_text())
    out["site_entries"] = sorted(p.name for p in (kernel_site.ROOT / "site").iterdir())


def site(mode: str, root: Path) -> None:
    store = kernel_cache.Store(kernel_cache.namespace(root, 0))
    for name in (kernel_sources.PYTHON, "sageattention"):
        for entry in (store.own / name).glob("*/entry.json"):
            key = json.loads(entry.read_text())["key"]
            if mode == "publish":
                kernel_site.publish(entry.parent, kernel_cache.Key.of(name, key["inputs"]))
                continue
            for item in (entry.parent / kernel_cache.SITE).iterdir():
                (kernel_site.ROOT / "site" / item.name).unlink(missing_ok=True)


def executor_warm(
    out: dict[str, Any], device: DeviceFacts, roots: dict[str, Any], ladder: tuple[str, ...]
) -> None:
    out["warm"] = attention.expect(device, list(ladder))
    constructed = attention.select(device, roots, choose=lambda _: ladder)
    out["construction"] = {
        "hosts": constructed.hosts,
        "skipped": constructed.skipped,
        "evidence": attention.evidence(constructed.totals()),
    }
    try:
        attention.pinned("sageattention", device)
    except attention.AttentionRefusal as exc:
        out["pin_while_compiling"] = {"code": exc.code, "detail": str(exc)}


def executor_serve(
    out: dict[str, Any],
    device: DeviceFacts,
    roots: dict[str, Any],
    ladder: tuple[str, ...],
    store: kernel_cache.Store,
) -> None:
    cuda = torch.device("cuda", 0)
    served = attention.select(device, roots, choose=lambda _: ladder)
    out["h3_hosts"] = served.hosts
    out["artifacts"] = served.artifacts
    # Sol's card object and the fused glue are this executor's own, submitted as a request
    # would; the steps before them run dense and eager.
    fusion.expect(cuda)
    capability = (device.sm // 10, device.sm % 10)
    for job in (attention_sol._job_for(TOKENS, capability), fusion.job(device.sm)):
        with store.building(job.key):
            pass
    site_module = roots["ref2va_dit"]
    hidden = torch.randn(1, TOKENS, 1024, device=cuda, dtype=torch.bfloat16)
    layout = AttentionLayout(TOKENS, 0, step=5, dense_until_step=4)
    with torch.no_grad(), attention_sol.observing() as counts, attention_scope(layout):
        sol = site_module(hidden)
    out["sol_counts"] = dict(counts)
    out["sol_site_finite"] = bool(torch.isfinite(sol).all())
    out["sol"] = sol_numerics(cuda, layout)
    out["evidence"] = attention.evidence(served.totals())
    tail = attention.select(device, roots, choose=lambda _: ladder[1:])
    out["sage"] = {"hosts": tail.hosts, "artifacts": tail.artifacts}
    with torch.no_grad():
        sage = site_module(hidden)
        attention.select(device, roots)
        floor = site_module(hidden)
    out["sage"]["finite"] = bool(torch.isfinite(sage).all())
    out["sage"]["rel_l2_to_sdpa"] = _rel(sage, floor)
    loaded = fusion.load(cuda)
    out["fusion"] = {"arch": loaded.arch, "kernels": len(loaded.compiled)}


def sol_numerics(cuda: Any, layout: AttentionLayout) -> dict[str, float]:
    """Sol's compiled object against upstream's own Triton implementation of the same sparse
    algorithm, on block-local attention (each query's keys in one 64-token block): what this
    card's build computes, apart from how far Sol's sparsity is from dense."""
    # Importable once the store's Python sources are on the path, as the runtime imports it.
    reference = import_module("sol_attn.triton_ref").sol_attn
    generator = torch.Generator(device=cuda).manual_seed(7)
    heads, blocks = 8, TOKENS // 64
    centers = torch.randn(1, blocks, heads, 128, generator=generator, device=cuda)
    noise = torch.randn(1, TOKENS, heads, 128, generator=generator, device=cuda)
    key = (centers.repeat_interleave(64, dim=1) + 0.3 * noise).bfloat16()
    target = torch.randint(0, blocks, (TOKENS,), generator=generator, device=cuda)
    query = centers[:, target].bfloat16().contiguous()
    value = torch.randn(1, TOKENS, heads, 128, generator=generator, device=cuda).bfloat16()
    token = attention_sol._SITE.set(attention_sol.Site("ref2va_dit", ""))
    try:
        with torch.no_grad(), attention_scope(layout):
            sol = attention_sol.sol_attention(query, key, value)
    finally:
        attention_sol._SITE.reset(token)
    upstream = reference(query, key, value, tau=1.0, thresh_type="exact", sink_start=0)
    dense = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (query, key, value))
    ).transpose(1, 2)
    return {"rel_l2_to_upstream": _rel(sol, upstream), "rel_l2_to_dense": _rel(sol, dense)}


def _producer(store: kernel_cache.Store, job: kernel_compile.Job) -> dict[str, Any]:
    entry = store.entry(job.key)
    if entry is None:
        return {}
    producer = json.loads((entry / "entry.json").read_text()).get("producer") or {}
    return {k: producer[k] for k in ("width", "extensions_at_once", "units_ms") if k in producer}


def _rel(value: Any, reference: Any) -> float:
    return float((value.float() - reference.float()).norm() / reference.float().norm())


if __name__ == "__main__":
    main()
