#!/usr/bin/env python
"""cr-005's live verification: fill a REAL SDXL UNet from a REAL CAS onto a REAL GPU.

Not a test suite (#160) — every section drives the actual system and prints what it
observed. Run in the corpus venv, which has torch and diffusers:

    nice -n 19 corpus/.venv/bin/python scripts/fill-real.py fill
    nice -n 19 corpus/.venv/bin/python scripts/fill-real.py arms
    nice -n 19 corpus/.venv/bin/python scripts/fill-real.py bench

The bytes come from `~/cozy_v2/tensorfs-bench/store`: a real tensorfs CAS holding
stabilityai/stable-diffusion-xl-base-1.0's fp16 UNet, 1,680 tensors / 4.782 GiB, ingested
by tfs-018 through the real store writer. That is deliberate — the same store, the same
working set and the same card as the tfs-018 benchmark table, so cr-005's fill numbers are
directly comparable to it instead of being a fresh unanchored measurement.

The store is READ-ONLY here (another agent owns that repo). Verified reads go through
`internal/fill.CasStore`, the tfs-007 interim seam.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus import sdxl_real  # noqa: E402

from cozy_runtime.author._errors import AuthorError, CapabilityError  # noqa: E402
from cozy_runtime.author._loader import Artifact, Config  # noqa: E402
from cozy_runtime.internal import canonical  # noqa: E402
from cozy_runtime.internal.derive import DeriveError, derive  # noqa: E402
from cozy_runtime.internal.fill import (  # noqa: E402
    CasStore,
    FillRefusal,
    ResourceLedger,
    StreamingFillBackend,
    read_plan,
    topology,
)

BENCH = Path("/home/fidika/cozy_v2/tensorfs-bench")
STORE = BENCH / "store"
PLAN = BENCH / "results" / "read.plan"
CONFIG = BENCH / "hf" / "sdxl" / "unet" / "config.json"
CHECKPOINT = BENCH / "results" / "checkpoint.txt"
RELEASE = "cozy/sdxl-unet@cr-005"
COMPONENT = "unet"
#: tfs-018's cross-arm identity sample: every arm that genuinely landed the same bytes
#: prints the same digest. Ours must equal it or the fill moved different bytes.
SAMPLE_EVERY = 211
TFS018_DIGEST = "12ddf85c5247"

PASS = 0
FAIL = 0
MIB = 1 << 20


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {label}" + (f" — {detail}" if detail else ""), flush=True)
    else:
        FAIL += 1
        print(f"  FAIL {label}" + (f" — {detail}" if detail else ""), flush=True)


def red(label: str, fn: Any, *, code: str) -> None:
    """Plant a violation, observe the typed refusal, require it to NAME the right thing."""
    global PASS, FAIL
    try:
        fn()
    except (FillRefusal, DeriveError, AuthorError) as exc:
        named = getattr(exc, "code", "") or getattr(exc, "construct", "") or ""
        if code == named or code in str(exc):
            PASS += 1
            print(f"  red  {label} [{named}] — {str(exc).splitlines()[0][:96]}", flush=True)
            return
        FAIL += 1
        print(f"  FAIL {label} — refused as {named!r}, wanted {code!r}", flush=True)
        return
    FAIL += 1
    print(f"  FAIL {label} — did NOT refuse", flush=True)


# ------------------------------------------------------------------------- fixtures


def snapshot() -> str:
    for line in CHECKPOINT.read_text().strip().split("\n"):
        name, _, value = line.partition("\t")
        if name == "checkpoint":
            return value
    raise SystemExit("no checkpoint digest in results/checkpoint.txt")


def artifact() -> Artifact:
    plan = read_plan(PLAN, COMPONENT)
    config = Config(json.loads(CONFIG.read_text()), "unet.config")
    return Artifact(snapshot(), topology(plan), config, custody="canonical")


def derive_contract(art: Artifact, **kwargs: Any) -> Any:
    return derive(
        sdxl_real.SdxlUnetModel(),
        art,
        release=RELEASE,
        application="corpus.sdxl_real:app",
        hardware_variant="sm89",
        # Upstream diffusers READS the device capability while constructing (attention
        # backend selection), so a real UNet is hardware-conditional construction and
        # derives PER DECLARED VARIANT. cr-004 predicted this from v1's sm89/sm90 kernel
        # gates; here it fires on an unmodified upstream library, on the first real run.
        declared_variants=("sm89", "sm90"),
        component_use=sdxl_real.COMPONENT_USE,
        **kwargs,
    )


def backend_for(art: Artifact, torch: Any, **kwargs: Any) -> StreamingFillBackend:
    plan = read_plan(PLAN, COMPONENT)
    store = CasStore(STORE, verify=kwargs.pop("verify", "record"))
    free, _total = torch.cuda.mem_get_info()
    ledger = ResourceLedger(budgets={"vram": int(free * 0.9), "pinned": 512 * MIB})
    backend = StreamingFillBackend(plan, store, ledger=ledger, **kwargs)
    return backend


def peak_rss() -> int:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return 0


def written_bytes() -> int:
    """The weight-materialization write counter, at the only boundary that cannot lie.

    §3.2 requires the PyTorch serve tier's `write_bytes == 0` to be proven by a counter on
    the write surface, not by assertion-of-absence. The kernel keeps exactly that counter:
    `/proc/self/io write_bytes` is bytes this process actually sent to storage. A fill that
    materialized a file anywhere — a cache, a temp decode, a "harmless" spill — moves it.
    """
    with open("/proc/self/io") as io:
        for line in io:
            if line.startswith("write_bytes:"):
                return int(line.split()[1])
    return -1


def plan_keys() -> list[str]:
    """Header order — the same order tfs-018's arms hashed their sample in."""
    return [row.key for row in read_plan(PLAN, COMPONENT)]


def sample_digest(torch: Any, live: Any, keys: list[str]) -> str:
    digest = hashlib.sha256()
    for key in keys[::SAMPLE_EVERY]:
        digest.update(key.split(".", 1)[1].encode())
        digest.update(live[key].view(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()[:32]


# --------------------------------------------------------------------------- device


class DevicePeak:
    """Device-level peak bytes, sampled — it sees allocations torch does not own."""

    def __init__(self, torch: Any, hz: int = 100) -> None:
        import threading

        self.torch, self.interval = torch, 1.0 / hz
        self.stop, self.peak = threading.Event(), 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            free, total = self.torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)
            self.stop.wait(self.interval)

    def __enter__(self) -> DevicePeak:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


# ------------------------------------------------------------------------- eviction

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def evict_tree(root: Path) -> int:
    """POSIX_FADV_DONTNEED the whole CAS. No passwordless sudo on this box, so this is the
    eviction primitive — the same one tfs-018 used, so cold means the same thing."""
    count = 0
    for folder, _dirs, files in os.walk(root):
        for name in files:
            fd = os.open(os.path.join(folder, name), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            count += 1
    return count


def resident_pages(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    if size == 0:
        return (0, 0)
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, 0x1, 0x02, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap")
        try:
            pages = (size + 4095) // 4096
            vec = ctypes.create_string_buffer(pages)
            if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore")
            return (sum(b & 1 for b in vec.raw), pages)
        finally:
            _libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


# ------------------------------------------------------------------------ the fill


def section_fill(args: argparse.Namespace) -> dict[str, Any]:
    """The real run: derive, fill for real, prove derive == serve ON BYTES, then compute."""
    import torch

    art = artifact()
    keys = plan_keys()
    total_bytes = sum(row.nbytes for row in read_plan(PLAN, COMPONENT))

    head("derive the real SDXL UNet contract (meta substrate, no GPU, no bytes)")
    started = time.perf_counter()
    derived = derive_contract(art)
    derive_ms = (time.perf_counter() - started) * 1000
    body = canonical.write(derived.contract.render())
    dset = derived.contract.destination_sets[0]
    check(
        "one complete DestinationSet",
        len(derived.contract.destination_sets) == 1,
        f"{dset.component}: {len(dset.keys)} keys, {dset.module_class}",
    )
    check(
        "logical tensors match the checkpoint",
        len(derived.contract.logical_tensors) == 1680,
        f"{len(derived.contract.logical_tensors)} tensors, {len(body)} contract bytes, "
        f"derived in {derive_ms:.1f} ms",
    )
    check(
        "no derived (non-persistent) tensors",
        not derived.contract.derived,
        f"{len(derived.contract.derived)} — nothing the fill plane cannot supply",
    )
    print(f"  contract {derived.contract.digest}")

    head(
        f"fill for real — {args.window // MIB} MiB windows, {args.slots} slots, "
        f"{args.readers} readers, verify={args.verify}"
    )
    torch.cuda.init()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()

    backend = backend_for(
        art,
        torch,
        window_bytes=args.window,
        slots=args.slots,
        readers=args.readers,
        verify=args.verify,
    )
    backend.expect({d.component: d.keys for d in derived.contract.destination_sets})
    written_before = written_bytes()
    with DevicePeak(torch) as peak:
        started = time.perf_counter()
        served = derive_contract(
            art, mode="serve", backend=backend, substrate_device=backend.device_str
        )
        elapsed = time.perf_counter() - started
    written = written_bytes() - written_before

    fill_s = backend.stats["fill_ms"] / 1000
    gbps = total_bytes / fill_s / 1e9
    print(
        f"  {total_bytes / 2**30:.3f} GiB in {fill_s:.3f} s = {gbps:.3f} GB/s "
        f"({total_bytes / fill_s / 2**30:.3f} GiB/s) — the disk->VRAM leg; "
        f"{elapsed:.3f} s including the meta construction"
    )
    print(f"  store {backend.store.stats()}")
    print(f"  ring  {json.dumps(backend.stats)}")
    print(f"  ledger {json.dumps(backend.ledger.rows())}")

    check("the transaction is READY", backend.state == "ready", backend.state)
    check(
        "every contract destination was filled",
        len(backend.enqueued) == 1680,
        f"{len(backend.enqueued)}/1680",
    )
    check(
        "zero rehashes on a verified store",
        backend.store.rehashes == 0,
        f"{backend.store.verifies} record hits, "
        f"{backend.store.verify_ns / 1e6:.1f} ms of verification",
    )
    check(
        "serve-tier write_bytes == 0, by COUNTER",
        written == 0,
        f"the kernel charged this process {written} B of storage writes across the whole "
        f"fill (/proc/self/io write_bytes) — measured, never asserted-by-absence",
    )

    head("derive == serve, on the real byte path")
    other = canonical.write(served.contract.render())
    check(
        "byte-identical contracts",
        body == other,
        f"{len(body)} bytes both, {served.contract.digest[:24]}…",
    )
    inline_bytes = sum(
        s.length for row in read_plan(PLAN, COMPONENT) for s in row.segments if s.oid is None
    )
    check(
        "the serve pass moved real bytes",
        backend.store.bytes + inline_bytes == total_bytes,
        f"{backend.store.bytes} B read from the CAS + {inline_bytes} B carried inline "
        f"by the header = {total_bytes} B",
    )

    head("the filled destinations vs the derived contract")
    live = {key: tensor for key, tensor in backend.enqueued.items()}
    rows = {row.key: row for row in read_plan(PLAN, COMPONENT)}
    bad_shape = [
        t.key for t in derived.contract.logical_tensors if tuple(live[t.key].shape) != t.shape
    ]
    bad_dtype = [
        t.key for t in derived.contract.logical_tensors if str(live[t.key].dtype) != "torch.float16"
    ]
    off_device = [
        t.key
        for t in derived.contract.logical_tensors
        if str(live[t.key].device) != backend.device_str
    ]
    check("shapes", not bad_shape, f"1680 destinations, {len(bad_shape)} wrong")
    check("dtypes", not bad_dtype, f"all float16, {len(bad_dtype)} wrong")
    check("placement", not off_device, f"all on {backend.device_str}")

    if args.verify_bytes:
        head("every destination's DEVICE bytes hash to its CAS object id")
        started = time.perf_counter()
        wrong, hashed, inline = [], 0, 0
        for key in keys:
            row = rows[key]
            body_bytes = live[key].view(-1).view(torch.uint8).cpu().numpy().tobytes()
            if len(row.segments) == 1 and row.segments[0].oid is not None:
                if hashlib.sha256(body_bytes).hexdigest() != row.segments[0].oid:
                    wrong.append(key)
                hashed += 1
            else:
                want = b"".join(s.inline or b"" for s in row.segments)
                if body_bytes != want:
                    wrong.append(key)
                inline += 1
        took = time.perf_counter() - started
        check(
            "on-device sha256 == object id, every tensor",
            not wrong,
            f"{hashed} hashed + {inline} inline in {took:.1f} s, {len(wrong)} wrong",
        )

    digest = sample_digest(torch, live, keys)
    check(
        "cross-arm identity with tfs-018's banked digest",
        digest.startswith(TFS018_DIGEST),
        f"{digest[:12]} vs {TFS018_DIGEST}",
    )

    out: dict[str, Any] = {
        "window": args.window,
        "slots": args.slots,
        "readers": args.readers,
        "verify": args.verify,
        "seconds": round(fill_s, 4),
        "bytes": total_bytes,
        "wall_seconds": round(elapsed, 4),
        "gbps": round(gbps, 3),
        "peak_vram": peak.peak,
        "peak_vram_gib": round(peak.peak / 2**30, 3),
        "torch_peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_rss": peak_rss(),
        "peak_rss_gib": round(peak_rss() / 2**30, 3),
        "digest": digest,
        "written_bytes": written,
        "ledger": backend.ledger.rows(),
        **backend.store.stats(),
        **backend.stats,
    }

    if args.forward:
        backend.require_ready("a forward pass")
        unet = backend.components[COMPONENT]
        unet.eval()

        def raw(*inputs: Any) -> Any:
            with torch.inference_mode():
                return unet(
                    inputs[0],
                    inputs[1],
                    encoder_hidden_states=inputs[2],
                    added_cond_kwargs={"text_embeds": inputs[3], "time_ids": inputs[4]},
                ).sample

        out.update(section_forward(torch, raw, backend.device, args))

    if args.release:
        head("unload: component release returns VRAM")
        # `live` deliberately still holds all 1,680 device tensors here: release must free
        # the ALLOCATION, not merely drop the plane's own references, or a supervisor
        # holding a handle would pin 4.78 GiB after an unload.
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        freed = backend.release(COMPONENT)
        after = torch.cuda.memory_allocated()
        check(
            "the allocator gave the weights back",
            freed >= total_bytes * 0.99 and after - base_alloc < 64 * MIB,
            f"released {freed / 2**30:.3f} GiB, allocator {before / 2**30:.3f} -> "
            f"{after / 2**30:.3f} GiB (baseline {base_alloc / 2**30:.3f})",
        )
        check(
            "the ledger agrees",
            backend.ledger.charged["vram"] == 0,
            json.dumps(backend.ledger.rows()["vram"]),
        )
        out["released_bytes"] = freed
    return out


def section_forward(torch: Any, call: Any, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    """One real UNet denoising step on the filled weights — the first real inference
    computation in v2. Fixed seed, fp16, deterministic algorithms where torch has them.

    `call` is either the raw filled module or the author's declared model method; both
    spellings must produce the SAME bits, which is what makes the guarded root a fence
    rather than a behaviour change.
    """
    head("one real denoising step on the filled weights")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    determinism = "cudnn.deterministic + benchmark=False"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        determinism += " + use_deterministic_algorithms(True, warn_only=True)"
    except Exception as exc:
        determinism += f" (use_deterministic_algorithms unavailable: {exc})"

    gen = torch.Generator(device=device).manual_seed(args.seed)
    side = args.latent
    latents = torch.randn(1, 4, side, side, generator=gen, device=device, dtype=torch.float16)
    prompt = torch.randn(1, 77, 2048, generator=gen, device=device, dtype=torch.float16)
    text_embeds = torch.randn(1, 1280, generator=gen, device=device, dtype=torch.float16)
    time_ids = torch.tensor(
        [[side * 8, side * 8, 0, 0, side * 8, side * 8]], device=device, dtype=torch.float16
    )
    timestep = torch.tensor([981], device=device)

    started = time.perf_counter()
    out = call(latents, timestep, prompt, text_embeds, time_ids)
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - started) * 1000

    body = out.view(-1).view(torch.uint8).cpu().numpy().tobytes()
    digest = hashlib.sha256(body).hexdigest()
    stats = out.float()
    check(
        "the step produced finite output",
        bool(torch.isfinite(out).all()),
        f"shape {tuple(out.shape)} {out.dtype}, mean {stats.mean().item():+.6f}, "
        f"std {stats.std().item():.6f}, min {stats.min().item():+.4f}, "
        f"max {stats.max().item():+.4f}",
    )
    check(
        "the output is not degenerate",
        stats.std().item() > 1e-3,
        "a constant or zero field would mean the weights never landed",
    )
    print(f"  latent {side}x{side} ({side * 8}px) · step {step_ms:.1f} ms · seed {args.seed}")
    print(f"  determinism: {determinism}")
    print(f"  output sha256 {digest}")
    return {
        "forward_digest": digest,
        "forward_ms": round(step_ms, 2),
        "forward_mean": stats.mean().item(),
        "forward_std": stats.std().item(),
        "determinism": determinism,
        "latent": side,
        "seed": args.seed,
        "forward_peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }


# --------------------------------------------------------------------------- serve


def section_serve(args: argparse.Namespace) -> None:
    """The REAL consumer: `ModelRegistry.acquire` on the real UNet, filled by cr-005.

    This is the whole vertical the fill plane exists for — one endpoint class, one binding
    path, weightless construction on the meta substrate, real bytes placed by the fill
    plane, guarded component roots, and the author's own declared method computing on the
    result. cr-007 drives exactly this from the supervisor; cr-008a wraps a request around
    it.
    """
    import torch

    from cozy_runtime.author import ModelRegistry
    from cozy_runtime.internal.derive import Observations, meta_substrate

    art = artifact()
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    head("ModelRegistry.acquire — weightless construction, cr-005 places the bytes")
    backend = backend_for(
        art, torch, window_bytes=args.window, slots=args.slots, readers=args.readers
    )
    plan = read_plan(PLAN, COMPONENT)
    backend.expect({COMPONENT: tuple(row.key for row in plan)})
    seen = Observations()
    registry = ModelRegistry(
        release=RELEASE,
        backend=backend,
        substrate=lambda: meta_substrate("sm89", seen),
    )
    started = time.perf_counter()
    model = registry.acquire("denoise.models.model", sdxl_real.SdxlUnetModel, art)
    elapsed = time.perf_counter() - started
    record = registry.generations()[0].record
    leg = backend.stats["fill_ms"] / 1000
    print(
        f"  acquired in {elapsed:.3f} s — {record.filled} destinations, "
        f"{record.filled_bytes / 2**30:.3f} GiB, fill leg {leg:.3f} s"
    )
    print(f"  identity {record.identity}")
    check(
        "the generation is READY only after the fill fence",
        model._cozy_ready and backend.state == "ready",
        f"backend {backend.state}, model ready={model._cozy_ready}",
    )
    check(
        "the component root is GUARDED",
        type(model.pipe.components[COMPONENT]).__name__ == "GuardedComponent",
        repr(model.pipe.components[COMPONENT]),
    )

    head("the author's declared method, on real weights")
    digest = section_forward(torch, model.denoise, backend.device, args)
    print("  through @uses_components('unet') on the guarded root")

    head("the guarded root: no component access outside an ACTIVE scope")
    red(
        "touching the UNet outside a declared scope",
        lambda: model.pipe.components[COMPONENT].config,
        code="undeclared_component",
    )
    red(
        "…and the generation is POISONED, not merely scolded",
        lambda: model.denoise(*(None,) * 5),
        code="poisoned_generation",
    )

    head("unload: the generation gives the device back")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    try:
        registry.unload_all()
    except CapabilityError as exc:
        print(f"  unload_all refused on the poisoned generation: {str(exc)[:88]}")
    freed = backend.release(COMPONENT) if COMPONENT in backend.components else 0
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    check(
        "the allocator is back to baseline",
        after - base < 64 * MIB,
        f"{before / 2**30:.3f} -> {after / 2**30:.3f} GiB (baseline {base / 2**30:.3f}), "
        f"release() returned {freed / 2**30:.3f} GiB",
    )
    check(
        "the ledger is back to zero",
        all(v["charged"] == 0 for v in backend.ledger.rows().values()),
        json.dumps(backend.ledger.rows()),
    )
    if args.marker:
        Path(args.marker).write_text(json.dumps(digest))


# --------------------------------------------------------------------------- arms


def section_arms(args: argparse.Namespace) -> None:
    """The refusal matrix, the transaction, the lease and the ledger — planted live."""
    import torch

    from cozy_runtime.author._loader import TensorSpec
    from cozy_runtime.internal.fill import CasStore as Store
    from cozy_runtime.internal.fill import StagingRing

    art = artifact()
    plan = read_plan(PLAN, COMPONENT)
    tiny = [row for row in plan if row.nbytes < 4096][:6]

    head("refusal matrix — each fires BEFORE a byte moves")

    def missing_object() -> None:
        store = Store(STORE)
        store.ensure_verified("f" * 64, 16)

    red("missing object", missing_object, code="missing_object")

    def digest_mismatch() -> None:
        """A real object, verified against the WRONG id: the store's own record for the
        real id cannot answer for it, so the read rehashes and the address disagrees."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "store"
            oid = tiny[0].segments[0].oid or ""
            wrong = ("0" * 8) + oid[8:]
            target = fake / "objects" / "sha256" / wrong[:2] / wrong[2:4] / wrong
            target.parent.mkdir(parents=True)
            shutil.copyfile(Store(STORE).object_path(oid), target)
            Store(fake).ensure_verified(wrong, tiny[0].nbytes)

    red("digest mismatch (content address is identity)", digest_mismatch, code="digest_mismatch")

    def length_mismatch() -> None:
        store = Store(STORE)
        row = tiny[0]
        store.ensure_verified(row.segments[0].oid or "", row.nbytes + (1 << 30))

    red(
        "length mismatch (header claims more than the object holds)",
        length_mismatch,
        code="length_mismatch",
    )

    def budget_overrun() -> None:
        ledger = ResourceLedger(budgets={"vram": 1 << 20, "pinned": 1 << 20})
        ledger.grant({"vram": 5 << 30}, "the whole UNet against a 1 MiB budget")

    red("budget overrun (named need and have)", budget_overrun, code="budget_overrun")

    def destination_absent() -> None:
        store = Store(STORE)
        backend = StreamingFillBackend(plan, store, slots=2, window_bytes=MIB)
        backend.fill(
            "unet.not_a_destination",
            TensorSpec((1,), "float16", nbytes=2),
            torch.empty(1, dtype=torch.float16, device="cuda"),
        )

    red("destination absent from the contract", destination_absent, code="destination_absent")

    def length_mismatch_dest() -> None:
        store = Store(STORE)
        backend = StreamingFillBackend(plan, store, slots=2, window_bytes=MIB)
        row = tiny[0]
        backend.fill(
            row.key,
            TensorSpec(row.shape, "float16", nbytes=row.nbytes),
            torch.empty(row.nbytes, dtype=torch.float16, device="cuda"),
        )

    red(
        "destination length disagrees with the checkpoint",
        length_mismatch_dest,
        code="length_mismatch",
    )

    def dtype_mismatch() -> None:
        """The SILENT case: the same byte count, read as a different dtype. Length agrees,
        so only an explicit dtype check stands between this and a plausible wrong tensor."""
        store = Store(STORE)
        backend = StreamingFillBackend(plan, store, slots=2, window_bytes=MIB)
        row = tiny[0]
        backend.fill(
            row.key,
            TensorSpec(row.shape, "float16", nbytes=row.nbytes),
            torch.empty(row.nbytes // 4, dtype=torch.float32, device="cuda"),
        )

    red(
        "same bytes, wrong dtype — a decode, and decoders are cr-006's",
        dtype_mismatch,
        code="dtype_mismatch",
    )

    def incomplete() -> None:
        store = Store(STORE)
        backend = StreamingFillBackend(plan, store, slots=2, window_bytes=MIB)
        backend.expect({COMPONENT: tuple(row.key for row in plan[:4])})
        row = plan[0]
        backend.fill(
            row.key,
            TensorSpec(row.shape, "float16", nbytes=row.nbytes),
            torch.empty(row.shape, dtype=torch.float16, device="cuda"),
        )
        backend.commit()

    red("a DestinationSet is COMPLETE or it does not commit", incomplete, code="incomplete_fill")

    def not_ready() -> None:
        store = Store(STORE)
        backend = StreamingFillBackend(plan, store, slots=2, window_bytes=MIB)
        backend.state = "staging"
        backend.require_ready("a forward pass")

    red("nothing is observable as Ready before the fence", not_ready, code="not_ready")

    def one_slot() -> None:
        StagingRing(torch, torch.device("cuda"), window_bytes=MIB, slots=1, ledger=ResourceLedger())

    red("a one-slot ring cannot double-buffer", one_slot, code="lease_live")

    def early_release() -> None:
        """The lease-lifetime arm: give a slot back with no completion event recorded,
        then claim it again. The ring refuses rather than racing the copy engine."""
        ring = StagingRing(
            torch,
            torch.device("cuda"),
            window_bytes=MIB,
            slots=2,
            ledger=ResourceLedger(),
            fence=False,
        )
        slot = ring.slots[0]
        ring.take(slot)
        ring.give_back(slot)  # fence=False: no event recorded — the planted defect
        ring.take(slot)

    red("a staging slot released before its H2D event", early_release, code="lease_live")

    head("the transaction: an interrupted real fill leaves nothing Ready")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    backend = backend_for(
        art,
        torch,
        window_bytes=args.window,
        slots=args.slots,
        readers=args.readers,
        interrupt_at=0.5,
    )
    backend.expect({COMPONENT: tuple(row.key for row in plan)})
    interrupted = None
    try:
        derive_contract(art, mode="serve", backend=backend, substrate_device=backend.device_str)
    except FillRefusal as exc:
        interrupted = exc
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    check(
        "the fill refused mid-stream",
        interrupted is not None,
        str(interrupted).splitlines()[0][:96] if interrupted else "",
    )
    check("nothing is Ready", backend.state == "rolled_back", backend.state)
    check(
        "the destinations were released",
        after - base < 64 * MIB,
        f"allocator {base / 2**30:.3f} -> {after / 2**30:.3f} GiB after rollback",
    )
    check(
        "the ledger is back to zero",
        backend.ledger.charged["vram"] == 0 and backend.ledger.charged["pinned"] == 0,
        json.dumps(backend.ledger.rows()),
    )
    check(
        "verified CAS is untouched",
        backend.store.rehashes == 0,
        f"{backend.store.verifies} objects verified, 0 rehashed, 0 written",
    )


def section_kill(args: argparse.Namespace) -> None:
    """Process death mid-fill: the OTHER half of transactionality, which no in-process
    rollback can prove. SIGKILL leaves no chance to unwind, so the evidence is external."""
    import torch

    head("SIGKILL mid-fill: no Ready marker, and the device gets its memory back")
    marker = Path(args.marker)
    marker.unlink(missing_ok=True)
    free_before, _ = torch.cuda.mem_get_info()
    child = subprocess.Popen(
        [
            "nice",
            "-n",
            "19",
            sys.executable,
            __file__,
            "fill",
            "--no-forward",
            "--no-release",
            "--no-verify-bytes",
            "--marker",
            str(marker),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(args.kill_after)
    used = 0
    for _ in range(20):
        free, _t = torch.cuda.mem_get_info()
        used = max(used, free_before - free)
        if used > 1 << 30:
            break
        time.sleep(0.1)
    child.kill()
    child.wait(timeout=30)
    time.sleep(1.0)
    free_after, _ = torch.cuda.mem_get_info()
    check(
        "the child was killed mid-fill",
        used > 1 << 30,
        f"{used / 2**30:.2f} GiB of destinations were live when SIGKILL landed",
    )
    check(
        "no Ready marker exists",
        not marker.exists(),
        "a killed fill never reaches the freeze point",
    )
    check(
        "the device is back to baseline",
        abs(free_after - free_before) < 64 * MIB,
        f"free {free_before / 2**30:.3f} -> {free_after / 2**30:.3f} GiB",
    )
    store = CasStore(STORE)
    row = read_plan(PLAN, COMPONENT)[0]
    store.ensure_verified(row.segments[0].oid or "", row.nbytes)
    check(
        "verified CAS survived the kill",
        store.rehashes == 0,
        "the killed fill wrote nothing, so the record still answers",
    )


# -------------------------------------------------------------------------- bench


def section_bench(args: argparse.Namespace) -> None:
    """Cold and warm arms, fresh process each, evicted between colds — the tfs-018 method,
    so the numbers land in the same table."""
    results = Path(args.out)
    configs = [(int(w) * MIB, s, r) for w, s, r in (c.split(":") for c in args.configs.split(","))]
    for window, slots, readers in configs:
        for mode in ["cold"] * args.colds + ["warm"]:
            evicted = pages = 0
            if mode == "cold":
                evicted = evict_tree(STORE / "objects")
                # Verify the eviction on the LARGEST object, the same way tfs-018 did:
                # fadvise is advisory, so a cold arm that never checks is a warm arm.
                biggest = max(read_plan(PLAN, COMPONENT), key=lambda r: r.nbytes)
                pages = resident_pages(CasStore(STORE).object_path(biggest.segments[0].oid or ""))[
                    0
                ]
            started = time.perf_counter()
            proc = subprocess.run(
                [
                    "nice",
                    "-n",
                    "19",
                    sys.executable,
                    __file__,
                    "fill",
                    "--no-forward",
                    "--no-verify-bytes",
                    "--no-release",
                    "--json",
                    "--window",
                    str(window),
                    "--slots",
                    str(slots),
                    "--readers",
                    str(readers),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"  FAIL {window // MIB}MiB s{slots} r{readers} {mode}: {proc.stderr[-800:]}")
                continue
            row = json.loads(proc.stdout.strip().split("\n")[-1])
            row.update(
                mode=mode,
                evicted_files=evicted,
                sample_resident_pages=pages,
                wall=round(time.perf_counter() - started, 2),
            )
            with open(results, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"  {window // MIB:>4} MiB s{slots} r{readers} {mode:4} "
                f"{row['seconds']:7.3f} s {row['gbps']:6.3f} GB/s  "
                f"vram {row['peak_vram_gib']:.2f} rss {row['peak_rss_gib']:.2f}  "
                f"{row['digest'][:12]}",
                flush=True,
            )


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=("fill", "serve", "arms", "kill", "bench"))
    parser.add_argument("--window", type=int, default=16 * MIB)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--verify", choices=("record", "rehash"), default="record")
    parser.add_argument("--seed", type=int, default=1005)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--no-forward", dest="forward", action="store_false")
    parser.add_argument("--no-release", dest="release", action="store_false")
    parser.add_argument("--no-verify-bytes", dest="verify_bytes", action="store_false")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--marker", default="")
    parser.add_argument("--kill-after", type=float, default=2.0)
    parser.add_argument("--configs", default="16:4:4,16:2:1,64:4:4,4:4:4")
    parser.add_argument("--colds", type=int, default=3)
    parser.add_argument("--out", default=str(ROOT / "results" / "fill.jsonl"))
    args = parser.parse_args()

    if args.section == "fill":
        row = section_fill(args)
        if args.marker and not FAIL:
            Path(args.marker).write_text(json.dumps(row))
        if args.json:
            row["checks"] = f"{PASS} passed, {FAIL} failed"
            print(f"\n{PASS} passed, {FAIL} failed")
            print(json.dumps(row))
            return 1 if FAIL else 0
    elif args.section == "serve":
        section_serve(args)
    elif args.section == "arms":
        section_arms(args)
    elif args.section == "kill":
        section_kill(args)
    else:
        section_bench(args)
        return 0

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
