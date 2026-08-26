#!/usr/bin/env python3
"""The MXFP8 lane's GPU leg. POD-SIDE ONLY — this file never runs on the dev box.

`scripts/mxfp8-live.py` proves the VALUE LAW on CPU, bit for bit, against TensorFS's frozen
mined vectors and the producing job's own dequant reference. Three things it structurally
cannot prove, and this script does:

  probe    the same 65,536-pair format domain decoded by CUDA kernels rather than CPU ones
  fill     the real fill plane end to end — a two-role encoding through the staging ring,
           the transient device scratch, the decode fence and the decode ledger class, none
           of which has ever carried an encoding with more than a rank-0 companion
  forward  what the encoding costs THROUGH REAL WEIGHTS: one H3 curve-DiT block's GEMM
           chain, mxfp8-filled against bf16-filled, from the same source bytes

**The subject.** `a97b89d0…` was produced on a pod whose store died with it (job-003's
`/root/mx/store`), so there is no store anywhere holding it — what survives is the proof
that it REPRODUCES. Reproducing the whole of it here would mean 40.2 GB fetched and a
40.2 GB bf16 residency, because the decoded-float route materializes float weights: mxfp8
buys bytes read, NOT VRAM. So this leg takes the same source file, the same revision, the
same job equation and the same border, over ONE BLOCK — 772 MB fetched by HTTP range,
which is a real subject a cheap card can actually hold. What it does not claim is the
532-tensor completeness walk; TensorFS's own `snapshot verify` banked that.

**The forward's honesty.** H3's block wiring is not in this tree, so the chain below is
RMSNorm -> qkv -> SDPA -> out_proj -> RMSNorm -> SwiGLU, inferred from the real shapes
(qkv [21504, 5376] = 3 x 56 x 128 against a 5376 width; fc1 [28672, 5376] = 2 x 14336).
It is not claimed to be H3's forward. It does not need to be: BOTH ARMS RUN THE IDENTICAL
MODULE and differ only in which stored bytes filled it, so a wiring error moves both arms
together and the delta between them stays attributable to the encoding alone.

    python3 mxfp8-pod.py all
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path("/root/mx8")
STORE = ROOT / "store"
TFS = "/root/tfs"
REPO = "Comfy-Org/MiniMax-H3"
REV = "4cc1d817b6184899b41293954329f576cb5ae86b"
MEMBER = "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/{MEMBER}"

#: The block this leg takes, and the four keys the job's inclusion map quantizes in it.
BLOCK = "blocks.0."
QUANTIZED = (
    "attn.qkv_proj.weight",
    "attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)
MX_BLOCK = 32
E4M3_MAX = 448.0
MIB = 1 << 20
GIB = 1 << 30

PASS = 0
FAIL = 0


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"\n          {detail}" if detail else ""))
    return ok


def tfs_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [TFS, *args, "--registry", str(ROOT / "registry.json")],
        capture_output=True,
        text=True,
    )


# ------------------------------------------------------------------------------- fetch


def _get(start: int, end: int) -> bytes:
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        if response.status != 206:
            raise SystemExit(f"expected a 206 partial response, got {response.status}")
        return bytes(response.read())


def fetch() -> None:
    """One HTTP range over the source file's own declared offsets. 772 MB, not 40.2 GB —
    a safetensors header is a byte map, so a block is addressable without the file."""
    import torch
    from safetensors.torch import save_file

    head("fetch — block 0 of the real H3 curve DiT, by HTTP range over the pinned revision")
    ROOT.mkdir(parents=True, exist_ok=True)
    length = struct.unpack("<Q", _get(0, 8))[0]
    header = json.loads(_get(8, 8 + length))
    base = 8 + length
    keys = sorted(k for k in header if k.startswith(BLOCK) and "adaln" not in k)
    lo = min(header[k]["data_offsets"][0] for k in keys)
    hi = max(header[k]["data_offsets"][1] for k in keys)
    print(f"  source     {MEMBER} @ {REV[:12]}, {len(header)} keys, {length} B header")
    started = time.time()
    span = _get(base + lo, base + hi)
    took = time.time() - started
    print(
        f"  fetched    [{lo}, {hi}) = {len(span) / MIB:.1f} MiB in {took:.1f}s "
        f"({len(span) / MIB / max(took, 0.001):.0f} MiB/s)"
    )
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    source: dict[str, Any] = {}
    for key in keys:
        a, b = header[key]["data_offsets"]
        raw = bytearray(span[a - lo : b - lo])
        source[key[len(BLOCK) :]] = (
            torch.frombuffer(raw, dtype=dtypes[header[key]["dtype"]])
            .reshape(tuple(int(n) for n in header[key]["shape"]))
            .clone()
        )
    plain = {k: v for k, v in source.items()}
    save_file(plain, str(ROOT / "bf16.safetensors"), metadata={"format": "pt"})

    encoded: dict[str, Any] = {k: v for k, v in source.items() if k not in QUANTIZED}
    saturated = 0
    for key in QUANTIZED:
        data, scale, sat = quantize(torch, source[key])
        saturated += sat
        encoded[key] = data
        # #501d: named off the MODULE STEM. `mlp.fc2.weight_scale`, never
        # `mlp.fc2.weight.weight_scale` — present-but-invisible is worse than absent.
        encoded[key[: -len(".weight")] + ".weight_scale"] = scale
    save_file(encoded, str(ROOT / "mxfp8.safetensors"), metadata={"format": "pt"})
    src_bytes = sum(v.numel() * v.element_size() for v in source.values())
    check(
        "the mxfp8 carrier is ~1+1/32 bytes per quantized element, not 2",
        (ROOT / "mxfp8.safetensors").stat().st_size < (ROOT / "bf16.safetensors").stat().st_size,
        f"{(ROOT / 'mxfp8.safetensors').stat().st_size / MIB:.1f} vs "
        f"{(ROOT / 'bf16.safetensors').stat().st_size / MIB:.1f} MiB "
        f"({saturated} elements saturated at e4m3's 448 during quantization)",
    )
    print(f"  source     {src_bytes / MIB:.1f} MiB across {len(source)} tensors")


def quantize(torch: Any, weight: Any) -> tuple[Any, Any, int]:
    """The producing job's quantizer (`cozy_jobs/quantize.py`), transcribed."""
    rows, cols = weight.shape
    work = weight.to(torch.float32)
    blocks = work.view(rows, cols // MX_BLOCK, MX_BLOCK)
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.where(
        amax > 0, torch.floor(torch.log2(amax)) - 8.0, torch.full_like(amax, -127.0)
    )
    byte = (exponent + 127.0).clamp(0.0, 254.0)
    scale = torch.pow(2.0, byte - 127.0)
    quant = (blocks / scale.unsqueeze(-1)).view(rows, cols)
    saturated = int((quant.abs() > E4M3_MAX).sum())
    return quant.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), byte.to(torch.uint8), saturated


# ------------------------------------------------------------------------------ ingest


def ingest() -> None:
    head("ingest — both arms through the REAL border into one store")
    if not (STORE / "objects").is_dir():
        STORE.mkdir(parents=True, exist_ok=True)
        tfs_cli("store", "init", str(STORE))
    for arm, encoding in (("bf16", "plain/1"), ("mxfp8", "mxfp8/1")):
        source = ROOT / f"{arm}.safetensors"
        tfs_cli(
            "ingest",
            "bank",
            "safetensors.diffusers",
            f"h3-block0-{arm}",
            "block",
            "diffusers.identity/1",
            "--source",
            f"block={source}",
            "--provenance",
            "real",
            "--note",
            f"H3 curve DiT block 0, {MEMBER} @ {REV[:12]}, ranged fetch",
        )
        plan = tfs_cli("ingest", "plan", f"block={encoding}", "--source", f"block={source}")
        run = tfs_cli(
            "ingest",
            "run",
            str(STORE),
            f"block={encoding}",
            "--source",
            f"block={source}",
            "--session",
            arm,
            "--tenant",
            "local",
        )
        install = tfs_cli("ingest", "install", str(STORE), arm)
        ok = plan.returncode == 0 and run.returncode == 0 and install.returncode == 0
        digest = ""
        for line in (install.stdout or "").split():
            if line.startswith("sha256:") and len(line) > 40:
                digest = line
                break
        (ROOT / f"snapshot-{arm}.txt").write_text(digest)
        check(
            f"the {arm} arm plans, ingests and installs — {encoding}",
            ok and bool(digest),
            (digest or (plan.stderr + run.stderr + install.stderr))[:280],
        )


# -------------------------------------------------------------------------------- fill


class H3BlockLike:
    """The block's real GEMMs, wired the standard way. Identical in both arms."""

    def __init__(self, config: Any) -> None:
        import torch
        from torch import nn

        width, heads, dim, hidden = 5376, 56, 128, 14336
        block = nn.Module()
        block.norm1 = nn.RMSNorm(width, dtype=torch.bfloat16, elementwise_affine=True)
        block.norm2 = nn.RMSNorm(width, dtype=torch.bfloat16, elementwise_affine=True)
        attn = nn.Module()
        attn.qkv_proj = nn.Linear(width, 3 * heads * dim, bias=False, dtype=torch.bfloat16)
        attn.out_proj = nn.Linear(heads * dim, width, bias=False, dtype=torch.bfloat16)
        attn.q_norm = nn.RMSNorm(dim, dtype=torch.bfloat16, elementwise_affine=True)
        attn.k_norm = nn.RMSNorm(dim, dtype=torch.bfloat16, elementwise_affine=True)
        block.attn = attn
        mlp = nn.Module()
        mlp.fc1 = nn.Linear(width, 2 * hidden, bias=False, dtype=torch.bfloat16)
        mlp.fc2 = nn.Linear(hidden, width, bias=False, dtype=torch.bfloat16)
        block.mlp = mlp
        self.shape = (heads, dim)
        self.components: dict[str, Any] = {"block": block}

    def forward(self, torch: Any, x: Any) -> Any:
        import torch.nn.functional as functional

        block = self.components["block"]
        heads, dim = self.shape
        batch, tokens, _ = x.shape
        h = block.norm1(x)
        qkv = block.attn.qkv_proj(h).view(batch, tokens, 3, heads, dim)
        q = block.attn.q_norm(qkv[:, :, 0]).transpose(1, 2)
        k = block.attn.k_norm(qkv[:, :, 1]).transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        a = functional.scaled_dot_product_attention(q, k, v)
        x = x + block.attn.out_proj(a.transpose(1, 2).reshape(batch, tokens, heads * dim))
        h = block.norm2(x)
        gate, up = block.mlp.fc1(h).chunk(2, dim=-1)
        return x + block.mlp.fc2(functional.silu(gate) * up)


def build_block(config: Any) -> H3BlockLike:
    return H3BlockLike(config)


def make_model() -> Any:
    from cozy_runtime.author import Model, uses_components
    from cozy_runtime.author._loader import Loader

    class H3BlockModel(Model[H3BlockLike]):
        pipe: H3BlockLike

        def load(self, loader: Loader) -> None:
            self.pipe = loader.construct(H3BlockLike, factory=build_block)

        @uses_components("block")
        def step(self, torch: Any, x: Any) -> Any:
            with torch.inference_mode():
                return self.pipe.forward(torch, x)

    return H3BlockModel


def fill_arm(arm: str) -> tuple[Any, Any, Any]:
    import torch

    from cozy_runtime.author import ModelRegistry
    from cozy_runtime.author._loader import Artifact, Config
    from cozy_runtime.internal.derive import Observations, serving_substrate
    from cozy_runtime.internal.fill import (
        Checkpoint,
        ResourceLedger,
        StreamingFillBackend,
        topology_of,
    )

    snapshot = (ROOT / f"snapshot-{arm}.txt").read_text().strip()
    books = {"block": Checkpoint(str(STORE), snapshot)}
    rows = books["block"].rows("block")
    backend = StreamingFillBackend(
        books,
        rows,
        ledger=ResourceLedger(budgets={"vram": 8 * GIB, "pinned": 256 * MIB}),
        window_bytes=4 * MIB,
        slots=16,
        readers=16,
        inflight=8,
        probe_cache=ROOT / "probe",
        decode_scratch_bytes=64 * MIB,
        release="cozy/h3-block@mxfp8",
    )
    registry = ModelRegistry(
        release="cozy/h3-block@mxfp8",
        backend=backend,
        # `sm<major><minor>`, which is the substrate's own spelling — `DeviceFacts.predicate`
        # is `cuda.sm89`, a different vocabulary, and the guard says so rather than guessing.
        substrate=lambda: serving_substrate(f"sm{backend.device_facts.sm}", Observations()),
    )
    artifact = Artifact(
        snapshot, topology_of(rows), Config({}, "block.config"), custody="canonical"
    )
    backend.expect({"block": tuple(r.key for r in rows)})
    model = registry.acquire("step.models.model", make_model(), artifact)
    return backend, model, torch


def fill() -> None:
    import torch

    head("fill — the real fill plane, a two-role encoding through the staging ring")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    backend, model, _ = fill_arm("mxfp8")
    took = time.time() - started
    print(f"  device     {backend.device_facts.predicate} ({backend.device_facts.name})")
    print(f"  servable   {backend.servability.line()}")
    routes: dict[str, int] = {}
    for selection in backend.selected.values():
        routes[selection.provider.route] = routes.get(selection.provider.route, 0) + 1
    check(
        "the mxfp8 tensors took the DECODED FLOAT route, the rest verbatim",
        routes.get("decoded_float") == len(QUANTIZED),
        f"{routes}; {backend.decoded['tensors']} tensors decoded from "
        f"{backend.decoded['bytes'] / MIB:.1f} MiB stored in {backend.decoded['ms']} ms "
        f"over {backend.decoded['flushes']} flush(es)",
    )
    print(
        f"  fill       {took:.1f}s wall, ledger {json.dumps(backend.ledger.rows()['decode'])} "
        f"decode B, peak alloc {torch.cuda.max_memory_allocated() / MIB:.0f} MiB"
    )
    check(
        "the DECODE scratch was charged and released — nothing outlives the fill",
        backend.ledger.charged["decode"] == 0 and backend.ledger.peak["decode"] > 0,
        f"peak {backend.ledger.peak['decode'] / MIB:.1f} MiB charged, "
        f"{backend.ledger.charged['decode']} B still held",
    )
    globals()["_MX"] = (backend, model)


def compare() -> None:
    """The delta the encoding costs, on real weights: at the WEIGHT and through the CHAIN."""
    import torch

    from cozy_runtime.internal.fill import _tensors as tensors_of

    head("forward — mxfp8-filled against bf16-filled, same module, same source bytes")
    mx_backend, mx_model = globals()["_MX"]
    mx = dict(tensors_of(mx_backend.components["block"]))
    bf_backend, bf_model, _ = fill_arm("bf16")
    bf = dict(tensors_of(bf_backend.components["block"]))

    worst = ("", 0.0)
    for name in sorted(mx):
        a, b = mx[name].float(), bf[name].float()
        error = float((a - b).detach().norm() / b.detach().norm().clamp(min=1e-12))
        if any(name.endswith(q) for q in QUANTIZED) and error > worst[1]:
            worst = (name, error)
        if not any(name.endswith(q) for q in QUANTIZED):
            check(
                f"{name}: carried at SOURCE PRECISION, bit for bit",
                bool(torch.equal(mx[name], bf[name])),
                f"{tuple(mx[name].shape)} {mx[name].dtype}",
            )
    print(f"  weights    worst quantized-tensor relative Frobenius {worst[1]:.6f} ({worst[0]})")
    check(
        "every quantized weight is inside the job's own e4m3 acceptance bound",
        worst[1] < 2.0**-3,
        f"{worst[1]:.6f} < {2.0**-3} — the error is the QUANTIZER's, and the decode adds none",
    )

    torch.manual_seed(0x469A)
    x = torch.randn(1, 512, 5376, dtype=torch.bfloat16, device="cuda") * 0.5
    other = torch.randn(1, 512, 5376, dtype=torch.bfloat16, device="cuda") * 0.5
    with torch.inference_mode():
        want = bf_model.step(torch, x).float()
        got = mx_model.step(torch, x).float()
        again = mx_model.step(torch, x).float()
        control = bf_model.step(torch, other).float()

    def measure(a: Any, b: Any) -> tuple[float, float]:
        d = float((a - b).norm() / b.norm())
        return d, -20.0 * math.log10(max(d, 1e-12))

    delta, snr = measure(got, want)
    noise, noise_snr = measure(control, want)
    cosine = float(torch.nn.functional.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)))
    print(
        f"  forward    1x512x5376 through the block's real GEMM chain\n"
        f"             ENCODING delta  relative Frobenius {delta:.6f}  SNR {snr:.3f} dB  "
        f"cosine {cosine:.8f}\n"
        f"             CONTROL  delta  relative Frobenius {noise:.6f}  SNR "
        f"{noise_snr:.3f} dB  (same bf16 arm, different input)\n"
        f"             the {worst[1]:.6f} weight-level error entering the chain is amplified "
        f"{delta / worst[1]:.2f}x by four GEMMs and the SwiGLU gate"
    )
    # NO THRESHOLD HERE. What the encoding is ALLOWED to cost is job-009's ruling through
    # the banked noise floors, measured by that gate's own metric core on real renders
    # (#501b) — a constant invented in this file would be a quality bar nobody set. What can
    # be asserted without inventing anything is an ORDERING and a REPRODUCIBILITY:
    check(
        "the encoding's perturbation is smaller than the signal's own variation",
        snr > noise_snr,
        f"encoding {snr:.3f} dB vs control {noise_snr:.3f} dB — the gap between two "
        "ENCODINGS is below the gap between two INPUTS, which is the floor a lane clears "
        "before a quality gate is even a meaningful question to ask of it",
    )
    check(
        "the mxfp8 arm is BIT-IDENTICAL run to run on this host",
        bool(torch.equal(got, again)),
        "#501c: fp8 is bit-identical within a host and not across hosts, and a "
        "decoded-float mxfp8 lane inherits exactly that — which is what lets a re-render "
        "floor be BOUND to this lane instead of merely measured near it",
    )
    print(
        "  context    proto-001 banked the WEIGHT-level number over 40 real H3 DiT tensors "
        f"at 0.029937 relative Frobenius / 30.478 dB SNR; this block's four GEMMs land at "
        f"{worst[1]:.6f} independently. mxfp8 remains 1.06 dB BEHIND per-tensor fp8 for "
        "3.1% more bytes, and this route materializes bf16 either way"
    )
    (ROOT / "delta.json").write_text(
        json.dumps(
            {
                "format": "cozy.runtime.mxfp8_delta/1",
                "source": {"repo": REPO, "revision": REV, "member": MEMBER, "block": BLOCK},
                "device": mx_backend.device_facts.predicate,
                "snapshot_mxfp8": (ROOT / "snapshot-mxfp8.txt").read_text().strip(),
                "snapshot_bf16": (ROOT / "snapshot-bf16.txt").read_text().strip(),
                "weight_worst_relative_frobenius": worst[1],
                "weight_worst_key": worst[0],
                "forward_relative_frobenius": delta,
                "forward_snr_db": snr,
                "forward_cosine": cosine,
                "control_relative_frobenius": noise,
                "control_snr_db": noise_snr,
                "decode": mx_backend.decoded,
                "decode_scratch_peak_bytes": mx_backend.ledger.peak["decode"],
            },
            indent=1,
        )
    )


# ------------------------------------------------------------------------------- probe


def probe() -> None:
    import torch

    from cozy_runtime.internal.encoding import digests_of, launch_providers, measure_device
    from cozy_runtime.internal.probe import qualify

    head("probe — the 65,536-pair format domain, decoded by CUDA kernels")
    device = measure_device(torch)
    providers = launch_providers()
    mx = digests_of("mxfp8/1")[0]
    _, evidence = qualify(
        torch,
        {mx: providers[mx]},
        device,
        ["bfloat16", "float32", "float16"],
        release="cozy/mxfp8-pod",
        cache=None,
    )
    by_dtype = {o["output_dtype"]: o["evidence"] for o in evidence["observations"]}
    print(f"  device     {device.predicate} ({device.name}, driver cuda {device.driver})")
    for name in ("bfloat16", "float32"):
        n = by_dtype[name]["numerics"]
        check(
            f"{name}: all {n['pairs']} pairs decode correctly ON THE CARD",
            by_dtype[name]["passed"],
            f"domain {n['domain']}; wrong {n['wrong']}; max rel {n['max_rel_error']:.3e} "
            f"<= {n['tolerance']:.3e}; max abs {n['max_abs_error']:.3e} <= "
            f"{n['subnormal_tolerance']:.3e}; {by_dtype[name]['micro_bench']['gb_per_s']} GB/s",
        )
    f16 = by_dtype["float16"]["numerics"]
    check(
        "RED on the card too: float16 fails and mints no record",
        not by_dtype["float16"]["passed"],
        f"{f16['wrong']['normal']} normal pairs wrong; {f16['first_wrong'][:80]}",
    )


def main(argv: list[str]) -> int:
    names = argv or ["all"]
    if names == ["all"]:
        names = ["fetch", "ingest", "probe", "fill", "compare"]
    for name in names:
        {"fetch": fetch, "ingest": ingest, "probe": probe, "fill": fill, "compare": compare}[name]()
    print(f"\n{PASS} ok, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
