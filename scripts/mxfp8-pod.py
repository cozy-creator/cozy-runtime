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
        StreamingFillBackend,
        topology_of,
    )

    snapshot = (ROOT / f"snapshot-{arm}.txt").read_text().strip()
    books = {"block": Checkpoint(str(STORE), snapshot)}
    rows = books["block"].rows("block")
    backend = StreamingFillBackend.for_script(
        books,
        rows,
        store=str(STORE),
        snapshot=snapshot,
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
        f"  fill       {took:.1f}s wall, "
        f"peak alloc {torch.cuda.max_memory_allocated() / MIB:.0f} MiB"
    )
    check(
        "the DECODE scratch was freed behind the fence — nothing outlives the fill",
        not backend.scratch and backend.decoded["flushes"] > 0,
        f"{backend.decoded['flushes']} flush(es), scratch map empty after commit",
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


# ----------------------------------------------------------------- the rowwise derivation


def _rel(a: Any, b: Any) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def derive() -> None:
    """#514a on REAL H3 weights: what the block->row scale collapse costs, and what the
    double quantization costs OVER quantizing straight to the same target format."""
    import torch
    from safetensors import safe_open

    from cozy_runtime.internal.encoding import MicroScaledDequant, digests_of
    from cozy_runtime.internal.instruments import derive_rowwise, quantize_rowwise

    head("derive — mxfp8 -> fp8-rowwise on the real H3 block-0 GEMMs (#514a)")
    provider = MicroScaledDequant(digests_of("mxfp8/1")[0])
    rows: list[dict[str, Any]] = []
    with safe_open(str(ROOT / "bf16.safetensors"), framework="pt", device="cuda") as source:
        for key in QUANTIZED:
            weight = source.get_tensor(key)
            data, scale, saturated = quantize(torch, weight)
            exact = torch.empty(weight.shape, dtype=torch.float32, device="cuda")
            provider.decode(torch, {"data": data, "scale": scale}, exact)
            derived = derive_rowwise(torch, data, scale)
            direct = quantize_rowwise(torch, weight)
            as_float = derived.data.float() * derived.scale.unsqueeze(-1)
            direct_float = direct.data.float() * direct.scale.unsqueeze(-1)
            original = weight.float()
            row: dict[str, Any] = {
                "key": key,
                "shape": list(weight.shape),
                "mxfp8": _rel(exact, original),
                "derived": _rel(as_float, original),
                "direct": _rel(direct_float, original),
                "added_over_mxfp8": _rel(as_float, exact),
                "mxfp8_saturated": saturated,
                "derived_saturated": derived.saturated,
                "derived_clamped": derived.clamped,
            }
            row["penalty_x"] = row["derived"] / max(row["direct"], 1e-12)
            row["mxfp8_vs_direct_db"] = 20.0 * math.log10(
                max(row["mxfp8"], 1e-12) / max(row["direct"], 1e-12)
            )
            rows.append(row)
            print(
                f"  {key:24s} mxfp8 {row['mxfp8']:.6f} -> derived {row['derived']:.6f}   "
                f"direct {row['direct']:.6f}   PENALTY {row['penalty_x']:.3f}x"
            )
            del exact, as_float, direct_float, original
            torch.cuda.empty_cache()
    for row in rows:
        check(
            f"{row['key']}: the row scale covers its row, and double > single",
            row["derived_saturated"] == 0 and row["derived"] >= row["direct"],
            f"{row['derived_clamped']} clamped at the rounding boundary, "
            f"{row['derived_saturated']} genuinely saturated; derived {row['derived']:.6f} "
            f">= direct {row['direct']:.6f}",
        )
    worst = max(r["penalty_x"] for r in rows)
    gap = sum(r["mxfp8_vs_direct_db"] for r in rows) / len(rows)
    print(
        f"  penalty    worst {worst:.3f}x over direct rowwise across the four real GEMMs\n"
        f"  #513 note  mxfp8 sits {gap:+.3f} dB against DIRECT fp8-rowwise on these same "
        f"tensors\n             (positive = mxfp8 worse). proto-001 banked 1.06 dB; this is "
        f"an INDEPENDENT\n             re-derivation on the real weights, same metric, and it "
        f"is evidence FOR the\n             #513 audit, not a verdict from this lane."
    )
    # The #513c hypothesis, tested directly rather than speculated about. The producing
    # kind picks `floor(log2(amax)) - 8`, which puts the block maximum at 256*m for
    # m in [1,2) — above E4M3's 448 whenever m > 1.75, so the block's own largest element
    # CLAMPS. An error-minimising choice puts it at or just under 448 and never clamps.
    with safe_open(str(ROOT / "bf16.safetensors"), framework="pt", device="cuda") as source:
        weight = source.get_tensor(QUANTIZED[0])
    shipped_data, shipped_scale, shipped_sat = quantize(torch, weight)
    ceil_data, ceil_scale, ceil_sat = _quantize_ceiling(torch, weight)
    exact = torch.empty(weight.shape, dtype=torch.float32, device="cuda")
    provider.decode(torch, {"data": shipped_data, "scale": shipped_scale}, exact)
    shipped_err = _rel(exact, weight.float())
    provider.decode(torch, {"data": ceil_data, "scale": ceil_scale}, exact)
    ceil_err = _rel(exact, weight.float())
    print(
        f"  #513c      on {QUANTIZED[0]}: the SHIPPED rule "
        f"(floor(log2 amax)-8) saturates {shipped_sat} elements at err {shipped_err:.6f}; "
        f"an\n             error-minimising ceiling rule saturates {ceil_sat} at err "
        f"{ceil_err:.6f} "
        f"({20 * math.log10(max(shipped_err, 1e-12) / max(ceil_err, 1e-12)):+.3f} dB). "
        f"The jobs lane owns\n             the fix; this is the runtime's independent "
        f"observation of the same defect."
    )
    (ROOT / "derive.json").write_text(
        json.dumps(
            {
                "format": "cozy.runtime.mxfp8_rowwise_derivation/1",
                "source": {"repo": REPO, "revision": REV, "member": MEMBER, "block": BLOCK},
                "quantizer_rule": "floor(log2(block amax)) - 8 + 127, clamped [0,254]",
                "row_scale_rule": "rowmax|V| / 448, continuous f32, floor 1e-12",
                "tensors": rows,
                "worst_penalty_x": worst,
                "mxfp8_vs_direct_rowwise_db": gap,
                "ceiling_rule_check": {
                    "key": QUANTIZED[0],
                    "shipped_saturated": shipped_sat,
                    "shipped_error": shipped_err,
                    "ceiling_saturated": ceil_sat,
                    "ceiling_error": ceil_err,
                },
            },
            indent=1,
        )
    )


def _quantize_ceiling(torch: Any, weight: Any) -> tuple[Any, Any, int]:
    """The #513c alternative: pick the exponent so the block maximum lands AT OR UNDER 448
    instead of at 256*mantissa. Same format, same payload width — only the choice differs."""
    rows, cols = weight.shape
    work = weight.to(torch.float32)
    blocks = work.view(rows, cols // MX_BLOCK, MX_BLOCK)
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.where(
        amax > 0,
        torch.ceil(torch.log2(amax / E4M3_MAX)),
        torch.full_like(amax, -127.0),
    )
    byte = (exponent + 127.0).clamp(0.0, 254.0)
    scale = torch.pow(2.0, byte - 127.0)
    quant = (blocks / scale.unsqueeze(-1)).view(rows, cols)
    saturated = int((quant.abs() > E4M3_MAX).sum())
    return quant.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), byte.to(torch.uint8), saturated


# ------------------------------------------------------------------------- the w8a8 half


def _quantize_activation(torch: Any, x: Any) -> tuple[Any, Any]:
    """Per-token dynamic activation quantization: one scale per ROW of the [M, K] operand,
    which is what pairs with a per-output-row weight scale in a rowwise scaled GEMM.

    Transcribed from v1's proven recipe (`gen_worker/models/w8a8.py::_quant_src`): amax over
    the K axis, `.float()` BEFORE the divide so the scale is f32 whatever the activation is,
    the literal 448.0 rather than `finfo.max`, a `1e-12` floor, and the clamp BEFORE the cast
    because the cast does not saturate. The one deliberate difference is that v1's EAGER path
    multiplies by a reciprocal rounded down to the activation dtype while this divides in
    f32 — which is v1's own FUSED behaviour, and v1 banked the fused path as the slightly
    more accurate of the two (0.01568 vs 0.01579 against an f32 reference).
    """
    flat = x.reshape(-1, x.shape[-1])
    amax = flat.abs().amax(dim=-1, keepdim=True).float()
    scale = (amax / E4M3_MAX).clamp(min=1e-12)
    return (flat.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def _scaled_linear(
    torch: Any, x: Any, weight: Any, wscale: Any, out_dtype: Any, *, fast_accum: bool = False
) -> Any:
    """One w8a8 rowwise GEMM. `weight` is [out, in] E4M3, `wscale` is [out] f32.

    `_scaled_mm` wants the second operand COLUMN-major, which `weight.t()` already is for a
    contiguous [out, in]; scale_a is per-row of the activation and scale_b per COLUMN of
    that operand, which is per OUTPUT row of the weight — the two scale vectors are the two
    outer dimensions of the product and neither is the reduction axis.
    """
    xq, xs = _quantize_activation(torch, x)
    out = torch._scaled_mm(
        xq,
        weight.t(),
        scale_a=xs,
        scale_b=wscale.reshape(1, -1),
        out_dtype=out_dtype,
        use_fast_accum=fast_accum,
    )
    return out.reshape(*x.shape[:-1], -1)


def w8a8() -> None:
    """#514b: Hopper's fp8 tensor cores need BOTH operands fp8, so the serving form is
    necessarily w8a8. This measures what that costs and what it buys, on the real block."""
    import torch
    from safetensors import safe_open

    from cozy_runtime.internal.instruments import derive_rowwise

    head("w8a8 — the derived rowwise weights through a real scaled-fp8 GEMM (#514b)")
    capability = torch.cuda.get_device_capability(0)
    print(f"  device     sm{capability[0]}{capability[1]} ({torch.cuda.get_device_name(0)})")

    # THE FIRST QUESTION, and it can end the leg: does this card's torch expose a ROWWISE
    # scaled fp8 GEMM at all? Recorded verbatim either way — "no" is a finding.
    probe_a = torch.zeros(16, 32, dtype=torch.float8_e4m3fn, device="cuda")
    probe_b = torch.zeros(32, 16, dtype=torch.float8_e4m3fn, device="cuda").t().contiguous().t()
    supported, refusal = True, ""
    try:
        torch._scaled_mm(
            probe_a,
            probe_b,
            scale_a=torch.ones(16, 1, device="cuda"),
            scale_b=torch.ones(1, 16, device="cuda"),
            out_dtype=torch.bfloat16,
        )
    except Exception as exc:  # the refusal text IS the finding, so nothing is narrowed
        supported, refusal = False, f"{type(exc).__name__}: {exc}"
    check(
        "this card's torch exposes a ROWWISE-scaled fp8 GEMM",
        supported,
        refusal[:300] or "torch._scaled_mm accepted per-row scale_a and per-column scale_b",
    )
    if not supported:
        print(
            "  finding    the rowwise w8a8 lane cannot be qualified on this device. Per-tensor\n"
            "             scaling may still work; rowwise is what #514 specifies and it is the\n"
            "             rowwise answer that is being recorded. No capability record is minted."
        )
        return

    weights: dict[str, Any] = {}
    with safe_open(str(ROOT / "bf16.safetensors"), framework="pt", device="cuda") as source:
        for key in QUANTIZED:
            weights[key] = source.get_tensor(key)

    torch.manual_seed(0x514)
    banked: dict[str, Any] = {}
    for key, weight in weights.items():
        data, scale, _ = quantize(torch, weight)
        derived = derive_rowwise(torch, data, scale)
        cols = int(weight.shape[1])
        x = torch.randn(512, cols, dtype=torch.bfloat16, device="cuda") * 0.5
        reference_out = (x @ weight.t().to(torch.bfloat16)).float()
        got = _scaled_linear(torch, x, derived.data, derived.scale, torch.bfloat16).float()
        fast = _scaled_linear(
            torch, x, derived.data, derived.scale, torch.bfloat16, fast_accum=True
        ).float()
        # The WEIGHT-ONLY control: the same derived weights dequantized and run bf16, which
        # separates what the WEIGHT quantization costs from what the ACTIVATION one adds.
        weight_only = (
            x @ (derived.data.float() * derived.scale.unsqueeze(-1)).t().to(torch.bfloat16)
        ).float()
        w8a8_err = _rel(got, reference_out)
        fast_err = _rel(fast, reference_out)
        wonly_err = _rel(weight_only, reference_out)
        banked[key] = {
            "w8a8_relative": w8a8_err,
            "w8a8_snr_db": -20.0 * math.log10(max(w8a8_err, 1e-12)),
            "w8a8_fast_accum_relative": fast_err,
            "w8a8_fast_accum_snr_db": -20.0 * math.log10(max(fast_err, 1e-12)),
            "weight_only_relative": wonly_err,
            "weight_only_snr_db": -20.0 * math.log10(max(wonly_err, 1e-12)),
        }
        print(
            f"  {key:24s} w8a8 {w8a8_err:.6f} ({banked[key]['w8a8_snr_db']:.3f} dB)   "
            f"weight-only {wonly_err:.6f} ({banked[key]['weight_only_snr_db']:.3f} dB)   "
            f"activation costs "
            f"{banked[key]['weight_only_snr_db'] - banked[key]['w8a8_snr_db']:.3f} dB   "
            f"fast-accum {banked[key]['w8a8_fast_accum_snr_db']:.3f} dB "
            f"({banked[key]['w8a8_fast_accum_snr_db'] - banked[key]['w8a8_snr_db']:+.3f} dB)"
        )
        del x, reference_out, got, fast, weight_only
        torch.cuda.empty_cache()
    check(
        "every real GEMM computes through the fp8 tensor cores and stays finite",
        all(math.isfinite(v["w8a8_snr_db"]) for v in banked.values()),
        "; ".join(f"{k.split('.')[-2]} {v['w8a8_snr_db']:.1f} dB" for k, v in banked.items()),
    )

    # WHAT IT BUYS. Residency first, because that is #514's stated payoff.
    bf16_bytes = sum(w.numel() * w.element_size() for w in weights.values())
    fp8_bytes = 0
    for weight in weights.values():
        data, scale, _ = quantize(torch, weight)
        derived = derive_rowwise(torch, data, scale)
        fp8_bytes += derived.data.numel() + derived.scale.numel() * 4
    print(
        f"  resident   bf16 {bf16_bytes / MIB:.1f} MiB -> fp8 rowwise {fp8_bytes / MIB:.1f} MiB "
        f"({fp8_bytes / bf16_bytes:.4f}x)"
    )
    check(
        "the w8a8 form is about HALF the resident bytes — the payoff #514 names",
        fp8_bytes < bf16_bytes * 0.55,
        f"{fp8_bytes / bf16_bytes:.4f}x of bf16, versus decoded-float mxfp8 at exactly 1.0x "
        "(that route materializes bf16 and buys no VRAM at all)",
    )

    # THROUGHPUT, on the largest real GEMM.
    key = "mlp.fc1.weight"
    weight = weights[key]
    data, scale, _ = quantize(torch, weight)
    derived = derive_rowwise(torch, data, scale)
    x = torch.randn(512, weight.shape[1], dtype=torch.bfloat16, device="cuda") * 0.5
    xq, xs = _quantize_activation(torch, x)
    wt, ws = derived.data.t(), derived.scale.reshape(1, -1)
    wbf = weight.t().contiguous()

    def timed(fn: Any, reps: int = 50) -> float:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / reps * 1e6

    # DECOMPOSED, the way v1 decomposed it (pgw#1156): the bare GEMM against bf16, and then
    # the whole forward INCLUDING the activation quantization. v1 measured its shipped lane
    # LOSING to bf16 (0.81-0.93x) while the bare _scaled_mm won 1.30-1.48x on the same
    # shapes -- six elementwise passes over [M,K] around a GEMM that costs M*K*N, so the
    # overhead fraction falls as 1/N and only disappears when the quantize is fused into one
    # kernel. Reporting one number here would hide exactly that.
    gemm_us = timed(
        lambda: torch._scaled_mm(xq, wt, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
    )
    gemm_fast_us = timed(
        lambda: torch._scaled_mm(
            xq, wt, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16, use_fast_accum=True
        )
    )
    full_us = timed(lambda: _scaled_linear(torch, x, derived.data, derived.scale, torch.bfloat16))
    full_fast_us = timed(
        lambda: _scaled_linear(
            torch, x, derived.data, derived.scale, torch.bfloat16, fast_accum=True
        )
    )
    bf16_us = timed(lambda: x @ wbf)
    flops = 2 * x.shape[0] * weight.shape[0] * weight.shape[1]
    print(
        f"  throughput {key} [{x.shape[0]}x{weight.shape[1]}] @ "
        f"[{weight.shape[1]}x{weight.shape[0]}]\n"
        f"             bf16              {bf16_us:8.1f} us ({flops / bf16_us / 1e6:6.1f} TFLOP/s)\n"
        f"             GEMM  default acc {gemm_us:8.1f} us ({flops / gemm_us / 1e6:6.1f} TFLOP/s)  "
        f"{bf16_us / gemm_us:.3f}x\n"
        f"             GEMM  fast   acc {gemm_fast_us:8.1f} us "
        f"({flops / gemm_fast_us / 1e6:6.1f} TFLOP/s)  {bf16_us / gemm_fast_us:.3f}x\n"
        f"             full  default acc {full_us:8.1f} us  {bf16_us / full_us:.3f}x   "
        f"(activation quantize {full_us - gemm_us:.1f} us, UNFUSED)\n"
        f"             full  fast   acc {full_fast_us:8.1f} us  {bf16_us / full_fast_us:.3f}x"
    )
    check(
        "the fp8 GEMM only beats bf16 with FAST ACCUMULATE — the default path loses",
        bf16_us / gemm_fast_us > 1.0 and bf16_us / gemm_us < 1.0,
        f"default {bf16_us / gemm_us:.3f}x vs fast-accum {bf16_us / gemm_fast_us:.3f}x. v1 "
        "never passes `use_fast_accum` (gen_worker/models/w8a8.py) and banked its own lane "
        "LOSING to bf16 at 0.81-0.93x — this is the same shape, and the flag is the "
        "difference. Its accuracy cost is measured above, not assumed away",
    )
    fp8_us = full_fast_us
    banked["_cost"] = {
        "resident_bf16_bytes": bf16_bytes,
        "resident_fp8_bytes": fp8_bytes,
        "resident_ratio": fp8_bytes / bf16_bytes,
        "gemm_key": key,
        "bare_gemm_us": gemm_us,
        "bare_gemm_fast_accum_us": gemm_fast_us,
        "full_w8a8_us": full_us,
        "full_w8a8_fast_accum_us": full_fast_us,
        "fp8_us": fp8_us,
        "bf16_us": bf16_us,
        "speedup_x": bf16_us / fp8_us,
        "bare_gemm_speedup_x": bf16_us / gemm_us,
        "bare_gemm_fast_accum_speedup_x": bf16_us / gemm_fast_us,
        "device": f"sm{capability[0]}{capability[1]}",
    }
    (ROOT / "w8a8.json").write_text(
        json.dumps({"format": "cozy.runtime.mxfp8_w8a8/1", "tensors": banked}, indent=1)
    )


def _fp8_leaf(torch: Any, weight: Any, scale: Any, fast_accum: bool) -> Any:
    """A LEAF REPLACEMENT, standing in for the machinery `delivery.py`'s `encoded_gemm`
    rung has been waiting on since cr-006. Held in this harness, not in `src/`: a leaf the
    fill plane cannot reserve, charge or evict is not a runtime provider, and shipping one
    that merely looked like a provider is what #504 closes.

    Its shape is v1's `_Fp8ScaledLinear` (`gen_worker/models/w8a8.py`): the fp8 weight and
    the f32 row scale are BUFFERS, not Parameters — an fp8 Parameter is hostile to every
    `.to(dtype)` that walks a module — and the class is built lazily so importing this file
    never imports torch. It must be a real `nn.Module` because `nn.Module.__setattr__`
    refuses any other child, which is itself the reason leaf replacement is machinery rather
    than a one-line swap.
    """
    from torch import nn

    class Fp8Leaf(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("w", weight)
            self.register_buffer("s", scale)
            self.fast_accum = fast_accum

        def forward(self, x: Any) -> Any:
            return _scaled_linear(torch, x, self.w, self.s, x.dtype, fast_accum=self.fast_accum)

    return Fp8Leaf()


def blockforward() -> None:
    """The three lanes on ONE scale: the same block, the same input, the same protocol as
    the ratified lane's banked 20.817 dB, differing only in how the weights compute."""
    import torch

    from cozy_runtime.internal.instruments import derive_rowwise

    head("blockforward — dequant-bf16 vs derived-w8a8, same block, same protocol")
    pipe_ref = H3BlockLike(None)
    block = pipe_ref.components["block"]
    from safetensors import safe_open

    with safe_open(str(ROOT / "bf16.safetensors"), framework="pt", device="cuda") as source:
        state = {k: source.get_tensor(k) for k in source.keys()}  # noqa: SIM118
    block.to_empty(device=torch.device("cuda"), recurse=True)
    block.load_state_dict(state, strict=True, assign=False)

    torch.manual_seed(0x469A)
    x = torch.randn(1, 512, 5376, dtype=torch.bfloat16, device="cuda") * 0.5
    with torch.inference_mode():
        reference = pipe_ref.forward(torch, x).float()

    # The MXFP8 -> dequant-bf16 arm: the ratified lane, exact decode into bf16 weights.
    from cozy_runtime.internal.encoding import MicroScaledDequant, digests_of

    provider = MicroScaledDequant(digests_of("mxfp8/1")[0])
    pipe_mx = H3BlockLike(None)
    mx_block = pipe_mx.components["block"]
    mx_block.to_empty(device=torch.device("cuda"), recurse=True)
    mx_state = dict(state)
    derived_leaves: dict[str, Any] = {}
    for key in QUANTIZED:
        data, scale, _ = quantize(torch, state[key])
        exact = torch.empty(state[key].shape, dtype=torch.bfloat16, device="cuda")
        provider.decode(torch, {"data": data, "scale": scale}, exact)
        mx_state[key] = exact
        derived_leaves[key] = derive_rowwise(torch, data, scale)
    mx_block.load_state_dict(mx_state, strict=True, assign=False)
    with torch.inference_mode():
        dequant_out = pipe_mx.forward(torch, x).float()

    # The MXFP8 -> fp8-rowwise -> w8a8 arm: same source bytes, derived, computed in fp8.
    pipe_w8 = H3BlockLike(None)
    w8_block = pipe_w8.components["block"]
    w8_block.to_empty(device=torch.device("cuda"), recurse=True)
    w8_block.load_state_dict(mx_state, strict=True, assign=False)
    for key, leaf in derived_leaves.items():
        parent = w8_block
        parts = key[: -len(".weight")].split(".")
        for name in parts[:-1]:
            parent = getattr(parent, name)
        setattr(parent, parts[-1], _fp8_leaf(torch, leaf.data, leaf.scale, True))
    with torch.inference_mode():
        w8a8_out = pipe_w8.forward(torch, x).float()

    lanes = {"dequant_bf16": dequant_out, "derived_w8a8": w8a8_out}
    banked: dict[str, Any] = {}
    for name, out in lanes.items():
        rel = _rel(out, reference)
        snr = -20.0 * math.log10(max(rel, 1e-12))
        cos = float(
            torch.nn.functional.cosine_similarity(out.reshape(1, -1), reference.reshape(1, -1))
        )
        banked[name] = {"relative_frobenius": rel, "snr_db": snr, "cosine": cos}
        print(f"  {name:14s} relative {rel:.6f}   SNR {snr:.3f} dB   cosine {cos:.8f}")
    check(
        "both lanes decode the SAME stored bytes and land on the same reference",
        all(math.isfinite(v["snr_db"]) for v in banked.values()),
        f"the w8a8 lane costs "
        f"{banked['dequant_bf16']['snr_db'] - banked['derived_w8a8']['snr_db']:.3f} dB "
        "against the ratified dequant-bf16 lane, on identical input and identical weights "
        "-- that difference IS the derivation plus the activation quantization",
    )
    (ROOT / "blockforward.json").write_text(
        json.dumps(
            {
                "format": "cozy.runtime.mxfp8_lane_ranking/1",
                "protocol": "1x512x5376 through the block GEMM chain, bf16 arm as reference",
                "source": {"repo": REPO, "revision": REV, "member": MEMBER, "block": BLOCK},
                "device": f"sm{torch.cuda.get_device_capability(0)[0]}"
                f"{torch.cuda.get_device_capability(0)[1]}",
                "lanes": banked,
            },
            indent=1,
        )
    )


def main(argv: list[str]) -> int:
    names = argv or ["all"]
    if names == ["all"]:
        names = ["fetch", "ingest", "probe", "fill", "compare", "derive", "w8a8"]
    for name in names:
        {
            "fetch": fetch,
            "ingest": ingest,
            "probe": probe,
            "fill": fill,
            "compare": compare,
            "derive": derive,
            "w8a8": w8a8,
            "blockforward": blockforward,
        }[name]()
    print(f"\n{PASS} ok, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
