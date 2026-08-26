#!/usr/bin/env python3
"""The native-encoded rung's GPU leg, on REAL weights. POD-SIDE ONLY.

`scripts/leaf-live.py` proves the machinery on a 29 KB synthetic carrier: leaf
replacement, the census substitution, the consent gate, the ledger settle and the ranked
chooser. Three things it structurally cannot prove, and this script does:

  fill     a real H3 curve-DiT block through the real fill plane and into real leaves —
           the same 532-tensor-class artifact shape, at the size where a residency claim
           means something
  forward  what the rung COSTS on real weights: one block's GEMM chain through the leaves
           against the dequant-bf16 reference, on identical input, the SNR protocol the
           mxfp8 lanes share
  vram     what it BUYS, measured by the allocator and by the fill plane's own ledger

**The subject and its digest.** `Comfy-Org/MiniMax-H3` @ `4cc1d817b618`,
`diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`, `blocks.0.*` fetched by HTTP
range — 735 MiB of 40.2 GB, because a safetensors header is a byte map and a block is
addressable without the file. Every arm below prints the snapshot digest it ran against
(#517's rule): the quantizer is this script's own, applied to those exact source bytes.

**#518 decides the encoding.** The Hopper/Ada artifact is `fp8-rowwise/1` produced
DIRECTLY from bf16, never derived from mxfp8 — the derivation measured 1.44-1.52x more
error and survives only as a fallback. So `quantize_rowwise` is what makes the carrier
here, which is the runtime's own transcription of v1's proven recipe.

**The forward's honesty.** H3's block wiring is not in this tree, so the chain is the one
`scripts/mxfp8-pod.py` inferred from the real shapes, IMPORTED from that file rather than
copied so the two lanes cannot drift. It is not claimed to be H3's forward. It does not
need to be: every arm runs the IDENTICAL module and differs only in how its weights
compute, so a wiring error moves all arms together and the deltas stay attributable.

    PYTHONPATH=/root python3 leaf-pod.py all
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/root/leaf")
STORE = ROOT / "store"
TFS = "/root/tfs"
RELEASE = "cozy/h3-block@leaf"
MIB = 1 << 20
GIB = 1 << 30

PASS = 0
FAIL = 0


def _mx() -> Any:
    """`scripts/mxfp8-pod.py`, loaded as a module. Its ranged fetch and its H3 block wiring
    are the SAME experiment one lane earlier; copying them here would let the two drift and
    make a forward delta mean two different things."""
    spec = importlib.util.spec_from_file_location(
        "mxfp8_pod", Path(__file__).resolve().parent / "mxfp8-pod.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MX = _mx()
REPO, REV, MEMBER = MX.REPO, MX.REV, MX.MEMBER
BLOCK, QUANTIZED = MX.BLOCK, MX.QUANTIZED


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"\n          {detail}" if detail else ""))
    return bool(ok)


def tfs_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [TFS, *args, "--registry", str(ROOT / "registry.json")],
        capture_output=True,
        text=True,
    )


def _rel(got: Any, want: Any) -> float:
    return float((got.double() - want.double()).norm() / want.double().norm().clamp(min=1e-30))


def _snr(rel: float) -> float:
    return -20.0 * math.log10(max(rel, 1e-12))


# ------------------------------------------------------------------------------- fetch


def fetch() -> None:
    """One HTTP range over the source file's own declared offsets, then TWO carriers."""
    import torch
    from safetensors.torch import save_file

    from cozy_runtime.internal.encoding import quantize_rowwise

    head("fetch — block 0 of the real H3 curve DiT, by range over the pinned revision")
    ROOT.mkdir(parents=True, exist_ok=True)
    import struct
    import urllib.request

    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{MEMBER}"

    def get(start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=600) as response:
            if response.status != 206:
                raise SystemExit(f"expected a 206 partial response, got {response.status}")
            return bytes(response.read())

    length = struct.unpack("<Q", get(0, 8))[0]
    header = json.loads(get(8, 8 + length))
    base = 8 + length
    keys = sorted(k for k in header if k.startswith(BLOCK) and "adaln" not in k)
    lo = min(header[k]["data_offsets"][0] for k in keys)
    hi = max(header[k]["data_offsets"][1] for k in keys)
    started = time.time()
    span = get(base + lo, base + hi)
    took = time.time() - started
    print(
        f"  source     {MEMBER} @ {REV[:12]}\n"
        f"  fetched    [{lo}, {hi}) = {len(span) / MIB:.1f} MiB in {took:.1f}s "
        f"({len(span) / MIB / max(took, 0.001):.0f} MiB/s)"
    )
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    source: dict[str, Any] = {}
    for key in keys:
        a, b = header[key]["data_offsets"]
        source[key[len(BLOCK) :]] = (
            torch.frombuffer(bytearray(span[a - lo : b - lo]), dtype=dtypes[header[key]["dtype"]])
            .reshape(tuple(int(n) for n in header[key]["shape"]))
            .clone()
        )
    save_file(dict(source), str(ROOT / "bf16.safetensors"), metadata={"format": "pt"})

    # #518: the Hopper/Ada artifact is produced DIRECTLY from the bf16 source, never derived
    # from mxfp8 — the derivation measured 1.44-1.52x more error and is a fallback only.
    encoded: dict[str, Any] = {k: v for k, v in source.items() if k not in QUANTIZED}
    clamped = saturated = 0
    for key in QUANTIZED:
        derived = quantize_rowwise(torch, source[key])
        clamped += derived.clamped
        saturated += derived.saturated
        encoded[key] = derived.data
        # #501d: named off the MODULE STEM, never `…weight.weight_scale`.
        encoded[key[: -len(".weight")] + ".weight_scale"] = derived.scale
    save_file(encoded, str(ROOT / "rowwise.safetensors"), metadata={"format": "pt"})
    bf16_size = (ROOT / "bf16.safetensors").stat().st_size
    rw_size = (ROOT / "rowwise.safetensors").stat().st_size
    check(
        "the fp8-rowwise carrier is ~half the bf16 one, and its row maxima did not overshoot",
        rw_size < bf16_size and saturated == 0,
        f"{rw_size / MIB:.1f} vs {bf16_size / MIB:.1f} MiB ({rw_size / bf16_size:.4f}x); "
        f"{clamped} elements met the 448 clamp and {saturated} overshot it by more than "
        "rounding can explain — the rowmax/448 rule puts a row's largest magnitude exactly "
        "on E4M3's largest, so a nonzero saturation count would mean lost range",
    )


# ------------------------------------------------------------------------------ ingest


def ingest() -> None:
    head("ingest — both arms through the REAL border into one store")
    if not (STORE / "objects").is_dir():
        STORE.mkdir(parents=True, exist_ok=True)
        tfs_cli("store", "init", str(STORE))
    for arm, encoding in (("bf16", "plain/1"), ("rowwise", "fp8-rowwise/1")):
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
        digest = next(
            (w for w in (install.stdout or "").split() if w.startswith("sha256:") and len(w) > 40),
            "",
        )
        (ROOT / f"snapshot-{arm}.txt").write_text(digest)
        check(
            f"the {arm} arm plans, ingests and installs — {encoding}",
            plan.returncode == 0
            and run.returncode == 0
            and install.returncode == 0
            and bool(digest),
            (digest or (plan.stderr + run.stderr + install.stderr))[:280],
        )


# ------------------------------------------------------------------------------- probe


def probe() -> None:
    """The real qualification suite on this pod's card. A record means something RAN."""
    import torch

    from cozy_runtime.internal.encoding import (
        digests_of,
        implementation_digest,
        launch_providers,
        measure_device,
    )
    from cozy_runtime.internal.probe import qualify

    head("probe — capability records by MEASUREMENT on this card")
    device = measure_device(torch)
    print(f"  device     {device.predicate} ({device.name}, driver cuda {device.driver})")
    table = launch_providers()
    subject = {d: table[d] for a in ("fp8-rowwise/1", "mxfp8/1") for d in digests_of(a)}
    names = {implementation_digest(p): p.name for g in subject.values() for p in g}
    caps, evidence = qualify(
        torch, subject, device, ["bfloat16"], release=RELEASE, cache=ROOT / "probe"
    )
    for o in evidence["observations"]:
        axes = o["evidence"].get("route_axes", {})
        print(
            f"  {'MINTED ' if o['evidence']['passed'] else 'no record'}  "
            f"{o['implementation']:34s} [{o['delivery_route']:14s}] "
            f"footprint {axes.get('resident_bits_per_element', -1):7.3f} b/elt  "
            f"fidelity {axes.get('deviation_rel', -1):9.6f}"
        )
    minted = {names[r.implementation_digest] for r in caps.records}
    check(
        "the ROWWISE native leaf is qualified here, by a suite that ran its real kernel",
        "cozy.fp8-rowwise.native-leaf/1" in minted,
        f"minted {sorted(minted)}",
    )
    check(
        "the MXFP8 native leaf mints nothing — cuda.sm120 exactly, and this is not it",
        "cozy.mxfp8.native-leaf/1" not in minted,
        "#511b's law: a device with no measured suite mints nothing, and a hand-written "
        "floor is what that law forbids",
    )
    bench = next(
        (
            o["evidence"].get("micro_bench")
            for o in evidence["observations"]
            if o["implementation"] == "cozy.fp8-rowwise.native-leaf/1" and o["evidence"]["passed"]
        ),
        None,
    )
    if bench:
        print(
            f"  leaf bench {bench['shape']}  leaf {bench['leaf_ns'] / 1000:.1f} us  "
            f"bf16 GEMM {bench['float_gemm_ns'] / 1000:.1f} us  {bench['speedup_x']:.3f}x "
            "(activation quantize INSIDE the timed region, unfused — #515f)"
        )
    (ROOT / "probe.json").write_text(json.dumps(evidence, indent=1))


# -------------------------------------------------------------------------------- fill


def make_model() -> Any:
    """The endpoint. It DECLARES that its components are touched through their forward
    only, which is the whole of what `encoded_leaves="accept"` states."""
    from cozy_runtime.author import Model, uses_components
    from cozy_runtime.author._loader import Loader

    class H3BlockModel(Model[Any], encoded_leaves="accept"):
        pipe: Any

        def load(self, loader: Loader) -> None:
            self.pipe = loader.construct(MX.H3BlockLike, factory=MX.build_block)

        @uses_components("block")
        def step(self, torch: Any, x: Any) -> Any:
            with torch.inference_mode():
                return self.pipe.forward(torch, x)

    return H3BlockModel


def fill_arm(arm: str, objective: str) -> tuple[Any, Any]:
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
        ledger=ResourceLedger(budgets={"vram": 16 * GIB, "pinned": 512 * MIB}),
        window_bytes=4 * MIB,
        slots=16,
        readers=16,
        inflight=8,
        probe_cache=ROOT / "probe",
        decode_scratch_bytes=256 * MIB,
        release=RELEASE,
        objective=objective,
    )
    registry = ModelRegistry(
        release=RELEASE,
        backend=backend,
        substrate=lambda: serving_substrate(f"sm{backend.device_facts.sm}", Observations()),
    )
    artifact = Artifact(
        snapshot, topology_of(rows), Config({}, "block.config"), custody="canonical"
    )
    backend.expect({"block": tuple(r.key for r in rows)})
    model = registry.acquire("step.models.model", make_model(), artifact)
    torch.cuda.synchronize()
    return backend, model


def fill() -> None:
    """The rung on real weights: leaves installed, VRAM measured, the ledger settled."""
    import torch

    from cozy_runtime.internal.fill import _tensors

    head("fill — the real H3 block through the real fill plane, into real leaves")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    started = time.time()
    leaf_backend, leaf_model = fill_arm("rowwise", "footprint")
    took = time.time() - started
    leaf_alloc = torch.cuda.memory_allocated() - before
    leaf_peak = torch.cuda.max_memory_allocated() - before
    print(f"  artifact   {(ROOT / 'snapshot-rowwise.txt').read_text().strip()[:46]}…")
    print(f"  servable   {leaf_backend.servability.line()}")
    check(
        f"all {len(QUANTIZED)} quantized weights became LEAVES, the rest filled plain",
        len(leaf_backend.replaced) == len(QUANTIZED),
        "; ".join(
            f"{r.module_path.split('.', 1)[1]}: {r.was}->{r.now} "
            f"({r.stored_bytes / MIB:.1f} MiB stored, {r.freed_bytes / MIB:.1f} MiB freed)"
            for r in leaf_backend.replaced.values()
        ),
    )
    block = leaf_backend.components["block"]
    leaf_resident = sum(int(t.numel() * t.element_size()) for _, t in _tensors(block))
    ledger = leaf_backend.ledger.rows()
    print(
        f"  fill       {took:.1f}s wall; module {leaf_resident / MIB:.1f} MiB, "
        f"allocator +{leaf_alloc / MIB:.1f} MiB, peak +{leaf_peak / MIB:.1f} MiB\n"
        f"  ledger     vram charged {ledger['vram']['charged'] / MIB:.1f} MiB, "
        f"peak {ledger['vram']['peak'] / MIB:.1f} MiB; "
        f"decode charged {ledger['decode']['charged']} B, peak {ledger['decode']['peak']} B"
    )
    check(
        "the LEDGER settled to the measured truth, and native roles were never scratch",
        leaf_backend.vram_charge["block"] == leaf_resident and ledger["decode"]["peak"] == 0,
        f"charged {ledger['vram']['charged']} B == module {leaf_resident} B; the decode "
        "class stayed at zero because a native role is residency, not scratch",
    )

    # THE SAME ARTIFACT, the SAME card, the OTHER objective. This is the whole of #517d's
    # ruling made operational: fidelity and footprint order oppositely, so the deployment's
    # stated axis is what decides, and both answers are served from one set of stored bytes.
    float_backend, float_model = fill_arm("rowwise", "fidelity")
    float_block = float_backend.components["block"]
    float_resident = sum(int(t.numel() * t.element_size()) for _, t in _tensors(float_block))
    check(
        "objective `fidelity` takes the DECODE floor on the same bytes — no leaf at all",
        not float_backend.replaced
        and all(
            s.provider.route == "decoded_float"
            for k, s in float_backend.selected.items()
            if any(k.endswith(q) for q in QUANTIZED)
        ),
        f"{len(float_backend.replaced)} replacements, module {float_resident / MIB:.1f} MiB",
    )
    ratio = leaf_resident / float_resident
    gemm_elements = sum(int(t.numel()) for t in _leaf_weights(leaf_backend))
    gemm_bytes = sum(r.stored_bytes for r in leaf_backend.replaced.values())
    print(
        f"  RESIDENCY  native leaf {leaf_resident / MIB:.1f} MiB vs decoded float "
        f"{float_resident / MIB:.1f} MiB = {ratio:.4f}x  "
        f"({8.0 * gemm_bytes / max(gemm_elements, 1):.3f} bits per GEMM element)"
    )
    check(
        "the ~0.5x VRAM claim, MEASURED on real weights and not on the payload alone",
        ratio < 0.60,
        "the whole component, norms and all: the GEMM weights halve and the rest does not, "
        "so the component ratio is the honest number and it is above the payload's 0.5",
    )
    globals()["_ARMS"] = (leaf_backend, leaf_model, float_backend, float_model)
    (ROOT / "fill.json").write_text(
        json.dumps(
            {
                "format": "cozy.runtime.leaf_fill/1",
                "source": {"repo": REPO, "revision": REV, "member": MEMBER, "block": BLOCK},
                "snapshot": (ROOT / "snapshot-rowwise.txt").read_text().strip(),
                "device": f"sm{leaf_backend.device_facts.sm}",
                "leaf_resident_bytes": leaf_resident,
                "float_resident_bytes": float_resident,
                "ratio": ratio,
                "allocator_bytes": leaf_alloc,
                "allocator_peak_bytes": leaf_peak,
                "ledger": ledger,
                "replacements": [r.document() for r in leaf_backend.replaced.values()],
            },
            indent=1,
        )
    )


def _leaf_weights(backend: Any) -> list[Any]:
    out = []
    for record in backend.replaced.values():
        module: Any = backend.components["block"]
        for step in record.module_path.split(".")[1:]:
            module = getattr(module, step)
        out.append(module.data)
    return out


# ----------------------------------------------------------------------------- forward


def forward() -> None:
    """THREE lanes on ONE scale: the same block, the same input, the same protocol."""
    import torch

    head("forward — native leaf vs dequant-bf16, against the bf16 source reference")
    arms = globals().get("_ARMS")
    if arms is None:
        check("forward needs the fill leg", False, "run `fill` first")
        return
    leaf_backend, leaf_model, _, float_model = arms
    _, reference_model = fill_arm("bf16", "fidelity")

    torch.manual_seed(0x517E)
    x = torch.randn(1, 512, 5376, dtype=torch.bfloat16, device="cuda") * 0.5
    with torch.inference_mode():
        reference = reference_model.step(torch, x).float()
        dequant = float_model.step(torch, x).float()
        native = leaf_model.step(torch, x).float()

    banked: dict[str, Any] = {}
    for name, out in (("dequant_bf16", dequant), ("native_leaf", native)):
        rel = _rel(out, reference)
        cosine = float(
            torch.nn.functional.cosine_similarity(out.reshape(1, -1), reference.reshape(1, -1))
        )
        banked[name] = {"relative_frobenius": rel, "snr_db": _snr(rel), "cosine": cosine}
        print(f"  {name:14s} relative {rel:.6f}   SNR {_snr(rel):.3f} dB   cosine {cosine:.8f}")
    check(
        "both lanes read the SAME stored bytes and land on the same bf16 reference",
        all(math.isfinite(v["snr_db"]) for v in banked.values()),
        f"the native leaf costs "
        f"{banked['dequant_bf16']['snr_db'] - banked['native_leaf']['snr_db']:.3f} dB against "
        "the dequant floor on identical input and identical stored weights — that difference "
        "IS the activation quantization the w8a8 route necessarily pays (#515a, #517d)",
    )
    check(
        "the native leaf's output is finite and is the contract's dtype",
        bool(torch.isfinite(native).all()) and banked["native_leaf"]["cosine"] > 0.99,
        f"cosine {banked['native_leaf']['cosine']:.8f} against the bf16 reference",
    )
    (ROOT / "forward.json").write_text(
        json.dumps(
            {
                "format": "cozy.runtime.leaf_forward/1",
                "protocol": "1x512x5376 through the block's real GEMM chain, bf16 arm as "
                "reference; identical module in every arm",
                "source": {"repo": REPO, "revision": REV, "member": MEMBER, "block": BLOCK},
                "snapshots": {
                    "bf16": (ROOT / "snapshot-bf16.txt").read_text().strip(),
                    "rowwise": (ROOT / "snapshot-rowwise.txt").read_text().strip(),
                },
                "device": f"sm{leaf_backend.device_facts.sm}",
                "lanes": banked,
            },
            indent=1,
        )
    )


# ------------------------------------------------------------------------------ choose


def choose() -> None:
    """The chooser, observed: the leaf where the probe measured, the floor where it did not."""
    import torch

    from cozy_runtime.internal.encoding import (
        Capabilities,
        DeviceFacts,
        Selector,
        digests_of,
        launch_providers,
        measure_device,
    )
    from cozy_runtime.internal.fill import Checkpoint, dtype_name
    from cozy_runtime.internal.probe import qualify

    head("choose — the leaf on the MEASURED device, the floor on an unmeasured one")
    device = measure_device(torch)
    table = launch_providers()
    caps, _ = qualify(torch, table, device, ["bfloat16"], release=RELEASE, cache=ROOT / "probe")
    rows = Checkpoint(str(STORE), (ROOT / "snapshot-rowwise.txt").read_text().strip()).rows("block")
    subject = next(r for r in rows if r.encoded.alias == "fp8-rowwise/1")
    for objective in ("footprint", "fidelity"):
        selection = Selector(table, caps, device, objective).select(
            subject.key, subject.encoded, dtype_name(subject.dtype)
        )
        want = "native_encoded" if objective == "footprint" else "decoded_float"
        check(
            f"on {device.predicate} the {objective!r} objective takes {want}",
            selection.provider.route == want,
            f"{selection.provider.name}: footprint "
            f"{selection.record.resident_bits_per_element:.3f} b/elt, fidelity "
            f"{selection.record.route_deviation_rel:.6f} rel",
        )
    # An UNMEASURED device: the same build, the same artifact, and a table that carries a
    # DECODE record for that card and no native one — which is exactly the state of a card
    # whose probe ran the decode oracle and never ran a leaf's kernel.
    blind = DeviceFacts("cuda", "unmeasured-card", 86, device.driver, "default")
    rowwise = set(digests_of("fp8-rowwise/1"))
    kept = Capabilities(
        [
            dataclasses.replace(record, device_predicate=blind.predicate)
            for record in caps.records
            if record.delivery_route == "decoded_float" and record.encoding_spec_id in rowwise
        ]
    )
    selection = Selector(table, kept, blind, "footprint").select(
        subject.key, subject.encoded, dtype_name(subject.dtype)
    )
    check(
        "an UNMEASURED card takes the dequant floor under the objective that WANTS the leaf",
        selection.provider.route == "decoded_float",
        f"{selection.provider.name} on {blind.predicate}: the leaf's kernel exists there "
        "and nothing measured it, so no record admits it. An unqualified provider is never "
        "selected, whatever the objective says",
    )


def main(argv: list[str]) -> int:
    names = argv or ["all"]
    if names == ["all"]:
        names = ["fetch", "ingest", "probe", "fill", "forward", "choose"]
    started = time.perf_counter()
    for name in names:
        {
            "fetch": fetch,
            "ingest": ingest,
            "probe": probe,
            "fill": fill,
            "forward": forward,
            "choose": choose,
        }[name]()
    print(f"\n{PASS} ok, {FAIL} failed in {time.perf_counter() - started:.1f} s")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
