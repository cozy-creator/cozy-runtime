#!/usr/bin/env python3
"""cr-006/#469a live verification: the MXFP8 lane, from frozen vectors to a real snapshot.

Not a test (tracker README #160): every leg runs the real provider, the real border, the
real header closure and the real fill plane, and prints numbers.

#469a made MXFP8 the preferred H3 DiT serving lane and said in the same breath that
PRODUCING an artifact is not SERVING one — the runtime's mxfp8 fill/compute lane is its own
honesty gate. #501a produced the artifact (21.56 GB curve-topology carrier, 532 tensors,
snapshot `a97b89d0…`). This script is the other half.

  vectors   the provider against TensorFS's FROZEN mined vectors, bit for bit
  probe     the qualification suite over the whole 65,536-pair format domain
  carrier   a real carrier through the real border, read back and dequantized
  arms      the red arms, each planted for real
  pod       the real `a97b89d0…` leg — GPU, pod-side only (see `mxfp8-pod.py`)

The CPU legs run on this box because they move kilobytes. The fill and the forward do not
run here: `GPU work happens ONLY on the pod`, and the weights-locality law is hard.

    uv run python scripts/mxfp8-live.py vectors probe carrier arms
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TFS = Path("/home/fidika/cozy_v2/tensorfs")
WORK = Path("/tmp/cozy-mxfp8")
STORE = WORK / "store"
CARRIER = WORK / "carrier.safetensors"

#: The registry's own block, transcribed from the spec relation `ceil_div(axis 1, by 32)`.
BLOCK = 32
E4M3_MAX = 448.0

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


def registry() -> str:
    """A WRITABLE COPY of the reviewed fingerprint registry — `tfs ingest bank` WRITES its
    argument, and the reviewed file is committed data in a repo this lane does not own."""
    local = WORK / "fingerprints.json"
    if not local.is_file():
        WORK.mkdir(parents=True, exist_ok=True)
        local.write_bytes((TFS / "vectors/fingerprints/registry.json").read_bytes())
    return str(local)


def tfs_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """The real `tfs` binary. The border is a PROCESS boundary and it stays one here."""
    return subprocess.run(
        [str(TFS / "target/release/tfs"), *args, "--registry", registry()],
        capture_output=True,
        text=True,
        cwd=str(TFS),
    )


# --------------------------------------------------------------------------- the encoder

#: The producing job's quantizer, transcribed from `cozy_jobs/quantize.py` and from nothing
#: else. It is here so this script's carriers are the bytes THAT job would write — a fixture
#: quantized by a second, friendlier equation would prove the reader agrees with itself.
QUANT_RULE = "e8m0 = floor(log2(rowblock amax)) - 8 + 127, clamped [0,254]; amax==0 -> byte 0"


def quantize(torch: Any, weight: Any) -> tuple[Any, Any]:
    rows, cols = weight.shape
    work = weight.to(torch.float32)
    blocks = work.view(rows, cols // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.where(
        amax > 0, torch.floor(torch.log2(amax)) - 8.0, torch.full_like(amax, -127.0)
    )
    byte = (exponent + 127.0).clamp(0.0, 254.0)
    scale = torch.pow(2.0, byte - 127.0)
    quant = (blocks / scale.unsqueeze(-1)).view(rows, cols)
    data = quant.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return data, byte.to(torch.uint8)


def reference(torch: Any, data: Any, scale: Any, dtype: Any) -> Any:
    """The NORMATIVE dequant, transcribed from `tfs conform.rs::mxfp8_inner`: multiply in
    f32 and round ONCE into the logical dtype. Byte 0xFF is E8M0's NaN, not 2**128."""
    rows, cols = data.shape
    mult = torch.where(
        scale == 255,
        torch.full(scale.shape, float("nan"), dtype=torch.float32),
        torch.pow(2.0, scale.to(torch.float32) - 127.0),
    )
    wide = data.to(torch.float32).view(rows, cols // BLOCK, BLOCK) * mult.unsqueeze(-1)
    return wide.view(rows, cols).to(dtype)


# --------------------------------------------------------------------------- the vectors


def vectors() -> None:
    """The strongest CPU proof available: TensorFS ships FROZEN expected logical bits for
    `mxfp8/1`, mined from an implementation written to the OCP spec by someone who had never
    seen this provider. Bit for bit or it is not a proof."""
    import torch

    from cozy_runtime.internal.encoding import MicroScaledDequant, digests_of

    head("vectors — the provider against TensorFS's frozen mined vectors, BIT FOR BIT")
    corpus = json.loads((TFS / "vectors/mined/mxfp8.json").read_text())
    provider = MicroScaledDequant(digests_of("mxfp8/1")[0])
    torch_dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    print(f"  corpus     {corpus['format']}  {len(corpus['cases'])} cases")
    for case in corpus["cases"]:
        shape = tuple(int(n) for n in case["logical"]["shape"])
        dtype = torch_dtype[case["logical"]["dtype"]]
        roles = {}
        for role, part in case["roles"].items():
            raw = base64.b64decode(part["bits"])
            flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            if part["dtype"] == "f8_e4m3fn":
                flat = flat.view(torch.float8_e4m3fn)
            roles[role] = flat.reshape(tuple(int(n) for n in part["shape"]))
        destination = torch.empty(shape, dtype=dtype)
        provider.decode(torch, roles, destination)
        want = base64.b64decode(case["expect_logical_bits"])
        got = bytes(destination.contiguous().view(torch.uint8).reshape(-1).tolist())
        wrong = [i for i, (a, b) in enumerate(zip(got, want, strict=True)) if a != b]
        check(
            f"{case['name']}: decoded bits == the corpus's expected bits",
            got == want,
            f"{len(want)} B, {len(wrong)} differing"
            + (f"; first at byte {wrong[0]}" if wrong else ""),
        )
    # The WRONG-BLOCK arm the corpus itself carries: reading these scales at block 16 is the
    # single most likely way to be plausibly wrong, so it is planted rather than trusted.
    case = corpus["cases"][0]
    shape = tuple(int(n) for n in case["logical"]["shape"])
    data = (
        torch.frombuffer(
            bytearray(base64.b64decode(case["roles"]["data"]["bits"])), dtype=torch.uint8
        )
        .view(torch.float8_e4m3fn)
        .reshape(shape)
    )
    scale = torch.frombuffer(
        bytearray(base64.b64decode(case["roles"]["scale"]["bits"])), dtype=torch.uint8
    ).reshape(tuple(int(n) for n in case["roles"]["scale"]["shape"]))
    # The corpus's own wrong-arm, `conform.rs::mxfp8_wrong_block`: the SAME bytes read with
    # `b = min(j // 16, blocks - 1)`. Every value is plausible and the tensor is wrong.
    index = torch.clamp(torch.arange(shape[1]) // 16, max=scale.shape[1] - 1)
    stretched = torch.empty(shape, dtype=torch.float32)
    stretched.copy_(data.to(torch.float32))
    stretched.mul_(torch.pow(2.0, scale.float() - 127)[:, index])
    misread = bytes(stretched.contiguous().view(torch.uint8).reshape(-1).tolist())
    check(
        "RED: the same bytes read at block 16 do NOT match — the corpus can tell",
        misread != base64.b64decode(case["expect_logical_bits"]),
        "a block size that is not 32 is a different tensor, and the vectors say so",
    )


# ----------------------------------------------------------------------------- the probe


def probe() -> None:
    import torch

    from cozy_runtime.internal.encoding import digests_of, launch_providers, measure_device
    from cozy_runtime.internal.probe import qualify

    head("probe — the qualification suite over the whole 65,536-pair format domain")
    device = measure_device(torch)
    providers = launch_providers()
    mx = digests_of("mxfp8/1")[0]
    check(
        "the launch provider set CLAIMS mxfp8/1, by digest",
        mx in providers,
        # digest -> ORDERED CANDIDATES since #517e: one encoding, two routes.
        f"{[p.name for p in providers.get(mx, ())] or 'ABSENT'} <- {mx[:26]}…",
    )
    capabilities, evidence = qualify(
        torch,
        {mx: providers[mx]},
        device,
        ["bfloat16", "float32", "float16"],
        release="cozy/mxfp8-live",
        cache=None,
    )
    print(f"  device     {device.predicate} ({device.name})")
    # Keyed on the IMPLEMENTATION as well as the dtype since #517e: one digest now carries
    # a decode floor and a native leaf, and this section is about the decode one's oracle.
    # A dtype-only key would silently read whichever observation came last.
    by_dtype = {
        o["output_dtype"]: o["evidence"]
        for o in evidence["observations"]
        if o["implementation"] == "cozy.mxfp8.dequant/1"
    }
    for name in ("bfloat16", "float32"):
        n = by_dtype[name]["numerics"]
        check(
            f"{name}: all {n['pairs']} (e4m3 x e8m0) pairs decode correctly",
            by_dtype[name]["passed"],
            f"domain {n['domain']}; wrong {n['wrong']}; "
            f"max rel {n['max_rel_error']:.3e} <= {n['tolerance']:.3e}, "
            f"max abs {n['max_abs_error']:.3e} <= {n['subnormal_tolerance']:.3e}; "
            f"{by_dtype[name]['micro_bench']['gb_per_s']} GB/s",
        )
    f16 = by_dtype["float16"]["numerics"]
    check(
        "RED, MEASURED: float16 FAILS the domain — 2^(b-127) flushes to zero below b=103",
        not by_dtype["float16"]["passed"] and f16["wrong"]["normal"] > 0,
        f"{f16['wrong']['normal']} normal + {f16['wrong']['subnormal']} subnormal pairs "
        f"decode to zero; first {f16['first_wrong'][:90]}",
    )
    minted = {r.output_dtype for r in capabilities.records}
    check(
        "...so float16 mints NO capability record and bf16/f32 do",
        minted == {"bfloat16", "float32"},
        f"records for {sorted(minted)} — fail-closed by EVIDENCE, not by a hand-written floor",
    )


# --------------------------------------------------------------------------- the carrier


def carrier() -> None:
    """A real mxfp8 carrier, quantized by the producing job's own equation, pushed through
    the real border, installed, and read back through the runtime's own header path."""
    import torch
    from safetensors.torch import save_file

    from cozy_runtime.internal.encoding import (
        MicroScaledDequant,
        Selector,
        digests_of,
        launch_providers,
        measure_device,
    )
    from cozy_runtime.internal.fill import Checkpoint, dtype_name, servability
    from cozy_runtime.internal.probe import qualify

    head("carrier — the producing job's own equation, through the real border and back")
    torch.manual_seed(0x469A)
    WORK.mkdir(parents=True, exist_ok=True)
    # Two shapes with the H3 DiT's own aspect: a fat projection and a wide fc2. Kilobytes,
    # real bytes, and the SAME mixed-component shape the real artifact has — rank-2 weights
    # encoded, everything else left at source precision.
    sources = {
        "block.attn.out_proj.weight": torch.randn(64, 224, dtype=torch.bfloat16),
        "block.mlp.fc2.weight": torch.randn(32, 448, dtype=torch.bfloat16) * 0.05,
    }
    out: dict[str, Any] = {"block.norm.weight": torch.randn(64, dtype=torch.bfloat16)}
    for key, weight in sources.items():
        data, scale = quantize(torch, weight)
        out[key] = data
        # #501d: the companion is named off the MODULE STEM. `out_proj.weight_scale`, never
        # `out_proj.weight.weight_scale` — present-but-invisible is worse than absent.
        out[key[: -len(".weight")] + ".weight_scale"] = scale
    save_file(out, str(CARRIER), metadata={"format": "pt"})
    print(f"  rule       {QUANT_RULE}")
    print(f"  carrier    {CARRIER.stat().st_size} B, {len(out)} keys")

    tfs_cli(
        "ingest",
        "bank",
        "safetensors.diffusers",
        "cr006-mxfp8",
        "tiny",
        "diffusers.identity/1",
        "--source",
        f"c={CARRIER}",
        "--provenance",
        "synthetic",
        "--note",
        "mxfp8 provider fixture",
    )
    plan = tfs_cli("ingest", "plan", "c=mxfp8/1", "--source", f"c={CARRIER}")
    line = next((ln.strip() for ln in plan.stdout.splitlines() if "mxfp8/1" in ln), plan.stderr)
    check(
        "the border SELECTS mxfp8/1 per tensor, folding the stem-named companions",
        plan.returncode == 0 and "mxfp8/1" in plan.stdout,
        line[:150],
    )
    if plan.returncode != 0:
        return
    STORE.mkdir(parents=True, exist_ok=True)
    if not (STORE / "objects").is_dir():
        tfs_cli("store", "init", str(STORE))
    session = "mx" + str(abs(hash(CARRIER.stat().st_mtime_ns)) % 100000)
    run = tfs_cli(
        "ingest",
        "run",
        str(STORE),
        "c=mxfp8/1",
        "--source",
        f"c={CARRIER}",
        "--session",
        session,
        "--tenant",
        "local",
    )
    install = tfs_cli("ingest", "install", str(STORE), session)
    if run.returncode or install.returncode:
        check("ingest", False, (run.stderr + install.stderr)[:300])
        return

    import tensorfs

    snapshot = str(tensorfs.ingest_installed(tensorfs.Store.open(str(STORE)))[-1])
    book = Checkpoint(str(STORE), snapshot)
    rows = book.rows("c")
    print(f"  snapshot   {snapshot[:30]}…  {len(rows)} tensors")
    encoded = [r for r in rows if r.encoded.alias == "mxfp8/1"]
    check(
        "the runtime reads the header's roles: e4m3 [r,c] data + u8 [r,c/32] scale",
        len(encoded) == 2
        and all(
            r.encoded.part("data").shape == r.shape
            and r.encoded.part("scale").shape == (r.shape[0], r.shape[1] // BLOCK)
            and r.encoded.part("scale").dtype == "u8"
            for r in encoded
        ),
        "; ".join(
            f"{r.name} data{r.encoded.part('data').shape} scale{r.encoded.part('scale').shape}"
            for r in encoded
        ),
    )
    check(
        "STORED bytes are 1 + 1/32 per element, not 2 — the carrier is the smaller one",
        all(r.stored_nbytes < r.nbytes for r in encoded),
        f"{sum(r.stored_nbytes for r in encoded)} stored vs "
        f"{sum(r.nbytes for r in encoded)} logical B "
        f"({sum(r.stored_nbytes for r in encoded) / sum(r.nbytes for r in encoded):.4f}x)",
    )

    # THE NUMERICS PROOF on real stored bytes: read each role back out of the store and
    # decode it with the provider, against the producing job's own reference equation.
    device = measure_device(torch)
    providers = launch_providers()
    capabilities, _ = qualify(
        torch,
        providers,
        device,
        sorted({dtype_name(r.dtype) for r in rows}),
        release="cozy/mxfp8-live",
        cache=None,
    )
    answer = servability(rows, Selector(providers, capabilities, device))
    check(
        "SERVABILITY, before a byte moves: this runtime's own answer", answer.serves, answer.line()
    )

    # THE STORED BYTES, read back through the VERIFIED LEASE — never from the carrier file
    # this script wrote. The store is the artifact identity, and a proof that reads its own
    # input back proves nothing about the store.
    provider = MicroScaledDequant(digests_of("mxfp8/1")[0])
    entries = json.loads(book.header_bytes)["components"]["c"]
    lease = book.acquire()
    worst_ref = 0
    worst_src = 0.0
    read_bytes = 0
    try:
        for row in encoded:
            roles: dict[str, Any] = {}
            for part in row.encoded.parts:
                stored = entries[row.name]["parts"][part.role]
                buffer = bytearray(part.nbytes)
                if "inline" in stored:
                    # tfs-002: a part of 256 B or less rides IN the header. It is still the
                    # store's byte, and the header it rode in on was verified against the
                    # snapshot digest before this line could run.
                    buffer[:] = base64.b64decode(stored["inline"])[: part.nbytes]
                else:
                    offset = 0
                    for segment in stored["segments"]:
                        length = int(segment["length"])
                        view = memoryview(buffer)[offset : offset + length]
                        lease.read_into(str(segment["sha256"]), length, 0, length, view)
                        offset += length
                read_bytes += part.nbytes
                flat = torch.frombuffer(buffer, dtype=torch.uint8)
                if part.dtype == "f8_e4m3fn":
                    flat = flat.view(torch.float8_e4m3fn)
                roles[part.role] = flat.reshape(part.shape)
            destination = torch.empty(row.shape, dtype=torch.bfloat16)
            provider.decode(torch, roles, destination)
            want = reference(torch, roles["data"], roles["scale"], torch.bfloat16)
            worst_ref = max(
                worst_ref, int((destination.view(torch.uint16) != want.view(torch.uint16)).sum())
            )
            source = sources[row.name].float()
            worst_src = max(worst_src, float((destination.float() - source).norm() / source.norm()))
    finally:
        lease.release()
    check(
        "the provider == the producing job's own dequant reference, BIT FOR BIT",
        worst_ref == 0,
        f"{worst_ref} differing bf16 words over "
        f"{sum(r.shape[0] * r.shape[1] for r in encoded)} elements decoded from "
        f"{read_bytes} B read back through the verified lease; the reference multiplies in "
        "f32 and rounds once, this route multiplies in bf16 — they agree exactly because a "
        "power-of-two scale rounds nothing",
    )
    check(
        "round-trip error is the QUANTIZER's, and inside the job's own e4m3 bound",
        worst_src < 2.0**-3,
        f"worst relative Frobenius {worst_src:.5f} (the job's per-tensor acceptance bound is "
        f"amax * 2^-3; e4m3 carries 3 mantissa bits)",
    )


# ------------------------------------------------------------------------------ the arms


def arms() -> None:
    import torch
    from safetensors.torch import save_file

    from cozy_runtime.internal.encoding import (
        Capabilities,
        Encoded,
        EncodingRefusal,
        RolePart,
        Selector,
        digests_of,
        launch_providers,
        measure_device,
    )
    from cozy_runtime.internal.fill import FillRefusal, servability, store_refusals
    from cozy_runtime.internal.probe import qualify
    from cozy_runtime.internal.worker.refusal import SIDE, classify

    head("red arms — each planted for real, observed, and removed")
    WORK.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1)
    weight = torch.randn(64, 96, dtype=torch.bfloat16)
    data, scale = quantize(torch, weight)
    bias = torch.randn(64, dtype=torch.bfloat16)

    # --- THE #501d INCIDENT, replanted at the border ------------------------------------
    plants = {
        # The scales are simply not there.
        "noscale": {"block.linear.weight": data, "block.linear.bias": bias},
        # The scales ARE there and INVISIBLE: named off the KEY instead of the MODULE STEM.
        # This is the exact defect #501d was written from, and the border's refusal — which
        # reads it as scale-free storage — was ruled CORRECT.
        "stem": {
            "block.linear.weight": data,
            "block.linear.weight.weight_scale": scale,
            "block.linear.bias": bias,
        },
        # The right roles at the WRONG BLOCK: a [64,2] grid where the spec derives [64,3].
        "blocks": {
            "block.linear.weight": data,
            "block.linear.weight_scale": scale[:, :2].contiguous(),
            "block.linear.bias": bias,
        },
        # The scale carried as f32 instead of the E8M0 u8 the spec's carrier set names.
        "scaledtype": {
            "block.linear.weight": data,
            "block.linear.weight_scale": scale.float(),
            "block.linear.bias": bias,
        },
    }
    good = WORK / "arm-good.safetensors"
    save_file(
        {
            "block.linear.weight": data,
            "block.linear.weight_scale": scale,
            "block.linear.bias": bias,
        },
        str(good),
        metadata={"format": "pt"},
    )
    tfs_cli(
        "ingest",
        "bank",
        "safetensors.diffusers",
        "cr006-mxarm",
        "tiny",
        "diffusers.identity/1",
        "--source",
        f"c={good}",
        "--provenance",
        "synthetic",
        "--note",
        "mxfp8 red-arm fixture",
    )
    green = tfs_cli("ingest", "plan", "c=mxfp8/1", "--source", f"c={good}")
    check(
        "GREEN first: the well-formed carrier PLANS, so the reds are not vacuous",
        green.returncode == 0 and "mxfp8/1" in green.stdout,
        next((ln.strip() for ln in green.stdout.splitlines() if "mxfp8" in ln), green.stderr)[:140],
    )
    expect = {
        "noscale": "scale-free",
        "stem": "scale-free",
        "blocks": "SHAPE_MISMATCH",
        "scaledtype": "DTYPE_MISMATCH",
    }
    for name, tensors in plants.items():
        path = WORK / f"arm-{name}.safetensors"
        save_file(tensors, str(path), metadata={"format": "pt"})
        red = tfs_cli("ingest", "plan", "c=mxfp8/1", "--source", f"c={path}")
        body = (red.stdout + red.stderr).strip()
        line = next((ln for ln in body.splitlines() if "REFUSED" in ln), "no refusal")
        check(
            f"a carrier with {name} refuses at the border, naming the candidates tried",
            red.returncode != 0 and expect[name] in body and "FINGERPRINT" not in line,
            line[:300],
        )
        path.unlink(missing_ok=True)
    good.unlink(missing_ok=True)

    # --- THE HEADER CLOSURE, which is the WORKER's own re-check --------------------------
    # Law 18: a RecordOwner's authority is not worker safety. The border above is the
    # RecordOwner; these plants go straight at the header a worker is handed, which is the
    # only version of this refusal that protects a worker from a store it did not fill.
    import tensorfs

    if (STORE / "objects").is_dir() and tensorfs.ingest_installed(tensorfs.Store.open(str(STORE))):
        store = tensorfs.Store.open(str(STORE))
        snapshot = str(tensorfs.ingest_installed(store)[-1])
        raw = store.snapshot(snapshot)["header"]

        def closure() -> Any:
            c = tensorfs.Closure()
            for ref in tensorfs.parse_header(raw)["encodings"]:
                c.insert(store.document(str(ref["sha256"])))
            return c

        key = next(
            k
            for k, v in json.loads(raw)["components"]["c"].items()
            if "scale" in v.get("parts", {})
        )
        mutations: tuple[tuple[str, Callable[[dict[str, Any]], object]], ...] = (
            (
                "a scale grid at the wrong block count",
                lambda e: e["parts"]["scale"].__setitem__("shape", [64, 2]),
            ),
            (
                "a scale role carried as f32",
                lambda e: e["parts"]["scale"].__setitem__("dtype", "f32"),
            ),
            ("the scale role dropped entirely", lambda e: e["parts"].pop("scale")),
        )
        for label, mutate in mutations:
            document = json.loads(raw)
            mutate(document["components"]["c"][key])
            planted = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
            code = detail = ""
            try:
                with store_refusals("planted header"):
                    tensorfs.parse_header(planted, closure())
            except FillRefusal as exc:
                code, detail = exc.code, str(exc)
            except Exception as exc:
                code, detail = f"UNTRANSLATED:{type(exc).__name__}", str(exc)
            want = "dtype_mismatch" if "f32" in label else "carrier_geometry"
            check(
                f"{label} refuses TYPED through the closed registry",
                code == want,
                f"[{code}] {detail[:190]}",
            )

    # --- THE SIDES, which decide whose remedy gets printed --------------------------------
    for code, side in (
        ("carrier_geometry", "data"),
        ("role_mismatch", "reader"),
        ("encoding_unqualified", "reader"),
        ("unknown_encoding", "reader"),
    ):
        sentence = classify(code, "planted", origin="platform").sentence()
        check(
            f"{code} is classified {side.upper()} and prints that remedy",
            SIDE.get(code) == side and side.upper()[:4].lower() in sentence.lower(),
            sentence[:170],
        )

    # --- THE SELECTOR, on this device -----------------------------------------------------
    device = measure_device(torch)
    providers = launch_providers()
    mx = digests_of("mxfp8/1")[0]
    capabilities, _ = qualify(
        torch, providers, device, ["bfloat16"], release="cozy/mxfp8-live", cache=None
    )
    selector = Selector(providers, capabilities, device)
    parts = (RolePart("data", "f8_e4m3fn", (64, 96), 6144), RolePart("scale", "u8", (64, 3), 192))

    def refuse(encoded: Encoded, dtype: str = "bfloat16") -> tuple[str, str]:
        try:
            selector.select("block.linear.weight", encoded, dtype)
        except EncodingRefusal as exc:
            return exc.code, str(exc)
        return "", "no refusal — the arm is GREEN when it should be RED"

    code, _ = refuse(Encoded(mx, "mxfp8/1", parts))
    check("GREEN: a well-formed mxfp8 tensor SELECTS on this device", code == "", "")
    code, detail = refuse(Encoded(mx, "mxfp8/1", (parts[0],)))
    check(
        "an mxfp8 tensor with NO scale role refuses before any byte moves",
        code == "role_mismatch",
        f"[{code}] {detail[:160]}",
    )
    code, detail = refuse(Encoded(mx, "mxfp8/1", (*parts, RolePart("input_scale", "f32", (), 4))))
    check(
        "an input_scale mxfp8 never carries refuses rather than being ignored",
        code == "role_mismatch",
        f"[{code}] {detail[:160]}",
    )
    code, detail = refuse(Encoded(mx, "mxfp8/1", parts), "float16")
    check(
        "an f16 destination refuses CAPABILITY, naming the card — never served zeros",
        code == "encoding_unqualified",
        f"[{code}] {detail[:200]}",
    )

    # --- THE SERVABILITY GATE, which is #501f's runtime twin ------------------------------
    from cozy_runtime.internal.fill import PlanRow

    row = PlanRow("c.w", "w", "c", "bf16", (64, 96), 12288, Encoded(mx, "mxfp8/1", parts))
    answer = servability([row], selector)
    check(
        "the servability answer is GREEN for bytes this build serves", answer.serves, answer.line()
    )
    blind = servability([row], Selector(providers, Capabilities(), device))
    check(
        "a worker with NO capability record says so TYPED, before any fill",
        not blind.serves and blind.code == "encoding_unqualified" and "mxfp8/1" in blind.line(),
        blind.line()[:230],
    )
    raised = ""
    try:
        blind.require()
    except FillRefusal as exc:
        raised = exc.code
    check(
        "...and `require()` speaks it in the FILL plane's closed vocabulary",
        raised == "encoding_unqualified",
        f"[{raised}] — a manifest that claims servability changes nothing here",
    )


# ------------------------------------------------------------------------ the derivation


def _mxfp8(torch: Any, weight: Any) -> tuple[Any, Any]:
    data, scale, _ = _quantize_counting(torch, weight)
    return data, scale


def _quantize_counting(torch: Any, weight: Any) -> tuple[Any, Any, int]:
    rows, cols = weight.shape
    work = weight.to(torch.float32)
    blocks = work.view(rows, cols // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.where(
        amax > 0, torch.floor(torch.log2(amax)) - 8.0, torch.full_like(amax, -127.0)
    )
    byte = (exponent + 127.0).clamp(0.0, 254.0)
    mult = torch.pow(2.0, byte - 127.0)
    quant = (blocks / mult.unsqueeze(-1)).view(rows, cols)
    saturated = int((quant.abs() > E4M3_MAX).sum())
    return quant.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), byte.to(torch.uint8), saturated


def derive() -> None:
    """#514a: collapsing per-32 block scales into one continuous row scale is a SECOND
    lossy quantization, and the number that matters is what it costs OVER a direct one."""
    import torch

    from cozy_runtime.internal.encoding import MicroScaledDequant, digests_of
    from cozy_runtime.internal.instruments import derive_rowwise, quantize_rowwise

    head("derive — mxfp8 -> fp8-rowwise, and what the SECOND quantization costs (#514a)")
    provider = MicroScaledDequant(digests_of("mxfp8/1")[0])
    torch.manual_seed(0x514)
    print(
        "  rule       row scale = rowmax|V| / 448 (continuous, f32); each block's payload "
        "rescaled\n             by mult/scale and RE-ROUNDED to E4M3 round-to-nearest-even, "
        "clamped at\n             +/-448 BEFORE the cast (the cast does not saturate)"
    )
    worst_penalty = 0.0
    for name, weight in (
        ("gaussian 256x512", torch.randn(256, 512, dtype=torch.bfloat16)),
        ("wide 128x4096", torch.randn(128, 4096, dtype=torch.bfloat16) * 0.05),
        (
            "heavy-tailed",
            torch.randn(256, 512, dtype=torch.bfloat16)
            * torch.exp(torch.randn(256, 512) * 1.5).bfloat16(),
        ),
    ):
        data, scale = _mxfp8(torch, weight)
        exact = torch.empty(weight.shape, dtype=torch.float32)
        provider.decode(torch, {"data": data, "scale": scale}, exact)
        derived = derive_rowwise(torch, data, scale)
        direct = quantize_rowwise(torch, weight)
        # The `fp8-rowwise/1` value law, from the registry entry: value = data * scale, one
        # scale per ROW. Written here rather than reused so the check has its own oracle.
        as_float = derived.data.float() * derived.scale.unsqueeze(-1)
        direct_float = direct.data.float() * direct.scale.unsqueeze(-1)
        original = weight.float()

        def rel(a: Any, b: Any) -> float:
            return float((a - b).norm() / b.norm().clamp(min=1e-12))

        err_mx, err_der, err_dir = (
            rel(exact, original),
            rel(as_float, original),
            rel(direct_float, original),
        )
        penalty = err_der / max(err_dir, 1e-12)
        worst_penalty = max(worst_penalty, penalty)
        print(
            f"  {name:17s} mxfp8 {err_mx:.6f} -> derived {err_der:.6f}   direct "
            f"{err_dir:.6f}   PENALTY {penalty:.3f}x   (derivation adds "
            f"{rel(as_float, exact):.6f} over mxfp8)"
        )
        check(
            f"{name}: the derived payload is finite everywhere",
            bool(torch.isfinite(as_float).all()) and derived.zero_rows == 0,
            f"peak offered to the cast {derived.max_rescaled:.6f}, {derived.clamped} clamped "
            f"of which {derived.saturated} beyond rounding",
        )
        check(
            f"{name}: the row scale COVERS its row — no genuine saturation",
            derived.saturated == 0,
            "the row maximum is supposed to land exactly on 448; anything materially past "
            "it means the scale did not cover the row and range was thrown away",
        )
        # NO THRESHOLD on the penalty itself — what it is ALLOWED to be is job-009's gate.
        # The ORDERING is structural and is asserted: one quantization to a target format
        # cannot be beaten by two quantizations ending at the same format.
        check(
            f"{name}: DOUBLE quantization costs more than direct — the penalty is real",
            err_der >= err_dir,
            f"{err_der:.6f} >= {err_dir:.6f}; mxfp8 carried a scale per 32 elements and "
            "fp8-rowwise carries one per row, so the block dynamic range is what is lost",
        )

    # The ZERO ROW, which is the divide this derivation would otherwise do by zero.
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    weight[7] = 0.0
    data, scale = _mxfp8(torch, weight)
    derived = derive_rowwise(torch, data, scale)
    as_float = derived.data.float() * derived.scale.unsqueeze(-1)
    check(
        "a row of exact zeros derives to zeros, not NaN — the scale floor holds",
        derived.zero_rows == 1
        and bool(torch.isfinite(as_float).all())
        and float(as_float[7].abs().max()) == 0.0,
        f"{derived.zero_rows} zero row(s) held off the divide by ROW_SCALE_FLOOR",
    )

    # E8M0's NaN byte, which the producing job never emits and a derivation must still get
    # right. It must poison its OWN 32 elements and stop: letting it set the row scale would
    # turn 32 bad elements into a bad row, a strictly larger claim than the bytes make.
    data = (torch.randn(8, 128) * 10).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    scale = torch.full((8, 4), 127, dtype=torch.uint8)
    scale[2, 1] = 255
    derived = derive_rowwise(torch, data, scale)
    as_float = derived.data.float() * derived.scale.unsqueeze(-1)
    poisoned = int(torch.isnan(as_float).sum())
    check(
        "an E8M0 NaN block poisons its OWN 32 elements and no more",
        bool(torch.isfinite(derived.scale).all())
        and poisoned == BLOCK
        and int(torch.isnan(as_float[2]).sum()) == BLOCK,
        f"{poisoned} NaN elements, all in the one block that carried byte 255; every row "
        "scale stayed finite",
    )

    # A RAGGED mxfp8 tail has no rowwise derivation and says so rather than mis-striding.
    ragged_data = torch.zeros(8, 48, dtype=torch.float8_e4m3fn)
    ragged_scale = torch.full((8, 2), 127, dtype=torch.uint8)
    refused = ""
    try:
        derive_rowwise(torch, ragged_data, ragged_scale)
    except AssertionError as exc:
        refused = str(exc)
    check(
        "RED: a ragged 48-column mxfp8 tensor refuses derivation rather than mis-striding",
        "ragged" in refused,
        f"{refused[:150]} — the producing job refuses non-divisible columns, so no real "
        "artifact carries one; a decoder that only works on its own producer's output is "
        "still a gap and is named as one",
    )
    print(
        f"  penalty    worst {worst_penalty:.3f}x over direct rowwise on these shapes. What "
        f"the\n             derivation is ALLOWED to cost is job-009's gate, not this file's."
    )


def main(argv: list[str]) -> int:
    for name in argv or ["vectors", "probe", "carrier", "arms", "derive"]:
        {
            "vectors": vectors,
            "probe": probe,
            "carrier": carrier,
            "arms": arms,
            "derive": derive,
        }[name]()
    print(f"\n{PASS} ok, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
