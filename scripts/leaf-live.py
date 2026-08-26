#!/usr/bin/env python3
"""The NATIVE-ENCODED FILL RUNG, verified: leaf replacement, and the chooser that picks it.

Not a test (tracker README #160): every leg runs the real provider, the real probe, the
real fill plane and the real module tree, and prints numbers.

Two lanes reached this rung independently and both stopped at the same place. #515b
qualified Blackwell-native MXFP8 on an RTX 5090 and minted NO record, correctly, because
a record for a fill that does not exist is the one thing the capability table forbids.
#517e reached it from Hopper's side, registered no provider, and named what was missing:
LEAF REPLACEMENT — a real `nn.Module` holding the encoded role set — plus two latent
defects that fire the moment an encoding has a second implementation. This script is the
evidence for the half that was missing.

  providers  digest -> ORDERED CANDIDATES: two routes under one encoding, two digests
  rank       `Capabilities.admit` RANKS by a stated objective; nothing rides dict order
  leaf       the module itself: its roles ARE its weight, and it has no `.weight`
  mixed      #543g: three encodings, three routes, ONE component — resolution is per TENSOR
  qualify    the real probe on THIS card: which records mint, and which correctly do not
  fill       a real store through the real fill plane; consent, census, ledger, residency
  arms       the red arms, each planted for real and observed refusing
  all        every section above, in order

The fill and qualify legs need a CUDA card because `fill.py`'s streaming path is CUDA
implementation code behind the `accel` boundary. Everything else is pure CPU. The subject
is a 29 KB synthetic carrier with the H3 DiT's own aspect — this leg proves the MACHINERY,
and `scripts/leaf-pod.py` proves what it costs on real weights.

    corpus/.venv/bin/python scripts/leaf-live.py all
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TFS = Path("/home/fidika/cozy_v2/tensorfs")
WORK = Path("/tmp/cozy-leaf")
STORE = WORK / "store"
CARRIER = WORK / "carrier.safetensors"
RELEASE = "cozy/leaf-live@cr-006"

#: The fixture's shapes, with the H3 DiT's own aspect: a square-ish projection and a wide
#: fc2. Kilobytes of real bytes through the real border, and the same MIXED shape the real
#: artifact has — rank-2 weights encoded, everything else at source precision.
SHAPES = {"attn.out_proj.weight": (64, 224), "mlp.fc2.weight": (32, 448)}

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
    return bool(ok)


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


def _rel(got: Any, want: Any) -> float:
    return float((got.double() - want.double()).norm() / want.double().norm().clamp(min=1e-30))


def _spec(row: Any) -> Any:
    """The code-topology spec for one plan row — what the loader would have handed `fill`."""
    from cozy_runtime.author._loader import TensorSpec
    from cozy_runtime.internal.fill import dtype_name

    return TensorSpec(row.shape, dtype_name(row.dtype), row.encoded.alias, row.nbytes)


def _snr(rel: float) -> float:
    return -20.0 * math.log10(max(rel, 1e-12))


# ------------------------------------------------------------------------ the candidates


def providers() -> None:
    """#517e's FIRST latent defect, and the shape that fixes it.

    `launch_providers()` was digest -> ONE provider, which was true for exactly as long as
    no encoding had two implementations. It has two now, twice: `fp8-rowwise/1` and
    `mxfp8/1` each carry a decode floor and a native leaf. A dict that could only hold one
    of them would have silently kept whichever was assigned last.
    """
    from cozy_runtime.internal.encoding import (
        digests_of,
        implementation_digest,
        launch_providers,
    )

    head("providers — one digest, ORDERED CANDIDATES (#517e's first defect)")
    table = launch_providers()
    for alias in ("plain/1", "fp8-scaled-scalar/1", "fp8-rowwise/1", "mxfp8/1"):
        for digest in digests_of(alias):
            group = table.get(digest, ())
            print(
                f"  {alias:22s} {digest[7:19]}  "
                + ", ".join(f"{p.name} [{p.route}]" for p in group)
            )
    rowwise = digests_of("fp8-rowwise/1")
    mx = digests_of("mxfp8/1")
    check(
        "BOTH physical variants of fp8-rowwise/1 carry a decode floor AND a native leaf",
        len(rowwise) == 2
        and all(
            sorted(p.route for p in table[d]) == ["decoded_float", "native_encoded"]
            for d in rowwise
        ),
        f"{len(rowwise)} digests: "
        + " | ".join(f"{d[7:19]} -> {[p.route for p in table[d]]}" for d in rowwise),
    )
    check(
        "mxfp8/1 carries both routes under its ONE digest",
        len(mx) == 1
        and sorted(p.route for p in table[mx[0]]) == ["decoded_float", "native_encoded"],
        f"{[p.name for p in table[mx[0]]]}",
    )
    everything = [p for group in table.values() for p in group]
    digests = {implementation_digest(p) for p in everything}
    check(
        "every implementation has a DISTINCT identity, so a record cannot admit the wrong one",
        len(digests) == len({p.name for p in everything}),
        f"{len(digests)} identities over {len(everything)} registrations, "
        f"{len({p.name for p in everything})} distinct names",
    )
    check(
        "the two providers of one encoding declare DIFFERENT probe oracles",
        {p.suite for p in table[mx[0]]} == {"mxfp8-decode", "mxfp8-native"},
        "a role-set-keyed suite table would have handed the native leaf the decode oracle: "
        "same roles {data, scale}, different routes — a suite that never runs the kernel, "
        "cannot see #515d's layout hazard, and mints a record saying a GEMM was measured",
    )


# ----------------------------------------------------------------------------- the rank


def _record(implementation: Any, predicate: str, footprint: float, fidelity: float) -> Any:
    from cozy_runtime.internal.encoding import CapabilityRecord, implementation_digest

    return CapabilityRecord(
        encoding_spec_id="sha256:fixture",
        implementation_digest=implementation_digest(implementation),
        runtime_release=RELEASE,
        device_predicate=predicate,
        output_dtype="bfloat16",
        delivery_route=implementation.route,
        evidence_digest="sha256:fixture-evidence",
        resident_bits_per_element=footprint,
        route_deviation_rel=fidelity,
    )


def rank() -> None:
    """#517e's SECOND latent defect: `Capabilities.admit` returned the FIRST match, unranked.

    The numbers below are #517d's, and they are the whole reason this cannot be a ladder
    position: on identical stored bytes the dequant lane measured 20.82 dB at 1.0x VRAM and
    the derived-w8a8 lane 16.72 dB at 0.5003x. Fidelity and footprint order OPPOSITELY, so
    "the better provider" is not a fact — it is a question the deployment answers.
    """
    from cozy_runtime.internal.encoding import (
        Capabilities,
        DeviceFacts,
        EncodingRefusal,
        RowwiseDequant,
        RowwiseNativeLeaf,
        implementation_digest,
    )

    head("rank — the objective decides, and registration order decides nothing (#517e)")
    decode = RowwiseDequant("sha256:fixture")
    native = RowwiseNativeLeaf("sha256:fixture")
    among = {implementation_digest(p): p for p in (decode, native)}
    sm89 = DeviceFacts("cuda", "fixture-sm89", 89, "13.0", "default")
    sm86 = DeviceFacts("cuda", "fixture-sm86", 86, "13.0", "default")
    # Both MEASURED here, on the same card. 16.0 bits is bf16 resident; 8.29 is the payload
    # plus its row scale. The deviations are #517d's, scaled to this fixture's frame.
    caps = Capabilities(
        [
            _record(decode, "cuda.sm89", 16.0, 0.00911),
            _record(native, "cuda.sm89", 8.286, 0.01459),
        ]
    )
    for objective, want in (("footprint", native.name), ("fidelity", decode.name)):
        record = caps.admit(
            "sha256:fixture", sm89, "bfloat16", "fp8-rowwise/1", objective=objective, among=among
        )
        check(
            f"objective {objective!r} selects {want.rsplit('.', 1)[0].rsplit('.', 1)[-1]}",
            among[record.implementation_digest].name == want,
            f"{among[record.implementation_digest].name} [{record.delivery_route}] "
            f"footprint {record.resident_bits_per_element:.3f} b/elt, "
            f"fidelity {record.route_deviation_rel:.5f} rel",
        )
    # ORDER-INDEPENDENCE, proved rather than asserted: the same two candidates in the other
    # registration order must give the same two answers.
    flipped = {implementation_digest(p): p for p in (native, decode)}
    reversed_caps = Capabilities(list(reversed(caps.records)))
    same = all(
        reversed_caps.admit(
            "sha256:fixture", sm89, "bfloat16", "fp8-rowwise/1", objective=o, among=flipped
        ).implementation_digest
        == caps.admit(
            "sha256:fixture", sm89, "bfloat16", "fp8-rowwise/1", objective=o, among=among
        ).implementation_digest
        for o in ("footprint", "fidelity")
    )
    check(
        "reversing BOTH the record order and the candidate order changes nothing",
        same,
        "first-match would have flipped both answers; a ranked table is a function of the "
        "measurements",
    )
    try:
        caps.admit(
            "sha256:fixture", sm89, "bfloat16", "fp8-rowwise/1", objective="latency", among=among
        )
        check("ARM: an unmeasurable objective refuses", False, "it did not refuse")
    except EncodingRefusal as exc:
        check(
            "ARM: `latency` has no axis measured across both routes, so it REFUSES",
            exc.code == "objective_unranked",
            f"[{exc.code}] {str(exc)[:230]}",
        )
    # THE UNMEASURED DEVICE. Only the decode floor has a record that sm86 satisfies, so
    # `footprint` — the objective that WANTS the leaf — takes the floor and does not sulk.
    floor = Capabilities([_record(decode, "cuda.sm86", 16.0, 0.00911), *caps.records])
    record = floor.admit(
        "sha256:fixture", sm86, "bfloat16", "fp8-rowwise/1", objective="footprint", among=among
    )
    check(
        "an UNMEASURED card takes the dequant floor even under the objective that wants the leaf",
        among[record.implementation_digest].name == decode.name,
        f"sm86 -> {among[record.implementation_digest].name}; the leaf's own floor is "
        f"{native.device_predicate} and no record was minted there. An unqualified provider "
        "is never selected, whatever the objective says",
    )
    # A record that survives a provider's DELETION must not select it.
    try:
        caps.admit(
            "sha256:fixture",
            sm89,
            "bfloat16",
            "fp8-rowwise/1",
            objective="footprint",
            among={implementation_digest(decode): decode},
        )
        record = caps.admit(
            "sha256:fixture",
            sm89,
            "bfloat16",
            "fp8-rowwise/1",
            objective="footprint",
            among={implementation_digest(decode): decode},
        )
        check(
            "a record for an implementation this build no longer carries admits NOTHING",
            among[record.implementation_digest].name == decode.name,
            "the native record is present and its implementation is not offered, so the "
            "ranking never sees it",
        )
    except EncodingRefusal as exc:  # pragma: no cover - would mean the filter is wrong
        check("a record for a departed implementation admits nothing", False, str(exc)[:160])


# ----------------------------------------------------------------------------- the leaf


def leaf() -> None:
    """The module. Its ROLES ARE ITS WEIGHT, and that is the whole of the rung.

    Everything else on the delivery ladder is invisible to author code by construction: the
    weights arrive as floats in the destinations the census named. This one cannot be. It
    is checked here at the smallest scale that is still real — a genuine `nn.Linear`
    replaced by a genuine leaf, on CPU, where `torch._scaled_mm`'s rowwise recipe happens
    to run too.
    """
    import torch
    from torch import nn

    from cozy_runtime.internal.encoding import (
        RowwiseDequant,
        RowwiseNativeLeaf,
        quantize_rowwise,
    )

    head("leaf — a real nn.Module whose weight is the encoded role set")
    torch.manual_seed(0x517E)
    linear = nn.Linear(224, 64, bias=True, dtype=torch.bfloat16)
    weight = linear.weight.detach().clone()
    derived = quantize_rowwise(torch, weight)
    roles = {"data": derived.data, "scale": derived.scale}
    provider = RowwiseNativeLeaf("sha256:fixture")

    holder = nn.Module()
    holder.out_proj = linear
    built = provider.leaf(torch, roles, linear, torch.bfloat16)
    check(
        "a leaf is a real nn.Module, which is WHY leaf replacement is machinery",
        isinstance(built, nn.Module),
        "`nn.Module.__setattr__` refuses any other child, so the payload cannot be assigned "
        "where the float weight was — the linear op has to be swapped whole",
    )
    holder.out_proj = built  # the swap `__setattr__` accepts precisely because it is a Module
    check(
        "the replaced leaf has NO `.weight` — which is what author consent is about",
        not hasattr(holder.out_proj, "weight") and hasattr(holder.out_proj, "data"),
        f"state_dict keys {sorted(holder.out_proj.state_dict())}",
    )
    check(
        "the roles are BUFFERS, never Parameters",
        not list(built.parameters()) and len(list(built.buffers())) >= 2,
        f"{len(list(built.parameters()))} parameters, {len(list(built.buffers()))} buffers — "
        "an fp8 Parameter is hostile to every `.to(dtype)` that walks a module",
    )

    x = torch.randn(96, 224, dtype=torch.bfloat16) * 0.5
    bias = linear.bias.detach()
    with torch.inference_mode():
        got = holder.out_proj(x)
    check(
        "the leaf's OUTPUT DTYPE is the contract's dtype, not the payload's",
        got.dtype == torch.bfloat16 and got.shape == (96, 64),
        f"{tuple(got.shape)} {got.dtype} — the contract dtype becomes the encoding's "
        "declared output dtype, and the destination is what declares it",
    )

    # THE THREE LANES ON ONE SCALE, the protocol #517d used: an f64 reference over the
    # exactly dequantized weights, the decode route, and the leaf.
    exact = derived.data.double() * derived.scale.double().reshape(-1, 1)
    reference = x.double() @ exact.t() + bias.double()
    decoded = torch.empty(64, 224, dtype=torch.bfloat16)
    RowwiseDequant("sha256:fixture").decode(torch, roles, decoded)
    with torch.inference_mode():
        float_out = x @ decoded.t() + bias
    leaf_rel, float_rel = _rel(got, reference), _rel(float_out, reference)
    print(
        f"  fidelity   dequant-bf16 {float_rel:.6f} ({_snr(float_rel):.2f} dB)   "
        f"native leaf {leaf_rel:.6f} ({_snr(leaf_rel):.2f} dB)   "
        f"activation quantization costs {_snr(float_rel) - _snr(leaf_rel):.2f} dB"
    )
    check(
        "both routes compute the SAME weights, and the leaf is the lossier of the two",
        float_rel < leaf_rel < 0.30,
        "that ordering IS #515a and #517d: the native route is w8a8, so it quantizes the "
        "activation, and the decode route never does. It is a real cost, measured, and the "
        "reason the chooser needs an objective rather than a ladder position",
    )
    stored = sum(int(t.numel() * t.element_size()) for t in roles.values())
    resident = int(decoded.numel() * decoded.element_size())
    print(
        f"  footprint  leaf {stored} B ({8.0 * stored / weight.numel():.3f} bits/element)  "
        f"vs decoded bf16 {resident} B (16.000)  ratio {stored / resident:.4f}"
    )
    check(
        "the leaf holds the weight at ~half the bytes, which the decode route cannot",
        stored / resident < 0.55,
        "the decode route's resident weights are float either way — that is what 'it buys "
        "bytes read and load time, NOT VRAM' has always meant",
    )


# -------------------------------------------------------------------------- the mixing


def mixed() -> None:
    """#543g: provider resolution is PER TENSOR, and one component may hold several.

    The `quantize-mixed` job kind emits carriers whose per-tensor encodings differ under one
    allocation map — nvfp4 beside fp8-rowwise beside plain, in one component. Those carriers
    cannot reach a snapshot yet (two TensorFS planner defects, fix dispatched), so this arm
    drives the runtime's own seams with a HAND-BUILT mixed row set instead of a store.

    What it proves is that per-tensor resolution is not a generalisation this design needs
    — it is the shape it already has. `Selector.select` takes ONE tensor's `Encoded` and
    keys the capability lookup on that tensor's own spec digest; `_stream` and `_decode`
    branch per row on the route THAT row selected. A component holding a native leaf beside
    a decoded float weight beside a verbatim copy is not a special case anywhere, which is
    why the fill leg's own fixture already is one.
    """
    import torch

    from cozy_runtime.internal.encoding import (
        Encoded,
        RolePart,
        Selector,
        digests_of,
        launch_providers,
        measure_device,
    )
    from cozy_runtime.internal.fill import PlanRow, servability, survey_routes
    from cozy_runtime.internal.probe import qualify as run_suite

    head("mixed — three encodings, three routes, ONE component (#543g)")
    device = measure_device(torch)
    table = launch_providers()
    caps, _ = run_suite(
        torch,
        {
            d: table[d]
            for group in ("plain/1", "fp8-rowwise/1", "mxfp8/1")
            for d in digests_of(group)
        },
        device,
        ["bfloat16"],
        release=RELEASE,
        cache=None,
    )
    selector = Selector(table, caps, device, "footprint")
    plain, rowwise, mx = (
        digests_of("plain/1")[0],
        digests_of("fp8-rowwise/1")[0],
        digests_of("mxfp8/1")[0],
    )
    rows = [
        PlanRow(
            "block.norm.weight",
            "norm.weight",
            "block",
            "bf16",
            (64,),
            128,
            Encoded(plain, "plain/1", (RolePart("value", "bf16", (64,), 128),)),
        ),
        PlanRow(
            "block.attn.out_proj.weight",
            "attn.out_proj.weight",
            "block",
            "bf16",
            (64, 224),
            28672,
            Encoded(
                rowwise,
                "fp8-rowwise/1",
                (
                    RolePart("data", "f8_e4m3fn", (64, 224), 14336),
                    RolePart("scale", "f32", (64,), 256),
                ),
            ),
        ),
        PlanRow(
            "block.mlp.fc1.weight",
            "mlp.fc1.weight",
            "block",
            "bf16",
            (128, 224),
            57344,
            Encoded(
                mx,
                "mxfp8/1",
                (
                    RolePart("data", "f8_e4m3fn", (128, 224), 28672),
                    RolePart("scale", "u8", (128, 7), 896),
                ),
            ),
        ),
    ]
    chosen = {r.key: selector.select(r.key, r.encoded, "bfloat16") for r in rows}
    for key, selection in chosen.items():
        print(
            f"  {key.split('.', 1)[1]:24s} {selection.encoded.alias:20s} -> "
            f"{selection.provider.name} [{selection.provider.route}]"
        )
    check(
        "ONE component resolves THREE providers on THREE routes, keyed per tensor's digest",
        {s.provider.route for s in chosen.values()}
        == {"verbatim", "decoded_float", "native_encoded"},
        "the capability lookup takes the TENSOR's spec digest, never the artifact's or the "
        "component's — a mixed carrier is not a special case, it is the ordinary path with "
        "more than one answer",
    )
    check(
        "the NATIVE leaf and the DECODED float sit side by side under one objective",
        chosen["block.attn.out_proj.weight"].provider.route == "native_encoded"
        and chosen["block.mlp.fc1.weight"].provider.route == "decoded_float",
        "fp8-rowwise has a qualified leaf on this card and mxfp8's leaf is cuda.sm120, so "
        "the SAME objective lands on different routes for different tensors — which is "
        "exactly what a per-artifact answer could not express",
    )
    routes, refusal = survey_routes(rows, selector)
    answer = servability(rows, selector)
    print(f"  survey     {routes}  refusal={refusal or 'none'}")
    print(f"  servable   {answer.line()}")
    from cozy_runtime.internal.delivery import eligible_rungs

    rungs = eligible_rungs(routes)
    check(
        "a MIXED artifact's route census reaches BOTH encoded rungs, and says so",
        {r.name for r in rungs} >= {"aot_decode", "native_encoded"} and answer.serves,
        f"eligible rungs {[r.name for r in rungs]} — a rung is a LABEL over a per-tensor "
        "fill; nothing downstream reads it to pick a provider",
    )


# ------------------------------------------------------------------------- the fixture


def carrier() -> None:
    """One synthetic `fp8-rowwise/1` carrier through the REAL border into a real store."""
    import torch
    from safetensors.torch import save_file

    from cozy_runtime.internal.encoding import quantize_rowwise

    head("carrier — a real fp8-rowwise artifact, ingested by the real border")
    WORK.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0x517E)
    out: dict[str, Any] = {"norm.weight": torch.randn(64, dtype=torch.bfloat16)}
    for key, shape in SHAPES.items():
        weight = torch.randn(*shape, dtype=torch.bfloat16) * 0.5
        derived = quantize_rowwise(torch, weight)
        out[key] = derived.data
        # #501d: the companion is named off the MODULE STEM. `out_proj.weight_scale`, never
        # `out_proj.weight.weight_scale` — present-but-invisible is worse than absent.
        out[key[: -len(".weight")] + ".weight_scale"] = derived.scale
    save_file(out, str(CARRIER), metadata={"format": "pt"})
    tfs_cli(
        "ingest",
        "bank",
        "safetensors.diffusers",
        "leaf-rowwise",
        "block",
        "diffusers.identity/1",
        "--source",
        f"block={CARRIER}",
        "--provenance",
        "synthetic",
        "--note",
        "native-encoded leaf fixture",
    )
    plan = tfs_cli("ingest", "plan", "block=fp8-rowwise/1", "--source", f"block={CARRIER}")
    check(
        "the border SELECTS fp8-rowwise/1 per tensor, folding the stem-named companions",
        plan.returncode == 0 and "fp8-rowwise/1" in plan.stdout,
        next(
            (ln.strip() for ln in plan.stdout.splitlines() if "fp8-rowwise/1" in ln),
            plan.stderr[:200],
        ),
    )
    STORE.mkdir(parents=True, exist_ok=True)
    if not (STORE / "objects").is_dir():
        tfs_cli("store", "init", str(STORE))
    session = "leaf" + str(CARRIER.stat().st_mtime_ns % 100000)
    run = tfs_cli(
        "ingest",
        "run",
        str(STORE),
        "block=fp8-rowwise/1",
        "--source",
        f"block={CARRIER}",
        "--session",
        session,
        "--tenant",
        "local",
    )
    install = tfs_cli("ingest", "install", str(STORE), session)
    if run.returncode or install.returncode:
        check("ingest", False, (run.stderr + install.stderr)[:300])
        raise SystemExit(1)
    import tensorfs

    snapshot = str(tensorfs.ingest_installed(tensorfs.Store.open(str(STORE)))[-1])
    (WORK / "snapshot.txt").write_text(snapshot)
    print(f"  snapshot   {snapshot[:30]}…  carrier {CARRIER.stat().st_size} B")


# ------------------------------------------------------------------------- the pipeline


class LeafBlock:
    """A block with two encoded GEMMs and one plain norm. The MIXED shape a real artifact
    has, at the smallest size that is still a real module tree."""

    def __init__(self, config: Any) -> None:
        import torch
        from torch import nn

        block = nn.Module()
        block.norm = nn.RMSNorm(64, dtype=torch.bfloat16, elementwise_affine=True)
        attn = nn.Module()
        attn.out_proj = nn.Linear(224, 64, bias=False, dtype=torch.bfloat16)
        block.attn = attn
        mlp = nn.Module()
        mlp.fc2 = nn.Linear(448, 32, bias=False, dtype=torch.bfloat16)
        block.mlp = mlp
        self.components: dict[str, Any] = {"block": block}

    def forward(self, torch: Any, x: Any) -> Any:
        block = self.components["block"]
        return block.attn.out_proj(x), block.mlp.fc2(x.new_zeros(x.shape[0], 448))


def build_block(config: Any) -> LeafBlock:
    return LeafBlock(config)


def make_model(consent: str | None) -> Any:
    """The endpoint, with and without the consent declaration. Nothing else differs."""
    from cozy_runtime.author import Model, uses_components
    from cozy_runtime.author._loader import Loader

    if consent is None:

        class Silent(Model[LeafBlock]):
            pipe: LeafBlock

            def load(self, loader: Loader) -> None:
                self.pipe = loader.construct(LeafBlock, factory=build_block)

            @uses_components("block")
            def step(self, torch: Any, x: Any) -> Any:
                with torch.inference_mode():
                    return self.pipe.forward(torch, x)

        return Silent

    class Consenting(Model[LeafBlock], encoded_leaves=consent):
        pipe: LeafBlock

        def load(self, loader: Loader) -> None:
            self.pipe = loader.construct(LeafBlock, factory=build_block)

        @uses_components("block")
        def step(self, torch: Any, x: Any) -> Any:
            with torch.inference_mode():
                return self.pipe.forward(torch, x)

    return Consenting


def _backend(objective: str, **kwargs: Any) -> Any:
    from cozy_runtime.internal.fill import (
        Checkpoint,
        ResourceLedger,
        StreamingFillBackend,
    )

    snapshot = (WORK / "snapshot.txt").read_text().strip()
    books = {"block": Checkpoint(str(STORE), snapshot)}
    rows = books["block"].rows("block")
    backend = StreamingFillBackend(
        books,
        rows,
        ledger=ResourceLedger(budgets={"vram": 1 << 30, "pinned": 64 << 20}),
        window_bytes=1 << 20,
        slots=4,
        readers=4,
        inflight=2,
        probe_cache=WORK / "probe",
        decode_scratch_bytes=1 << 20,
        release=RELEASE,
        objective=objective,
        **kwargs,
    )
    return backend, rows


def _acquire(backend: Any, rows: Any, consent: str | None) -> Any:
    from cozy_runtime.author import ModelRegistry
    from cozy_runtime.author._loader import Artifact, Config
    from cozy_runtime.internal.derive import Observations, serving_substrate
    from cozy_runtime.internal.fill import topology_of

    registry = ModelRegistry(
        release=RELEASE,
        backend=backend,
        substrate=lambda: serving_substrate(f"sm{backend.device_facts.sm}", Observations()),
    )
    artifact = Artifact(
        (WORK / "snapshot.txt").read_text().strip(),
        topology_of(rows),
        Config({}, "block.config"),
        custody="canonical",
    )
    backend.expect({"block": tuple(r.key for r in rows)})
    return registry.acquire("step.models.model", make_model(consent), artifact)


# ------------------------------------------------------------------------ qualification


def qualify() -> None:
    """The real probe on THIS card. What mints, and — more importantly — what does not."""
    import torch

    from cozy_runtime.internal.encoding import (
        digests_of,
        implementation_digest,
        launch_providers,
        measure_device,
    )
    from cozy_runtime.internal.probe import qualify as run_suite

    head("qualify — capability records by MEASUREMENT, on the card that is actually here")
    device = measure_device(torch)
    print(f"  device     {device.predicate} ({device.name}, driver cuda {device.driver})")
    table = launch_providers()
    subject = {d: table[d] for d in (*digests_of("fp8-rowwise/1"), *digests_of("mxfp8/1"))}
    names = {implementation_digest(p): p.name for group in subject.values() for p in group}
    started = time.perf_counter()
    caps, evidence = run_suite(torch, subject, device, ["bfloat16"], release=RELEASE, cache=None)
    took = time.perf_counter() - started
    minted = {names[r.implementation_digest] for r in caps.records}
    for observation in evidence["observations"]:
        verdict = "MINTED " if observation["evidence"]["passed"] else "no record"
        axes = observation["evidence"].get("route_axes", {})
        print(
            f"  {verdict}  {observation['implementation']:34s} "
            f"[{observation['delivery_route']:14s}] "
            f"footprint {axes.get('resident_bits_per_element', -1):7.3f} b/elt  "
            f"fidelity {axes.get('deviation_rel', -1):9.6f}"
        )
        if not observation["evidence"]["passed"]:
            print(f"             why: {str(observation['evidence'].get('note', ''))[:150]}")
    print(f"  suite      {evidence['suite_ms']} ms ({took:.1f} s wall)")
    check(
        "the mxfp8 NATIVE leaf mints NOTHING here — its floor is cuda.sm120 and this is not",
        "cozy.mxfp8.native-leaf/1" not in minted,
        f"minted: {sorted(minted)}. #515c put the predicate at cuda.sm120 EXACTLY, not "
        "cuda.sm100+, because a hand-written floor is what #511b's fail-closed-by-"
        "measurement rule forbids — and no B200 has ever run",
    )
    check(
        "every minted record carries BOTH measured ranking axes",
        bool(caps.records)
        and all(
            r.resident_bits_per_element > 0 and r.route_deviation_rel >= 0 for r in caps.records
        ),
        f"{len(caps.records)} records, "
        + ", ".join(
            f"{names[r.implementation_digest].rsplit('.', 2)[-2]}="
            f"{r.resident_bits_per_element:.2f}b/{r.route_deviation_rel:.5f}"
            for r in caps.records
        ),
    )
    if device.sm >= 89:
        check(
            "the rowwise NATIVE leaf IS qualified here, by a suite that ran its real kernel",
            "cozy.fp8-rowwise.native-leaf/1" in minted,
            "#517c measured rowwise `_scaled_mm` working on sm89, which makes v1's "
            "W8A8_ROWWISE_MIN_SM=90 a PROFITABILITY floor mislabelled as a capability one",
        )
        bench = next(
            (
                o["evidence"].get("micro_bench")
                for o in evidence["observations"]
                if o["implementation"] == "cozy.fp8-rowwise.native-leaf/1"
                and o["evidence"]["passed"]
            ),
            None,
        )
        if bench:
            print(
                f"  leaf bench {bench['shape']}  leaf {bench['leaf_ns'] / 1000:.1f} us  "
                f"bf16 GEMM {bench['float_gemm_ns'] / 1000:.1f} us  {bench['speedup_x']:.3f}x "
                "(activation quantize INSIDE the timed region, unfused — #515f)"
            )
    (WORK / "qualify.json").write_text(json.dumps(evidence, indent=1))


# ------------------------------------------------------------------------------- the fill


def fill() -> None:
    """The rung itself: a real artifact, the real fill plane, real leaves installed."""
    import torch

    from cozy_runtime.internal.fill import FillRefusal, _tensors, dtype_name

    head("fill — the native rung end to end: consent, replacement, census, ledger")

    # ARM, RED FIRST: an endpoint that never declared consent gets the typed refusal, and
    # gets it before a destination is reserved.
    backend, rows = _backend("footprint")
    if not any(
        backend.selector.select(r.key, r.encoded, dtype_name(r.dtype)).provider.route
        == "native_encoded"
        for r in rows
    ):
        check(
            "this card qualifies the native leaf",
            False,
            "no row takes the native route here, so the rung cannot be exercised on this "
            "device; the chooser correctly took the decode floor",
        )
        return
    try:
        _acquire(backend, rows, None)
        check("ARM: an endpoint that did not consent refuses", False, "it did not refuse")
    except FillRefusal as exc:
        check(
            "ARM: an endpoint that never declared consent REFUSES, before any reservation",
            exc.code == "leaf_unconsented" and not backend.components,
            f"[{exc.code}] {str(exc)[:260]}",
        )
    except Exception as exc:  # the refusal may arrive wrapped by the loader's own boundary
        cause = getattr(exc, "__cause__", None) or exc
        check(
            "ARM: an endpoint that never declared consent REFUSES, before any reservation",
            getattr(cause, "code", "") == "leaf_unconsented",
            f"[{getattr(cause, 'code', type(cause).__name__)}] {str(cause)[:260]}",
        )
    backend.rollback([])

    # ...and the SAME artifact, the SAME card, with the declaration present.
    backend, rows = _backend("footprint")
    before_alloc = torch.cuda.memory_allocated()
    model = _acquire(backend, rows, "accept")
    block = backend.components["block"]
    print(f"  servable   {backend.servability.line()}")
    print(
        "  routes     "
        + ", ".join(
            f"{k.split('.', 1)[1]} -> {v.provider.route}"
            for k, v in sorted(backend.selected.items())
        )
    )
    check(
        "every encoded tensor became a LEAF, and the plain one did not",
        len(backend.replaced) == len(SHAPES) and backend.decoded["leaves"] == len(SHAPES),
        "; ".join(
            f"{r.module_path.split('.', 1)[1]}: {r.was} -> {r.now} "
            f"({r.stored_bytes} B stored, {r.freed_bytes} B freed)"
            for r in backend.replaced.values()
        ),
    )
    check(
        "the module tree HOLDS the leaves — the swap is real, not a bookkeeping entry",
        type(block.attn.out_proj).__name__ == "EncodedLinear"
        and not hasattr(block.attn.out_proj, "weight"),
        f"block.attn.out_proj is {type(block.attn.out_proj).__name__} with "
        f"{sorted(block.attn.out_proj.state_dict())}",
    )
    substitution = sorted(k for r in backend.replaced.values() for k in r.role_keys)
    census_keys = sorted(f"block.{k}" for k in block.state_dict())
    check(
        "the CENSUS is exactly the declared substitution: replaced gone, roles arrived",
        all(k in census_keys for k in substitution)
        and not any(r.contract_key in census_keys for r in backend.replaced.values()),
        f"census {census_keys}",
    )
    resident = sum(int(t.numel() * t.element_size()) for _, t in _tensors(block))
    float_bytes = sum(r.freed_bytes for r in backend.replaced.values())
    stored = sum(r.stored_bytes for r in backend.replaced.values())
    ledger = backend.ledger.rows()
    print(
        f"  residency  encoded weights {stored} B vs bf16 {float_bytes} B "
        f"({stored / float_bytes:.4f}x); whole component {resident} B; "
        f"ledger vram charged {ledger['vram']['charged']} B, peak {ledger['vram']['peak']} B"
    )
    check(
        "the LEDGER settled to the measured truth, not to arithmetic",
        backend.vram_charge["block"] == resident
        and ledger["vram"]["charged"] == resident
        and ledger["vram"]["peak"] > resident,
        f"charged {ledger['vram']['charged']} B == module {resident} B, and the peak "
        f"{ledger['vram']['peak']} B is higher because the float destinations and the "
        "payload are BOTH live for the length of one flush",
    )
    check(
        "the decode scratch class was never charged for bytes that stayed",
        ledger["decode"]["charged"] == 0 and ledger["decode"]["peak"] == 0,
        "a native role is not scratch: charging it to `decode` and releasing it in the "
        "`finally` would tell the ledger the component gave back memory it still holds",
    )
    allocator = torch.cuda.memory_allocated() - before_alloc
    print(f"  allocator  +{allocator} B live after the fill (ledger says {resident} B)")

    x = torch.randn(96, 224, dtype=torch.bfloat16, device="cuda") * 0.5
    with torch.inference_mode():
        out, _ = model.step(torch, x)
    check(
        "the endpoint's own forward runs THROUGH the leaves and returns the contract dtype",
        out.dtype == torch.bfloat16 and out.shape == (96, 64) and bool(torch.isfinite(out).all()),
        f"{tuple(out.shape)} {out.dtype}, finite",
    )
    globals()["_FILLED"] = backend


# ------------------------------------------------------------------------------ the arms


def arms() -> None:
    """Every law of the rung, planted for real and observed refusing."""
    import torch
    from torch import nn

    from cozy_runtime.internal.encoding import (
        Capabilities,
        DeviceFacts,
        EncodingRefusal,
        MicroScaledNativeLeaf,
        RowwiseNativeLeaf,
        implementation_digest,
        quantize_rowwise,
    )
    from cozy_runtime.internal.fill import FillRefusal, PlanRow

    head("arms — the rung's laws, each planted")

    backend = globals().get("_FILLED")
    if backend is None:
        check("arms need the fill leg's backend", False, "run `fill` first")
        return

    # ARM 1 — a destination that is not a `.weight` has no linear op to replace.
    row = next(iter(backend.plan.values()))
    for key, plan_row, why in (
        (
            "block.attn.out_proj.bias",
            PlanRow(
                "block.attn.out_proj.bias",
                "attn.out_proj.bias",
                "block",
                "bf16",
                (64,),
                128,
                row.encoded,
            ),
            "not a `.weight`",
        ),
        (
            "block.attn.out_proj.weight",
            PlanRow(
                "block.attn.out_proj.weight",
                "attn.out_proj.weight",
                "block",
                "bf16",
                (2, 4, 8),
                128,
                row.encoded,
            ),
            "rank 3",
        ),
    ):
        try:
            backend._leaf_site(key, plan_row, torch.zeros(1))
            check(f"ARM: a leaf site that is {why} refuses", False, "it did not refuse")
        except FillRefusal as exc:
            check(
                f"ARM: a leaf site that is {why} refuses TYPED",
                exc.code == "leaf_topology",
                f"[{exc.code}] {exc.args[0][:190]}",
            )

    # ARM 2 — a TIED or re-wired weight resolves to a module whose `.weight` is not this
    # tensor. Installing there would compute a different model in silence.
    real = backend.plan["block.attn.out_proj.weight"]
    try:
        backend._leaf_site(
            "block.attn.out_proj.weight", real, torch.zeros(64, 224, dtype=torch.bfloat16)
        )
        check("ARM: a weight the named module does not own refuses", False, "it did not refuse")
    except FillRefusal as exc:
        check(
            "ARM: a destination the named module does not OWN refuses typed",
            exc.code == "leaf_topology",
            f"[{exc.code}] {exc.args[0][:190]}",
        )

    # ARM 3 — an EVICTION of a component holding leaves. The re-fill would walk destinations
    # the header does not supply, so the honest answer is a refusal, not a half-build.
    try:
        backend.evict("block")
        check("ARM: evicting an encoded-leaf component refuses", False, "it did not refuse")
    except FillRefusal as exc:
        check(
            "ARM: a component holding leaves cannot be evicted, and says exactly why",
            exc.code == "leaf_topology" and "substitution map" in exc.args[0],
            f"[{exc.code}] {exc.args[0][:210]}",
        )

    # ARM 4 — the CENSUS SUBSTITUTION law. Plant a leaf that lost a role after the fill
    # declared it, and the reconciliation must catch the topology the ledger is pricing.
    block = backend.components["block"]
    holder, victim = block.attn, block.attn.out_proj
    delattr(victim, "scale")
    victim._cozy_roles = ("data",)
    try:
        backend._reconcile_leaves()
        check("ARM: a leaf that lost a declared role refuses", False, "it did not refuse")
    except FillRefusal as exc:
        check(
            "ARM: a substitution the census cannot confirm is a typed refusal",
            exc.code == "leaf_topology",
            f"[{exc.code}] {exc.args[0][:210]}",
        )
    finally:
        victim.register_buffer("scale", torch.ones(64, dtype=torch.float32, device="cuda"))
        victim._cozy_roles = ("data", "scale")
        holder.out_proj = victim

    # ARM 5 — the mxfp8 native leaf on a card that is not sm120. The record is the ONLY
    # thing that can admit it, and #515c put the predicate at sm120 exactly.
    mx = MicroScaledNativeLeaf("sha256:fixture")
    among = {implementation_digest(mx): mx}
    for sm, name in ((89, "L40S-class"), (100, "B200")):
        try:
            Capabilities([]).admit(
                "sha256:fixture",
                DeviceFacts("cuda", name, sm, "13.0", "default"),
                "bfloat16",
                "mxfp8/1",
                objective="footprint",
                among=among,
            )
            check(f"ARM: sm{sm} admits no mxfp8 native record", False, "it ADMITTED one")
        except EncodingRefusal as exc:
            check(
                f"ARM: sm{sm} ({name}) admits NO mxfp8 native record",
                exc.code == "encoding_unqualified",
                f"[{exc.code}] {str(exc)[-190:]}",
            )

    # ARM 6 — EXPECT/COMMIT COMPLETENESS, on a component whose destinations become leaves.
    # A DestinationSet is COMPLETE or the component does not commit, and leaf replacement
    # does not get an exemption: a module tree half of whose linear ops were swapped is the
    # one state no request may ever observe.
    partial, rows = _backend("footprint")
    partial.expect({"block": tuple(r.key for r in rows)})
    from cozy_runtime.author._loader import census as walk_census

    pipe = LeafBlock(None)
    live = partial.materialize(pipe, walk_census(pipe), encoded_leaves="accept")
    first = next(k for k in live if k.endswith("out_proj.weight"))
    partial.fill(first, _spec(partial.plan[first]), live[first])
    try:
        partial.commit()
        check("ARM: an incomplete encoded component refuses", False, "it did not refuse")
    except FillRefusal as exc:
        check(
            "ARM: a DestinationSet missing a leaf's own destination REFUSES at commit",
            exc.code == "incomplete_fill",
            f"[{exc.code}] {exc.args[0][:190]}",
        )
    partial.rollback(list(live))

    # ARM 7 — a PARKED encoded component. Staging re-fills the checkpoint's own keys, which
    # stop being this module's keys the moment a leaf lands, so the two plans are refused
    # together rather than half-served.
    parking, rows = _backend("footprint")
    parking.resident_budget = 1024
    parking.expect({"block": tuple(r.key for r in rows)})
    pipe = LeafBlock(None)
    try:
        parking.materialize(pipe, walk_census(pipe), encoded_leaves="accept")
        check("ARM: parking an encoded-leaf component refuses", False, "it did not refuse")
    except FillRefusal as exc:
        check(
            "ARM: a placement plan that would PARK an encoded-leaf component refuses typed",
            exc.code == "leaf_topology",
            f"[{exc.code}] {exc.args[0][:200]}",
        )
    parking.rollback([])

    # ARM 8 — the OUTPUT DTYPE law. A leaf built for one contract dtype must not quietly
    # emit another; the probe checks it and so does this.
    linear = nn.Linear(224, 64, bias=False, dtype=torch.bfloat16)
    derived = quantize_rowwise(torch, linear.weight.detach())
    for dtype in (torch.bfloat16, torch.float16):
        built = RowwiseNativeLeaf("sha256:fixture").leaf(
            torch, {"data": derived.data, "scale": derived.scale}, linear, dtype
        )
        with torch.inference_mode():
            emitted = built(torch.randn(32, 224, dtype=dtype))
        check(
            f"ARM: a leaf built for {dtype} emits {dtype} and nothing else",
            emitted.dtype == dtype,
            f"{emitted.dtype} — the contract dtype becomes the ENCODING's declared output "
            "dtype, and a capability record exists per output dtype for exactly this reason",
        )


# ------------------------------------------------------------------------------ main


def main(argv: list[str]) -> int:
    sections = {
        "providers": providers,
        "rank": rank,
        "leaf": leaf,
        "mixed": mixed,
        "carrier": carrier,
        "qualify": qualify,
        "fill": fill,
        "arms": arms,
    }
    names = argv or ["providers"]
    if names == ["all"]:
        names = list(sections)
    started = time.perf_counter()
    for name in names:
        if name not in sections:
            raise SystemExit(f"unknown section {name!r}; have {sorted(sections)}")
        sections[name]()
    print(f"\n{PASS} ok, {FAIL} failed in {time.perf_counter() - started:.1f} s")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
