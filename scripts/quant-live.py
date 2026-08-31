#!/usr/bin/env python3
"""cr-006 live verification: a REAL fp8-scaled-scalar artifact, filled and served.

Not a test (tracker README #160): every leg below runs the real border, the real fill
plane and the real package, and prints numbers.

The proof needs an artifact in the encoding proto-001 SELECTED (`fp8-scaled-scalar/1`,
the Comfy `scaled_fp8` dialect). The H3 DiT that carries it is 72.90 GiB and its weights
do not transit this box, so this script MAKES one from bytes that are already here: the
real SDXL UNet, quantized by the registry spec's own value law, through the real ingest
border, into the real store. That is a weaker provenance than producer-mined bytes and it
is stated rather than hidden — what it proves is the RUNTIME's decode and the fill plane's
seam, not ComfyUI's quantizer.

  produce   quantize the real SDXL UNet -> a scaled_fp8-dialect carrier
  ingest    bank the dialect fingerprint, plan, run and install it (the real border)
  dequant   fill on the 4070 and check every decoded weight two ways
  arms      the red arms, each planted for real
  image     a real SDXL generation on decoded weights vs the fp16 baseline
  bench     fill wall / VRAM / step time, fp8 against plain
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

HF = Path("/home/fidika/cozy_v2/tensorfs-bench/hf/sdxl")
TFS = Path("/home/fidika/cozy_v2/tensorfs")
WORK = Path("/tmp/cozy-cr006")
STORE = WORK / "store"
CARRIER = WORK / "unet_fp8_scaled.safetensors"
#: The plain fp16 SDXL store cr-008b built. Read-only here: it is the BASELINE arm.
PLAIN_STORE = Path("/tmp/cozy-sdxl4/store")
PLAIN_SNAPSHOTS = Path("/tmp/cozy-sdxl4/snapshots.json")

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
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def gpu_used_mib() -> int:
    """Device memory held by everyone EXCEPT this process.

    Discounting our own context is not cosmetic. The bench arms run in one process, so by
    the second arm this process already holds its own ~200 MiB CUDA context — and a
    threshold read off the raw total then waits for the card to be quiet OF ITSELF, which
    it never will be. Measured the direct way: a `bench` run sat for twelve minutes
    waiting on its own 204 MiB.
    """
    import os

    try:
        total = int(
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return -1
    mine = 0
    for line in apps.strip().splitlines():
        pid, _, used = line.partition(",")
        if pid.strip().isdigit() and int(pid) == os.getpid():
            mine += int(used.strip() or 0)
    return max(total - mine, 0)


def require_quiet_card(seconds: float = 3600.0) -> int:
    """8 GiB fits ONE resident pipeline. A run that starts on a busy card measures the
    neighbour, so waiting is the honest response."""
    deadline = time.monotonic() + seconds
    first = True
    while time.monotonic() < deadline:
        used = gpu_used_mib()
        if used < 0 or used <= 400:
            if not first:
                print("  the card is quiet again", flush=True)
            return max(used, 0)
        if first:
            print(f"  waiting for a quiet card: {used} MiB held by a neighbour", flush=True)
            first = False
        time.sleep(10)
    raise SystemExit("the card never went quiet")


def registry() -> str:
    """A WRITABLE COPY of the reviewed fingerprint registry.

    `tfs ingest bank` WRITES its argument, and the reviewed registry is committed data in
    a repo this issue does not own. Banking against a copy is not a weakening: the bank is
    the review act, and reviewing a fixture this script produced into another repo's
    evidence file would be the actual error. (Found the direct way — the first run edited
    `tensorfs/vectors/fingerprints/registry.json`; it was reverted.)
    """
    local = WORK / "fingerprints.json"
    if not local.is_file():
        WORK.mkdir(parents=True, exist_ok=True)
        local.write_bytes((TFS / "vectors/fingerprints/registry.json").read_bytes())
    return str(local)


def tfs_cli(*args: str, cwd: Path = TFS) -> subprocess.CompletedProcess[str]:
    """The real `tfs` binary. The border is a PROCESS boundary and it stays one here."""
    return subprocess.run(
        [str(TFS / "target/release/tfs"), *args, "--registry", registry()],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )


def construction_order() -> Path:
    """Materialize the current real SDXL construction census as an ingest recipe input."""
    from corpus.sdxl_real import SdxlUnetPipeline

    from cozy_runtime.author._loader import Config
    from cozy_runtime.author._loader import census as walk_census
    from cozy_runtime.internal.derive import placing

    with placing():
        construction = SdxlUnetPipeline(Config(unet_config(), "unet.config"))
    rows: list[list[str]] = []
    for destination in walk_census(construction).destinations:
        component, separator, local = destination.key.partition(".")
        if not separator or component != destination.component:
            raise RuntimeError(f"malformed construction destination {destination.key!r}")
        rows.append([component, local])
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "construction-order.json"
    path.write_text(json.dumps(rows, separators=(",", ":")))
    return path


# --------------------------------------------------------------------------- produce

#: ComfyUI's `scaled_fp8` emits an `input_scale` beside the weight scale for the modules
#: it runs fp8 matmuls on, and the weight scale alone for the rest. On the measured H3 DiT
#: that split is 150 both / 50 `mlp.fc2` weight-only. This fixture mirrors the SHAPE of
#: that split on SDXL's own fc2-class leaf so ONE artifact carries BOTH spec digests,
#: which is the case the registry says a real artifact carries and the case a per-tensor
#: selector has to get right.
WEIGHT_ONLY_SUFFIX = "ff.net.2.weight"


def eligible(key: str, shape: tuple[int, ...]) -> bool:
    """Rank-2 weights only — the encoding spec's own `logical_rank`. Everything else
    (biases, norms, the 51 conv kernels) stays plain/1, which is what makes this a MIXED
    component and not a uniformly-encoded toy."""
    return len(shape) == 2 and key.endswith(".weight")


def produce() -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    head("produce — the real SDXL UNet, quantized by the registry spec's own value law")
    source = HF / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    WORK.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, Any] = {}
    started = time.perf_counter()
    encoded = plain = with_input = 0
    src_bytes = enc_bytes = 0
    worst = 0.0
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 — safe_open has no __iter__
            tensor = handle.get_tensor(key)
            shape = tuple(tensor.shape)
            src_bytes += tensor.numel() * tensor.element_size()
            if not eligible(key, shape):
                out[key] = tensor
                plain += 1
                enc_bytes += tensor.numel() * tensor.element_size()
                continue
            work = tensor.to(device, torch.float32)
            # THE SPEC's law, transcribed from the registry entry and nowhere else:
            # value = data * weight_scale, weight_scale = amax(|W|)/448 floored at 1e-12,
            # and the clamp happens BEFORE the e4m3 cast because that cast does not
            # saturate — an unclamped 449 becomes NaN, not 448.
            scale = torch.clamp(work.abs().amax() / E4M3_MAX, min=1e-12)
            data = torch.clamp(work / scale, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
            back = data.to(torch.float32) * scale
            # RELATIVE FROBENIUS error, not a per-element ratio: an element the scale
            # underflows to zero has relative error 1.0 and says nothing about the
            # tensor, which is exactly why the per-element metric is the wrong one.
            worst = max(worst, float((back - work).norm().item() / work.norm().clamp(min=1e-12)))
            # THE CARRIER's naming convention, which is NOT the same as appending to the
            # key: the companion sits beside `<module>.weight` as `<module>.weight_scale`.
            # The border's reviewed role map says exactly that, and getting it wrong is a
            # REFUSAL there rather than a silent misread here — observed, first attempt.
            module = key[: -len(".weight")]
            out[key] = data.cpu()
            out[f"{module}.weight_scale"] = scale.to(torch.float32).cpu().reshape(())
            enc_bytes += data.numel() + 4
            encoded += 1
            if not key.endswith(WEIGHT_ONLY_SUFFIX):
                # The activation scale of the NATIVE fp8 matmul. This fixture has no
                # calibration corpus, so the VALUE is a placeholder 1.0 and is recorded as
                # one: the decoded-float route never reads it, and what is under test is
                # that its PRESENCE selects the other spec digest per tensor.
                out[f"{module}.input_scale"] = torch.ones((), dtype=torch.float32)
                enc_bytes += 4
                with_input += 1
    save_file(out, str(CARRIER), metadata={"format": "pt"})
    took = time.perf_counter() - started
    print(f"  source     {source.name}  {src_bytes / GIB:.3f} GiB")
    print(f"  carrier    {CARRIER.name}  {CARRIER.stat().st_size / GIB:.3f} GiB  in {took:.1f}s")
    print(
        f"  encoded    {encoded} rank-2 weights ({with_input} with input_scale, "
        f"{encoded - with_input} weight-scale-only)"
    )
    print(f"  plain      {plain} tensors left at plain/1 (biases, norms, {51} conv kernels)")
    print(
        f"  quantizer  worst per-tensor relative Frobenius error {worst:.5f} "
        f"(e4m3 carries 3 mantissa bits; ~2^-4 per element is its floor)"
    )
    check(
        "the carrier is smaller than the source",
        CARRIER.stat().st_size < src_bytes,
        f"{CARRIER.stat().st_size / GIB:.3f} vs {src_bytes / GIB:.3f} GiB "
        f"({100 * (1 - CARRIER.stat().st_size / src_bytes):.1f}% off)",
    )
    check(
        "both spec role sets are present in ONE artifact",
        with_input > 0 and encoded - with_input > 0,
        f"{with_input} data+weight_scale+input_scale, {encoded - with_input} data+weight_scale",
    )


# ---------------------------------------------------------------------------- ingest


def ingest() -> None:
    head("ingest — the real border: inspect, bank the dialect, plan, run, install")
    STORE.mkdir(parents=True, exist_ok=True)
    if not (STORE / "objects").is_dir():
        run = tfs_cli("store", "init", str(STORE))
        print(
            f"  store init: {run.stdout.strip().splitlines()[-1] if run.stdout else run.returncode}"
        )

    inspect = tfs_cli("ingest", "inspect", "--source", f"unet={CARRIER}", "--markers")
    print("  --- inspect")
    for line in inspect.stdout.splitlines()[:14]:
        print(f"    {line}")
    if inspect.returncode != 0:
        print(inspect.stderr[:2000])
        raise SystemExit("inspect refused")

    # NO BANK STEP. The carrier's KEYSET digest is `sha256:2436aeff…`, byte for byte the
    # one the plain fp16 SDXL UNet already banked — because an encoding rung changes no
    # LOGICAL keyset, which is the same fact that makes every rung of one model share a
    # `topology_digest`. The reviewed fingerprint that authorizes the fp16 UNet therefore
    # authorizes this one, and inventing a second entry for it would be evidence-shaped
    # noise. (Its TOPOLOGY digest does differ, and that is the dtype: f8 elements under
    # an f16 logical declaration are two different logical dtypes.)
    plan = tfs_cli("ingest", "plan", "unet=fp8-scaled-scalar/1", "--source", f"unet={CARRIER}")
    print("  --- plan")
    for line in (plan.stdout or plan.stderr).splitlines()[:20]:
        print(f"    {line}")
    if plan.returncode != 0:
        raise SystemExit("plan refused")

    run = tfs_cli(
        "ingest",
        "run",
        str(STORE),
        "unet=fp8-scaled-scalar/1",
        "--source",
        f"unet={CARRIER}",
        "--order",
        str(construction_order()),
        "--session",
        "cr006",
        "--tenant",
        "local",
    )
    print("  --- run")
    for line in (run.stdout or run.stderr).splitlines()[-14:]:
        print(f"    {line}")
    if run.returncode != 0:
        raise SystemExit("ingest run refused")

    install = tfs_cli("ingest", "install", str(STORE), "cr006")
    print("  --- install")
    for line in (install.stdout or install.stderr).splitlines()[-10:]:
        print(f"    {line}")
    if install.returncode != 0:
        raise SystemExit("install refused")


# --------------------------------------------------------------------------- dequant


def snapshot() -> str:
    """The installed fp8 checkpoint, read from the store rather than transcribed."""
    import tensorfs

    store = tensorfs.Store.open(str(STORE))
    installed = getattr(store, "resolve_local")("cr006")
    if installed is None:
        raise SystemExit("nothing installed in the cr-006 store — run `ingest` first")
    return str(installed)


def unet_config() -> dict[str, Any]:
    carriers = ("_name_or_path", "_diffusers_version", "_commit_hash")
    raw = json.loads((HF / "unet" / "config.json").read_text())
    return {k: v for k, v in raw.items() if k not in carriers}


def _fill(store: str, snap: str, *, scratch: int = 512 * MIB) -> tuple[Any, Any, Any]:
    """Construct + fill the real SDXL UNet from one snapshot. Returns (backend, model, torch)."""
    import torch
    from corpus.sdxl_real import SdxlUnetModel

    from cozy_runtime.author import ModelRegistry
    from cozy_runtime.author._loader import Artifact, Config
    from cozy_runtime.internal.derive import Observations, serving_substrate
    from cozy_runtime.internal.fill import (
        Checkpoint,
        StreamingFillBackend,
        topology_of,
    )

    checkpoints = {"unet": Checkpoint(store, snap)}
    rows = checkpoints["unet"].rows("unet")
    backend = StreamingFillBackend.for_script(
        checkpoints,
        rows,
        store=store,
        snapshot=snap,
        window_bytes=4 * MIB,
        slots=16,
        readers=16,
        inflight=8,
        decode_scratch_bytes=scratch,
        release="cozy/sdxl-unet@cr-006",
    )
    registry = ModelRegistry(
        release="cozy/sdxl-unet@cr-006",
        backend=backend,
        substrate=lambda: serving_substrate("sm89", Observations()),
    )
    artifact = Artifact(
        snap, topology_of(rows), Config(unet_config(), "unet.config"), custody="canonical"
    )
    backend.expect({"unet": tuple(r.key for r in rows)})
    model = registry.acquire("denoise.models.model", SdxlUnetModel, artifact)
    return backend, model, torch


def dequant() -> None:
    import torch
    from safetensors import safe_open

    from cozy_runtime.internal.fill import _tensors as tensors_of

    require_quiet_card()
    head("dequant — fill the fp8 checkpoint on the 4070 and check every decoded weight")
    snap = snapshot()
    print(f"  snapshot   {snap}")
    torch.cuda.reset_peak_memory_stats()
    backend, _model, _ = _fill(str(STORE), snap)

    routes: dict[str, int] = {}
    digests: dict[str, int] = {}
    for selection in backend.selected.values():
        route, short = selection.provider.route, selection.encoded.encoding[:19]
        routes[route] = routes.get(route, 0) + 1
        digests[short] = digests.get(short, 0) + 1
    print(f"  device     {backend.device_facts}")
    print(
        f"  probe      suite {backend.probe['suite_ms']} ms, "
        f"{len(backend.capabilities.records)} capability records minted"
    )
    for observation in backend.probe["observations"]:
        evidence = observation["evidence"]
        if evidence["kind"] == "decode":
            n = evidence["numerics"]
            print(
                f"    {observation['delivery_route']:>14} -> {observation['output_dtype']}: "
                f"max_rel {n['max_rel_error']:.3e} <= {n['tolerance']:.3e} over "
                f"{n['patterns']} bit patterns, NaN preserved {n['nan_preserved']}, "
                f"{evidence['micro_bench']['gb_per_s']:.1f} GB/s"
            )
    print(f"  routes     {routes}")
    print(f"  digests    {digests}")
    print(f"  decode     {backend.stats['decode']}")
    print(
        f"  fill       {backend.stats['fill_ms']} ms "
        f"({backend.stats['stream_ms']} stream + {backend.stats['materialize_ms']} materialize)"
    )
    peak = torch.cuda.max_memory_allocated() / GIB
    print(f"  peak vram  {peak:.3f} GiB allocated")

    check(
        "every destination selected a provider by its own digest",
        len(backend.selected) == 1680,
        f"{len(backend.selected)} of 1680",
    )
    check(
        "the border's per-tensor split survives into the runtime's selection",
        digests.get("sha256:c60935ae4df7") == 673
        and digests.get("sha256:55f85da3ae53") == 70
        and digests.get("sha256:88c2f114d68d") == 937,
        f"{digests}",
    )

    member = backend.components["unet"]
    filled = dict(tensors_of(member))

    # ORACLE ONE — the PLUMBING. Recompute every logical tensor from the CARRIER's own
    # stored bytes and require bit equality. This catches the silent class: right bytes in
    # the wrong tensor, a scale paired with a neighbour's data, a role read at the wrong
    # offset. It cannot catch a wrong value LAW, which is oracle two's job.
    # ORACLE TWO — the NUMERICS, already run by the probe against an INDEPENDENT e4m3
    # decoder written from the OCP format (probe.e4m3_value), printed above.
    wrong_plain: list[str] = []
    wrong_decoded: list[str] = []
    worst_frob = 0.0
    worst_key = ""
    frobs: list[float] = []
    compared_plain = compared_decoded = 0
    with safe_open(str(CARRIER), framework="pt", device="cuda") as carrier:
        keys = set(carrier.keys())
        for key, tensor in filled.items():
            if key not in keys:
                continue
            stored = carrier.get_tensor(key)
            if stored.dtype is torch.float8_e4m3fn:
                module = key[: -len(".weight")]
                scale = carrier.get_tensor(f"{module}.weight_scale")
                want = stored.to(tensor.dtype).mul_(scale.to(tensor.dtype).reshape(()))
                compared_decoded += 1
                if not torch.equal(want.view(torch.uint8), tensor.view(torch.uint8)):
                    wrong_decoded.append(key)
                original = _original(key)
                if original is not None:
                    error = float(
                        (tensor.float() - original).norm().item()
                        / original.norm().clamp(min=1e-12).item()
                    )
                    frobs.append(error)
                    if error > worst_frob:
                        worst_frob, worst_key = error, key
            else:
                compared_plain += 1
                if not torch.equal(stored.view(torch.uint8), tensor.view(torch.uint8)):
                    wrong_plain.append(key)

    check(
        "every PLAIN tensor is byte-identical to the carrier's stored bytes",
        not wrong_plain,
        f"{compared_plain} compared, {len(wrong_plain)} differing",
    )
    check(
        "every DECODED tensor equals the spec's value law over its own stored roles",
        not wrong_decoded,
        f"{compared_decoded} compared bit-for-bit, {len(wrong_decoded)} differing"
        + (f", first {wrong_decoded[0]}" if wrong_decoded else ""),
    )
    mean = sum(frobs) / len(frobs) if frobs else 0.0
    check(
        "decoded weights track the PRE-QUANTIZATION fp16 baseline within e4m3's bound",
        worst_frob < 0.0625,
        f"worst relative Frobenius {worst_frob:.5f} on {worst_key}, mean {mean:.5f} over "
        f"{len(frobs)} weights; e4m3's per-element floor is 2^-4 = 0.0625",
    )
    print(
        f"  BANKED     decode {backend.stats['decode']['tensors']} tensors / "
        f"{backend.stats['decode']['bytes'] / GIB:.3f} GiB stored in "
        f"{backend.stats['decode']['ms']} ms across "
        f"{backend.stats['decode']['flushes']} flush(es)"
    )
    backend.close()
    for name in list(backend.components):
        backend.evict(name)
    torch.cuda.empty_cache()


_ORIGINAL: dict[str, Any] = {}


def _original(key: str) -> Any:
    """One pre-quantization fp16 weight, from the untouched source, on the card."""
    from safetensors import safe_open

    handle = _ORIGINAL.get("handle")
    if handle is None:
        handle = safe_open(
            str(HF / "unet" / "diffusion_pytorch_model.fp16.safetensors"),
            framework="pt",
            device="cuda",
        ).__enter__()
        _ORIGINAL["handle"] = handle
    return handle.get_tensor(key).float()


# ----------------------------------------------------------------------------- image

RELEASE = "cozy/sdxl-pipeline@cr-006"
COMPONENTS = ("text_encoder", "text_encoder_2", "unet", "vae")
PIPE_CONFIG = Path("/tmp/cozy-sdxl4/pipeline.config.json")


def plan_id_for(entrypoint: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(f"ebp/{RELEASE}/{entrypoint}".encode()).hexdigest()


def pipeline_snapshots(unet: str) -> dict[str, str]:
    """The four-component snapshot map with ONE component swapped for its fp8 rung.

    The other three are byte-identical between the two arms, which is the point: the only
    thing that differs between the baseline image and the decoded one is the UNet's
    encoding, so any deviation is the encoding's and nothing else's.
    """
    have = dict(json.loads(PLAIN_SNAPSHOTS.read_text()))
    have["unet"] = unet
    return have


def binding_record(unet: str) -> dict[str, Any]:
    del unet  # the artifact's snapshot map is the index row's, not the record's (#567e)
    return {
        "entrypoint_binding_plan_id": plan_id_for("generate"),
        "project": str(ROOT / "corpus" / "pipeline"),
        "model_class": "SdxlPipelineModel",
        "model_binding_path": "generate.models.model",
        "model_parameter_name": "model",
        "artifact_ref": RELEASE,
        "components": list(COMPONENTS),
        "release": RELEASE,
        "host_bytes": 2 * GIB,
        "pinned_bytes": 256 * MIB,
        "entrypoint": "generate",
        "model_construction_digest": "",
        "resident_budget_bytes": 2 * GIB,
        "tenancy": "local",
    }


def install_into(root: Path) -> str:
    """Put the fp8 UNet in the four-component store, through the real border again.

    Additive and content-addressed: the 937 plain tensors DEDUP against the fp16 UNet
    already there, so this writes only the fp8 elements and the scales. The alternative —
    a second store for one component — would make the two image arms differ in where
    their bytes came from as well as in the encoding under test.
    """
    session = f"cr006-{root.name}"
    run = tfs_cli(
        "ingest",
        "run",
        str(root),
        "unet=fp8-scaled-scalar/1",
        "--source",
        f"unet={CARRIER}",
        "--order",
        str(construction_order()),
        "--session",
        session,
        "--tenant",
        "local",
    )
    if run.returncode != 0:
        print((run.stdout or run.stderr)[-2000:])
        raise SystemExit("ingest run refused")
    install = tfs_cli("ingest", "install", str(root), session)
    if install.returncode != 0:
        print((install.stdout or install.stderr)[-2000:])
        raise SystemExit("install refused")
    for line in run.stdout.splitlines():
        if "snapshot" in line and "sha256:" in line:
            return line.split("sha256:")[-1].strip()
    raise SystemExit("no snapshot digest in the ingest output")


def exec_one(spec_path: Path, out_path: Path) -> None:
    """ONE request through the WHOLE real path — worker, executor, package."""
    import dataclasses
    import os

    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions

    spec = json.loads(spec_path.read_text())
    workspace = Path(spec["workspace"])
    home = workspace / "home"
    home.mkdir(parents=True, exist_ok=True)
    config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(home),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
        }
    )
    outcome = run_slice(
        config,
        LocalRequest(
            entrypoint="generate",
            payload=spec["payload"],
            outputs=("image",),
            request_id=spec["request_id"],
        ),
        [binding_record(spec["unet"])],
        workspace=workspace,
        package_revision_digest=RELEASE,
        options=WorkerOptions(root=workspace / "worker", release_id=RELEASE, no_warm=True),
        delete_workspace=False,
    )
    out_path.write_text(json.dumps(dataclasses.asdict(outcome), default=str))


def took_the_card(outcome: dict[str, Any]) -> bool:
    """True when a NEIGHBOUR took the card, not when this code failed.

    `require_quiet_card` only proves the card was quiet when it was asked. Another agent's
    worker legitimately arrives a second later and takes 5.4 GiB, and the `device_shortfall`
    that follows is completely true and completely uninformative about the arm. Retrying is
    honest; reporting it as a failure would not be, and neither would hiding it. (Observed
    here: an fp16 arm refused with `short by 5110547976 B` while a neighbour held 5410 MiB.)
    """
    parts = [str((outcome.get("terminal") or {}).get("safe_message", ""))]
    parts += [f"{f.get('reason', '')} {f.get('detail', '')}" for f in outcome.get("faults") or ()]
    message = " ".join(parts).lower()
    return "device_shortfall" in message or "out of memory" in message


def spawn(label: str, unet: str, payload: dict[str, Any], tries: int = 4) -> dict[str, Any]:
    """Run one arm in its OWN process, so peak RSS and peak VRAM are that arm's alone."""
    for attempt in range(1, tries + 1):
        require_quiet_card()
        outcome = _spawn_once(label, unet, payload)
        if not took_the_card(outcome) or attempt == tries:
            return outcome
        print(f"    a neighbour took the card during {label}; retrying ({attempt}/{tries})")
    raise SystemExit("unreachable")


def _spawn_once(label: str, unet: str, payload: dict[str, Any]) -> dict[str, Any]:
    import os
    import shutil

    workspace = WORK / f"run-{label}"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    spec_path = workspace / "spec.json"
    out_path = workspace / "outcome.json"
    spec_path.write_text(
        json.dumps(
            {"workspace": str(workspace), "unet": unet, "payload": payload, "request_id": label}
        )
    )
    log = workspace / "run.log"
    with open(log, "wb") as handle:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "exec", str(spec_path), str(out_path)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}"),
            cwd=str(ROOT),
        )
    process.wait(timeout=3600)
    if not out_path.is_file():
        raise SystemExit(
            f"{label}: no outcome (exit {process.returncode})\n"
            f"{log.read_text(errors='replace')[-3000:]}"
        )
    outcome: dict[str, Any] = json.loads(out_path.read_text())
    outcome["_log"] = str(log)
    return outcome


def image() -> None:
    """The real product question: does a decoded checkpoint still make the picture?"""
    head("image — one real SDXL generation on decoded weights against the fp16 baseline")
    fp8 = install_into(PLAIN_STORE)
    print(f"  fp8 unet in the four-component store: sha256:{fp8}")
    plain = json.loads(PLAIN_SNAPSHOTS.read_text())["unet"]
    # 512 px, 20 steps. The SIZE is the card's, not the encoding's: cr-008b measured this
    # 8 GiB device fitting the four-component pipeline at 512 and a 1024 VAE decode OOMing
    # on BOTH arms alike (observed here first, identically for fp16 and fp8 — a capacity
    # fact, not an encoding one). The step count is raised to 20 so the two arms diverge
    # over a real trajectory rather than a single step.
    payload = {
        "prompt": "a photograph of an astronaut riding a horse",
        "steps": 20,
        "size": 512,
        "seed": 1005,
    }

    arms: dict[str, dict[str, Any]] = {}
    for label, unet in (("fp16", plain), ("fp8", f"sha256:{fp8}")):
        started = time.perf_counter()
        outcome = spawn(label, unet, payload)
        took = time.perf_counter() - started
        terminal = outcome.get("terminal") or {}
        # The typed function result is the OUTCOME's, decoded from the terminal's inline
        # ResultEnvelope by the local slice; the ASSET is a separate granted write and its
        # path is in `outputs`. Reading the digest off the terminal alone finds nothing.
        result = outcome.get("result") or {}
        arms[label] = {
            "outcome": outcome,
            "result": result,
            "wall": took,
            "path": (outcome.get("outputs") or {}).get("image", ""),
        }
        print(
            f"  {label:>5}  {outcome.get('status', '?')}  {took:.1f}s  "
            f"pixel digest {str(result.get('digest', ''))[:23]}  "
            f"peak vram {terminal.get('metrics', {}).get('peak_vram_bytes', 0) / GIB:.3f} GiB"
        )
        if not result.get("digest"):
            print(f"    {json.dumps(terminal)[:1200]}")

    ok = all(a["result"].get("digest") for a in arms.values())
    check("both arms produced a real image through the whole attempt kernel", ok)
    if not ok:
        return
    check(
        "the encoding CHANGED the pixels (a no-op decode would be the real failure)",
        arms["fp16"]["result"]["digest"] != arms["fp8"]["result"]["digest"],
        "same seed, same prompt, same three other components",
    )
    _compare_images(arms)


def _compare_images(arms: dict[str, dict[str, Any]]) -> None:
    import numpy as np
    from PIL import Image

    paths = {}
    for label, arm in arms.items():
        found = Path(str(arm["path"]))
        if not found.is_file():
            candidates = sorted((WORK / f"run-{label}" / "outputs").glob("*"))
            if not candidates:
                check(f"{label}: the PNG is on disk", False, str(arm["path"]))
                return
            found = candidates[-1]
        paths[label] = found
    base = np.asarray(Image.open(paths["fp16"]).convert("RGB"), dtype=np.float64)
    other = np.asarray(Image.open(paths["fp8"]).convert("RGB"), dtype=np.float64)
    if base.shape != other.shape:
        check("the two arms are the same geometry", False, f"{base.shape} vs {other.shape}")
        return
    mse = float(((base - other) ** 2).mean())
    psnr = math.inf if mse == 0 else 10.0 * math.log10(255.0**2 / mse)
    mae = float(np.abs(base - other).mean())
    print(f"  baseline   {paths['fp16']}")
    print(f"  decoded    {paths['fp8']}")
    print(
        f"  BANKED     PSNR {psnr:.2f} dB · MAE {mae:.3f}/255 · MSE {mse:.3f} "
        f"over {base.size} subpixels"
    )
    # NO PASS/FAIL ON PSNR, deliberately. Twenty sampler steps are a chaotic map: a weight
    # perturbation of 0.027 relative is amplified by the trajectory, and the arms end at
    # two DIFFERENT valid samples rather than one sample plus noise. An absolute dB bar on
    # that quantity is a bar that means nothing — the closeness claim belongs to the
    # SINGLE-STEP measurement in `bench` (relative L2 0.037, cosine 0.999338), where the
    # trajectory has not compounded. What is worth checking here is STRUCTURE: did the two
    # arms make the same picture?
    structure = _structure(np, base, other)
    print(f"  BANKED     low-frequency structure correlation {structure:.4f}")
    control = _control_image(np, Image)
    if control is None:
        check("a discriminating control image exists", False, "no third arm")
        return
    baseline_control = _structure(np, base, control)
    print(f"  control    fp16 vs a DIFFERENT-SEED fp16 run: {baseline_control:.4f}")
    check(
        "the decoded arm made the SAME PICTURE, and the measure can tell when it does not",
        structure > 0.90 and baseline_control < structure - 0.25,
        f"fp8 {structure:.4f} vs a different-seed control {baseline_control:.4f} — same "
        "prompt, same seed, same three other components; the control is the falsifier",
    )


def _structure(np: Any, a: Any, b: Any) -> float:
    """Pearson correlation of 32x32 block means over luma: composition, not pixels."""
    grids = []
    for image in (a, b):
        luma = image.mean(axis=2)
        h, w = luma.shape[0] // 32 * 32, luma.shape[1] // 32 * 32
        grids.append(luma[:h, :w].reshape(h // 32, 32, w // 32, 32).mean(axis=(1, 3)).ravel())
    x, y = grids
    return float(np.corrcoef(x, y)[0, 1])


def _control_image(np: Any, image_module: Any) -> Any:
    """A THIRD fp16 run at a different seed — the falsifier for the structure measure.

    Without it "0.98 correlation" is a number with no scale. With it, the same measure over
    two genuinely different pictures shows what a MISS looks like.
    """
    plain = json.loads(PLAIN_SNAPSHOTS.read_text())["unet"]
    outcome = spawn(
        "control",
        plain,
        {
            "prompt": "a photograph of an astronaut riding a horse",
            "steps": 20,
            "size": 512,
            "seed": 90210,
        },
    )
    path = (outcome.get("outputs") or {}).get("image", "")
    if not path or not Path(str(path)).is_file():
        return None
    return np.asarray(image_module.open(str(path)).convert("RGB"), dtype=np.float64)


# ----------------------------------------------------------------------------- bench


def drop_page_cache(root: Path) -> dict[str, int]:
    """COLD, without root: `posix_fadvise(DONTNEED)` over every object in the store.

    Copied from cr-008b's harness, which is where it was measured: `off_t` is 64-bit and
    ctypes defaults to `c_int`, so without the argtypes a 4.78 GiB file has its length
    TRUNCATED and a "cold" run is mostly warm.
    """
    import ctypes
    import os

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
    files = nbytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            libc.posix_fadvise(fd, 0, size, 4)  # POSIX_FADV_DONTNEED
            files += 1
            nbytes += size
        finally:
            os.close(fd)
    return {"files": files, "bytes": nbytes}


def _one_forward(torch: Any, unet: Any, reps: int = 5) -> tuple[Any, float]:
    """One real SDXL denoise step on FIXED inputs, plus its median-ish wall.

    Deliberately the RAW module rather than the guarded author method: what is being
    measured is the WEIGHTS, and holding the inputs bit-identical across two processes is
    the whole experiment. The guarded path is cr-008b's proof and is exercised by `image`.
    """
    generator = torch.Generator(device="cuda").manual_seed(4242)
    latents = torch.randn(2, 4, 64, 64, generator=generator, device="cuda", dtype=torch.float16)
    prompt = torch.randn(2, 77, 2048, generator=generator, device="cuda", dtype=torch.float16)
    pooled = torch.randn(2, 1280, generator=generator, device="cuda", dtype=torch.float16)
    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]] * 2, device="cuda", dtype=torch.float16)
    timestep = torch.tensor(500, device="cuda")
    added = {"text_embeds": pooled, "time_ids": time_ids}
    with torch.inference_mode():
        out = unet(latents, timestep, encoder_hidden_states=prompt, added_cond_kwargs=added).sample
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        for _ in range(reps):
            unet(latents, timestep, encoder_hidden_states=prompt, added_cond_kwargs=added)
        torch.cuda.synchronize()
    return out.float().cpu(), (time.perf_counter_ns() - started) / reps / 1e6


def bench() -> None:
    """fill wall / VRAM / step time, fp8 against plain, both COLD and both alone."""
    import torch

    head("bench — cold fill, VRAM and step time: the fp8 rung against the fp16 one")
    plain_snap = json.loads(PLAIN_SNAPSHOTS.read_text())["unet"]
    rows: dict[str, dict[str, Any]] = {}
    outputs: dict[str, Any] = {}
    for label, store, snap in (
        ("fp16 plain/1", str(PLAIN_STORE), plain_snap),
        ("fp8 scaled-scalar", str(STORE), snapshot()),
    ):
        require_quiet_card()
        evicted = drop_page_cache(Path(store))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        backend, _model, _ = _fill(store, snap)
        unet = backend.components["unet"]
        prediction, step_ms = _one_forward(torch, unet)
        rows[label] = {
            "stored_gib": backend.stats["lease_bytes"] / GIB,
            "fill_ms": backend.stats["fill_ms"],
            "stream_ms": backend.stats["stream_ms"],
            "materialize_ms": backend.stats["materialize_ms"],
            "decode_ms": backend.stats["decode"]["ms"],
            "decoded_tensors": backend.stats["decode"]["tensors"],
            "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
            "weights_gib": sum(backend.vram_charge.values()) / GIB,
            "step_ms": step_ms,
            "evicted_gib": evicted["bytes"] / GIB,
        }
        outputs[label] = prediction
        backend.close()
        for name in list(backend.components):
            backend.evict(name)
        del backend, unet
        torch.cuda.empty_cache()

    print(
        f"  {'arm':<20}{'stored':>9}{'cold fill':>11}{'of which decode':>17}"
        f"{'peak vram':>11}{'scratch':>10}{'step':>10}"
    )
    for label, row in rows.items():
        print(
            f"  {label:<20}{row['stored_gib']:>8.3f}G{row['fill_ms']:>10.0f}ms"
            f"{row['decode_ms']:>15.0f}ms{row['peak_vram_gib']:>10.3f}G"
            f"{row['scratch_mib']:>9.0f}M{row['step_ms']:>9.1f}ms"
        )

    a, b = rows["fp16 plain/1"], rows["fp8 scaled-scalar"]
    print(
        f"  BANKED     bytes read {100 * (1 - b['stored_gib'] / a['stored_gib']):.1f}% less; "
        f"cold fill {a['fill_ms'] / max(b['fill_ms'], 1):.2f}x; "
        f"resident weights {b['weights_gib']:.3f} vs {a['weights_gib']:.3f} GiB; "
        f"step {b['step_ms'] / a['step_ms']:.3f}x"
    )
    check(
        "the encoded rung reads FEWER bytes",
        b["stored_gib"] < a["stored_gib"],
        f"{b['stored_gib']:.3f} vs {a['stored_gib']:.3f} GiB of verified lease",
    )
    check(
        "DEVICE HONESTY: resident weights are IDENTICAL — this route buys bytes, not VRAM",
        abs(b["weights_gib"] - a["weights_gib"]) < 1e-6,
        f"{b['weights_gib']:.3f} GiB both; the decode scratch "
        f"({b['scratch_mib']:.0f} MiB) is the only extra, and it is transient",
    )
    check(
        "step time is UNCHANGED — the decoded route runs the same fp16 kernels",
        abs(b["step_ms"] / a["step_ms"] - 1.0) < 0.10,
        f"{b['step_ms']:.1f} vs {a['step_ms']:.1f} ms per UNet forward "
        "(sm89 has no fp8 tensor core; native scaled-mm is the sm90+ seam)",
    )

    # THE SINGLE-STEP DEVIATION: one real inference step, identical inputs, two weight
    # sets. Isolated from sampler chaos on purpose — the 20-step image PSNR measures
    # TRAJECTORY divergence, which compounds; this measures what the weights did.
    ref, got = outputs["fp16 plain/1"], outputs["fp8 scaled-scalar"]
    rel = float((got - ref).norm().item() / ref.norm().item())
    cosine = float((got.flatten() @ ref.flatten()).item() / (got.norm().item() * ref.norm().item()))
    print(
        f"  BANKED     one denoise step, identical inputs: relative L2 {rel:.5f}, "
        f"cosine {cosine:.6f}, |eps| ref {ref.norm().item():.1f}"
    )
    check(
        "one real inference step on decoded weights tracks the fp16 baseline",
        rel < 0.05 and cosine > 0.998,
        f"relative L2 {rel:.5f} against the quantizer's own {0.0269:.4f} weight error",
    )


# ------------------------------------------------------------------------------ arms


def tiny_carrier(name: str, mutate: str) -> Path:
    """A 2-tensor scaled_fp8 carrier with ONE deliberate deviation. Kilobytes, real bytes."""
    import torch
    from safetensors.torch import save_file

    out: dict[str, Any] = {}
    weight = torch.randn(64, 32, dtype=torch.float16)
    scale = (weight.abs().amax().float() / E4M3_MAX).clamp(min=1e-12)
    data = (weight.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    out["block.linear.weight"] = data
    out["block.linear.bias"] = torch.randn(64, dtype=torch.float16)
    if mutate == "rank1_scale":
        # The rank-1 [out] geometry of the OTHER fp8 variant, offered under the SCALAR
        # spec's alias. Two digests exist precisely because a reader that assumes one
        # mis-strides the other.
        out["block.linear.weight_scale"] = scale.repeat(64)
    elif mutate == "no_weight_scale":
        out["block.linear.input_scale"] = torch.ones((), dtype=torch.float32)
    else:
        out["block.linear.weight_scale"] = scale.reshape(())
    path = WORK / f"arm-{name}.safetensors"
    save_file(out, str(path), metadata={"format": "pt"})
    return path


def arms() -> None:
    import torch

    from cozy_runtime.internal.encoding import (
        Encoded,
        EncodingRefusal,
        RolePart,
        Selector,
        digests_of,
        launch_providers,
        measure_device,
    )
    from cozy_runtime.internal.probe import qualify

    head("red arms — each planted for real, observed, and removed")

    # --- the BORDER's arms: a carrier that cannot be what it says it is -------------
    # The GREEN arm first, and it is what makes the two red ones mean anything. A carrier
    # with no reviewed fingerprint refuses BEFORE encoding selection is even reached
    # (observed: `UNREGISTERED_FINGERPRINT` on the first attempt), so the fingerprint is
    # banked once — the three variants share ONE key set, since role siblings are not
    # logical keys — and only then does the refusal that fires belong to this issue.
    good = tiny_carrier("ok", "none")
    bank = tfs_cli(
        "ingest",
        "bank",
        "safetensors.diffusers",
        "cr006-arm",
        "tiny",
        "diffusers.identity/1",
        "--source",
        f"c={good}",
        "--provenance",
        "synthetic",
        "--note",
        "cr-006 red-arm fixture: 2 logical keys, one encoded",
    )
    plan = tfs_cli("ingest", "plan", "c=fp8-scaled-scalar/1", "--source", f"c={good}")
    check(
        "GREEN first: the well-formed tiny carrier PLANS, so the reds are not vacuous",
        plan.returncode == 0 and "fp8-scaled-scalar/1" in plan.stdout,
        next((ln.strip() for ln in plan.stdout.splitlines() if "fp8-scaled" in ln), bank.stderr)[
            :150
        ],
    )
    for name, mutate, expect in (
        ("rank1-scale", "rank1_scale", "the rank-1 [out] scale geometry"),
        ("no-weight-scale", "no_weight_scale", "an input_scale with NO weight_scale companion"),
    ):
        path = tiny_carrier(name, mutate)
        red = tfs_cli("ingest", "plan", "c=fp8-scaled-scalar/1", "--source", f"c={path}")
        message = (red.stdout + red.stderr).strip().splitlines()
        line = next((ln for ln in message if "REFUSED" in ln), "no refusal")
        check(
            f"{expect} refuses at the border, naming the candidates tried",
            red.returncode != 0 and "REFUSED" in line and "FINGERPRINT" not in line,
            line[:420],
        )
        path.unlink(missing_ok=True)

    unregistered = tfs_cli("ingest", "plan", "c=not-a-real-encoding/9", "--source", f"c={good}")
    body = (unregistered.stdout + unregistered.stderr).strip()
    check(
        "an UNREGISTERED encoding refuses at ingest NAMING ITSELF",
        unregistered.returncode != 0 and "not-a-real-encoding/9" in body,
        body.splitlines()[0][:200] if body else "silent",
    )
    good.unlink(missing_ok=True)

    # --- the RUNTIME's arms: selection on the real card ----------------------------
    device = measure_device(torch)
    providers = launch_providers()
    capabilities, probe = qualify(torch, providers, device, ["float16"], release="cozy/arms@cr-006")
    selector = Selector(providers, capabilities, device)
    parts = (
        RolePart("data", "f8_e4m3fn", (64, 32), 2048),
        RolePart("weight_scale", "f32", (), 4),
    )

    def refusal(encoded: Encoded) -> tuple[str, str]:
        try:
            selector.select("block.linear.weight", encoded, "float16")
        except EncodingRefusal as exc:
            return exc.code, str(exc)
        return "", "no refusal — the arm is GREEN when it should be RED"

    code, detail = refusal(Encoded("sha256:" + "de" * 32, "unregistered", parts))
    check(
        "an encoding digest the registry does not carry refuses at bind",
        code == "unknown_encoding",
        f"[{code}] {detail[:190]}",
    )

    # #469a: mxfp8 IS built now (`cozy.mxfp8.dequant/1`), so the SM100 arm moved to an
    # encoding that is still only NAMED. The mxfp8 digest keeps an arm of its own here,
    # because handing it THIS artifact's role set — a rank-0 f32 `weight_scale` where the
    # block spec declares a u8 `scale` grid — is the way a per-tensor selector goes wrong
    # quietly, and it must refuse on the ROLES rather than be tolerated as a variant.
    nvfp4 = digests_of("nvfp4-w4a4/1")[0]
    code, detail = refusal(Encoded(nvfp4, "nvfp4-w4a4/1", parts))
    check(
        "an SM100-only encoding refuses TYPED, naming this card's facts",
        code == "encoding_unqualified" and f"sm{device.sm}" in detail and "sm100" in detail,
        f"[{code}] {detail[:260]}",
    )

    mxfp8 = digests_of("mxfp8/1")[0]
    code, detail = refusal(Encoded(mxfp8, "mxfp8/1", parts))
    check(
        "the mxfp8 provider refuses this artifact's fp8 role set on the ROLES, not silently",
        code == "role_mismatch" and "scale" in detail,
        f"[{code}] {detail[:260]}",
    )

    svdq = digests_of("svdq-microscale/1")[0]
    code, detail = refusal(Encoded(svdq, "svdq-microscale/1", parts))
    check(
        "the W4A4/SVDQuant lane refuses naming its owner, never silently decoded",
        code == "encoding_unqualified",
        f"[{code}] {detail[:180]}",
    )

    scalar = digests_of("fp8-scaled-scalar/1")[0]
    short = (RolePart("data", "f8_e4m3fn", (64, 32), 2048),)
    code, detail = refusal(Encoded(scalar, "fp8-scaled-scalar/1", short))
    check(
        "a decodable encoding MISSING a required role refuses before any byte moves",
        code == "role_mismatch",
        f"[{code}] {detail[:190]}",
    )

    stray = (*parts, RolePart("zero_point", "f32", (), 4))
    code, detail = refusal(Encoded(scalar, "fp8-scaled-scalar/1", stray))
    check(
        "a role no decoder accounts for refuses rather than being ignored",
        code == "role_mismatch",
        f"[{code}] {detail[:190]}",
    )

    # --- the QUALIFICATION arm: a wrong decoder mints NO record --------------------
    broken = dict(providers)
    broken[scalar] = (_WrongScale(scalar),)
    caps, evidence = qualify(
        torch,
        {scalar: broken[scalar]},
        device,
        ["float16"],
        release="cozy/arms@cr-006",
    )
    numerics = evidence["observations"][0]["evidence"]["numerics"]
    check(
        "a decoder that DROPS the scale fails the numerics self-check",
        not numerics["passed"],
        f"max_rel {numerics['max_rel_error']:.3e} > tolerance {numerics['tolerance']:.3e} "
        f"over {numerics['patterns']} bit patterns",
    )
    check(
        "a failed self-check mints NO capability record, so the fill refuses",
        not caps.records,
        f"{len(caps.records)} records",
    )
    code, detail = "", ""
    try:
        Selector(broken, caps, device).select(
            "block.linear.weight", Encoded(scalar, "fp8-scaled-scalar/1", parts), "float16"
        )
    except EncodingRefusal as exc:
        code, detail = exc.code, str(exc)
    check(
        "...and that refusal is the CAPABILITY one, not a silent float fallback",
        code == "encoding_unqualified",
        f"[{code}] {detail[:200]}",
    )
    print(
        f"  probe      the GOOD suite: {probe['suite_ms']} ms, {len(capabilities.records)} records"
    )

    # --- the FIT arm, on the real 4.78 GiB artifact --------------------------------
    # "too-small scope refuses before allocation; no OOM-as-control-flow" (done-when).
    # Since #613 the fill plane keeps no counter: ADMISSION is the executor's arithmetic
    # over `price_envelope`, so this arm proves the envelope carries the decode scratch
    # BESIDE the destinations — a plan admitted at its weights alone is the defect.
    from cozy_runtime.internal.fill import Checkpoint, price_envelope

    header_rows = Checkpoint(str(STORE), snapshot()).rows("unet")
    env = price_envelope(
        [
            (
                r.component,
                int(r.nbytes),
                int(r.stored_nbytes),
                "verbatim" if r.encoded.alias == "plain/1" else "decoded_float",
            )
            for r in header_rows
        ],
        64 * MIB,
    )
    check(
        "the admission envelope prices the decode scratch BESIDE the destinations",
        env["decode_peak_bytes"] > 0
        and env["required_bytes"]
        == env["destination_bytes"] + env["native_bytes"] + env["decode_peak_bytes"],
        f"destinations {env['destination_bytes']} B + decode {env['decode_peak_bytes']} B "
        f"= required {env['required_bytes']} B",
    )


class _WrongScale:
    """A PLANT: the value law with the scale dropped. Lives in this script, never in
    `src/` — a broken provider that ships is not an arm, it is a defect."""

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding

    name = "cozy.PLANTED-wrong-scale/0"
    route = "decoded_float"
    suite = "fp8-scalar-decode"
    device_predicate = "any"
    required = frozenset({"data", "weight_scale"})
    optional = frozenset({"input_scale"})
    # #549.4/#549.9's two additions to the provider surface. A plant has to carry them or it
    # is not the shape a real provider is, and the arm would be exercising the wrong thing.
    code_revision = "0"

    @property
    def reviewed_specs(self) -> frozenset[str]:
        return frozenset({self.encoding})

    def supports(self, shape: object) -> str:
        return ""

    def decode(self, torch: Any, parts: dict[str, Any], destination: Any) -> None:
        destination.copy_(parts["data"].view(destination.shape).to(destination.dtype))


def main(argv: list[str]) -> int:
    if argv and argv[0] == "exec":
        exec_one(Path(argv[1]), Path(argv[2]))
        return 0
    names = argv or ["produce"]
    for name in names:
        {
            "produce": produce,
            "ingest": ingest,
            "dequant": dequant,
            "image": image,
            "arms": arms,
            "bench": bench,
        }[name]()
    print(f"\n{PASS} ok, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
