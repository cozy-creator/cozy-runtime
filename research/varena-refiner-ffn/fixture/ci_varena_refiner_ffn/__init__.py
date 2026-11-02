"""Real H3-width Diffusers/int8 FFN consumer; run only in the reviewed GPU cohort."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import msgspec
import tensorfs
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TokenRefinerBlock

import cozy_runtime
from cozy_runtime.author import App, Context

app = App()


class Probe(msgspec.Struct, forbid_unknown_fields=True):
    runtime_modules: dict[str, str]
    extension_sha256: str
    shim_sha256: str
    gpu_uuid: str


class Cleanup(Probe):
    expected_pid: int
    keys: list[str]


class Report(msgspec.Struct):
    passed: bool
    detail: str


RUNTIME = (
    "internal/execution/native_refiner_ffn.py", "internal/execution/cuts.py",
    "internal/execution/native_linear_split.py", "internal/execution/native_residual_ffn.py",
    "internal/execution/catch.py", "internal/execution/phase.py", "internal/residency.py",
    "internal/encoding/leaves.py", "internal/int8_linear.py", "internal/prebuilt.py",
    "author/kernels.py", "lora.py", "_kernels/_gemm.abi3.so",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual(payload: Probe, ctx: Context):
    from tensorfs import _ext
    from tensorfs.varena import torch as vt

    from cozy_runtime.internal.residency import _REQUEST, Request

    root = Path(cozy_runtime.__file__).resolve().parent
    assert set(payload.runtime_modules) == set(RUNTIME)
    for name, expected in payload.runtime_modules.items():
        assert digest(root / name) == expected, name
    assert digest(Path(_ext.__file__)) == payload.extension_sha256
    assert digest(Path(tensorfs.__file__).parent / "_varena_torch.so") == payload.shim_sha256
    assert ctx.device.type == "cuda" and torch.cuda.device_count() == 1
    uuid = str(torch.cuda.get_device_properties(ctx.device).uuid)
    assert uuid.removeprefix("GPU-").lower() == payload.gpu_uuid.removeprefix("GPU-").lower()
    request = _REQUEST.get()
    assert isinstance(request, Request) and request.id == ctx.request_id and not request.idle
    assert not request.memory.models
    return request, vt


def network(dtype, device):
    from cozy_runtime.internal.encoding.leaves import Int8ConvRotNativeLeaf

    # Real package geometry. Meta construction avoids allocating dense source weights.
    with torch.device("meta"):
        refiner = MiniMaxH3TokenRefinerBlock(5376, 56, 128, 14336, 1e-6, 1e-6).eval()
    ff = refiner.ff
    generator = torch.Generator().manual_seed(13829)
    for parent, name in ((ff.net[0], "proj"), (ff.net, "2")):
        original = getattr(parent, name)
        data = torch.randint(-31, 32, original.weight.shape, dtype=torch.int8, generator=generator)
        scale = torch.full((original.out_features,), .001, dtype=torch.float32)
        leaf = Int8ConvRotNativeLeaf().leaf(
            torch, {"weight": data.to(device), "scale": scale.to(device)}, original, dtype
        ).eval()
        setattr(parent, name, leaf)
    return refiner


def profile_call(ff, x):
    from cozy_runtime.internal.execution import catch
    from cozy_runtime.internal.execution.native_refiner_ffn import _execute

    counts = {"composites": 0, "legacy_cuts": 0}
    assert sys.getprofile() is None

    def profile(frame, event, arg):
        if event == "call":
            if frame.f_code is _execute.__code__:
                counts["composites"] += 1
            if frame.f_code is catch.cut.__code__:
                counts["legacy_cuts"] += 1

    sys.setprofile(profile)
    try:
        output = ff(x)
    finally:
        sys.setprofile(None)
    assert counts == {"composites": 1, "legacy_cuts": 0}, counts
    return output, counts


@app.entrypoint(accelerator=True)
def consumer(payload: Probe, ctx: Context) -> Report:
    from cozy_runtime.internal import execution, int8_linear, prebuilt, verify
    from cozy_runtime.internal.execution.native_refiner_ffn import declare, split_ffn

    request, vt = actual(payload, ctx)
    device = ctx.device.index or 0
    result = {"pid": os.getpid(), "scope": "H3-width FFN consumer only; no model qualification",
              "torch": str(torch.__version__), "cases": [], "keys": []}
    try:
        assert int8_linear.native(torch, device) is not None
        assert int8_linear.library(device).family == "mma.sync"
        assert vt.split_context(device)["eligible"]
        with torch.inference_mode():
            for dtype in (torch.bfloat16, torch.float16):
                refiner = network(dtype, ctx.device)
                ff = refiner.ff
                inputs, references = [], []
                for rows in (5, 17, 513):
                    value = torch.randn((1, rows, 5376), dtype=dtype,
                                        generator=torch.Generator().manual_seed(13831)).to(ctx.device)
                    inputs.append(value)
                    references.append(ff(value).cpu())
                execution.install(refiner)
                contract = declare(ff)
                assert contract is not None
                for value, reference in zip(inputs, references, strict=True):
                    # Returning None for unsupported contracts precedes any execution.
                    old_bias = ff.net[2].bias
                    ff.net[2].bias = torch.zeros(5376, dtype=dtype, device=ctx.device)
                    before = sum(row["selections"] for row in request.working_splits.values())
                    assert split_ffn(contract, value) is None
                    after = sum(row["selections"] for row in request.working_splits.values())
                    assert after == before
                    ff.net[2].bias = old_bias
                    output, counts = profile_call(ff, value)
                    torch.cuda.synchronize(device)
                    error = verify.error(torch, output.cpu(), reference)
                    assert error <= prebuilt.BUDGET, error
                    records = {key: row for key, row in request.working_splits.items()
                               if row.get("site") == "refiner-swiglu-ffn"}
                    assert records
                    result["cases"].append({"dtype": str(dtype), "rows": value.shape[1],
                                            "relative_l2": error, "limit": prebuilt.BUDGET,
                                            "counts": counts,
                                            "records": json.loads(json.dumps(records))})
                    result["keys"].extend(key for key in records if key not in result["keys"])
                    del output
                del contract, ff, refiner, inputs, references, value, reference
                gc.collect()
        torch.cuda.synchronize(device)
        result["passed"] = True
    except Exception:
        result["passed"] = False
        result["error"] = traceback.format_exc()
    return Report(result["passed"], json.dumps(result, sort_keys=True))


@app.entrypoint(accelerator=True)
def cleanup(payload: Cleanup, ctx: Context) -> Report:
    request, vt = actual(payload, ctx)
    assert os.getpid() == payload.expected_pid and not request.working_phase_keys
    removed = {key: vt.phase_forget(ctx.device.index or 0, key) for key in payload.keys}
    assert removed and not any(removed.values()), removed
    return Report(True, json.dumps({"passed": True, "prior_keys_already_retired": removed}))
