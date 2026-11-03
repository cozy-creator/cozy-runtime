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
    uuid = str(torch.cuda.get_device_properties(torch.device(str(ctx.device))).uuid)
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
    from cozy_runtime import _kernels
    from cozy_runtime.internal.execution import catch
    from cozy_runtime.internal.execution.native_linear_split import launch_projection
    from cozy_runtime.internal.execution.native_refiner_ffn import _execute

    import threading

    originals = {name: getattr(_kernels._gemm, name)
                 for name in ("gemm", "gemv", "quantize", "int8_plan")}
    counts = {"composites": 0, "legacy_cuts": 0, "native_errors": 0,
              **{name: 0 for name in originals}}
    thread = threading.get_ident()

    def observe_native(name, actual):
        # nanobind.nb_func does not emit CPython c_return/c_exception profile
        # events. Forward the real callable unchanged and count this thread.
        def observed(*args, **kwargs):
            try:
                result = actual(*args, **kwargs)
            except BaseException:
                if threading.get_ident() == thread:
                    counts["native_errors"] += 1
                raise
            if threading.get_ident() == thread:
                counts[name] += 1
            return result
        return observed
    dispatches = []
    assert sys.getprofile() is None

    def profile(frame, event, arg):
        if event == "call":
            if frame.f_code is _execute.__code__:
                counts["composites"] += 1
            if frame.f_code is catch.cut.__code__:
                counts["legacy_cuts"] += 1
            if frame.f_code is launch_projection.__code__:
                dispatches.append(dict(vars(frame.f_locals["dispatch"])))

    for name, actual in originals.items():
        setattr(_kernels._gemm, name, observe_native(name, actual))
    sys.setprofile(profile)
    try:
        output = ff(x)
    finally:
        sys.setprofile(None)
        for name, actual in originals.items():
            setattr(_kernels._gemm, name, actual)
    assert counts["composites"] == 1 and counts["legacy_cuts"] == 0, counts
    assert counts["int8_plan"] > 0 and counts["native_errors"] == 0, counts
    assert counts["gemm"] == sum(part["kind"] == 0 for part in dispatches), counts
    assert counts["gemv"] == sum(part["kind"] == 1 for part in dispatches), counts
    assert counts["quantize"] == counts["gemm"], counts
    assert counts["gemm"] + counts["gemv"] == len(dispatches) > 0, counts
    return output, counts, dispatches


@app.entrypoint(accelerator=True)
def consumer(payload: Probe, ctx: Context) -> Report:
    from cozy_runtime.author.kernels import INT8_CONVROT
    from cozy_runtime.internal import execution, int8_linear, prebuilt, verify
    from cozy_runtime.internal.encoding.leaves import _native_leaf_class
    from cozy_runtime.internal.execution.native_refiner_ffn import declare, split_ffn

    request, vt = actual(payload, ctx)
    device = ctx.device.index or 0
    torch_device = torch.device(str(ctx.device))
    result = {"pid": os.getpid(), "scope": "H3-width FFN consumer only; no model qualification",
              "torch": str(torch.__version__), "cases": [], "keys": [],
              "instrumentation": "transparent actual-native call observers; not performance evidence"}
    try:
        assert int8_linear.native(torch, device) is not None
        library = int8_linear.library(device)
        assert library.family == "mma.sync" and int8_linear.serving(device) == int8_linear.OP
        result["selected_library"] = {"identity": library.identity, "family": library.family,
                                      "serving": int8_linear.serving(device)}
        assert vt.split_context(device)["eligible"]
        with torch.inference_mode():
            for dtype in (torch.bfloat16, torch.float16):
                refiner = network(dtype, torch_device)
                ff = refiner.ff
                leaves = (ff.net[0].proj, ff.net[2])
                assert all(type(leaf) is _native_leaf_class(torch) for leaf in leaves)
                assert all(leaf.provider == INT8_CONVROT and leaf.out_dtype == dtype
                           for leaf in leaves)
                providers = [{"class": type(leaf).__qualname__, "provider": leaf.provider,
                              "encoding": leaf.encoding, "dtype": str(leaf.out_dtype),
                              "shape": tuple(leaf.data.shape)} for leaf in leaves]
                inputs, references = [], []
                for rows in (5, 17, 513):
                    value = torch.randn((1, rows, 5376), dtype=dtype,
                                        generator=torch.Generator().manual_seed(13831)).to(torch_device)
                    inputs.append(value)
                    references.append(ff(value).cpu())
                execution.install(refiner)
                contract = declare(ff)
                assert contract is not None
                for value, reference in zip(inputs, references, strict=True):
                    # Returning None for unsupported contracts precedes any execution.
                    old_bias = ff.net[2].bias
                    ff.net[2].bias = torch.zeros(5376, dtype=dtype, device=torch_device)
                    before = sum(row["selections"] for row in request.working_splits.values())
                    assert split_ffn(contract, value) is None
                    after = sum(row["selections"] for row in request.working_splits.values())
                    assert after == before
                    ff.net[2].bias = old_bias
                    prior = {key: row["selections"] for key, row in request.working_splits.items()}
                    output, counts, dispatches = profile_call(ff, value)
                    torch.cuda.synchronize(device)
                    error = verify.error(torch, output.cpu(), reference)
                    assert error <= prebuilt.BUDGET, error
                    records = {key: row for key, row in request.working_splits.items()
                               if row.get("site") == "refiner-swiglu-ffn"}
                    assert records
                    selected = [row for key, row in records.items()
                                if row["selections"] == prior.get(key, 0) + 1]
                    assert len(selected) == 1
                    active = selected[0]
                    chunk = active["row_chunk"]
                    expected_dispatches = []
                    for at in range(0, value.shape[1], chunk):
                        suffix = "" if min(chunk, value.shape[1] - at) == chunk else "_tail"
                        expected_dispatches.extend(active["dispatch"][name + suffix]
                                                   for name in ("up", "down"))
                    assert dispatches == expected_dispatches
                    result["cases"].append({"dtype": str(dtype), "rows": value.shape[1],
                                            "relative_l2": error, "limit": prebuilt.BUDGET,
                                            "counts": counts,
                                            "providers": providers,
                                            "native_dispatches": dispatches,
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
