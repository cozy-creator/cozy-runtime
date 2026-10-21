"""The production Comfy provider and serving route use the shared TensorFS decoder.

The immutable producer-mined corpus supplies independent bytes. CPU is mandatory;
the one GPU case requires the existing explicit --real-gpu opt-in. No model download.
"""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.author._loader import TENSORFS_DTYPES
from cozy_runtime.internal import probe
from cozy_runtime.internal.encoding import launch_providers
from cozy_runtime.internal.encoding.comfy import SCHEMES, ComfyDequant
from cozy_runtime.internal.encoding.formats import TORCH_DTYPES, params_of

torch = pytest.importorskip("torch")

CASES = [row for row in probe.vector_cases() if row.encoding.scheme in SCHEMES]
POSITIVE = [row for row in CASES if not row.refuse]
NEGATIVE = [row for row in CASES if row.refuse]
COUNTS = {"comfy.asym_w4a8_int8": 10, "comfy.w6a8_int8": 5, "int4.convrot": 3}


def provider(case: probe.VectorCase) -> ComfyDequant:
    choices = [
        value
        for value in launch_providers()[case.encoding.scheme]
        if isinstance(value, ComfyDequant)
    ]
    assert len(choices) == 1
    return replace(choices[0], params=params_of(case.encoding.params))


def vectors(device: str) -> None:
    checked: Counter[str] = Counter()
    for case in POSITIVE:
        got = probe.decode_case(torch, provider(case), case, torch.device(device))
        assert got is not None and got.device.type == device
        assert probe.same_bits(torch, got, case.expect) == 0, f"{case.component}/{case.key}"
        checked[case.encoding.scheme] += 1
    assert dict(checked) == COUNTS and sum(checked.values()) == 18


def test_cpu_vectors_through_registered_native_consumers() -> None:
    assert len(POSITIVE) == 18 and len(NEGATIVE) == 16
    vectors("cpu")


@pytest.mark.parametrize("case", NEGATIVE, ids=lambda row: row.key)
def test_frozen_refusals_survive_the_native_provider(case: probe.VectorCase) -> None:
    parts = {
        part.role: probe.vector_tensor(torch, part, torch.device("cpu")) for part in case.parts
    }
    dtype = getattr(torch, TORCH_DTYPES[TENSORFS_DTYPES[case.dtype]])
    destination = torch.full(case.shape, 99, dtype=dtype)
    before = destination.view(torch.uint8).clone()
    with pytest.raises(tensorfs.DecodeError):
        provider(case).decode(torch, parts, destination)
    if case.key != "s-rel-nan":
        assert torch.equal(destination.view(torch.uint8), before), case.key


@pytest.mark.parametrize("scheme", sorted(SCHEMES))
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_independent_constant_q_oracle_and_golden_qualification(scheme: str, dtype: str) -> None:
    bound = ComfyDequant(scheme, params_of(probe.PROBE_PARAMS[scheme]))
    sweep = probe.DECODE_SUITES[bound.suite](
        torch, bound, torch.device("cpu"), getattr(torch, dtype)
    )
    assert sweep.passed and sweep.nan_preserved and sweep.patterns == 16 * 256
    golden = probe._spec_vectors(torch, bound, torch.device("cpu"))
    assert golden.passed and not golden.skipped and len(golden.cases) == COUNTS[scheme]
    assert all(row.passed and not row.skipped for row in golden.cases)


@pytest.mark.parametrize("dtype", ["float64", "float8_e4m3fn"])
def test_plain_only_output_dtypes_leave_comfy_unmeasured(dtype: str) -> None:
    # The global qualifier visits these candidates even when the model is plain.
    # An unsupported encoded destination must not abort or mint a capability.
    evidence = probe._measured(
        torch, ComfyDequant("int4.convrot"), torch.device("cpu"), getattr(torch, dtype)
    )
    assert isinstance(evidence, probe.UnmeasuredEvidence)


def test_real_backend_serves_original_logical_dtype_from_resident_parts(tmp_path: Path) -> None:
    from tensorfs.varena.torch import Memory

    from cozy_runtime.internal.fill import Checkpoint
    from cozy_runtime.internal.weights import Backend

    case = POSITIVE[0]
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("net", cls=("torch", "", "Linear"), config={"bias": False})
    component.put_encoded(
        "weight",
        dtype=case.dtype,
        shape=case.shape,
        scheme=case.encoding.scheme,
        parts={
            part.role: (part.dtype, part.shape, base64.b64decode(part.bits)) for part in case.parts
        },
        params=case.encoding.params,
    )
    native = writer.commit()
    del component, writer
    checkpoint = Checkpoint(native)
    rows = checkpoint.rows("net")
    memory = Memory.open(torch.device("cpu"))
    backend = Backend.for_script(
        {"net": checkpoint},
        rows,
        torch=torch,
        memory=memory,
        device=torch.device("cpu"),
        construction="comfy-native",
        release="comfy-native-proof",
    )
    logical = getattr(torch, TORCH_DTYPES[TENSORFS_DTYPES[case.dtype]])
    with torch.device("meta"):
        net = torch.nn.Linear(case.shape[1], case.shape[0], bias=False, dtype=logical)
    model = backend._register("net", net, [row.key for row in rows])
    try:
        assert isinstance(backend.selected["net.weight"].provider, ComfyDequant)
        assert net.weight.dtype == logical and tuple(net.weight.shape) == tuple(case.shape)
        want = torch.frombuffer(bytearray(base64.b64decode(case.expect)), dtype=torch.uint8)
        want = want.view(logical).reshape(case.shape).clone()
        native.close()
        x = (torch.arange(case.shape[1], dtype=torch.float32) / 256).reshape(1, -1).to(logical)
        with torch.no_grad(), memory.request("comfy-native-forward"):
            assert torch.equal(net(x), torch.nn.functional.linear(x, want))
    finally:
        model.close()
        backend.close()
        native.close()


@pytest.mark.real_gpu
def test_gpu_vectors_through_registered_native_consumers() -> None:
    assert torch.cuda.is_available(), "the explicit GPU proof requires a CUDA device"
    vectors("cuda")
    torch.cuda.synchronize()
