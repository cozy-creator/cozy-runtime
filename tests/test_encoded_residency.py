"""A native encoded leaf on the weight plane: installed only with consent, under the hooks of
the linear op it replaced, computing from exactly the stored bytes."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tensorfs
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author._loader import census
from cozy_runtime.derive.microscale import encode_fp8_rowwise
from cozy_runtime.internal import accel, plane
from cozy_runtime.internal.encoding import SPEC_PLAIN, SPEC_ROWWISE, LeafProvider
from cozy_runtime.internal.fill import Checkpoint, FillRefusal
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights
from cozy_runtime.internal.weights_sink import weights_transaction_id

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not accel.present(torch, "cuda") or not plane.available(),
    reason="the weight plane on a CUDA card",
)

WIDTH = 1024


def test_a_native_leaf_serves_the_stored_bytes_only_with_consent(tmp_path: Path) -> None:
    values = np.random.default_rng(91).normal(0, 0.1, (WIDTH, WIDTH)).astype(np.float32)
    bias = np.random.default_rng(92).normal(0, 0.1, WIDTH).astype(np.float32)
    encoded = encode_fp8_rowwise("generated", values)
    store = tensorfs.Store.ensure(str(tmp_path / "store"))
    writer = store.begin_derived(
        weights_transaction_id("encoded-proof", "proof", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={
                "a": Target(
                    add={
                        "linear.weight": Tensor(
                            "bf16",
                            (WIDTH, WIDTH),
                            SPEC_ROWWISE,
                            {
                                "data": Part("f8_e4m3fn", (WIDTH, WIDTH)),
                                "scale": Part("f32", (WIDTH,)),
                            },
                        ),
                        "linear.bias": Tensor(
                            "f32", (WIDTH,), SPEC_PLAIN, {"value": Part("f32", (WIDTH,))}
                        ),
                    }
                )
            },
            configs={"model": NativeConfig("add")},
            order=(("a", "linear.weight"), ("a", "linear.bias")),
        ).native_arguments(4 << 20),
        work_fingerprint="sha256:" + "2" * 64,
    )
    writer.add_part("a", "linear.weight", "data", io.BytesIO(encoded.payload))
    writer.add_part("a", "linear.weight", "scale", io.BytesIO(encoded.companions[0].raw))
    writer.add_part("a", "linear.bias", "value", io.BytesIO(bias.tobytes()))
    writer.add_config("model", io.BytesIO(b"{}"))
    checkpoint = Checkpoint(tmp_path / "store", "sha256:" + writer.commit()["manifest"]["sha256"])
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    backend = PlaneBackend.for_script(
        {"a": checkpoint},
        checkpoint.rows("a"),
        weights=weights,
        construction="encoded",
        release="encoded-proof/1",
        objective="footprint",
    )
    module = torch.nn.Module()
    module.linear = original = torch.nn.Linear(WIDTH, WIDTH, device="meta", dtype=torch.bfloat16)
    original.bias = torch.nn.Parameter(torch.empty(WIDTH, device="meta"))
    inputs: list[Any] = []
    original.register_forward_pre_hook(lambda _module, args: inputs.append(args[0]))
    original.register_forward_hook(lambda _module, _args, output: -output)
    holder = SimpleNamespace(components={"a": module})
    walked = census(holder)
    try:
        if backend.resolved["a.linear.weight"].route != "encoded_gemm":
            pytest.skip("this card serves fp8-rowwise decoded, not through a native leaf")
        backend.expect({"a": [d.key for d in walked.destinations]})
        with pytest.raises(FillRefusal) as refused:
            backend.materialize(holder, walked, encoded_leaves="refuse")
        assert refused.value.code == "leaf_unconsented" and not weights.components
        live = backend.materialize(holder, walked, encoded_leaves="accept")
        for destination in walked.destinations:
            backend.fill(destination.key, destination.spec, live[destination.key])
        backend.commit()
        leaf = module.linear
        assert leaf is not original and not tuple(leaf.parameters())
        assert original.weight.is_meta and original.bias.is_meta
        # The leaf built directly from the encoder's own bytes, beside the plane's.
        reference_linear = torch.nn.Linear(WIDTH, WIDTH, device="cuda", dtype=torch.bfloat16)
        reference_linear.bias = torch.nn.Parameter(torch.from_numpy(bias).cuda())
        parts = {
            "data": torch.frombuffer(bytearray(encoded.payload), dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .reshape(WIDTH, WIDTH)
            .cuda(),
            "scale": torch.frombuffer(
                bytearray(encoded.companions[0].raw), dtype=torch.float32
            ).cuda(),
        }
        provider = backend.selected["a.linear.weight"].provider
        assert isinstance(provider, LeafProvider)
        reference = provider.leaf(torch, parts, reference_linear, torch.bfloat16)
        x = torch.randn(16, WIDTH, device="cuda", dtype=torch.bfloat16)
        residency = WeightResidency(weights, backend.components, {"run": ("a",)})
        residency.admit("run", ("a",))
        try:
            with torch.no_grad():
                served, expected = module.linear(x), reference(x)
        finally:
            residency.release("run", ("a",))
        # The replaced module's hooks run around the leaf: its input, and its output negated.
        assert len(inputs) == 1 and inputs[0] is x
        assert torch.equal(served, -expected)
    finally:
        backend.close()
