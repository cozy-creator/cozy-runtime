"""Generated <=16MB CUDA native-TensorFS proof; no model download or substituted provider."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorfs
import torch
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author._loader import census
from cozy_runtime.derive.microscale import encode_fp8_rowwise
from cozy_runtime.internal.encoding import SPEC_PLAIN, SPEC_ROWWISE
from cozy_runtime.internal.fill import Checkpoint, FillRefusal, StreamingFillBackend, _Holder
from cozy_runtime.internal.residency import ComponentResidency, ResidencyRefusal
from cozy_runtime.internal.weights_sink import weights_transaction_id


def main(root: Path) -> None:
    assert torch.cuda.is_available()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((512 << 20) / total)
    torch.set_num_threads(4)
    torch.manual_seed(897)
    root.mkdir(parents=True, exist_ok=True)
    store = tensorfs.Store.ensure(str(root / "store"))
    values = np.random.default_rng(91).normal(0, 0.1, (1024, 1024)).astype(np.float32)
    encoded = encode_fp8_rowwise("generated", values)
    bias = np.random.default_rng(92).normal(0, 0.1, 1024).astype(np.float32)
    targets = {
        "linear.weight": Tensor(
            "bf16",
            (1024, 1024),
            SPEC_ROWWISE,
            {
                "data": Part("f8_e4m3fn", (1024, 1024)),
                "scale": Part("f32", (1024,)),
            },
        ),
        "linear.bias": Tensor("f32", (1024,), SPEC_PLAIN, {"value": Part("f32", (1024,))}),
    }
    bound = 2 * (len(encoded.payload) + len(encoded.companions[0].raw) + bias.nbytes) + 4096
    transaction = store.begin_derived(
        weights_transaction_id("encoded-proof", "proof", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={name: Target(add=targets) for name in ("a", "b")},
            configs={"model": NativeConfig("add")},
            order=tuple((name, key) for name in ("a", "b") for key in targets),
        ).native_arguments(bound),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for name in ("a", "b"):
        transaction.add_part(name, "linear.weight", "data", io.BytesIO(encoded.payload))
        transaction.add_part(name, "linear.weight", "scale", io.BytesIO(encoded.companions[0].raw))
        transaction.add_part(name, "linear.bias", "value", io.BytesIO(bias.tobytes()))
    transaction.add_config("model", io.BytesIO(b"{}"))
    receipt = transaction.commit()
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    checkpoint = Checkpoint(root / "store", manifest)
    rows = [row for name in ("a", "b") for row in checkpoint.rows(name)]
    backend = StreamingFillBackend.for_script(
        checkpoint,
        rows,
        release="encoded-proof/1",
        store="fixture",
        snapshot=manifest,
        placement="component_staged",
        objective="footprint",
        window_bytes=1 << 20,
        slots=3,
        readers=2,
        inflight=2,
    )
    assert all(
        t.route == "encoded_gemm" for key, t in backend.resolved.items() if key.endswith("weight")
    ), backend.execution.document()
    print(
        json.dumps(
            {
                "phase": "qualified",
                "device": torch.cuda.get_device_name(),
                "routes": {k: r.route for k, r in backend.resolved.items()},
            }
        ),
        flush=True,
    )

    def model() -> Any:
        module = torch.nn.Module()
        module.linear = torch.nn.Linear(1024, 1024, device="meta", dtype=torch.bfloat16)
        module.linear.bias = torch.nn.Parameter(
            torch.empty(1024, device="meta", dtype=torch.float32)
        )
        return module

    models = {name: model() for name in ("a", "b")}
    original = {name: member.linear for name, member in models.items()}
    hook_calls = {name: 0 for name in models}
    handles = []
    for name, linear in original.items():

        def check_input(module: Any, args: Any, kwargs: Any, name: str = name) -> None:
            assert isinstance(args[0], torch.Tensor)
            hook_calls[name] += 1

        def negate_output(module: Any, args: Any, kwargs: Any, output: Any) -> Any:
            return -output

        handles.append(linear.register_forward_pre_hook(check_input, with_kwargs=True))
        handles.append(linear.register_forward_hook(negate_output, with_kwargs=True))
    holder = _Holder(models)
    walked = census(holder)
    backend.expect(
        {name: [d.key for d in walked.destinations if d.component == name] for name in models}
    )
    backend.resident_budget = 2_110_000  # one original float component, second parked
    backend.hold_lease = True
    live = backend.materialize(holder, walked, encoded_leaves="accept")
    for d in walked.destinations:
        backend.fill(d.key, d.spec, live[d.key])
    backend.commit()
    del live, walked
    assert set(backend.parked) == {"b"}
    assert original["a"].weight.is_meta and original["a"].bias.is_meta
    residency = ComponentResidency(backend=backend, torch=torch, placement="component_staged")
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    # A scope may need both components: old logical addresses must not become false
    # alias refusals when the allocator reuses their released blocks for the second.
    residency.admit("both", ("a", "b"))
    with torch.no_grad():
        assert torch.equal(models["a"].linear(x), models["b"].linear(x))
    residency.release("both", ("a", "b"))
    torch.cuda.synchronize()
    expected = None
    records = []
    for name in ("a", "b", "a", "b", "a"):
        torch.cuda.synchronize()
        residency.admit("forward", (name,))
        torch.cuda.synchronize()
        with torch.no_grad():
            output = models[name].linear(x).cpu()
        assert torch.isfinite(output).all()
        if expected is None:
            expected = output
        else:
            assert torch.equal(output, expected)
        assert original[name].weight.is_meta and original[name].bias.is_meta
        assert models[name].linear.provider == "cozy.fp8-rowwise.native-leaf/1"
        memory = backend.component_memory[name]
        measured = sum(t.numel() * t.element_size() for t in models[name].buffers())
        assert measured == memory.settled == backend.vram_charge[name]
        residency.release("forward", (name,))
        records.append(
            {
                "component": name,
                "allocated": torch.cuda.memory_allocated(),
                "fill_peak": memory.peak,
                "settled": memory.settled,
            }
        )
        torch.cuda.synchronize()
    assert records[0]["allocated"] == records[-1]["allocated"], records
    assert all(count >= 3 for count in hook_calls.values()), hook_calls
    # Handles registered on the original modules still control their replacements.
    for handle in handles:
        handle.remove()
    assert expected is not None
    expected = -expected
    # Read the peak immediately around the SAME backend.stage: admit resets CUDA's
    # peak at its end to start the forward-activation meter, so reading it afterwards
    # would misleadingly report zero staging overhead.
    residency.vacate()
    torch.cuda.synchronize()
    restored = residency.restore()
    assert set(restored["restored"]) == {"a", "b"}, restored
    with torch.no_grad():
        assert torch.equal(models["a"].linear(x).cpu(), expected)
        assert torch.equal(models["b"].linear(x).cpu(), expected)
    residency.vacate()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    backend.stage("a")
    torch.cuda.synchronize()
    stage_peak = torch.cuda.max_memory_allocated() - before
    assert stage_peak >= backend.component_memory["a"].peak
    assert stage_peak <= backend.component_memory["a"].peak + 65536
    backend.evict("a")
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    # A real impossible headroom refuses before any payload/allocation and does not poison.
    residency.scope_headrooms["short"] = total
    try:
        residency.admit("short", ("b",))
    except ResidencyRefusal as exc:
        assert exc.code == "device_shortfall" and not bool(residency.poisoned)
        assert original["b"].weight.is_meta
        assert torch.cuda.memory_allocated() == before
    else:
        raise AssertionError("impossible measured headroom admitted")
    # A genuine malformed topology fails after allocation; no subsequent admission may serve it.
    assert all(t.is_meta for m in original.values() for t in m.parameters())
    models["b"].linear.out_features = 1000
    try:
        residency.admit("broken", ("b",))
    except ResidencyRefusal as exc:
        assert exc.code == "leaf_schema" and bool(residency.poisoned), str(exc)
    else:
        raise AssertionError("invalid logical topology was served")
    try:
        residency.admit("retry", ("a",))
    except ResidencyRefusal as exc:
        assert exc.code == "residency_poisoned"
    else:
        raise AssertionError("poisoned generation was reused")
    backend.poison(tuple(backend.enqueued))
    backend.close()
    assert not backend.leases.held and not backend.scratch
    tied = StreamingFillBackend(
        checkpoint,
        rows,
        plan=backend.execution,
        qualified=(backend.capabilities, backend.probe),
        release="encoded-proof/1",
        window_bytes=1 << 20,
        slots=3,
        readers=2,
        inflight=2,
    )
    tied_models = {name: model() for name in ("a", "b")}
    tied_holder = _Holder(tied_models)
    walked = census(tied_holder)
    tied.expect(
        {name: [d.key for d in walked.destinations if d.component == name] for name in tied_models}
    )
    tied.materialize(tied_holder, walked, encoded_leaves="accept")
    tied_models["b"].linear.weight = tied_models["a"].linear.weight
    fresh = census(tied_holder)
    try:
        for destination in fresh.destinations:
            tied.fill(destination.key, destination.spec, fresh.live[destination.key])
    except FillRefusal as exc:
        assert exc.code == "leaf_schema" and "shares storage" in str(exc)
    else:
        raise AssertionError("shared native weight admitted twice")
    tied.poison(tuple(tied.enqueued))
    tied.close()
    print(
        json.dumps(
            {
                "phase": "passed",
                "records": records,
                "failure": "poisoned",
                "staging_peak_bytes": stage_peak,
                "staging_bound_bytes": backend.component_memory["a"].peak,
                "native_bytes": bound,
                "peak_reserved": torch.cuda.max_memory_reserved(),
            }
        ),
        flush=True,
    )
    # No rollback: process exit is the reclaim authority for the poisoned allocation.


if __name__ == "__main__":
    main(Path(sys.argv[1]))
