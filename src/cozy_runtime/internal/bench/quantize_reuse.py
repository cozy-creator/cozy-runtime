"""The Runtime's per-component quantization across an Output's lifetime (`cozy_runtime.bench.quantize_reuse`; the release
gate's `svdq-calibration` controller). Each case reopens its own Output by store, slot and replay key, as only the
Runtime does when an attempt retries, so no package can drive it. CPU only.

- `components`: two components converted one call each under one full-plan identity (svdq.nvfp4 with real calibration
  hooks): neither call writes the other's tensor, and an unknown component refuses before any write. A finished Output
  reuses both with no reads or writes and refuses a different calibration. A noncalibrated encoding (fp8.rowwise) binds
  the same identity: reused when unchanged, refused with another encoding or a smaller plan.
- `resume`: an unfinished Output closed after component A and reopened converts only B; the finished one reuses A with
  no reads or writes. A changed quality bound refuses, finished or not. A tensor written by hand can't seed a native
  plan, and a base's inherited tensor can't take its acceptance.

The selected weights' bytes are the gate's to check, through the public Output conversion (`svdq-calibration`'s
calibrate and predict)."""

from __future__ import annotations

import hashlib
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import tensorfs
import torch

from cozy_runtime.author import Context, Telemetry, UnsupportedInput
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.derive import native_quantization
from cozy_runtime.derive.quantization import (
    ArtifactQuantizationPlan,
    ArtifactQuantizationRequest,
    QuantizationTensor,
    quantize_component_into,
)
from cozy_runtime.derive.svdq import Calibration
from cozy_runtime.weights_store import Derivation, Output

if TYPE_CHECKING:
    from cozy_runtime.derive.native_quantization import QuantizationStats

CASES = ("components", "resume")
WORKSPACE = 64 << 20
BOUND = 2 << 20
type Facts = dict[str, object]


def encoded_bytes(shape: tuple[int, int]) -> int:
    """One svdq.nvfp4 weight's stored bytes: residual, block scales, global scale, smoothing, both rank-32 factors."""
    n, k = shape
    return n * k // 2 + n * k // 16 + 8 + k * 4 + (n + k) * 32 * 2


def identity_refused(attempt: Callable[[], object]) -> str:
    """`attempt`'s refusal: a finished scope's changed identity."""
    try:
        attempt()
    except CapabilityError as error:
        assert error.code == "quantization_identity", error.code
        return str(error)
    raise AssertionError("a finished Output reused changed work")


def components(ctx: Context, tel: Telemetry, seed: int) -> Facts:
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(prefix="cozy-svdq-components-") as temporary:
        directory = Path(temporary)
        store = directory / "store"
        writer = tensorfs.Writer(directory / "source.cozytensors", cache_dir=store)
        modules: dict[str, torch.nn.Sequential] = {}
        for name in ("a", "b"):
            weight = torch.from_numpy(rng.normal(0, 0.0625, (64, 256)).astype("<f4")).bfloat16()
            linear = torch.nn.Linear(256, 64, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                linear.weight.copy_(weight)
            modules[name] = torch.nn.Sequential(OrderedDict([("proj", linear)]))
            writer.component(name).put(
                "proj.weight", weight.view(torch.uint8).numpy().tobytes(), dtype="BF16", shape=[64, 256]
            )
        source = writer.commit()
        plan = ArtifactQuantizationPlan(
            ["a", "b"],
            [(name, "proj.weight") for name in ("a", "b")],
            [QuantizationTensor(name, "proj.weight", "bf16", (64, 256)) for name in ("a", "b")],
        )

        def collect() -> Calibration:
            calibration = Calibration(torch, modules, ("a.proj.weight", "b.proj.weight"))
            with calibration, torch.inference_mode():
                for _ in range(2):
                    for module in modules.values():
                        module(torch.from_numpy(rng.normal(0, 0.125, (16, 256)).astype("<f4")).bfloat16())
            return calibration

        def opened(label: str, work: Derivation) -> Output:
            transaction = "sha256:" + hashlib.sha256(f"{ctx.request_id}:{label}".encode()).hexdigest()
            return Output(store, label, transaction, work, BOUND)

        def convert(
            output: Output, component: str, encoding: str = "svdq.nvfp4", chosen: ArtifactQuantizationPlan = plan,
            calibration: Calibration | None = None,
        ) -> QuantizationStats:
            return quantize_component_into(
                source, output, ctx, ArtifactQuantizationRequest(fit_workspace_bytes=WORKSPACE), tel,
                encoding=encoding, plan=chosen, component=component, logical_dtype=lambda row: row.logical_dtype,
                calibration=calibration,
            )

        calibration = collect()
        rows = native_quantization.resolved(source, plan, "svdq.nvfp4", lambda row: row.logical_dtype)
        with native_quantization.snapshot(source, plan, "svdq.nvfp4", calibration, ctx) as snapshot:
            assert snapshot is not None
            work = Derivation(
                base=source, sources=(snapshot,), order=plan.order,
                provenance=native_quantization.provenance(source, rows, snapshot),
            )
            with opened("components", work) as output:
                assert not output.has("a", "proj.weight") and not output.has("b", "proj.weight")
                try:
                    convert(output, "missing", calibration=calibration)
                except UnsupportedInput:
                    assert output.written == 0
                else:
                    raise AssertionError("an unknown component did not refuse before writes")
                for name in ("a", "b"):
                    measured = convert(output, name, calibration=calibration)
                    assert measured.encoded_keys == 1 and measured.reused_keys == 0
                    assert measured.source_bytes_read == 64 * 256 * 2
                    assert measured.new_bytes_written == encoded_bytes((64, 256))
                    assert len(measured.tensors) == 1 and measured.tensors[0]["component"] == name
                    assert output.has(name, "proj.weight")
                    if name == "a":
                        assert not output.has("b", "proj.weight"), "the call for A wrote B"
                        assert output.written == encoded_bytes((64, 256))
                assert output.written == 2 * encoded_bytes((64, 256))
                fitted = output.commit()
            assert fitted.verify().ok
            fitted.close()
            # A finished Output's reuse validates the same complete plan and scope.
            with opened("components", work) as reopened:
                assert reopened.done is not None
                for name in ("a", "b"):
                    reused = convert(reopened, name, calibration=calibration)
                    assert reused.encoded_keys == 0 and reused.reused_keys == 1
                    assert reused.source_bytes_read == 0 and reused.new_bytes_written == 0
                    assert reused.worst_relative_frobenius is None
                changed = collect()
                calibration_refused = identity_refused(lambda: convert(reopened, "a", calibration=changed))
        # Noncalibrated native outputs bind the same complete producer identity.
        plain_rows = native_quantization.resolved(source, plan, "fp8.rowwise", lambda row: row.logical_dtype)
        plain_work = Derivation(
            base=source, order=plan.order, provenance=native_quantization.provenance(source, plain_rows, None)
        )
        with opened("rowwise", plain_work) as output:
            for name in ("a", "b"):
                plain = convert(output, name, "fp8.rowwise")
                assert plain.encoded_keys == 1 and plain.reused_keys == 0
                assert plain.source_bytes_read == 64 * 256 * 2
                assert plain.new_bytes_written == 64 * (256 + 4)
                assert len(plain.tensors) == 1 and plain.tensors[0]["component"] == name
                if name == "a":
                    assert not output.has("b", "proj.weight")
            output.commit().close()
        with opened("rowwise", plain_work) as reopened:
            for name in ("a", "b"):
                again = convert(reopened, name, "fp8.rowwise")
                assert again.encoded_keys == 0 and again.reused_keys == 1
                assert again.source_bytes_read == 0 and again.new_bytes_written == 0
                assert again.worst_relative_frobenius is None
            reduced = ArtifactQuantizationPlan(["a"], [("a", "proj.weight")], plan.tensors[:1])
            nonfit_refused = [
                identity_refused(lambda: convert(reopened, "a", "int8.convrot")),
                identity_refused(lambda: convert(reopened, "a", "fp8.rowwise", reduced)),
            ]
        source.close()
    return {
        "different_finished_calibration_refused": calibration_refused,
        "nonfit_identity_refusals": nonfit_refused,
        "full_plan_single_identity": True,
        "future_component_not_written": True,
    }


def resume(ctx: Context, tel: Telemetry, seed: int) -> Facts:
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(prefix="cozy-runtime-component-resume-") as temporary:
        directory = Path(temporary)
        store = directory / "store"
        writer = tensorfs.Writer(directory / "source.cozytensors", cache_dir=store)
        values: dict[str, npt.NDArray[np.float32]] = {}
        for name in ("a", "b"):
            values[name] = rng.normal(0, 0.0625, (64, 256)).astype("<f4")
            writer.component(name).put("weight", values[name], dtype="F32", shape=[64, 256])
        source = writer.commit()
        plan = ArtifactQuantizationPlan(
            ["a", "b"],
            [(name, "weight") for name in ("a", "b")],
            [QuantizationTensor(name, "weight", "f32", (64, 256)) for name in ("a", "b")],
        )
        work = Derivation(base=source, order=plan.order)

        def args(label: str) -> tuple[Path, str, str, Derivation, int]:
            transaction = "sha256:" + hashlib.sha256(f"{ctx.request_id}:{label}".encode()).hexdigest()
            return (store, label, transaction, work, BOUND)

        def convert(output: Output, component: str, bound: float = 0.2) -> QuantizationStats:
            return quantize_component_into(
                source, output, ctx, ArtifactQuantizationRequest(max_relative_frobenius=bound), tel,
                encoding="fp8.rowwise", plan=plan, component=component, logical_dtype=lambda row: row.logical_dtype,
            )

        resumed_args = args("resumed")
        with Output(*resumed_args) as first:
            a = convert(first, "a")
            assert a.encoded_keys == 1 and a.reused_keys == 0
            assert first.has("a", "weight") and not first.has("b", "weight")
        # The first Output and its native helper really ended. A is never converted again on this unfinished writer:
        # only B gets a call after the reopen.
        with Output(*resumed_args) as resumed:
            assert resumed.done is None and resumed.has("a", "weight")
            assert not resumed.has("b", "weight")
            b = convert(resumed, "b")
            assert b.encoded_keys == 1 and b.reused_keys == 0
            assert b.source_bytes_read == 64 * 256 * 4
            assert b.new_bytes_written == 64 * (256 + 4)
            assert len(b.tensors) == 1 and b.tensors[0]["component"] == "b"
            completed = resumed.commit()
        with Output(*resumed_args) as finished:
            assert finished.done is not None
            again = convert(finished, "a")
            assert again.encoded_keys == 0 and again.reused_keys == 1
            assert again.source_bytes_read == again.new_bytes_written == 0
            assert again.worst_relative_frobenius is None
            changed_finished = [identity_refused(lambda: convert(finished, "a", bound)) for bound in (0.0, 0.4)]
        assert completed.verify().ok
        finished_id = completed.id
        completed.close()

        policy_args = args("changed-policy")
        with Output(*policy_args) as first:
            convert(first, "a")
        with Output(*policy_args) as resumed:
            for bound in (0.0, 0.4):
                try:
                    convert(resumed, "a", bound)
                except tensorfs.InvalidError as error:
                    assert "selected by another conversion" in str(error)
                else:
                    raise AssertionError("a native resume accepted a changed quality bound")
            assert resumed.has("a", "weight") and not resumed.has("b", "weight")

        controls: dict[str, dict[str, str | int]] = {}
        for label, manual in (("manual-before-binding", True), ("inherited-base", False)):
            control_args = args(label)
            with Output(*control_args) as output:
                if manual:
                    output.put("a", "weight", "F32", (64, 256), values["a"])
                assert output.has("a", "weight") is manual
                if manual:
                    try:
                        convert(output, "b")
                    except tensorfs.InvalidError as error:
                        assert "selected tensor was written without this quantization identity" in str(error)
                        controls[label] = {"native_refusal": str(error)}
                    else:
                        raise AssertionError("a hand-written A was admitted by the first native binding")
                    assert not output.has("b", "weight")
                else:
                    only_b = convert(output, "b")
                    assert only_b.encoded_keys == 1 and not output.has("a", "weight")
                    controls[label] = {"only_b_encoded": only_b.encoded_keys}
                committed = output.commit()
            with Output(*control_args) as finished:
                controls[label]["finished_a_refused"] = identity_refused(lambda: convert(finished, "a"))
            committed.close()
        source.close()
    return {
        "checkpoint": finished_id,
        "a_encoded": a.encoded_keys,
        "b_after_reopen": {"encoded": b.encoded_keys, "reused": b.reused_keys, "read_bytes": b.source_bytes_read,
                           "new_bytes": b.new_bytes_written},
        "a_finished_reuse": {"encoded": again.encoded_keys, "reused": again.reused_keys,
                             "read_bytes": again.source_bytes_read, "new_bytes": again.new_bytes_written,
                             "worst_relative_frobenius": again.worst_relative_frobenius},
        "changed_finished_policy_refusals": changed_finished,
        "changed_unfinished_policy_refusals": [0.0, 0.4],
        "controls": controls,
    }


def exercise(ctx: Context, tel: Telemetry, case: str, seed: int) -> Facts:
    if case not in CASES:
        raise ValueError(f"no quantize_reuse case {case!r}; the cases are {', '.join(CASES)}")
    found = components(ctx, tel, seed) if case == "components" else resume(ctx, tel, seed)
    return {"case": case, "seed": seed, **found}
