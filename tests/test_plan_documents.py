"""A plan has no document cap: a real 40,000-tensor checkpoint resolves and digests."""

from __future__ import annotations

import io
import json
from pathlib import Path

import msgspec
from tensorfs import Store
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.internal import canonical, qualification
from cozy_runtime.internal.encoding import SPEC_PLAIN, launch_providers
from cozy_runtime.internal.fill import Checkpoint, dtype_name
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.probe import QualificationResult
from cozy_runtime.internal.resolution import ConstructionFacts, Variant, resolve
from cozy_runtime.internal.weights_sink import weights_transaction_id

TENSORS = 40_000
#: A shipped table answers for its card without measuring, so nothing reads torch.
NOT_TORCH = object()


def _checkpoint(root: Path, keys: list[str]) -> Checkpoint:
    store = Store.ensure(str(root / "store"))
    tensor = Tensor("bf16", (1,), SPEC_PLAIN, {"value": Part("bf16", (1,))})
    writer = store.begin_derived(
        weights_transaction_id("plan-documents", "create", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={"model": Target(add=dict.fromkeys(keys, tensor))},
            configs={"model": Config("add")},
            order=tuple(("model", key) for key in keys),
        ).native_arguments(1 << 30),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for key in keys:
        writer.add_part("model", key, "value", io.BytesIO(b"\x80\x3f"))
    writer.add_config("model", io.BytesIO(b"{}"))
    receipt = writer.commit()
    return Checkpoint.from_store(store, "sha256:" + receipt["manifest"]["sha256"])


def test_a_40000_tensor_plan_resolves_and_digests(tmp_path: Path) -> None:
    """Its plan identity passes 8 MiB, where the old document cap refused the load with
    `size_cap` (from about 30,000 tensors)."""
    keys = [
        f"model.language_model.layers.{i // 128:03d}.mlp.experts.{i % 128:03d}"
        ".gate_up_proj.weight_scale_inv"
        for i in range(TENSORS)
    ]
    checkpoint = _checkpoint(tmp_path, keys)
    path = next(qualification.SHIPPED.glob("cuda.sm89-*.json"))
    table = msgspec.convert(json.loads(path.read_bytes()), QualificationResult)
    plain = {SPEC_PLAIN: launch_providers()[SPEC_PLAIN]}
    admitted = qualification.qualified(
        NOT_TORCH, plain, table.device, ("bfloat16",), release="t/1", runtime=table.runtime
    )
    manifest = checkpoint.manifest_id
    variant = Variant("sm89", str(tmp_path / "store"), manifest, {"model": manifest}, True)
    plan = resolve(
        variants=[variant],
        rows_for={"sm89": checkpoint.rows("model")},
        providers=launch_providers(),
        capabilities=admitted.capabilities,
        device=table.device,
        runtime=table.runtime,
        construction=ConstructionFacts("t/1", "wide:Model", ("model",)),
        facts=PlanFacts(),
        dtype_name=dtype_name,
    )
    assert len(plan.tensors) == TENSORS
    assert len(canonical.write(plan.identity())) > 8 << 20
    assert plan.digest().startswith("sha256:")
    assert plan.seam_document()["tensors_count"] == TENSORS
