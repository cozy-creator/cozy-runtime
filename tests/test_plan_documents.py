"""A plan has no document cap: a real 40,000-tensor checkpoint resolves and digests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorfs

from cozy_runtime.internal import canonical, probe
from cozy_runtime.internal.encoding import launch_providers
from cozy_runtime.internal.fill import Checkpoint, dtype_name
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.resolution import ConstructionFacts, Variant, resolve
from store_fixtures import home
from test_leaf_speedup_record import sm86

TENSORS = 40_000


def _checkpoint(store: Path, keys: list[str]) -> Checkpoint:
    writer = tensorfs.Writer(cache_dir=store)
    model = writer.component("model", cls=("", "", ""), config={})
    one = np.array([0x3F80], np.uint16)
    for key in keys:
        model.put(key, memoryview(one), dtype="BF16", shape=[1])
    return Checkpoint(writer.commit())


def test_a_40000_tensor_plan_resolves_and_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its plan identity passes 8 MiB, where the old document cap refused the load with
    `size_cap` (from about 30,000 tensors)."""
    keys = [
        f"model.language_model.layers.{i // 128:03d}.mlp.experts.{i % 128:03d}"
        ".gate_up_proj.weight_scale_inv"
        for i in range(TENSORS)
    ]
    store = home(tmp_path, monkeypatch)
    checkpoint = _checkpoint(store, keys)
    table = sm86()
    ref = checkpoint.id
    variant = Variant("sm86", str(tmp_path), ref, {"model": ref}, True)
    plan = resolve(
        variants=[variant],
        rows_for={"sm86": checkpoint.rows("model")},
        providers=launch_providers(),
        capabilities=probe.admission_table(table),
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
