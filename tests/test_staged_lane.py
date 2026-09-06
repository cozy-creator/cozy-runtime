"""Actual Worker assignment, executor preparation and native component staging on CUDA.

The positive package has five independent scopes; the negative package declares one
scope holding them all. Both have the SAME complete native tensor geometry larger than
the driver's actual total VRAM. No capacity reading or declared binding is patched.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from cozy_runtime.author import WeightsConfig, WeightsPart, WeightsTarget, WeightsTensor
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal import accel
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
from test_shared_lane import TenantStore, _drive, _turn, _workspace, needs_tenant

PACKAGE = Path(__file__).resolve().parent.parent / "corpus" / "staged-lane"


def write_staged(root: Path, component_bytes: int) -> TenantStore:
    import tensorfs

    store = tensorfs.Store.ensure(str(root / "store"))
    names = ("a", "b", "c", "d", "e")
    elements = component_bytes // 2
    config = json.dumps({"elements": elements}).encode()
    bound = component_bytes * len(names) + len(config)
    host = WeightsTransactionHost(
        store=store,
        owner_scope="staged-lane-proof",
        request_id="create",
        invocation_spec_digest="sha256:" + "a" * 64,
        writer_session_id=1,
        allowed_sources={},
        output_bounds={"model": bound},
    )
    writer = host.open(
        WeightsCommit(
            output_slot="model",
            sources={},
            targets={
                name: WeightsTarget(
                    add={
                        "weight": WeightsTensor(
                            "bf16",
                            (elements,),
                            SPEC_PLAIN,
                            {"value": WeightsPart("bf16", (elements,))},
                        )
                    }
                )
                for name in names
            },
            configs={"model": WeightsConfig(data=config)},
            order=tuple((name, "weight") for name in names),
            max_new_bytes=bound,
        )
    )
    # A sparse input file supplies real zero bytes. TensorFS hashes/objectizes the
    # complete stream; equal 64 MiB objects deduplicate, without a host state dict.
    with (root / "zeros").open("w+b") as source:
        source.truncate(component_bytes)
        for name in names:
            source.seek(0)
            writer.add_part(name, "weight", "value", source)
    (root / "zeros").unlink()
    writer.add_config("model", config, len(config))
    receipt = json.loads(writer.commit().tensorfs_receipt)
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    length = receipt["manifest"]["length"]
    operation = store.begin_operation("source", "cozytest", "staged-lane")
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1", "bf16", manifest, length)
    return TenantStore(root / "store", component_bytes * len(names), {"elements": elements})


@needs_tenant
def test_complete_model_larger_than_vram_reaches_bounded_prepare_and_stages() -> None:
    memory = accel.device_memory("0", accel.host_backend_family())
    assert memory.state == "measured"
    # Five components, each just over one quarter of real VRAM; the total cannot
    # fit even on an otherwise empty card. Round native geometry to whole 64 MiB parts.
    part = 64 << 20
    component_bytes = (memory.total_bytes // 4 // part + 1) * part
    with _workspace() as root:
        store = write_staged(root, component_bytes)
        assert store.weight_bytes > memory.total_bytes
        package = root / "package"
        shutil.copytree(PACKAGE, package)
        owner, worker, notes = _drive(store, [_turn("plc-staged", "staged", 0)], root, package)
        assert owner.ended == "terminal", (owner.ended, owner.error, notes)
        assert owner.turns[0].outcome["status"] == 1, owner.turns[0].outcome
        row = owner.turns[0].rows["plc-staged"]
        assert row["evicted"], row
        walk = row["walk"]
        assert walk["declared_bytes"] == store.weight_bytes, walk
        assert worker.lanes.lanes[0].outstanding_bytes() == 0
        print(
            json.dumps(
                {
                    "declared": store.weight_bytes,
                    "device_total": memory.total_bytes,
                    "component": component_bytes,
                    "walk": walk,
                }
            ),
            flush=True,
        )
        # Same native weights, but a real author scope requires all five together.
        # A distinct package source identity carries that requirement through prepare.
        negative = root / "negative"
        shutil.copytree(PACKAGE, negative)
        (negative / "staged_lane.py").rename(negative / "whole_lane.py")
        manifest = negative / "package.toml"
        manifest.write_text(manifest.read_text().replace("staged_lane:app", "whole_lane:app"))
        source = negative / "whole_lane.py"
        source.write_text(
            source.read_text().replace(
                '@uses_components("a")', '@uses_components("a", "b", "c", "d", "e")'
            )
        )
        refused, _worker, refusal_notes = _drive(
            store, [_turn("plc-whole", "whole", 0)], root / "whole", negative
        )
        assert refused.ended == "terminal", (refused.ended, refused.error, refusal_notes)
        assert refused.turns[0].outcome["status"] != 1, refused.turns[0].outcome
        assert "device_shortfall" in refused.turns[0].outcome["cause"]["detail"]
        negative_walk = refused.turns[0].rows["plc-whole"]["walk"]
        assert negative_walk["largest_declared_set_bytes"] == store.weight_bytes, negative_walk
        # The existing single-tenant uncalibrated policy may deepen to a bounded
        # attempt. Its scope guard must still refuse before the method accesses weights.
        assert refused.turns[0].outcome["execution_started"]
        print(
            json.dumps(
                {
                    "negative": refused.turns[0].outcome["cause"],
                    "largest_scope": negative_walk["largest_declared_set_bytes"],
                }
            ),
            flush=True,
        )
