"""Actual Worker assignment, executor preparation and native component staging on CUDA.

The positive package has five independent scopes; the negative package declares one
scope holding them all. Both have the SAME complete native tensor geometry larger than
the driver's actual total VRAM. No capacity reading or declared binding is patched.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.internal import accel
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.weights_sink import weights_transaction_id
from test_shared_lane import TenantStore, _drive, _turn, _workspace, needs_tenant

PACKAGE = Path(__file__).resolve().parent.parent / "corpus" / "staged-lane"


def write_staged(root: Path, component_bytes: int) -> TenantStore:
    import tensorfs

    store = tensorfs.Store.ensure(str(root / "store"))
    names = ("a", "b", "c", "d", "e")
    elements = component_bytes // 2
    config = json.dumps({"elements": elements}).encode()
    bound = component_bytes * len(names) + len(config)
    writer = store.begin_derived(
        weights_transaction_id("staged-lane-proof", "create", "sha256:" + "a" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={
                name: Target(
                    add={
                        "weight": Tensor(
                            "bf16", (elements,), SPEC_PLAIN, {"value": Part("bf16", (elements,))}
                        )
                    }
                )
                for name in names
            },
            configs={"model": NativeConfig("add")},
            order=tuple((name, "weight") for name in names),
        ).native_arguments(bound),
        work_fingerprint="sha256:" + "b" * 64,
    )
    # A sparse input file supplies real zero bytes. TensorFS hashes/objectizes the
    # complete stream; equal 64 MiB objects deduplicate, without a host state dict.
    with (root / "zeros").open("w+b") as source:
        source.truncate(component_bytes)
        for name in names:
            source.seek(0)
            writer.add_part(name, "weight", "value", source)
    (root / "zeros").unlink()
    writer.add_config("model", io.BytesIO(config))
    receipt = writer.commit()
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


@needs_tenant
def test_staged_rung_confesses_this_attempt_not_the_generations_history() -> None:
    """The staged-rung confession's counts and its milliseconds are the SAME scope (cr-134).

    `Residency` keeps two ledgers: `stages`/`evictions`/`staged_ms`/`evicted_ms` accumulate
    for the life of the executor generation, and the `attempt_` counters are zeroed at every
    attempt. The confession read `attempt_stages` beside `staged_ms` — one count from each —
    and reported a production H3 request as "6 component stage(s) costing 31220.3 ms" when
    its six stages cost 5542.85 ms. The 31.2 s was the pod's whole history and became the
    case for changing residency policy.

    Two turns on one placement is the smallest arm that separates the ledgers: the second
    attempt's `attempt_staged_ms` is strictly below the generation's `staged_ms`, because
    the first attempt's staging is in one and not the other.
    """
    memory = accel.device_memory("0", accel.host_backend_family())
    assert memory.state == "measured"
    part = 64 << 20
    component_bytes = (memory.total_bytes // 4 // part + 1) * part
    with _workspace() as root:
        store = write_staged(root, component_bytes)
        assert store.weight_bytes > memory.total_bytes
        package = root / "package"
        shutil.copytree(PACKAGE, package)
        owner, worker, notes = _drive(
            store,
            [_turn("plc-staged", "staged-first", 0), _turn("plc-staged", "staged-second", 1)],
            root,
            package,
        )
        assert owner.ended == "terminal", (owner.ended, owner.error, notes)
        assert [turn.outcome["status"] for turn in owner.turns] == [1, 1], [
            turn.outcome for turn in owner.turns
        ]
        bundles = sorted(Path(worker.bundles.root).glob("*.json"))
        assert len(bundles) == 2, bundles
        seen = []
        for path in bundles:
            bundle = json.loads(path.read_text())
            residency = bundle["measurements"]["residency"]
            confessions = [
                row["value"]
                for row in bundle["confessions"]
                if "component stage(s)" in row["value"]
            ]
            assert len(confessions) == 1, bundle["confessions"]
            seen.append((residency, confessions[0]))
            # THE CONFESSION QUOTES THE ATTEMPT'S OWN NUMBERS, both halves.
            assert (
                f"{residency['attempt_stages']} component stage(s) costing "
                f"{residency['attempt_staged_ms']} ms" in confessions[0]
            ), confessions[0]
            assert (
                f"{residency['attempt_evictions']} eviction(s) costing "
                f"{residency['attempt_evicted_ms']} ms" in confessions[0]
            ), confessions[0]
            assert residency["attempt_stages"] > 0, residency
            assert residency["attempt_staged_ms"] > 0, residency
            # ... and never the generation's, which is a superset of them.
            assert residency["attempt_stages"] <= residency["stages"], residency
            assert residency["attempt_staged_ms"] <= residency["staged_ms"], residency
        # THE RED ARM: the second attempt inherits the first's history, so the two ledgers
        # have measurably different values. Without that, quoting either would pass.
        later = seen[-1][0] if seen[-1][0]["stages"] > seen[0][0]["stages"] else seen[0][0]
        assert later["attempt_stages"] < later["stages"], later
        assert later["attempt_staged_ms"] < later["staged_ms"], later
        assert str(later["staged_ms"]) not in seen[-1][1] + seen[0][1], (later, seen)
        print(
            json.dumps(
                {
                    "component": component_bytes,
                    "device_total": memory.total_bytes,
                    "attempts": [
                        {
                            "attempt_stages": row["attempt_stages"],
                            "attempt_staged_ms": row["attempt_staged_ms"],
                            "stages": row["stages"],
                            "staged_ms": row["staged_ms"],
                            "confession": line,
                        }
                        for row, line in seen
                    ],
                }
            ),
            flush=True,
        )
