"""Memory facts the planner reads: a cgroup's page cache and shared memory in either cgroup
version, and the executor's plane document as telemetry on the ledger.
"""

from __future__ import annotations

import os
from pathlib import Path

from cozy_runtime.internal import proctree
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    PLANE_PLACEMENT,
    DeclaredBinding,
    PlanChooser,
    PreparedModel,
    PreparedRequest,
)

GiB = 1 << 30


def test_a_cgroups_page_cache_and_shared_memory_are_read_in_either_version(
    tmp_path: Path,
) -> None:
    """A rented pod is cgroup v1 (its subtree's counters under `total_`), a desktop v2."""
    (tmp_path / "memory.stat").write_text(
        "cache 900\nshmem 7\ntotal_inactive_file 300\ntotal_active_file 200\ntotal_shmem 100\n"
    )
    v1 = {"inactive_file": 300, "active_file": 200, "shmem": 100}
    assert proctree._cgroup_stat(tmp_path) == v1
    (tmp_path / "memory.stat").write_text("anon 5\nfile 900\nshmem 100\ninactive_file 300\n")
    assert proctree._cgroup_stat(tmp_path) == {"inactive_file": 300, "active_file": 0, "shmem": 100}


def test_a_plane_document_is_telemetry_and_a_new_generation_forgets_it() -> None:
    ledger = Ledger(worker_pid=os.getpid())
    ledger.begin_generation(1)
    ledger.observe_plane(
        {
            "committed_bytes": 3 * GiB,
            "resident": {"unet": 3 * GiB},
            "h2d_gbps": "fast",
            "from_a_newer_executor": 1,
        }
    )
    assert ledger.plane is not None
    assert (ledger.plane.committed_bytes, ledger.plane.h2d_gbps) == (3 * GiB, 0.0)
    ledger.observe_plane(None)
    assert ledger.plane is not None and ledger.plane.budget_bytes == -1, "unreadable, not 0"
    ledger.device_reserved, ledger.device_allocated = 3 * GiB, GiB
    assert ledger.slack == 2 * GiB
    ledger.begin_generation(2)
    assert ledger.slack == 0 and ledger.plane is None


def test_a_plane_tenant_plans_weight_plane_whatever_room_it_was_given() -> None:
    ledger = Ledger(worker_pid=os.getpid())
    ledger.begin_generation(1)
    ledger.observe_construction({"resident": {"unet": GiB}, "parked": ["vae"]})
    binding = DeclaredBinding(
        entrypoint_binding_digest="sha256:" + "0" * 64,
        entrypoint="serve",
        model_class="M",
        model_binding_path="serve.models.m",
        model_parameter_name="m",
        release="r",
        logical_weight_bytes=GiB,
    )
    chooser, model = PlanChooser(ledger), PreparedModel(delivery_rung="verbatim")
    prepared = PreparedRequest.unresolved("serve", {"width": 512})
    assert chooser.choose(binding, model, prepared, fits=False).placement == "component_staged"
    ledger.observe_construction({"layouts": {"unet": {"common": GiB // 2, "blocks": [GiB]}}})
    ledger.observe_plane({"resident": {"unet": GiB // 2}})
    plan = chooser.choose(binding, model, prepared, fits=False)
    assert (plan.placement, plan.resident_bytes) == (PLANE_PLACEMENT, GiB // 2)
    assert plan.model_weight_bytes == 3 * GiB // 2, "the plane's weight sets, whole"
