"""An invalid later adapter refuses the entire CPU preparation without publishing a view."""

from pathlib import Path

import pytest
from test_lora_composition import Value, fixture, header, model

from cozy_runtime.internal.worker import machine_adapter_views, machine_model_defaults
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from test_device_lanes import _config


def test_later_invalid_adapter_cannot_publish_or_fall_back_to_the_base(tmp_path: Path) -> None:
    store, base, first, _second = fixture(tmp_path)
    invalid = model(
        store,
        tmp_path,
        "invalid",
        "adapter",
        {
            "proj.lora_A.weight": Value((1, 2), (1, 2)),
            "proj.lora_B.weight": Value((2, 1), (1, 2)),
        },
    )
    for name, artifact in (("base", base), ("style", first), ("invalid", invalid)):
        store.replace_local(
            None,
            name,
            artifact.tensorfs_receipt_digest,
            artifact.manifest.digest,
            artifact.manifest.length,
        )
    original = header(store, base)
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(
            root=tmp_path / "worker",
            tensorfs_root=Path(store.root),
            accelerator_backend="none",
            devices="",
        ),
        InMemoryControlHost(),
    )
    selected = {
        "parameter": "model",
        "repository": "local/base",
        "manifest": {"digest": base.manifest.digest, "length": base.manifest.length},
    }
    stack = [
        machine_adapter_views.Adapter(
            machine_model_defaults.Selected(
                parameter=name,
                repository="local/" + name,
                manifest=machine_model_defaults.Manifest(
                    artifact.manifest.digest, artifact.manifest.length
                ),
            ),
            "transformer",
            scale="0.5",
        )
        for name, artifact in (("style", first), ("invalid", invalid))
    ]
    try:
        before = store.repo_get("local", "base")
        with pytest.raises(WorkspaceRefusal, match="adapter_shape"):
            machine_adapter_views.compose(worker, selected, stack, check=lambda: None)
        assert store.repo_get("local", "base") == before
        assert header(store, base) == original
        assert worker.gpu.view() == {"leases": {}, "grants": {}, "waiting": {}}
        assert not worker.engine.live
        assert worker.supervision.current is None
        # Repeating the same invalid selection still refuses; no partly applied cached view
        # becomes a successful answer. Selecting only the valid adapter subsequently succeeds.
        with pytest.raises(WorkspaceRefusal, match="adapter_shape"):
            machine_adapter_views.compose(worker, selected, stack, check=lambda: None)
        result = machine_adapter_views.compose(worker, selected, stack[:1], check=lambda: None)
        assert result["manifest"] == selected["manifest"]
        composed = result["composed"]
        assert isinstance(composed, dict)
        assert composed["manifest"] != base.manifest.digest
        assert len(result["adapters"]) == 1
        assert store.repo_get("local", "base") == before
        assert header(store, base) == original
    finally:
        worker.shutdown()
