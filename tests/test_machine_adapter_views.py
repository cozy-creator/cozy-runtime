"""Adapter graph preparation uses a real CPU-only worker and native local custody."""

from pathlib import Path

from cozy_runtime.internal.worker import machine_adapter_views, machine_model_defaults
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from test_device_lanes import _config
from test_lora_composition import fixture


def test_cpu_worker_prepares_and_reuses_owned_adapter_view(tmp_path: Path) -> None:
    store, base, first, _second = fixture(tmp_path)
    for name, artifact in (("base", base), ("style", first)):
        store.replace_local(
            None,
            name,
            artifact.tensorfs_receipt_digest,
            artifact.manifest.digest,
            artifact.manifest.length,
        )
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
    adapter = machine_adapter_views.Adapter(
        machine_model_defaults.Selected(
            parameter="style",
            repository="local/style",
            manifest=machine_model_defaults.Manifest(first.manifest.digest, first.manifest.length),
        ),
        "transformer",
        scale="0.5",
    )

    def check() -> None:
        assert not worker.stop.is_set()

    result = machine_adapter_views.compose(worker, selected, [adapter], check=check)
    again = machine_adapter_views.compose(worker, selected, [adapter], check=check)
    assert again == result
    assert result["manifest"] == selected["manifest"]
    composed = result["composed"]
    assert isinstance(composed, dict)
    assert composed["manifest"] != base.manifest.digest
    assert result["adapters"] == [
        {
            "model": "local/style",
            "manifest": first.manifest.digest,
            "manifest_length": first.manifest.length,
            "component": "transformer",
            "source_component": "adapter",
            "scale": "0.5",
            "release": "",
            "lane": "",
        }
    ]
    # There is no accelerator to reserve on this worker; graph preparation still completed.
    assert worker.options.accelerator_backend == "none"
    assert worker.stages.view() == {"leases": {}, "waiting": {}, "holders": {}, "demands": {}}
    assert not worker.engine.live
    assert worker.supervision.current is None
