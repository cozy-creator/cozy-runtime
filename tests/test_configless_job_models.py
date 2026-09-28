"""A real configless adapter is a valid job source, not an inference construction."""

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.internal import base_observation, census_cache, package_interface
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    Selected,
    Selection,
    _entrypoints,
    _selected_models,
    selections,
)
from cozy_runtime.protocol import documents


def checkpoint(tmp_path: Path, *, config: bool = False) -> tuple[Path, str, int, str]:
    root = tmp_path / "store"
    store = tensorfs.Store.ensure(str(root))
    value = b"\x00\x3c" * (64 * 64)
    identity = "sha256:" + hashlib.sha256(value + bytes([config])).hexdigest()
    transaction = store.begin_derived(
        identity,
        1,
        *Derivation(
            sources={},
            targets={
                "adapter": Target(
                    add={
                        key: Tensor(
                            "f16",
                            (64, 64),
                            SPEC_PLAIN,
                            {
                                "value": Part("f16", (64, 64)),
                            },
                        )
                        for key in ("layer.lora_A.weight", "layer.lora_B.weight")
                    }
                )
            },
            configs={"model": Config("add")} if config else {},
            order=(("adapter", "layer.lora_A.weight"), ("adapter", "layer.lora_B.weight")),
        ).native_arguments(32768),
        work_fingerprint=identity,
    )
    transaction.add_part("adapter", "layer.lora_A.weight", "value", io.BytesIO(value))
    transaction.add_part("adapter", "layer.lora_B.weight", "value", io.BytesIO(value))
    if config:
        transaction.add_config("model", io.BytesIO(b'{"width":64}'))
    receipt = transaction.commit()
    return (
        root,
        "sha256:" + receipt["manifest"]["sha256"],
        receipt["manifest"]["length"],
        hashlib.sha256(value).hexdigest(),
    )


def selection(manifest: str, slot: str, **extra: Any) -> Selection:
    row = {"package": "example/convert", "model": "example/adapter", "slot": slot}
    return selections([{**row, "manifest": manifest, **extra}])[0]


def interface() -> package_interface.PackageInterface:
    def modeled(name: str, path: str) -> package_interface.CallableDoc:
        slot = package_interface.ModelSlot(path=path, class_name="Adapter", component_use={})
        return package_interface.CallableDoc(name=name, request={}, result={}, models=(slot,))

    return package_interface.PackageInterface(
        format=package_interface.SCHEMA,
        application="convert",
        entrypoints=(modeled("generate", "generate.models.model"),),
        jobs=(modeled("main", "main.models.source"),),
    )


def test_configless_job_source_prepares_without_a_constructor_or_fabricated_config(
    tmp_path: Path,
) -> None:
    root, manifest, length, _payload = checkpoint(tmp_path)
    models, entrypoints = _entrypoints(
        interface(),
        package_name="example/convert",
        selections=[selection(manifest, "main.models.source")],
        tensorfs_root=root,
        verified=lambda *_args: None,
        census=None,
    )
    assert entrypoints == [] and len(models) == 1
    assert documents.spell(models[0].manifest.digest) == manifest
    assert models[0].manifest.length == length
    selected = _selected_models(
        "example/convert",
        [selection(manifest, "main.models.source")],
        root,
        lambda *_args: None,
        construction_slots=set(),
    )
    assert selected["main.models.source"].construction is None
    header = tensorfs.Store.open(str(root)).manifest(manifest)["header"]
    assert header is not None
    assert not tensorfs.parse_header(bytes(header)).get("configs")


@pytest.mark.parametrize("also_job", [False, True])
def test_same_configless_manifest_still_refuses_when_bound_for_serving(
    tmp_path: Path, also_job: bool
) -> None:
    root, manifest, _length, _payload = checkpoint(tmp_path)
    selected = [selection(manifest, "generate.models.model")]
    if also_job:
        selected.append(selection(manifest, "main.models.source"))
    with pytest.raises(PreparationRefusal, match="model_config_mismatch"):
        _entrypoints(
            interface(),
            package_name="example/convert",
            selections=selected,
            tensorfs_root=root,
            verified=lambda *_args: None,
            census=None,
        )


def test_shared_configured_manifest_derives_only_its_serving_slot(tmp_path: Path) -> None:
    root, manifest, _length, _payload = checkpoint(tmp_path, config=True)
    observed = []

    def census(
        wanted: Mapping[str, tuple[Selected, package_interface.ModelSlot]],
    ) -> dict[str, census_cache.Census]:
        for path, (model, _declared) in wanted.items():
            built = model.construction
            assert built is not None
            observed.append((path, built.config_bytes, built.assets(), built.tensor_dtypes))
        return {path: census_cache.Census(("adapter",), (), True, "") for path in wanted}

    selected = [
        selection(manifest, slot) for slot in ("main.models.source", "generate.models.model")
    ]
    models, entries = _entrypoints(
        interface(),
        package_name="example/convert",
        selections=selected,
        tensorfs_root=root,
        verified=lambda *_args: None,
        census=census,
    )
    assert len(models) == 2 and [entry.name for entry in entries] == ["generate"]
    assert observed == [
        (
            "generate.models.model",
            b'{"width":64}',
            {},
            {
                "adapter.layer.lora_A.weight": "f16",
                "adapter.layer.lora_B.weight": "f16",
            },
        )
    ]


def test_job_source_retains_the_declared_slot_guard(tmp_path: Path) -> None:
    root, manifest, _length, _payload = checkpoint(tmp_path)
    with pytest.raises(PreparationRefusal):
        _entrypoints(
            interface(),
            package_name="example/convert",
            selections=[selection(manifest, "missing.models.source")],
            tensorfs_root=root,
            verified=lambda *_args: None,
            census=None,
        )


def test_job_source_length_comes_from_the_held_manifest(tmp_path: Path) -> None:
    root, manifest, length, _payload = checkpoint(tmp_path)
    models, _ = _entrypoints(
        interface(),
        package_name="example/convert",
        selections=[selection(manifest, "main.models.source", manifest_length=length + 1)],
        tensorfs_root=root,
        verified=lambda *_args: None,
        census=None,
    )
    assert models[0].manifest.length == length


def test_a_corrupt_closure_is_refused_by_the_one_acquisition_walk(tmp_path: Path) -> None:
    """Preparation takes no closure lease; the placement's acquisition walk is the verifier."""

    root, manifest, length, payload = checkpoint(tmp_path)
    models, _ = _entrypoints(
        interface(),
        package_name="example/convert",
        selections=[selection(manifest, "main.models.source", manifest_length=length)],
        tensorfs_root=root,
        verified=lambda *_args: None,
        census=None,
    )
    stored = next(root.rglob(payload))
    stored.chmod(stored.stat().st_mode | 0o200)
    stored.write_bytes(b"x" * stored.stat().st_size)
    for name in ("cache", "environment", "selection"):
        (tmp_path / name).mkdir()
    acquirer = Acquirer(
        cache_root=tmp_path / "cache",
        install_root=tmp_path / "environment",
        selection_root=tmp_path / "selection",
        tensorfs_root=root,
        python=Path("/usr/bin/python3"),
        base=base_observation.current(),
    )
    placement: Any = type(
        "P", (), {"models": [{"manifest": {"digest": manifest, "length": length}}]}
    )()
    assert len(models) == 1
    with pytest.raises(AcquisitionRefusal):
        acquirer._models(placement)
