"""A real configless adapter is a valid job source, not an inference construction."""

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest
import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    _entrypoints,
    _selected_models,
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


def selection(manifest: str, slot: str, **extra: Any) -> dict[str, Any]:
    return {
        "package": "example/convert",
        "model": "example/adapter",
        "slot": slot,
        "manifest": manifest,
        **extra,
    }


def interface() -> dict[str, Any]:
    return {
        "entrypoints": [{"name": "generate", "models": [{"path": "generate.models.model"}]}],
        "jobs": [{"name": "main", "models": [{"path": "main.models.source"}]}],
    }


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
        derive=None,
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
    assert set(selected["main.models.source"]) == {"selection", "manifest_length"}
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
            derive=None,
        )


def test_shared_configured_manifest_derives_only_its_serving_slot(tmp_path: Path) -> None:
    root, manifest, _length, _payload = checkpoint(tmp_path, config=True)
    observed = []

    def derive(path: str, config: bytes, assets: Any, dtypes: Any) -> tuple[str, ...]:
        observed.append((path, config, assets, dtypes))
        return ("adapter",)

    selected = [
        selection(manifest, slot) for slot in ("main.models.source", "generate.models.model")
    ]
    models, entries = _entrypoints(
        interface(),
        package_name="example/convert",
        selections=selected,
        tensorfs_root=root,
        verified=lambda *_args: None,
        derive=derive,
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


@pytest.mark.parametrize("failure", ["length", "closure", "undeclared"])
def test_job_source_retains_manifest_closure_and_declared_slot_guards(
    tmp_path: Path, failure: str
) -> None:
    root, manifest, length, payload = checkpoint(tmp_path)
    chosen = selection(manifest, "main.models.source", manifest_length=length)
    if failure == "length":
        chosen["manifest_length"] = length + 1
    elif failure == "undeclared":
        chosen["slot"] = "missing.models.source"
    else:
        stored = next(root.rglob(payload))
        stored.chmod(stored.stat().st_mode | 0o200)
        stored.write_bytes(b"x" * stored.stat().st_size)
    with pytest.raises(PreparationRefusal):
        _entrypoints(
            interface(),
            package_name="example/convert",
            selections=[chosen],
            tensorfs_root=root,
            verified=lambda *_args: None,
            derive=None,
        )
