"""One registration kind (xs-040): every keyword on any function, `Weights[M]`, one list.

The production fixture mirrors the 26 registrations that were `app.job`; both readers must
agree on it (`test_static_describe`), and this file pins what the interface says about it.
"""

from __future__ import annotations

import shutil
import typing
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import (
    App,
    CapabilityError,
    ConformanceError,
    Context,
    ImageAsset,
    ModelArtifact,
    Outputs,
    RegistrationError,
    Weights,
    WeightsOutput,
    describe,
    invocable,
)
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.package_interface import StalePackageInterface
from cozy_runtime.internal.static_interface import StaticRefusal

PRODUCTION = Path(__file__).resolve().parent / "testdata" / "static_describe" / "production"
GPU = {"bf16-pruned", "fp8-pruned", "mxfp8-pruned", "retable", "compute-adaln-tables", "turbo-lora"}
ORCHESTRATORS = {
    "fl2va",
    "fl2va_turbo",
    "ref2va",
    "ref2va_turbo",
    "long_form",
    "long_form_cuts",
    "assemble_video",
}


def _entries(doc: dict[str, typing.Any]) -> dict[str, dict[str, typing.Any]]:
    return {entry["name"]: entry for entry in doc["entrypoints"]}


def test_production_registrations_are_one_entrypoints_list() -> None:
    doc = static_interface.build(PRODUCTION)
    assert set(doc) == {"format", "application", "entrypoints"}
    entries = _entries(doc)
    assert len(entries) == 26
    assert {name for name, entry in entries.items() if "accelerator" not in entry} == GPU
    assert all(entry.get("accelerator", False) is False for entry in entries.values())
    for name, entry in entries.items():
        assert "publishes" not in entry and entry["invocable"]["memoize"] is False
        if name in ORCHESTRATORS:
            assert "models" not in entry and "weights_outputs" not in entry
            assert "save" in entry["invocable"]["capabilities"]
        else:
            assert entry["weights_outputs"] and "weights" in entry["invocable"]["capabilities"]
            assert entry["models"] and all(slot["weights"] is True for slot in entry["models"])
            # Raw arguments are model slots of the serving plane, not request fields.
            fields = {field["name"] for field in entry["request"]["fields"]}
            assert not fields & {slot["path"].rpartition(".")[2] for slot in entry["models"]}
    (sdxl,) = entries["sdxl-fp8"]["models"]
    assert sdxl["class"] == "SdxlModel" and sdxl["encoded_leaves"] == "accept"
    assert sdxl["component_use"]["denoise"] == ["unet"]
    dit, vae = entries["seedvr2_normalize"]["models"]
    assert dit["class"] == vae["class"] == "Model" and dit["component_use"] == {}
    ids = {package_interface.descriptor_id(doc, name) for name in entries}
    assert len(ids) == 26
    assert package_interface.parse(package_interface.canonical_bytes(doc)).function("fp8")


class _Lane(msgspec.Struct):
    prompt: str


class _Image(msgspec.Struct):
    image: ImageAsset


async def _derive(ctx: Context, *, source: Weights[typing.Any]) -> ModelArtifact:
    raise NotImplementedError


def _publish(payload: _Lane, out: Outputs) -> _Image:
    raise NotImplementedError


@pytest.mark.parametrize(
    "register",
    [
        lambda app: app.entrypoint(
            invocable(memoize=True)(_derive), weights=(WeightsOutput("model", 1 << 20),)
        ),
        lambda app: app.entrypoint(_publish, memoize=True, emits_media=True),
        lambda app: app.entrypoint(_publish, memoize=True, publishes=True),
    ],
    ids=["weights", "emits_media", "publishes"],
)
def test_describe_refuses_a_memoized_writer(register: typing.Any) -> None:
    app = App()
    register(app)
    with pytest.raises(ConformanceError) as caught:
        describe(app)
    assert caught.value.code == "memoized_writer"


def test_the_static_reader_refuses_a_memoized_writer(tmp_path: Path) -> None:
    project = tmp_path / "production"
    shutil.copytree(PRODUCTION, project, ignore=shutil.ignore_patterns("__pycache__"))
    init = project / "production_package" / "__init__.py"
    init.write_text(
        init.read_text().replace(
            "app.entrypoint(h3.fl2va, emits_media=True,",
            "app.entrypoint(h3.fl2va, memoize=True, emits_media=True,",
        )
    )
    with pytest.raises(StaticRefusal) as caught:
        static_interface.build(project)
    assert caught.value.code == "memoized_writer"


def test_accelerator_false_is_the_only_value(tmp_path: Path) -> None:
    with pytest.raises(RegistrationError):
        App().entrypoint(_publish, accelerator=True)  # type: ignore[call-overload]
    project = tmp_path / "production"
    shutil.copytree(PRODUCTION, project, ignore=shutil.ignore_patterns("__pycache__"))
    init = project / "production_package" / "__init__.py"
    init.write_text(
        init.read_text().replace('name="bf16-pruned",', 'name="bf16-pruned", accelerator=True,')
    )
    with pytest.raises(StaticRefusal, match="accelerator=False is the only value"):
        static_interface.build(project)


def test_a_document_with_a_memoized_writer_is_malformed() -> None:
    doc = static_interface.build(PRODUCTION)
    entry = _entries(doc)["fp8"]
    entry["invocable"]["memoize"] = True
    with pytest.raises(StalePackageInterface, match="not memoized"):
        package_interface.parse(package_interface.canonical_bytes(doc))


def test_unbound_weights_refuse_reads_and_canonical_identity() -> None:
    with pytest.raises(ConformanceError) as invalid:
        Weights("sha256:" + "A" * 64)
    assert invalid.value.code == "weights_manifest_invalid"
    weights: Weights[typing.Any] = Weights("sha256:" + "a" * 64)
    with pytest.raises(CapabilityError) as unbound:
        weights.components()
    assert unbound.value.code == "weights_source_ungranted"
