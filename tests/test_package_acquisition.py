"""PackageInterface admission at the worker's real acquisition boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from cozy_runtime.internal import base_observation, canonical, package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Placement, read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def _schema() -> dict[str, Json]:
    return {"fields": []}


def _entrypoint(name: str) -> dict[str, Json]:
    return {"name": name, "request": _schema(), "result": _schema()}


def _job(name: str) -> dict[str, Json]:
    return {**_entrypoint(name), "publishes": False}


def _interface(entrypoints: list[Json]) -> dict[str, Json]:
    return {
        "application": "job_only:app",
        "entrypoints": entrypoints,
        "format": package_interface.SCHEMA,
        "jobs": [_job("produce")],
    }


def _bindings(
    tmp_path: Path,
    body: dict[str, Json],
    selected: tuple[str, ...] = (),
    *,
    development: bool = False,
    models: tuple[dict[str, Json], ...] = (),
    prepared: Placement | None = None,
) -> tuple[Mapping[str, DeclaredBinding], int]:
    raw = canonical.write(body)
    digest = canonical.digest(body)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / digest.removeprefix("sha256:")).write_bytes(raw)
    for name in ("environment", "selection", "store"):
        (tmp_path / name).mkdir()
    acquirer = Acquirer(
        cache_root=cache,
        install_root=tmp_path / "environment",
        selection_root=tmp_path / "selection",
        tensorfs_root=tmp_path / "store",
        python=Path("/usr/bin/python3"),
        base=base_observation.current(),
    )
    placement = Placement(
        placement_id="package-job-only",
        placement_set_digest=b"\x11" * 32,
        package={}
        if development
        else {"project_wheel": {"ref": {"digest": "sha256:" + "22" * 32}}},
        development={"source_digest": "sha256:" + "22" * 32} if development else {},
        environment_digest=b"\x33" * 32,
        package_interface=(digest, len(raw)),
        bindings_digest=documents.digest_of(canonical.write({"entrypoints": [], "models": []})),
        models=models,
        entrypoints=tuple(
            {
                "name": name,
                "entrypoint_binding_digest": canonical.digest({"entrypoint": name}),
                "slots": [],
            }
            for name in selected
        ),
        environment={},
        document={},
    )
    return acquirer._bindings(prepared or placement)


@pytest.mark.parametrize("development", [False, True])
def test_job_only_package_set_has_no_serving_bindings(tmp_path: Path, development: bool) -> None:
    bindings, documents = _bindings(tmp_path, _interface([]), development=development)

    assert bindings == {}
    assert documents == 1


@pytest.mark.parametrize(
    "entrypoints",
    [
        [_entrypoint("serve"), _entrypoint("serve")],
        [{"request": _schema(), "result": _schema()}],
    ],
    ids=["duplicate", "malformed"],
)
def test_package_set_refuses_invalid_interface_entrypoints(
    tmp_path: Path, entrypoints: list[Json]
) -> None:
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(tmp_path, _interface(entrypoints))

    assert refused.value.code == "package_interface_invalid"


@pytest.mark.parametrize("selected", [("first",), ("second",), ("first", "second")])
def test_package_set_admits_selected_interface_entrypoints(
    tmp_path: Path, selected: tuple[str, ...]
) -> None:
    bindings, count = _bindings(
        tmp_path, _interface([_entrypoint("first"), _entrypoint("second")]), selected
    )
    assert {binding.entrypoint for binding in bindings.values()} == set(selected)
    assert count == 1


@pytest.mark.parametrize(
    ("selected", "code"),
    [((), "binding_entrypoint_set_mismatch"), (("absent",), "binding_package_interface_invalid")],
)
@pytest.mark.parametrize("development", [False, True])
def test_package_set_refuses_empty_or_unknown_selection(
    tmp_path: Path, selected: tuple[str, ...], code: str, development: bool
) -> None:
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(
            tmp_path,
            _interface([_entrypoint("first"), _entrypoint("second")]),
            selected,
            development=development,
        )
    assert refused.value.code == code


def test_h3_prepared_code_only_set_has_no_executable_bindings(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "testdata" / "h3-code-only"
    raw = (fixture / "placement-set.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "384edb7272a8b760fa30b5c500b282c6726aede1613be17c9c52f5f3c6257991"
    )
    desired = pb.DesiredPlacementSet(
        placement_set_digest=documents.digest_of(raw), placement_set_canonical_bytes=raw
    )
    (entry,) = read_placement_set(desired)
    assert entry["development"]["local_revision_digest"] == (
        "sha256:2b96d8193a4933d9a62a4bdc51d9fdd2f8d80695025142e103c284f578a095aa"
    )
    assert entry["entrypoints"] == entry["models"] == []
    interface = (fixture / "package-interface.json").read_bytes()
    assert entry["package_interface"] == {
        "digest": "sha256:" + hashlib.sha256(interface).hexdigest(),
        "length": len(interface),
    }
    body = json.loads(interface)
    assert all(row["models"] for row in body["entrypoints"])
    placement = Placement(
        placement_id=entry["placement_id"],
        placement_set_digest=desired.placement_set_digest,
        package={},
        development=entry["development"],
        environment_digest=documents.raw(entry["environment_digest"]),
        package_interface=(
            entry["package_interface"]["digest"],
            entry["package_interface"]["length"],
        ),
        bindings_digest=documents.raw(entry["bindings_digest"]),
        models=tuple(entry["models"]),
        entrypoints=tuple(entry["entrypoints"]),
        environment=entry["environment"],
        document=entry,
    )
    bindings, count = _bindings(tmp_path, body, prepared=placement)
    assert bindings == {} and count == 1


@pytest.mark.parametrize("development", [False, True])
def test_empty_selection_cannot_drop_declared_weightless_binding(
    tmp_path: Path, development: bool
) -> None:
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(tmp_path, _interface([_entrypoint("serve")]), development=development)
    assert refused.value.code == "binding_entrypoint_set_mismatch"


@pytest.mark.parametrize("development", [False, True])
def test_empty_selection_cannot_hide_selected_models(tmp_path: Path, development: bool) -> None:
    modeled = {
        **_entrypoint("serve"),
        "models": [{"class": "Model", "path": "serve.models.model", "component_use": {}}],
    }
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(
            tmp_path, _interface([modeled]), development=development, models=({"id": "selected"},)
        )
    assert refused.value.code == "binding_entrypoint_set_mismatch"


@pytest.mark.parametrize("development", [False, True])
def test_modeled_code_only_selection_has_no_execution_authority(
    tmp_path: Path, development: bool
) -> None:
    modeled = {
        **_entrypoint("serve"),
        "models": [{"class": "Model", "path": "serve.models.model", "component_use": {}}],
    }
    bindings, count = _bindings(tmp_path, _interface([modeled]), development=development)
    assert bindings == {} and count == 1


def test_code_only_exception_cannot_hide_missing_model_slots(tmp_path: Path) -> None:
    modeled = {
        **_entrypoint("serve"),
        "models": [{"class": "Model", "path": "serve.models.model", "component_use": {}}],
    }
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(tmp_path, _interface([modeled]), ("serve",), development=True)
    assert refused.value.code == "binding_slot_set_mismatch"
