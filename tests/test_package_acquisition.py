"""Package-descriptor admission at the worker's real acquisition boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.internal import base_observation, canonical, descriptor
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal
from cozy_runtime.internal.worker.downloads import DownloadPlan
from cozy_runtime.internal.worker.session import Placement
from cozy_runtime.protocol import documents


def _schema() -> dict[str, Json]:
    return {"fields": []}


def _entrypoint(name: str) -> dict[str, Json]:
    return {"name": name, "request": _schema(), "result": _schema()}


def _job(name: str) -> dict[str, Json]:
    return {**_entrypoint(name), "publishes": False}


def _descriptor(entrypoints: list[Json]) -> dict[str, Json]:
    return {
        "application": "job_only:app",
        "entrypoints": entrypoints,
        "format": descriptor.SCHEMA,
        "jobs": [_job("produce")],
    }


def _bindings(tmp_path: Path, body: dict[str, Json]) -> tuple[object, int]:
    raw = canonical.write(body)
    digest = canonical.digest(body)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / digest.removeprefix("sha256:")).write_bytes(raw)
    for name in ("environment", "selection", "store"):
        (tmp_path / name).mkdir()
    acquirer = Acquirer(
        DownloadPlan(),
        cache_root=cache,
        environment_root=tmp_path / "environment",
        selection_root=tmp_path / "selection",
        artifact_store=tmp_path / "store",
        python=Path("/usr/bin/python3"),
        base=base_observation.current(),
    )
    placement = Placement(
        placement_id="package-job-only",
        placement_set_digest=b"\x11" * 32,
        package={"release_digest": "sha256:" + "22" * 32},
        development={},
        environment_digest=b"\x33" * 32,
        package_descriptor=(digest, len(raw)),
        bindings_digest=documents.digest_of(canonical.write({"entrypoints": [], "models": []})),
        models=(),
        entrypoints=(),
        environment={},
        document={},
    )
    return acquirer._bindings(placement)


def test_job_only_package_set_has_no_serving_bindings(tmp_path: Path) -> None:
    bindings, documents = _bindings(tmp_path, _descriptor([]))

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
def test_package_set_refuses_invalid_descriptor_entrypoints(
    tmp_path: Path, entrypoints: list[Json]
) -> None:
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(tmp_path, _descriptor(entrypoints))

    assert refused.value.code == "package_descriptor_invalid"
