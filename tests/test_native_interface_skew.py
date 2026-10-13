"""Native operations are admitted per operation between independently versioned peers."""

from __future__ import annotations

import msgspec

from cozy_runtime.author import ObjectRef
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal import effect_interfaces, native_interfaces, schema


class _Older(msgspec.Struct, frozen=True):
    destination: str
    release: str
    lanes: dict[str, CheckpointRef]


class _NewerOptional(_Older, frozen=True):
    mirror: str = ""


class _NewerRequired(_Older, frozen=True):
    mirror: str


class _WithoutRelease(msgspec.Struct, frozen=True):
    destination: str
    lanes: dict[str, CheckpointRef]


class _Checkpoint(msgspec.Struct, frozen=True):
    destination: str
    checkpoint: str
    manifest: ObjectRef
    publication: str
    observation: str
    listing: str


def _stating(
    version: str,
    request: type[msgspec.Struct] | None = None,
    result: type[msgspec.Struct] | None = None,
) -> dict[str, object]:
    """A hello stating `publish_release` / `upload_checkpoint` rendered from these types."""
    stated = native_interfaces.stated()
    operations = stated[effect_interfaces.MODULE]["operations"]
    if request is not None:
        operations["publish_release"]["request"] = schema.render(request)
    if result is not None:
        operations["upload_checkpoint"]["result"] = schema.render(result)
    return {"native_interfaces": stated, "runtime_version": version}


def _unavailable(hello: dict[str, object], export: str) -> str:
    row = next(row for row in native_interfaces.rows(hello) if row["export"] == export)
    return str(row.get("unavailable", ""))


def test_additive_optional_fields_are_admitted_in_both_directions() -> None:
    assert not any("unavailable" in row for row in native_interfaces.rows({}))
    assert not _unavailable(_stating("0.18.1", _Older), "publish_release")
    assert not _unavailable(_stating("99.0.0", _NewerOptional), "publish_release")


def test_required_fields_are_enforced_naming_the_side_to_update() -> None:
    newer = _unavailable(_stating("99.0.0", _NewerRequired), "publish_release")
    assert "request mirror is not read" in newer and "cozy rental update" in newer
    older = _unavailable(_stating("0.18.1", _WithoutRelease), "publish_release")
    assert "request release is not sent" in older
    assert "publish the package against cozy-runtime>=" in older
    # The refusal is confined to that operation.
    rows = native_interfaces.rows(_stating("0.18.1", _WithoutRelease))
    assert [row["export"] for row in rows if "unavailable" in row] == ["publish_release"]


def test_results_need_what_the_executor_requires() -> None:
    refused = _unavailable(_stating("0.18.1", result=_Checkpoint), "upload_checkpoint")
    assert "result listing is not produced" in refused and "result destination" not in refused


def test_an_operation_this_worker_lacks_is_refused_alone() -> None:
    hello = _stating("99.0.0")
    operations = hello["native_interfaces"][effect_interfaces.MODULE]["operations"]  # type: ignore[index]
    operations["publish_dataset"] = operations["publish_release"]
    rows = native_interfaces.rows(hello)
    (refused,) = [row for row in rows if "unavailable" in row]
    assert refused["export"] == "publish_dataset"
    assert "cozy rental update" in str(refused["unavailable"])
