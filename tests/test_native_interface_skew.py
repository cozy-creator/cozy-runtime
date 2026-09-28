"""Native operations are admitted per operation between independently versioned peers."""

from __future__ import annotations

import msgspec

from cozy_runtime.author import Tree
from cozy_runtime.internal import native_interfaces, schema, source_interfaces


class _Older(msgspec.Struct, frozen=True):
    repository: str
    revision: str


class _NewerOptional(msgspec.Struct, frozen=True):
    repository: str
    revision: str
    mirror: str = ""


class _NewerRequired(msgspec.Struct, frozen=True):
    repository: str
    revision: str
    mirror: str


class _WithoutRevision(msgspec.Struct, frozen=True):
    repository: str


class _Files(msgspec.Struct, frozen=True):
    files: Tree
    listing: str


def _stating(
    version: str,
    request: type[msgspec.Struct] | None = None,
    result: type[msgspec.Struct] | None = None,
) -> dict[str, object]:
    """A hello stating `download_huggingface` / `source_files` rendered from these types."""
    stated = native_interfaces.stated()
    operations = stated[source_interfaces.MODULE]["operations"]
    if request is not None:
        operations["download_huggingface"]["request"] = schema.render(request)
    if result is not None:
        operations["source_files"]["result"] = schema.render(result)
    return {"native_interfaces": stated, "runtime_version": version}


def _unavailable(hello: dict[str, object], export: str) -> str:
    row = next(row for row in native_interfaces.rows(hello) if row["export"] == export)
    return str(row.get("unavailable", ""))


def test_additive_optional_fields_are_admitted_in_both_directions() -> None:
    assert not any("unavailable" in row for row in native_interfaces.rows({}))
    assert not _unavailable(_stating("0.18.1", _Older), "download_huggingface")
    assert not _unavailable(_stating("99.0.0", _NewerOptional), "download_huggingface")


def test_required_fields_are_enforced_naming_the_side_to_update() -> None:
    newer = _unavailable(_stating("99.0.0", _NewerRequired), "download_huggingface")
    assert "request mirror is not read" in newer and "cozy rental update" in newer
    older = _unavailable(_stating("0.18.1", _WithoutRevision), "download_huggingface")
    assert "request revision is not sent" in older
    assert "publish the package against cozy-runtime>=" in older
    # The refusal is confined to that operation.
    rows = native_interfaces.rows(_stating("0.18.1", _WithoutRevision))
    assert [row["export"] for row in rows if "unavailable" in row] == ["download_huggingface"]


def test_results_need_what_the_executor_requires() -> None:
    refused = _unavailable(_stating("0.18.1", result=_Files), "source_files")
    assert "result listing is not produced" in refused and "result files" not in refused


def test_an_operation_this_worker_lacks_is_refused_alone() -> None:
    hello = _stating("99.0.0")
    operations = hello["native_interfaces"][source_interfaces.MODULE]["operations"]  # type: ignore[index]
    operations["download_dataset"] = operations["download_huggingface"]
    rows = native_interfaces.rows(hello)
    (refused,) = [row for row in rows if "unavailable" in row]
    assert refused["export"] == "download_dataset"
    assert "cozy rental update" in str(refused["unavailable"])
