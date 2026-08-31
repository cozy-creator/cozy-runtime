#!/usr/bin/env python3
"""Focused regression matrix for PackageDescriptor semantic compatibility."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from cozy_runtime.internal import canonical, descriptor
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.descriptor_compat import Result, classify_bytes


def _base() -> dict[str, Json]:
    request: dict[str, Json] = {
        "fields": [
            {"name": "prompt", "type": "str"},
            {
                "name": "options",
                "type": {"fields": [{"name": "seed", "type": "int"}]},
                "wire": "optional",
            },
            {
                "name": "batch",
                "type": {"list": {"fields": [{"name": "item", "type": "str"}]}},
                "wire": "optional",
            },
            {
                "name": "choice",
                "type": {
                    "union": [
                        {"fields": [{"name": "text", "type": "str"}], "tag": "text"},
                        {"fields": [{"name": "count", "type": "int"}], "tag": "count"},
                    ],
                    "tag_field": "kind",
                },
                "wire": "optional",
            },
        ]
    }
    result: dict[str, Json] = {"fields": [{"name": "message", "type": "str"}]}
    return {
        "format": "cozy.package.descriptor/1",
        "application": "example",
        "entrypoints": [
            {
                "name": "generate",
                "request": request,
                "result": result,
                "models": [
                    {
                        "path": "generate.models.model",
                        "class": "ExampleModel",
                        "stamps": {"family": "example"},
                        "component_use": {"run": ["model"]},
                    }
                ],
            },
            {
                "name": "inspect",
                "request": {"fields": []},
                "result": {"fields": [{"name": "ok", "type": "bool"}]},
            },
        ],
        "jobs": [
            {
                "name": "archive",
                "request": {"fields": [{"name": "source", "type": "str"}]},
                "result": {"fields": [{"name": "stored", "type": "bool"}]},
                "publishes": True,
                "resources": {"gpu_count": 1},
            }
        ],
        "model_productions": [],
    }


def _classify(change: Callable[[dict[str, Json]], None]) -> Result:
    before = _base()
    after = copy.deepcopy(before)
    change(after)
    return classify_bytes(canonical.write(before), canonical.write(after))


def _entrypoint(doc: dict[str, Json], name: str = "generate") -> dict[str, Json]:
    rows = doc["entrypoints"]
    assert isinstance(rows, list)
    return next(row for row in rows if isinstance(row, dict) and row["name"] == name)


def _job(doc: dict[str, Json]) -> dict[str, Json]:
    rows = doc["jobs"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    return rows[0]


def _fields(callable_doc: dict[str, Json], side: str = "request") -> list[Json]:
    schema = callable_doc[side]
    assert isinstance(schema, dict)
    fields = schema["fields"]
    assert isinstance(fields, list)
    return fields


def _expect(
    name: str,
    expected: str,
    change: Callable[[dict[str, Json]], None],
    reason: str | None = None,
) -> None:
    result = _classify(change)
    assert result.classification == expected, (name, result)
    if reason is not None:
        assert any(reason in item for item in result.reasons), (name, result)


def main() -> int:
    cases = 0

    def check(
        name: str,
        expected: str,
        change: Callable[[dict[str, Json]], None],
        reason: str | None = None,
    ) -> None:
        nonlocal cases
        _expect(name, expected, change, reason)
        cases += 1

    check("identical", "patch", lambda doc: None, "public interface identical")
    check(
        "callable ordering",
        "patch",
        lambda doc: doc["entrypoints"].reverse(),
        "public interface identical",
    )
    check(
        "explicit empty models",
        "patch",
        lambda doc: _entrypoint(doc, "inspect").update(models=[]),
    )

    def add_entrypoint(doc: dict[str, Json]) -> None:
        rows = doc["entrypoints"]
        assert isinstance(rows, list)
        rows.append(
            {
                "name": "health",
                "request": {"fields": []},
                "result": {"fields": [{"name": "ok", "type": "bool"}]},
            }
        )

    check("new entrypoint", "minor", add_entrypoint, "added entrypoint health")

    def add_job(doc: dict[str, Json]) -> None:
        rows = doc["jobs"]
        assert isinstance(rows, list)
        rows.append(
            {
                "name": "cleanup",
                "request": {"fields": []},
                "result": {"fields": []},
                "publishes": False,
            }
        )

    check("new job", "minor", add_job, "added job cleanup")

    def add_production(doc: dict[str, Json]) -> None:
        rows = doc["model_productions"]
        assert isinstance(rows, list)
        rows.append(
            {
                "name": "copy-model",
                "source": "source",
                "nodes": [
                    {
                        "name": "copy",
                        "callable": "example/model/copy",
                        "models": {"model": "source"},
                        "outputs": ["model"],
                        "assets": {},
                    }
                ],
                "outputs": [
                    {
                        "name": "model",
                        "source": "copy.model",
                        "lane_key": "bf16",
                        "required_contract": {
                            "tasks": ["generate"],
                            "structure": "h3/full",
                            "encodings": ["plain/1"],
                            "stamps": {},
                        },
                    }
                ],
            }
        )

    check("new model production", "minor", add_production, "added model production copy-model")

    before_with_production = _base()
    add_production(before_with_production)
    changed_production = copy.deepcopy(before_with_production)
    rows = changed_production["model_productions"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    outputs = rows[0]["outputs"]
    assert isinstance(outputs, list) and isinstance(outputs[0], dict)
    outputs[0]["lane_key"] = "fp8"
    result = classify_bytes(
        canonical.write(before_with_production), canonical.write(changed_production)
    )
    assert result.classification == "major"
    assert "model production copy-model changed" in result.reasons
    cases += 1
    check(
        "optional request field",
        "minor",
        lambda doc: _fields(_entrypoint(doc)).append(
            {"name": "style", "type": "str", "wire": "optional"}
        ),
        "added optional field style",
    )
    check(
        "omissible request field",
        "minor",
        lambda doc: _fields(_entrypoint(doc)).append(
            {"name": "quality", "type": "int", "wire": "omissible"}
        ),
        "added omissible field quality",
    )

    def add_nested_optional(doc: dict[str, Json]) -> None:
        options = _fields(_entrypoint(doc))[1]
        assert isinstance(options, dict) and isinstance(options["type"], dict)
        nested = options["type"]["fields"]
        assert isinstance(nested, list)
        nested.append({"name": "temperature", "type": "float", "wire": "optional"})

    check("nested optional request field", "minor", add_nested_optional, "temperature")

    def add_list_item_optional(doc: dict[str, Json]) -> None:
        batch = _fields(_entrypoint(doc))[2]
        assert isinstance(batch, dict) and isinstance(batch["type"], dict)
        item = batch["type"]["list"]
        assert isinstance(item, dict) and isinstance(item["fields"], list)
        item["fields"].append({"name": "label", "type": "str", "wire": "optional"})

    check("list item optional request field", "minor", add_list_item_optional, "label")

    def add_union_member_optional(doc: dict[str, Json]) -> None:
        choice = _fields(_entrypoint(doc))[3]
        assert isinstance(choice, dict) and isinstance(choice["type"], dict)
        branches = choice["type"]["union"]
        assert isinstance(branches, list) and isinstance(branches[0], dict)
        fields = branches[0]["fields"]
        assert isinstance(fields, list)
        fields.append({"name": "language", "type": "str", "wire": "omissible"})

    check("union member optional request field", "minor", add_union_member_optional, "language")

    check(
        "application changed",
        "patch",
        lambda doc: doc.update(application="renamed"),
        "public interface identical",
    )
    check(
        "removed callable",
        "major",
        lambda doc: doc.update(entrypoints=[_entrypoint(doc)]),
        "removed entrypoint inspect",
    )

    def change_kind(doc: dict[str, Json]) -> None:
        rows = doc["entrypoints"]
        jobs = doc["jobs"]
        assert isinstance(rows, list) and isinstance(jobs, list)
        moved = _entrypoint(doc, "inspect")
        rows.remove(moved)
        moved["publishes"] = False
        jobs.append(moved)

    check("callable kind", "major", change_kind, "changed kind")
    check(
        "result changed",
        "major",
        lambda doc: _fields(_entrypoint(doc), "result").append(
            {"name": "detail", "type": "str", "wire": "optional"}
        ),
        "result schema",
    )
    check(
        "model metadata changed",
        "major",
        lambda doc: _entrypoint(doc)["models"][0]["stamps"].update(family="other"),
        "model metadata",
    )
    check(
        "job publishes changed",
        "major",
        lambda doc: _job(doc).update(publishes=False),
        "job publication metadata",
    )
    check(
        "job resources changed",
        "major",
        lambda doc: _job(doc)["resources"].update(gpu_count=2),
        "job resource metadata",
    )
    check(
        "required request field added",
        "major",
        lambda doc: _fields(_entrypoint(doc)).append({"name": "style", "type": "str"}),
        "added required field style",
    )
    check(
        "request field removed",
        "major",
        lambda doc: _fields(_entrypoint(doc)).pop(),
        "removed or reordered",
    )
    check(
        "request field reordered",
        "major",
        lambda doc: _fields(_entrypoint(doc)).reverse(),
        "removed or reordered",
    )
    check(
        "request type changed",
        "major",
        lambda doc: _fields(_entrypoint(doc))[0].update(type="int"),
        "prompt schema changed",
    )
    check(
        "request constraint changed",
        "major",
        lambda doc: _fields(_entrypoint(doc))[0].update(constraints={"max_length": 8}),
        "changed field prompt",
    )
    check(
        "request wire changed",
        "major",
        lambda doc: _fields(_entrypoint(doc))[1].update(wire="omissible"),
        "changed field options",
    )

    def add_union_branch(doc: dict[str, Json]) -> None:
        field = _fields(_entrypoint(doc))[1]
        assert isinstance(field, dict)
        field["type"] = {"union": [field["type"], "null"]}

    check("other request schema change", "major", add_union_branch, "options schema changed")

    def change_union_tag(doc: dict[str, Json]) -> None:
        choice = _fields(_entrypoint(doc))[3]
        assert isinstance(choice, dict) and isinstance(choice["type"], dict)
        choice["type"]["tag_field"] = "type"

    check("union metadata changed", "major", change_union_tag, "choice schema changed")

    def mixed(doc: dict[str, Json]) -> None:
        add_entrypoint(doc)
        _fields(_entrypoint(doc), "result").append({"name": "detail", "type": "str"})

    check("major wins over minor", "major", mixed, "result schema")

    before = _base()
    many = copy.deepcopy(before)
    rows = many["entrypoints"]
    assert isinstance(rows, list)
    template = copy.deepcopy(_entrypoint(many, "inspect"))
    for index in range(20):
        row = copy.deepcopy(template)
        row["name"] = f"extra-{index:02d}"
        rows.append(row)
    result = classify_bytes(canonical.write(many), canonical.write(before))
    assert result.classification == "major" and len(result.reasons) == 16
    assert result.reasons[-1].endswith("more changes")
    cases += 1

    huge = _base()
    rows = huge["entrypoints"]
    assert isinstance(rows, list)
    added = copy.deepcopy(_entrypoint(huge, "inspect"))
    added["name"] = "n" * 1024
    rows.append(added)
    result = classify_bytes(canonical.write(before), canonical.write(huge))
    assert result.classification == "minor" and len(result.reasons[0]) == 256
    cases += 1

    ambiguous = _base()
    jobs = ambiguous["jobs"]
    assert isinstance(jobs, list)
    jobs.append(
        {
            "name": "generate",
            "request": {"fields": []},
            "result": {"fields": []},
            "publishes": False,
        }
    )
    try:
        classify_bytes(canonical.write(before), canonical.write(ambiguous))
    except descriptor.StalePackageDescriptor as exc:
        assert "both an entrypoint and a job" in str(exc)
    else:
        raise AssertionError("cross-kind duplicate callable name was accepted")
    cases += 1

    with tempfile.TemporaryDirectory(prefix="cozy-descriptor-compat-") as raw:
        root = Path(raw)
        previous = root / "previous.json"
        candidate = root / "candidate.json"
        previous.write_bytes(canonical.write(_base()))
        changed = _base()
        add_entrypoint(changed)
        candidate.write_bytes(canonical.write(changed))
        command = [
            sys.executable,
            "-m",
            "cozy_runtime.cli.descriptor_compat",
            str(previous),
            str(candidate),
        ]
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        assert process.returncode == 0 and process.stderr == "", process
        assert json.loads(process.stdout) == {
            "classification": "minor",
            "reasons": ["added entrypoint health"],
        }
        assert process.stdout == (
            '{"classification":"minor","reasons":["added entrypoint health"]}\n'
        )
        candidate.write_text("{}")
        refused = subprocess.run(command, check=False, capture_output=True, text=True)
        assert refused.returncode == 2 and refused.stdout == ""
        assert "descriptor compatibility refused" in refused.stderr
        cases += 2

    print(f"descriptor compatibility: {cases}/{cases} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
