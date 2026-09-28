"""Runtime's native operations, admitted per operation between independently versioned peers.

The executor states each native operation it expects as a request and result schema. The
worker serves an operation when it can read that request and the executor can read its
result: additive optional fields are tolerated both ways and required fields are enforced.
A mismatch refuses only that operation. No whole-interface digest is compared.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
from packaging.version import InvalidVersion, Version

from cozy_runtime import __version__ as RUNTIME_VERSION
from cozy_runtime import canonical_json
from cozy_runtime.author._calls import _CallType
from cozy_runtime.internal import effect_interfaces, source_interfaces
from cozy_runtime.internal.canonical import Json

if TYPE_CHECKING:
    Schema = Json
else:
    Schema = object  # msgspec cannot analyze the recursive schema grammar; it stays JSON


class Operation(msgspec.Struct, frozen=True):
    request: Schema
    result: Schema


class Statement(msgspec.Struct, frozen=True):
    """One native interface document; members a newer writer adds are ignored."""

    operations: dict[str, Operation]


_SERVICES = {
    source_interfaces.MODULE: (source_interfaces, "native_source"),
    effect_interfaces.MODULE: (effect_interfaces, "native_effect"),
}

UPDATE_RENTAL = "run `cozy rental update <rental>`"


def stated() -> dict[str, Json]:
    """This Runtime's native interface documents, as its executor states them at hello."""
    return {module: canonical_json.decode(s.DOCUMENT) for module, (s, _) in _SERVICES.items()}


def _offered(module: str) -> Statement:
    return canonical_json.decode_as(_SERVICES[module][0].DOCUMENT, Statement)


def bindings() -> dict[tuple[str, str], _CallType]:
    return {
        (module, name): binding
        for module, (service, _) in _SERVICES.items()
        for name, binding in service.bindings().items()
    }


def unserved(module: str, export: str) -> _CallType:
    """This Runtime's own operation, refused because the worker did not offer it."""
    detail = (
        f'native operation "{export}" is not offered by this machine\'s Runtime '
        f"(executor needs it); {UPDATE_RENTAL}"
    )
    return dataclasses.replace(bindings()[(module, export)], unavailable=detail)


def rows(hello: Mapping[str, object]) -> list[dict[str, object]]:
    """Broker rows for every native operation the executor expects.

    Each expected operation is listed so that the executor binds it. One this worker cannot
    serve carries `unavailable`, and only a call to it is refused.
    """
    expected = hello.get("native_interfaces")
    if not isinstance(expected, dict):
        return []
    result: list[dict[str, object]] = []
    for module, document in expected.items():
        if module not in _SERVICES:
            continue
        try:
            statement = msgspec.convert(document, Statement, strict=True)
        except msgspec.ValidationError:
            continue
        service, flag = _SERVICES[module]
        offered = _offered(module).operations
        for name, wanted in statement.operations.items():
            row: dict[str, object] = {"module": module, "export": name, flag: True}
            try:
                problem = _refusal(module, name, wanted, offered.get(name))
            except (KeyError, TypeError, AttributeError):
                problem = f"{module}.{name} is stated in a form this Runtime cannot read"
            if problem:
                row["unavailable"] = f"{problem}; {_remedy(hello)}"[:1024]
            else:
                request = service.TYPES[name][0]
                row["request_fields"] = sorted(
                    f.encode_name for f in msgspec.structs.fields(request)
                )
            result.append(row)
    return result


def _refusal(module: str, name: str, wanted: Operation, offered: Operation | None) -> str:
    if offered is None:
        return (
            f'native operation "{name}" is not offered by this machine\'s Runtime '
            f"{RUNTIME_VERSION} (executor needs it)"
        )
    # The worker reads the request and ignores unknown options only when they are optional:
    # dropping a required one would change what was asked. The executor ignores unknown
    # result fields.
    missing, unread = _gaps(wanted.request, offered.request, ignore="optional")
    absent, extra = _gaps(offered.result, wanted.result, ignore="all")
    problems = [
        *(f"request {field} is not sent" for field in missing),
        *(f"request {field} is not read" for field in unread),
        *(f"result {field} is not produced" for field in absent),
        *(f"result {field} is not read" for field in extra),
    ]
    if not problems:
        return ""
    return (
        f'native operation "{name}" on this machine\'s Runtime {RUNTIME_VERSION} cannot '
        f"serve this executor: {'; '.join(problems)}"
    )


def _gaps(written: Json, reader: Json, *, ignore: str, at: str = "") -> tuple[list[str], list[str]]:
    """(fields the reader requires that the writer lacks, written fields the reader cannot take).

    ``ignore`` names which unknown written fields the reader drops: "optional", "all" or "".
    """
    if _struct(written) and _struct(reader):
        if (written.get("tag"), written.get("tag_field")) != (
            reader.get("tag"),
            reader.get("tag_field"),
        ):
            return [], [at.rstrip(".") or "value"]
        wrote = {f["name"]: f for f in written["fields"]}
        reads = {f["name"]: f for f in reader["fields"]}
        missing = [at + n for n, f in reads.items() if n not in wrote and "wire" not in f]
        unread = [
            at + n
            for n, f in wrote.items()
            if n not in reads and not (ignore == "all" or (ignore == "optional" and "wire" in f))
        ]
        for n in wrote.keys() & reads.keys():
            inner = _gaps(wrote[n]["type"], reads[n]["type"], ignore=ignore, at=f"{at}{n}.")
            missing += inner[0]
            unread += inner[1]
        return missing, unread
    if _list(written) and _list(reader):
        return _gaps(written["list"], reader["list"], ignore=ignore, at=at)
    return ([], []) if written == reader else ([], [at.rstrip(".") or "value"])


def _struct(node: Json) -> bool:
    return isinstance(node, dict) and isinstance(node.get("fields"), list)


def _list(node: Json) -> bool:
    return isinstance(node, dict) and node.keys() == {"list"}


def _remedy(hello: Mapping[str, object]) -> str:
    executor = str(hello.get("runtime_version", ""))
    try:
        newer = Version(executor) > Version(RUNTIME_VERSION)
    except InvalidVersion:
        newer = False
    if newer:
        return UPDATE_RENTAL
    return (
        f"publish the package against cozy-runtime>={RUNTIME_VERSION} "
        f"(its lock pins cozy-runtime {executor or 'unknown'})"
    )
