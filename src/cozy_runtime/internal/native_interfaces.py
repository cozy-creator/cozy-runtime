"""Runtime's native operations, admitted per operation between independently versioned peers.

The executor states each native operation it expects as a request and result schema. The
worker serves an operation when it can read that request and the executor can read its
result: additive optional fields are tolerated both ways and required fields are enforced.
A mismatch refuses only that operation. No whole-interface digest is compared.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgspec
from packaging.version import InvalidVersion, Version

from cozy_runtime import __version__ as RUNTIME_VERSION
from cozy_runtime import canonical_json
from cozy_runtime.author._calls import _CallType
from cozy_runtime.internal import effect_interfaces, source_interfaces

_SERVICES = {
    source_interfaces.MODULE: (source_interfaces, "native_source"),
    effect_interfaces.MODULE: (effect_interfaces, "native_effect"),
}

#: The documents each released executor that does not state its own expects, keyed by the
#: first release carrying them (0.18.30-0.18.49). Executors before 0.18.39 also require each
#: row to echo their document digest and refuse unknown result fields; later ones refuse a
#: row naming an operation they lack. Frozen: every later executor states its documents.
UNSTATED: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).with_name("native-interfaces-unstated.json")).read_bytes()
)
UPDATE_RENTAL = "run `cozy rental update <rental>`"


def released(version: str) -> dict[str, Any]:
    try:
        running = Version(version)
    except InvalidVersion:
        running = Version("0")
    carried = [release for release in UNSTATED if Version(release) <= running]
    # An unknown executor gets the fewest operations: a row it lacks fails its whole broker.
    return UNSTATED[max(carried, key=Version) if carried else min(UNSTATED, key=Version)]


def stated() -> dict[str, Any]:
    """This Runtime's native interface documents, as its executor states them at hello."""
    return {module: canonical_json.decode(s.DOCUMENT) for module, (s, _) in _SERVICES.items()}


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


def rows(hello: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Broker rows for every native operation the executor expects.

    Each expected operation is listed so that the executor binds it. One this worker cannot
    serve carries `unavailable`, and only a call to it is refused.
    """
    expected = hello.get("native_interfaces")
    strict = not isinstance(expected, dict)
    if strict:
        expected = released(str(hello.get("runtime_version", "")))
    assert isinstance(expected, dict)
    result: list[dict[str, Any]] = []
    for module, document in expected.items():
        operations = document.get("operations") if isinstance(document, dict) else None
        if module not in _SERVICES or not isinstance(operations, dict):
            continue
        service, flag = _SERVICES[module]
        offered = canonical_json.decode(service.DOCUMENT)["operations"]
        try:
            # Echoed for executors that compare it; this Runtime compares nothing.
            digest = canonical_json.digest(document)
        except ValueError:
            digest = ""
        for name, wanted in operations.items():
            row: dict[str, Any] = {
                "interface_digest": digest,
                "module": module,
                "export": name,
                flag: True,
            }
            try:
                problem = _refusal(module, name, wanted, offered.get(name), strict=strict)
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


def _refusal(module: str, name: str, wanted: Any, offered: Any, *, strict: bool) -> str:
    if offered is None:
        return (
            f'native operation "{name}" is not offered by this machine\'s Runtime '
            f"{RUNTIME_VERSION} (executor needs it)"
        )
    # The worker reads the request and ignores unknown options only when they are optional:
    # dropping a required one would change what was asked. The executor ignores unknown
    # result fields unless it is a strict reader.
    missing, unread = _gaps(wanted["request"], offered["request"], ignore="optional")
    absent, extra = _gaps(offered["result"], wanted["result"], ignore="" if strict else "all")
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


def _gaps(written: Any, reader: Any, *, ignore: str, at: str = "") -> tuple[list[str], list[str]]:
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


def _struct(node: Any) -> bool:
    return isinstance(node, dict) and isinstance(node.get("fields"), list)


def _list(node: Any) -> bool:
    return isinstance(node, dict) and node.keys() == {"list"}


def _remedy(hello: Mapping[str, Any]) -> str:
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
