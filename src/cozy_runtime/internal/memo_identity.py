"""Pure operation identities from source ASTs and installed metadata bytes.

No imports, resource loading or discovery hooks execute on this path. Both static
inspection and the already-imported execution path use these exact normalizations.
"""

from __future__ import annotations

import ast
import copy
import email.parser
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import msgspec

MAX_SOURCE_BYTES = 1 << 20
MAX_METADATA_BYTES = 8 << 20


class _Archive(msgspec.Struct, frozen=True):
    hash: str = ""
    hashes: dict[str, str] = {}


class _Vcs(msgspec.Struct, frozen=True):
    commit_id: str = ""


class _DirectUrl(msgspec.Struct, frozen=True):
    """PEP 610 `direct_url.json`: exactly one of the three origins is present."""

    archive_info: _Archive | None = None
    vcs_info: _Vcs | None = None
    dir_info: dict[str, object] | None = None


def definitions_identity(version: str, definitions: list[tuple[str, str]]) -> str:
    """The operation identity of a callable's definitions, however they were read."""
    document = json.dumps(
        {"operation_version": version, "definitions": definitions},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return "sha256:" + hashlib.sha256(document).hexdigest()


def normalized(source: str | bytes, *, callable_only: bool = False) -> str:
    node = ast.parse(source)
    if callable_only:
        if len(node.body) != 1 or not isinstance(
            node.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            raise ValueError("operation source is not one definition")
        return normalized_definition(node.body[0])
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def normalized_definition(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """One definition as its own module, decorators dropped: its behavior, not its placement."""
    bare = copy.copy(node)
    bare.decorator_list = []
    module = ast.Module(body=[bare], type_ignores=[])
    return ast.dump(module, annotate_fields=True, include_attributes=False)


def distribution_identity(
    version: str, wheel: str | None, direct: str | None, cache_tag: str | None
) -> str | None:
    """An installed distribution's build as metadata states it, read from its files only."""
    try:
        if not version or not wheel:
            return None
        tags = sorted(line for line in wheel.splitlines() if line.startswith(("Tag:", "Build:")))
        if not tags:
            return None
        provenance: dict[str, object] = {}
        if direct:
            source = msgspec.json.decode(direct, type=_DirectUrl)
            archive, vcs = source.archive_info, source.vcs_info
            if source.dir_info is not None or (vcs is None and archive is None):
                return None
            # uv may record a local immutable wheel with archive_info={}.
            # Its version and wheel tags still supply the declared native build
            # contract; include archive hashes when the installer retains them.
            if archive is not None and archive.hashes:
                provenance["archive_hashes"] = archive.hashes
            elif archive is not None and archive.hash:
                provenance["archive_hash"] = archive.hash
            if vcs is not None:
                if not vcs.commit_id:
                    return None
                provenance["commit"] = vcs.commit_id
        return json.dumps(
            {
                "version": version,
                "wheel": tags,
                "build": provenance,
                "abi": cache_tag,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    except (OSError, TypeError, ValueError):
        return None


def installed_distribution_identity(
    name: str, roots: Sequence[Path], cache_tag: str | None
) -> str | None:
    """Read one declared wheel's metadata directly, without importlib finder hooks."""

    def normalize(value: str) -> str:
        return re.sub(r"[-_.]+", "-", value).lower()

    expected = normalize(name)
    matches = {
        info
        for root in dict.fromkeys(path.resolve() for path in roots)
        for info in root.glob("*.dist-info")
        if normalize(info.name.removesuffix(".dist-info").rpartition("-")[0]) == expected
    }
    if len(matches) != 1:
        return None
    info = matches.pop()

    def read(filename: str) -> str | None:
        path = info / filename
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            data = stream.read(MAX_METADATA_BYTES + 1)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("installed metadata exceeds bound")
        return data.decode("utf-8")

    try:
        metadata = read("METADATA")
        if metadata is None:
            return None
        headers = email.parser.Parser().parsestr(metadata, headersonly=True)
        if normalize(headers.get("Name", "")) != expected:
            return None
        return distribution_identity(
            headers.get("Version", ""), read("WHEEL"), read("direct_url.json"), cache_tag
        )
    except (OSError, ValueError):
        return None
