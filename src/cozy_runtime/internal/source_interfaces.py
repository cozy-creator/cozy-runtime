"""The fixed native source interface; not a package build or arbitrary RPC registry."""

from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, SourceArtifact
from cozy_runtime.author._calls import _CallType
from cozy_runtime.author._output_commit import CommittedFile
from cozy_runtime.author.sources import _SourceFilesResult
from cozy_runtime.internal import schema

MODULE = "cozy_runtime.author.sources"


class HuggingFace(msgspec.Struct, frozen=True):
    repository: str
    revision: str
    profiles: tuple[str, ...] = ()
    carriers: tuple[str, ...] = ()
    files: tuple[str, ...] = ()


class Civitai(msgspec.Struct, frozen=True):
    version: int
    file: str = ""


class Convert(msgspec.Struct, frozen=True):
    source: SourceArtifact
    profile: str


class SourceFiles(msgspec.Struct, frozen=True):
    source: SourceArtifact


class CommitFile(msgspec.Struct, frozen=True):
    slot: str
    digest: str
    size_bytes: int
    media_type: str


_TYPES: dict[str, tuple[type[msgspec.Struct], type[msgspec.Struct]]] = {
    "download_huggingface": (HuggingFace, SourceArtifact),
    "download_civitai": (Civitai, SourceArtifact),
    "convert_cozytensors": (Convert, ModelArtifact),
    "source_files": (SourceFiles, _SourceFilesResult),
    "commit_file": (CommitFile, CommittedFile),
}
DOCUMENT = canonical_json.encode(
    {
        "format": "cozy.native-source-interface/1",
        "module": MODULE,
        "operations": {
            name: {"request": schema.render(request), "result": schema.render(result)}
            for name, (request, result) in _TYPES.items()
        },
    }
)
DIGEST = "sha256:" + hashlib.sha256(DOCUMENT).hexdigest()


def bindings() -> dict[tuple[str, str, str], _CallType]:
    return {
        (prefix, MODULE, name): _CallType(DIGEST, MODULE, name, request, result)
        for name, (request, result) in _TYPES.items()
        for prefix in ("", DIGEST)
    }


def rows() -> list[dict[str, object]]:
    return [
        {"interface_digest": DIGEST, "module": MODULE, "export": name, "native_source": True}
        for name in _TYPES
    ]
