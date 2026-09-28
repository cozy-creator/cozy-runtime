"""The fixed native source interface; not a package build or arbitrary RPC registry."""

from __future__ import annotations

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, SourceArtifact
from cozy_runtime.author._calls import _CallType
from cozy_runtime.author._output_commit import CommittedFile
from cozy_runtime.author.publication import CheckpointRef
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
    profiles: tuple[str, ...] = ()


class UploadHuggingFace(msgspec.Struct, frozen=True):
    repository: str
    revision: str
    destination: str
    profiles: tuple[str, ...] = ()
    carriers: tuple[str, ...] = ()


class UploadCivitai(msgspec.Struct, frozen=True):
    version: int
    destination: str
    profiles: tuple[str, ...] = ()
    file: str = ""


class SourceFiles(msgspec.Struct, frozen=True):
    source: SourceArtifact


class CommitFile(msgspec.Struct, frozen=True):
    slot: str
    digest: str
    size_bytes: int
    media_type: str


TYPES: dict[str, tuple[type[msgspec.Struct], type[msgspec.Struct]]] = {
    "download_huggingface": (HuggingFace, SourceArtifact),
    "download_civitai": (Civitai, SourceArtifact),
    "convert_cozytensors": (Convert, ModelArtifact),
    "source_files": (SourceFiles, _SourceFilesResult),
    "commit_file": (CommitFile, CommittedFile),
    "upload_huggingface": (UploadHuggingFace, CheckpointRef),
    "upload_civitai": (UploadCivitai, CheckpointRef),
}
#: Operations that stream a provider source through conversion into a publication.
UPLOADS = ("upload_huggingface", "upload_civitai")
#: Each operation's output semantics, bumped by hand when the same inputs would produce
#: a different result. Native source memo keys carry this, never a source-tree hash.
OPERATION_VERSIONS: dict[str, int] = {name: 1 for name in TYPES}
DOCUMENT = canonical_json.encode(
    {
        "format": "cozy.native-source-interface/1",
        "module": MODULE,
        "operations": {
            name: {"request": schema.render(request), "result": schema.render(result)}
            for name, (request, result) in TYPES.items()
        },
    }
)


def bindings() -> dict[str, _CallType]:
    return {
        name: _CallType("", MODULE, name, request, result)
        for name, (request, result) in TYPES.items()
    }
