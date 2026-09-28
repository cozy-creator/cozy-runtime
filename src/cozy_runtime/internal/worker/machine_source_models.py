"""A release root's provider-source Model, made on this machine (release_root_sources).

The upload pipeline resolves the source headers-first, lands only the members of the named
profiles (or of the one TensorFS selects) within the free disk and converts them. The model
stays here as the local repository ``local/<name>``, where the name is the source and its
profiles: an identical resubmission reads that repository and touches no provider. Nothing
is killed by the clock; a pass that makes no measured progress refuses.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.internal import fill
from cozy_runtime.internal.source_interfaces import UploadCivitai, UploadHuggingFace
from cozy_runtime.protocol import worker_pb2 as pb

from . import source_steps, source_upload, upload_plan, workspace_partial
from .source_calls import UPLOAD_CHILD, diagnosis
from .source_steps import Answer, Launch, Refused, UploadStep
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .session import Worker

FORMAT = "cozy.machine-source-model/1"
_HF = re.compile(r"hf://([^/@]+/[^/@]+)@([^/]+)(?:/(.+))?")
_CIVITAI = re.compile(r"civitai://([1-9][0-9]*)(?:/(.+))?")
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

type Note = Callable[[str, int, int], None]


class Evidence(msgspec.Struct, frozen=True):
    """The model an identical choice made here, kept beside its local repository."""

    manifest: ObjectRef
    resolved: str
    profiles: list[str]
    selection: str


def name(source: str, profiles: Sequence[str]) -> str:
    digest = hashlib.sha256(canonical_json.encode([FORMAT, source, sorted(profiles)]))
    return "source-" + digest.hexdigest()[:40]


def _request(
    source: str, profiles: tuple[str, ...], local: str
) -> tuple[upload_plan.Request, bool]:
    """The upload this source is, and whether its spelling is immutable."""
    if match := _HF.fullmatch(source):
        repository, revision, member = match.groups()
        request = UploadHuggingFace(
            repository=repository,
            revision=revision,
            destination="local/" + local,
            profiles=profiles,
            carriers=(unquote(member),) if member else (),
        )
        return request, re.fullmatch(r"[0-9a-f]{40}", revision) is not None
    if match := _CIVITAI.fullmatch(source):
        version, file = match.groups()
        destination = "local/" + local
        return UploadCivitai(int(version), destination, profiles, file or ""), True
    raise WorkspaceRefusal(
        f"model source {source[:200]!r} is not hf://org/repo@revision or civitai://"
    )


def slot(
    worker: Worker, choice: pb.ModelChoice, credentials: Mapping[str, str], note: Note
) -> dict[str, object]:
    """One source choice as the slot row serving preparation binds, made once per machine."""
    if choice.repository or choice.HasField("manifest"):
        raise WorkspaceRefusal(f"{choice.parameter} names a source and a catalog checkpoint")
    local = name(choice.source, choice.profiles)
    request, pinned = _request(choice.source, tuple(choice.profiles), local)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(local, threading.Lock())
    with lock:
        facts = _held(worker, local) if pinned else None
        if facts is None:
            facts = _make(worker, local, request, credentials, note)
    return {
        "parameter": choice.parameter,
        "public_origin": "",
        "gpu": "*",
        "gpus": 0,
        "repository": "local/" + local,
        "manifest": msgspec.to_builtins(facts.manifest),
        "release": "",
        "lane": "",
        "source": choice.source,
        "resolved": facts.resolved,
        "profiles": facts.profiles,
    }


def _evidence(worker: Worker, local: str) -> Path:
    assert worker.workspace is not None
    return worker.workspace.directory / "source-models" / f"{local}.json"


def _held(worker: Worker, local: str) -> Evidence | None:
    """The model an earlier identical choice made here, when its bytes are all still held."""
    assert worker.workspace is not None
    store = fill.store(worker.workspace.store_root)
    try:
        facts = msgspec.json.decode(_evidence(worker, local).read_bytes(), type=Evidence)
        row = store.resolve_local(local)
        if facts.manifest != ObjectRef(row["manifest_digest"], row["manifest_length"]):
            return None
        store.verify_checkpoint_source(
            "local/" + local, facts.manifest.digest, facts.manifest.length
        )
    except (FileNotFoundError, ValueError, fill.tensorfs_module().errors.Refusal):
        return None
    return facts


def _make(
    worker: Worker,
    local: str,
    request: upload_plan.Request,
    credentials: Mapping[str, str],
    note: Note,
) -> Evidence:
    workspace, calls = worker.workspace, worker.source_calls
    assert workspace is not None
    store = fill.store(workspace.store_root)
    native_owner = "sha256:" + hashlib.sha256(canonical_json.encode([FORMAT, local])).hexdigest()
    provider = "huggingface" if isinstance(request, UploadHuggingFace) else "civitai"
    credential = credentials.get(provider, "")

    def child[A: Answer](launch: Launch[UploadStep], answer: type[A]) -> A:
        return _child(credential, launch, answer)

    upload = source_upload.Upload(
        store_root=workspace.store_root,
        native_owner=native_owner,
        service_id=local,
        request=request,
        access=source_steps.access(calls.endpoints if calls is not None else {}, credential),
        registry=calls.native_registry if calls is not None else None,
        client=None,
        hooks=source_upload.Hooks(
            child=child,
            canceled=lambda: False,
            active=calls.active if calls is not None else lambda: set(),
            live=lambda: set(),
            evict_retired=partial(workspace_partial.reclaim, workspace),
            consume=lambda _service: False,
            progress=note,
        ),
        directory=workspace.directory,
    )
    note("resolving model source headers", 0, 1)
    selection = upload.resolve()
    if not upload.slots:
        upload.select(selection)
    manifest = ObjectRef(selection.content_manifest_digest, selection.content_manifest_length)
    content = {"content_manifest": msgspec.to_builtins(manifest)}
    computation = hashlib.sha256(
        canonical_json.encode(
            [FORMAT, content, upload.slots, upload.recipe.name if upload.recipe else ""]
        )
    ).digest()
    model = upload.convert(
        selection, manifest, "sha256:" + computation.hex(), time.time_ns() // 1_000_000
    )
    try:
        current: bytes | None = store.repo_get("local", local)
    except fill.tensorfs_module().errors.Refusal:
        current = None
    pin = "sha256:" + selection.selection_sha256
    store.replace_local(current, local, pin, model.digest, model.length)
    facts = Evidence(model, selection.canonical, [profile for _, profile in upload.slots], pin)
    path = _evidence(worker, local)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        staged.write(canonical_json.encode(msgspec.to_builtins(facts)))
        staged.flush()
        os.fsync(staged.fileno())
    os.replace(staged.name, path)
    # The local repository holds the model; the source and the conversion session go.
    source_upload.release(store, workspace.directory, local, native_owner, set())
    return facts


def _child[A: Answer](credential: str, launch: Launch[UploadStep], answer: type[A]) -> A:
    """One upload step in its own process; no wall clock ends it."""
    status, output, stderr = source_steps.run(UPLOAD_CHILD, launch)
    if status or not output:
        raise WorkspaceRefusal(
            "model source preparation failed: " + diagnosis(status, stderr, credential)
        )
    got = source_steps.answer(output, answer)
    if isinstance(got, Refused):
        raise WorkspaceRefusal(f"model source preparation refused: {got.code}")
    return got


_PROVIDERS = {
    pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE: "huggingface",
    pb.NATIVE_SOURCE_OPERATION_CIVITAI: "civitai",
}


def credentials(given: Sequence[pb.SourceCredential]) -> dict[str, str]:
    """The submission's provider credentials, by the provider an upload operation names."""
    return {_PROVIDERS[row.provider]: row.credential for row in given if row.provider in _PROVIDERS}
