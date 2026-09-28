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
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill
from cozy_runtime.protocol import worker_pb2 as pb

from . import source_upload, workspace_partial
from .source_calls import _exchange, _Tail, diagnosis
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .session import Worker

FORMAT = "cozy.machine-source-model/1"
_HF = re.compile(r"hf://([^/@]+/[^/@]+)@([^/]+)(?:/(.+))?")
_CIVITAI = re.compile(r"civitai://([1-9][0-9]*)(?:/(.+))?")
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

type Note = Callable[[str, int, int], None]


def name(source: str, profiles: Sequence[str]) -> str:
    digest = hashlib.sha256(canonical_json.encode([FORMAT, source, sorted(profiles)]))
    return "source-" + digest.hexdigest()[:40]


def _operation(source: str, profiles: list[str], local: str) -> tuple[str, dict[str, Any], bool]:
    """The upload operation this source is, and whether its spelling is immutable."""
    if match := _HF.fullmatch(source):
        repository, revision, member = match.groups()
        arguments = {
            "repository": repository,
            "revision": revision,
            "destination": "local/" + local,
            "profiles": profiles,
            "carriers": [unquote(member)] if member else [],
        }
        return "upload_huggingface", arguments, re.fullmatch(r"[0-9a-f]{40}", revision) is not None
    if match := _CIVITAI.fullmatch(source):
        version, file = match.groups()
        arguments = {
            "version": int(version),
            "destination": "local/" + local,
            "profiles": profiles,
            "file": file or "",
        }
        return "upload_civitai", arguments, True
    raise WorkspaceRefusal(
        f"model source {source[:200]!r} is not hf://org/repo@revision or civitai://"
    )


def slot(
    worker: Worker,
    parameter: str,
    choice: Mapping[str, Any],
    credentials: Mapping[str, str],
    note: Note,
) -> dict[str, Any]:
    """One source choice as the slot row serving preparation binds, made once per machine."""
    source, profiles = str(choice["source"]), [str(p) for p in choice.get("profiles", [])]
    if choice.get("repository") or choice.get("manifest"):
        raise WorkspaceRefusal(f"{parameter} names a source and a catalog checkpoint")
    local = name(source, profiles)
    operation, arguments, pinned = _operation(source, profiles, local)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(local, threading.Lock())
    with lock:
        facts = _held(worker, local) if pinned else None
        if facts is None:
            facts = _make(worker, local, operation, arguments, credentials, note)
    return {
        "parameter": parameter,
        "public_origin": "",
        "gpu": "*",
        "gpus": 0,
        "repository": "local/" + local,
        "manifest": facts["manifest"],
        "release": "",
        "lane": "",
        "source": source,
        "resolved": facts["resolved"],
        "profiles": facts["profiles"],
    }


def _evidence(worker: Worker, local: str) -> Path:
    assert worker.workspace is not None
    return worker.workspace.directory / "source-models" / f"{local}.json"


def _held(worker: Worker, local: str) -> dict[str, Any] | None:
    """The model an earlier identical choice made here, when its bytes are all still held."""
    assert worker.workspace is not None
    store = fill.store(worker.workspace.store_root)
    try:
        facts = canonical_json.decode(_evidence(worker, local).read_bytes())
        row = store.resolve_local(local)
        digest, length = row["manifest_digest"], row["manifest_length"]
        if facts.get("manifest") != {"digest": digest, "length": length}:
            return None
        store.verify_checkpoint_source("local/" + local, digest, length)
    except (FileNotFoundError, ValueError, fill.tensorfs_module().errors.Refusal):
        return None
    return dict(facts)


def _make(
    worker: Worker,
    local: str,
    operation: str,
    arguments: dict[str, Any],
    credentials: Mapping[str, str],
    note: Note,
) -> dict[str, Any]:
    workspace, calls = worker.workspace, worker.source_calls
    assert workspace is not None
    store = fill.store(workspace.store_root)
    endpoints = dict(calls.endpoints) if calls is not None else {}
    native_owner = "sha256:" + hashlib.sha256(canonical_json.encode([FORMAT, local])).hexdigest()
    credential = credentials.get(operation.removeprefix("upload_"), "")
    upload = source_upload.Upload(
        store_root=workspace.store_root,
        native_owner=native_owner,
        service_id=local,
        operation=operation,
        arguments=arguments,
        envelope={
            "store": str(workspace.store_root),
            "parent_pid": os.getpid(),
            "operation": operation,
            "arguments": arguments,
            "native_owner": native_owner,
            "credential": credential,
            **endpoints,
            "allow_local": bool(endpoints),
        },
        registry=calls.native_registry if calls is not None else None,
        client=None,
        hooks=source_upload.Hooks(
            child=partial(_child, credential),
            canceled=lambda: False,
            active=calls.active if calls is not None else lambda: set(),
            live=lambda: set(),
            evict_retired=partial(workspace_partial.reclaim, workspace),
            consume=lambda _service: False,
            progress=note,
        ),
        allow_local=bool(endpoints),
        directory=workspace.directory,
    )
    note("resolving model source headers", 0, 1)
    selection = upload.resolve()
    if not upload.slots:
        upload.select(selection)
    content = {
        "content_manifest": {
            "digest": selection["content_manifest_digest"],
            "length": selection["content_manifest_length"],
        }
    }
    computation = hashlib.sha256(
        canonical_json.encode(
            [FORMAT, content, upload.slots, upload.recipe.name if upload.recipe else ""]
        )
    ).digest()
    model = upload.convert(
        selection, content, "sha256:" + computation.hex(), time.time_ns() // 1_000_000
    )
    manifest = {"digest": model["digest"], "length": int(model["length"])}
    try:
        current = store.repo_get("local", local)
    except fill.tensorfs_module().errors.Refusal:
        current = None
    store.replace_local(
        current,
        local,
        "sha256:" + selection["selection_sha256"],
        manifest["digest"],
        manifest["length"],
    )
    facts = {
        "manifest": manifest,
        "resolved": selection["canonical"],
        "profiles": [profile for _, profile in upload.slots],
        "selection": "sha256:" + selection["selection_sha256"],
    }
    path = _evidence(worker, local)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        staged.write(canonical_json.encode(facts))
        staged.flush()
        os.fsync(staged.fileno())
    os.replace(staged.name, path)
    # The local repository holds the model; the source and the conversion session go.
    source_upload.release(
        store, workspace.directory, {"service_id": local, "native_owner": native_owner}, set()
    )
    return facts


def _child(credential: str, envelope: dict[str, Any]) -> dict[str, Any]:
    """One upload step in its own process; no wall clock ends it."""
    encoded = canonical_json.encode(envelope)
    process = subprocess.Popen(
        [sys.executable, "-m", "cozy_runtime.internal.upload_child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stderr is not None
    tail = _Tail(process.stderr)
    try:
        output = _exchange(process, encoded, lambda _sample: None)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
    if process.returncode or not output:
        raise WorkspaceRefusal(
            "model source preparation failed: "
            + diagnosis(process.returncode, tail.text(), credential)
        )
    answer = canonical_json.decode(output)
    if not answer.get("ok"):
        raise WorkspaceRefusal(f"model source preparation refused: {answer.get('code')}")
    return dict(answer)


_PROVIDERS = {
    pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE: "huggingface",
    pb.NATIVE_SOURCE_OPERATION_CIVITAI: "civitai",
}


def credentials(given: Sequence[pb.SourceCredential]) -> dict[str, str]:
    """The submission's provider credentials, by the provider an upload operation names."""
    return {_PROVIDERS[row.provider]: row.credential for row in given if row.provider in _PROVIDERS}
