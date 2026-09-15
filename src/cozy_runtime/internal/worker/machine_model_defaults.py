"""Frozen public Model selections; only omitted inference arguments consume them."""

from __future__ import annotations

import ipaddress
import re
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from cozy_runtime.internal import fill, hostfacts
from cozy_runtime.protocol import CAPTURED_MODEL_DEFAULTS_WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import grants
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .machine_child_target import Target
    from .session import Worker
    from .workspace_calls import Call


def local_origin(origin: str) -> bool:
    try:
        host = urlsplit(origin).hostname or ""
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def valid_origin(origin: str, *, local: bool) -> bool:
    try:
        value = urlsplit(origin)
        return bool(
            0 < len(origin) <= 512
            and origin == origin.strip()
            and not any(ord(c) <= 32 or ord(c) == 127 for c in origin)
            and value.hostname
            and value.port != 0
            and not value.username
            and not value.password
            and not value.path
            and not value.query
            and not value.fragment
            and (
                value.scheme == "https"
                or (local and value.scheme == "http" and local_origin(origin))
            )
        )
    except ValueError:
        return False


def verify(
    worker: Worker,
    request: pb.MachineExecutionSubmit,
    capture: dict[str, Any],
    interfaces: dict[str, dict[str, Any]],
) -> None:
    rows = capture.get("model_defaults", [])
    if not rows:
        return
    if request.claim.wire_minor < CAPTURED_MODEL_DEFAULTS_WIRE_MINOR:
        raise WorkspaceRefusal("captured Model defaults require wire56")
    expected: set[tuple[str, str, str]] = set()
    for binding in capture["bindings"]:
        revision, entrypoint = binding["callee_revision_digest"], binding["entrypoint"]
        for entry in interfaces[revision]["entrypoints"]:
            if entry["name"] == entrypoint:
                expected.update(
                    (revision, entrypoint, model["path"].rpartition(".")[2])
                    for model in entry.get("models", [])
                )
    keys = []
    for row in rows:
        if not {"callee_revision_digest", "entrypoint", "parameter"} <= row.keys():
            raise WorkspaceRefusal("captured Model default has no exact slot identity")
        key = row["callee_revision_digest"], row["entrypoint"], row["parameter"]
        keys.append(key)
        rungs, unavailable = row.get("rungs", []), row.get("unavailable_code", "")
        if unavailable:
            if (
                unavailable
                not in {
                    "model_default_unavailable",
                    "model_default_unbound",
                    "model_default_origin_unsupported",
                }
                or rungs
                or row.get("public_origin")
            ):
                raise WorkspaceRefusal("captured Model default has conflicting availability")
            continue
        if not 1 <= len(rungs) <= 32 or not valid_origin(
            row.get("public_origin", ""), local=bool(worker.config.bootstrap_credential)
        ):
            raise WorkspaceRefusal("captured Model default has no exact public origin or ladder")
        selectors = []
        for rung in rungs:
            if not {"gpu", "repository", "manifest"} <= rung.keys():
                raise WorkspaceRefusal("captured Model default has incomplete rung facts")
            gpu, repository, manifest = rung["gpu"], rung["repository"], rung["manifest"]
            if (
                not 0 < len(gpu) <= 128
                or any(ord(c) < 32 for c in gpu)
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}", repository
                )
                is None
                or repository.startswith("local/")
                or manifest["length"] <= 0
            ):
                raise WorkspaceRefusal("captured Model default has invalid exact rung facts")
            documents.raw(manifest["digest"])
            selectors.append(gpu.lower())
        if len(set(selectors)) != len(selectors):
            raise WorkspaceRefusal("captured Model default repeats a GPU selector")
    if len(keys) > 4096 or keys != sorted(set(keys)) or set(keys) != expected:
        raise WorkspaceRefusal("captured Model default inventory is incomplete or ambiguous")


def matches(selector: str, observed: str) -> bool:
    """The existing Creator GPU ladder's case-insensitive token subsequence rule."""
    if selector == "*":
        return True
    wanted = re.findall(r"[^\W_]+", selector.lower())
    actual = iter(re.findall(r"[^\W_]+", observed.lower()))
    return bool(wanted) and all(any(token == candidate for candidate in actual) for token in wanted)


def select(
    worker: Worker, owner: str, call: Call, target: Target, arguments: dict[str, Any]
) -> list[dict[str, Any]]:
    missing = {
        model["path"].rpartition(".")[2]
        for model in target.declaration.get("models", [])
        if arguments.get(model["path"].rpartition(".")[2]) is None
    }
    if not missing:
        return []
    assert worker.executions is not None
    root = worker.executions.capture_root(owner, call.parent_request)
    rows = worker.executions.capture(owner, root).get("model_defaults", [])
    defaults = {
        row["parameter"]: row
        for row in rows
        if (row["callee_revision_digest"], row["entrypoint"])
        == (target.revision, target.entrypoint)
    }
    for parameter in sorted(missing):
        row = defaults.get(parameter)
        if row is None or row.get("unavailable_code"):
            raise WorkspaceRefusal("captured Model default is unavailable for " + parameter)
    facts = hostfacts.measure(worker.options.accelerator_backend)
    if "gpu_name" in facts.unreadable:
        raise WorkspaceRefusal("captured Model default cannot select an unreadable accelerator")
    selected = []
    for parameter in sorted(missing):
        row = defaults[parameter]
        rung = next((rung for rung in row["rungs"] if matches(rung["gpu"], facts.gpu_name)), None)
        if rung is None:
            raise WorkspaceRefusal("captured Model default has no rung for this accelerator")
        selected.append({"parameter": parameter, "public_origin": row["public_origin"], **rung})
    return selected


def materialize(worker: Worker, rows: list[dict[str, Any]], access: ExitStack | None) -> None:
    """Verify exact sources; lease bytes only when the preparer will read them.

    A reused placement consumes its registered binding document. Its fresh child
    admission still verifies and retains every checkpoint before submission.
    """
    if not rows:
        return
    assert worker.workspace is not None
    tensorfs = fill.tensorfs_module()
    store = tensorfs.Store.open(str(worker.workspace.store_root))
    for row in rows:
        manifest = row["manifest"]
        try:
            # Existing cached custody is sufficient. A cold use fetches this exact
            # public checkpoint; neither a release alias nor a credential is sent.
            try:
                store.verify_checkpoint_source(
                    row["repository"], manifest["digest"], manifest["length"]
                )
            except tensorfs.errors.Refusal:
                tensorfs.pull(
                    store,
                    row["public_origin"],
                    row["repository"] + "@" + manifest["digest"],
                    allowed_hosts=worker.config.object_storage_hosts,
                    allow_local=bool(worker.config.bootstrap_credential)
                    and local_origin(row["public_origin"]),
                )
                store.verify_checkpoint_source(
                    row["repository"], manifest["digest"], manifest["length"]
                )
            if access is not None:
                access.enter_context(store.acquire_cozytensors(manifest["digest"]))
        except tensorfs.errors.Refusal as error:
            raise WorkspaceRefusal(
                "captured Model default materialization refused: " + str(error.code)
            ) from error


def inputs(rows: list[dict[str, Any]]) -> tuple[list[pb.InputBinding], list[pb.InputAccess]]:
    bindings, accesses = [], []
    for row in rows:
        manifest, input_id = row["manifest"], grants.MODEL_PREFIX + row["parameter"]
        bindings.append(
            pb.InputBinding(
                input_id=input_id,
                digest=manifest["digest"],
                length=manifest["length"],
                kind_mime=grants.MODEL_MIME,
            )
        )
        accesses.append(
            pb.InputAccess(
                input_id=input_id,
                url="model://" + manifest["digest"],
                catalog_model=pb.CatalogModelSource(repository=row["repository"]),
            )
        )
    return bindings, accesses
