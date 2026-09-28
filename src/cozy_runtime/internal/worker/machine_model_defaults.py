"""Frozen public Model selections; only omitted inference arguments consume them."""

from __future__ import annotations

import ipaddress
import re
import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import grants, machine_model_resolve
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


def local_development(worker: Worker) -> bool:
    """A machine whose own Hub, as its registration grant names it, is on loopback: a
    development machine, which may reach the private addresses that Hub hands it
    (checkpoint pulls, object uploads). A production Hub is never loopback, and a job
    never names the grant."""
    authority = worker.options.publication_authority
    return authority is not None and local_origin(authority.origin)


def _same_origin(a: str, b: str) -> bool:
    try:
        left, right = urlsplit(a.rstrip("/")), urlsplit(b.rstrip("/"))
        return (left.scheme, left.hostname, left.port) == (right.scheme, right.hostname, right.port)
    except ValueError:
        return False


def credential(worker: Worker, origin: str) -> str:
    """This machine's worker capability, for its own Hub only: there it reads its owner's
    unpublished checkpoints. Any other origin gets nothing, and a refusal is never retried
    anonymously."""
    authority = worker.options.publication_authority
    if authority is None or not _same_origin(authority.origin, origin):
        return ""
    return f"worker {authority.worker_id} {authority.worker_token}"


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
    expected: set[tuple[str, str, str]] = set()
    for binding in capture["bindings"]:
        revision, entrypoint = callee(binding), binding["entrypoint"]
        for entry in interfaces[revision]["entrypoints"]:
            if entry["name"] == entrypoint:
                expected.update(
                    (revision, entrypoint, model["path"].rpartition(".")[2])
                    for model in entry.get("models", [])
                )
    keys = []
    for row in rows:
        if not callee(row) or not {"entrypoint", "parameter"} <= row.keys():
            raise WorkspaceRefusal("captured Model default has no exact slot identity")
        key = callee(row), row["entrypoint"], row["parameter"]
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
            row.get("public_origin", ""), local=local_development(worker)
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
            gpus = rung.get("gpus", 0)
            if type(gpus) is not int or gpus < 0:
                raise WorkspaceRefusal("captured Model default has invalid GPU count")
            selectors.append((gpu.lower(), gpus))
        if len(set(selectors)) != len(selectors):
            raise WorkspaceRefusal("captured Model default repeats a GPU and count selector")
    if len(keys) > 4096 or keys != sorted(set(keys)) or set(keys) != expected:
        raise WorkspaceRefusal("captured Model default inventory is incomplete or ambiguous")


def callee(row: dict[str, Any]) -> str:
    """A binding or default row's callee: its installation, or a deferred release key."""
    return str(row.get("callee_installation_id") or row.get("callee_deferred_key") or "")


def matches(selector: str, observed: str) -> bool:
    """The existing Creator GPU ladder's case-insensitive token subsequence rule."""
    if selector == "*":
        return True
    wanted = re.findall(r"[^\W_]+", selector.lower())
    actual = iter(re.findall(r"[^\W_]+", observed.lower()))
    return bool(wanted) and all(any(token == candidate for candidate in actual) for token in wanted)


def fitter(worker: Worker) -> Callable[[Mapping[str, Any]], bool]:
    """Whether a rung's GPU selector and count fit this machine's devices."""
    facts = worker.host_facts()
    if "gpu_name" in facts.unreadable:
        raise WorkspaceRefusal("captured Model default cannot select an unreadable accelerator")
    entries = getattr(getattr(worker, "lanes", None), "entries", None)

    def fits(candidate: Mapping[str, Any]) -> bool:
        if not matches(str(candidate.get("gpu", "")), facts.gpu_name):
            return False
        count = candidate.get("gpus", 0)
        if count and entries is None and "gpu_count" in facts.unreadable:
            raise WorkspaceRefusal("counted Model default requires a readable granted GPU count")
        return not count or count <= (len(entries) if entries is not None else facts.gpu_count)

    return fits


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
    capture = worker.executions.capture(owner, root)
    rows = capture.get("model_defaults", [])
    defaults = {
        row["parameter"]: row
        for row in rows
        if (callee(row), row["entrypoint"])
        == (target.callee or target.installation_id, target.entrypoint)
    }
    # A slot the capture left open is resolved here, at this machine's own Hub.
    for parameter in sorted(missing):
        row = defaults.get(parameter)
        if row is not None and row.get("unavailable_code"):
            raise WorkspaceRefusal("captured Model default is unavailable for " + parameter)
    open_slots = any(parameter not in defaults for parameter in missing)
    catalog = machine_model_resolve.own(worker) if open_slots else None
    fits = fitter(worker)
    package = str(
        (
            target.prepared_installation.get("placement", {}).get("package")
            or target.prepared_installation.get("placement", {}).get("development")
            or {}
        ).get("package", "")
    )
    declared = {
        model["path"].rpartition(".")[2]: model for model in target.declaration.get("models", [])
    }
    selected = []
    for parameter in sorted(missing):
        row = defaults.get(parameter)
        if row is None:
            assert catalog is not None
            selected.append(
                machine_model_resolve.slot(
                    catalog, package, declared[parameter], parameter, None, fits
                )
            )
            continue
        fitting = [candidate for candidate in row["rungs"] if fits(candidate)]
        # The widest rung this machine holds: the width itself is the declared degree.
        rung = max(fitting, key=lambda candidate: candidate.get("gpus", 0), default=None)
        if rung is None:
            raise WorkspaceRefusal("captured Model default has no rung for this accelerator")
        selected.append({"parameter": parameter, "public_origin": row["public_origin"], **rung})
    return selected


def materialize(
    worker: Worker,
    rows: list[dict[str, Any]],
    access: ExitStack | None,
    *,
    downloading: Callable[[], AbstractContextManager[None]] = nullcontext,
    cancellation: Any = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Verify exact sources; lease bytes only when the preparer will read them.

    A reused placement consumes its registered binding document. Its fresh child
    admission still verifies and retains every checkpoint before submission.
    """
    if not rows:
        return
    assert worker.workspace is not None
    tensorfs = fill.tensorfs_module()
    store = fill.store(worker.workspace.store_root)
    for row in rows:
        if cancellation is not None and cancellation.cancelled:
            raise WorkspaceRefusal("model preparation canceled")
        manifest = row["manifest"]
        try:
            # Existing cached custody is sufficient. A cold use fetches this exact
            # checkpoint; only this machine's own Hub sees its worker capability.
            with worker.model_transfer_lock:
                transfer = worker.model_transfers.setdefault(manifest["digest"], threading.Lock())
            with transfer:
                if cancellation is not None and cancellation.cancelled:
                    raise WorkspaceRefusal("model preparation canceled")
                try:
                    store.verify_checkpoint_source(
                        row["repository"], manifest["digest"], manifest["length"]
                    )
                except tensorfs.errors.Refusal:
                    with downloading():
                        tensorfs.pull(
                            store,
                            row["public_origin"],
                            row["repository"] + "@" + manifest["digest"],
                            credential=credential(worker, row["public_origin"]),
                            allowed_hosts=worker.config.object_storage_hosts,
                            allow_local=local_development(worker),
                            streams=8,
                            cancellation=cancellation,
                            progress=progress,
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
