"""Model slots this machine resolves itself: the caller's explicit choice, else the owner's
binding, else the callee's authored default. The rung is chosen for this machine's devices
before anything is read, so a slot costs one public catalog resolution of that rung's lane,
once per package: a warm run reads nothing, whatever release or snapshot of the package it
runs. The owner's rebinding, which this machine cannot see, reaches it as
`Resolutions.forget` from the controller that made it."""

from __future__ import annotations

import json
import re
import ssl
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode, urlsplit

from packaging.version import InvalidVersion, Version

from cozy_runtime.author import ConformanceError
from cozy_runtime.author._model_defaults import default_ladder, reference
from cozy_runtime.internal import egress, fill
from cozy_runtime.protocol import documents

from .machine_publication import PublicationRefusal, origin_key
from .workspace import WorkspaceRefusal
from .workspace_executions import ExecutionWorkspaceRefusal

if TYPE_CHECKING:
    from .machine_publication import PublicationAuthority
    from .session import Worker

MAX_METADATA = 1 << 20
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

type Fits = Callable[[Mapping[str, Any]], bool]
type Note = Callable[[str, int, int], None]
#: A repository named without a lane serves full precision first when the slot ranks none.
_FULL_PRECISION = ("bf16", "fp16", "fp32")
#: One catalog answer as read: a binding row, or a lane's checkpoint.
type Row = dict[str, Any]


@dataclass
class Resolved:
    """What one package's slots read at the Hub: the owner's bindings, and each lane's or
    digest's checkpoint. Every release and installation of the package shares it."""

    bindings: dict[str, Row] = field(default_factory=dict)
    checkpoints: dict[tuple[str, str, str], Row] = field(default_factory=dict)


class Resolutions:
    """What this machine read of each package at each Hub (`hub_key`), for its lifetime: its
    newest release, and its slots' resolutions."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.held: dict[tuple[str, str], Resolved] = {}
        self.newest: dict[tuple[str, str], str] = {}

    def of(self, package: str, hub: str = "") -> Resolved:
        with self.lock:
            return self.held.setdefault((hub, package), Resolved())

    def forget(self, package: str) -> None:
        """Drop what was read of a package at every Hub; its next run reads it once more."""
        with self.lock:
            for held in (self.newest, self.held):
                for key in [key for key in held if key[1] == package]:
                    del held[key]


class Catalog:
    """Reads at one catalog origin, each at most once for the package it resolves for. At
    the machine's own Hub each read presents its worker capability, which reads its owner's
    unpublished checkpoints; a refusal is never retried anonymously."""

    def __init__(
        self,
        origin: str,
        context: ssl.SSLContext | None = None,
        credential: Mapping[str, str] | None = None,
        resolved: Resolved | None = None,
    ) -> None:
        parts = urlsplit(origin)
        self.origin = origin
        self.host, self.port = parts.hostname or "", parts.port
        self.tls = (context or ssl.create_default_context()) if parts.scheme == "https" else None
        self.credential = dict(credential or {})
        self.resolved = resolved or Resolved()
        self.bindings = self.resolved.bindings

    def text(self, path: str, limit: int = MAX_METADATA) -> bytes | None:
        """One bounded document's bytes, or None when the catalog has none."""
        try:
            status, raw = egress.request_catalog_metadata(
                self.host, self.port, self.tls, path, limit=limit, credential=self.credential
            )
        except egress.EgressRefusal as exc:
            raise WorkspaceRefusal(f"catalog read failed: {exc.code}") from exc
        if status == 404:
            return None
        if status != 200:
            raise WorkspaceRefusal(f"catalog answered {status} {_code(raw)} for {path[:200]}")
        return raw

    def get(self, path: str, limit: int = MAX_METADATA) -> Any:
        raw = self.text(path, limit)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise WorkspaceRefusal("catalog answered invalid JSON") from exc

    def binding(self, package: str, slot: str) -> dict[str, Any] | None:
        """The owner's current binding of one slot, or None."""
        if package not in self.bindings:
            org, _, name = package.partition("/")
            value = self.get(f"/v1/packages/{quote(org)}/{quote(name)}/bindings") or {}
            rows = value.get("bindings", []) if isinstance(value, dict) else None
            if not isinstance(rows, list):
                raise WorkspaceRefusal("model catalog answered invalid bindings")
            self.bindings[package] = {
                row["slot"]: row for row in rows if isinstance(row, dict) and "slot" in row
            }
        return self.bindings[package].get(slot)

    def lanes(self, model: str, release: str) -> tuple[str, dict[str, Row]]:
        """A release's lanes by name: the named release, else the newest one not yanked."""
        org, _, name = model.partition("/")
        card = self.get(f"/v1/models/{quote(org)}/{quote(name)}") or {}
        rows = [row for row in card.get("releases", []) if isinstance(row, dict)]
        if release:
            rows = [row for row in rows if str(row.get("release", "")).lower() == release.lower()]
        else:
            rows = sorted(
                (row for row in rows if not row.get("yanked")), key=_version, reverse=True
            )
        if not rows:
            raise WorkspaceRefusal(f"model catalog has no release {release or '(any)'} of {model}")
        lanes = {str(row["lane"]): row for row in rows[0].get("lanes") or [] if row.get("lane")}
        return str(rows[0].get("release", release)), lanes

    def resolve(self, model: str, release: str, lane: str) -> dict[str, Any]:
        """One lane's exact checkpoint: its manifest digest and length."""
        held = self.resolved.checkpoints.get((model, release, lane))
        if held is None:
            held = self.resolved.checkpoints[(model, release, lane)] = self._resolve(
                model, release, lane
            )
        return held

    def _resolve(self, model: str, release: str, lane: str) -> Row:
        spec = model + ("@" + release if release else "")
        query = {"ref": spec, **({"lane": lane} if lane else {})}
        value = self.get("/v1/models/resolve?" + urlencode(query))
        # Names are case-insensitive: "Fidika/Anima" is the Hub's "fidika/anima".
        named = str(value.get("model", "")) if isinstance(value, dict) else ""
        if not isinstance(value, dict) or named.casefold() != model.casefold():
            raise WorkspaceRefusal(f"model catalog cannot resolve {spec}/{lane}")
        digest, length = value.get("manifest_id"), value.get("manifest_length")
        try:
            documents.raw(str(digest))
        except documents.DocumentError as exc:
            raise WorkspaceRefusal(f"model catalog returned no manifest for {spec}/{lane}") from exc
        if type(length) is not int or length <= 0:
            length = self.measure(named, str(digest))
        return {
            "repository": named,
            "manifest": {"digest": digest, "length": length},
            "release": str(value.get("release") or release),
            "lane": str(value.get("lane") or lane),
        }

    def measure(self, model: str, digest: str) -> int:
        """An unstated manifest length, measured from the exact bytes its digest names."""
        org, _, name = model.partition("/")
        path = f"/v1/models/{quote(org)}/{quote(name)}/checkpoints/{digest}"
        try:
            status, length, sha256 = egress.measure_catalog_document(
                self.host,
                self.port,
                self.tls,
                path,
                limit=fill.tensorfs_module().manifest_max_bytes(),
                credential=self.credential,
            )
        except egress.EgressRefusal as exc:
            raise WorkspaceRefusal(f"model catalog manifest read failed: {exc.code}") from exc
        if status != 200:
            raise WorkspaceRefusal(f"model catalog answered {status} for manifest {digest}")
        if sha256 != documents.raw(digest):
            raise WorkspaceRefusal(f"model catalog manifest bytes do not match {digest}")
        return length


def same_origin(a: str, b: str) -> bool:
    try:
        return origin_key(a) == origin_key(b)
    except ValueError:
        return False


def registration(worker: Worker, origin: str) -> PublicationAuthority | None:
    """Read this machine's current scoped access, including a renewed delegated token."""
    if not origin:
        default = worker.options.publication_authority
        if default is None:
            return None
        origin = default.origin
    for held in authorities(worker):
        if same_origin(held.origin, origin):
            if held.access_token:
                with worker.hub_access_lock:
                    principal = worker.hub_access_principals.setdefault(
                        origin_key(held.origin), held.principal
                    )
                if principal != held.principal:
                    raise ExecutionWorkspaceRefusal(
                        "hub_access_principal_conflict",
                        "this machine's Hub access names another account; renew the original account's access",
                    )
            return held
    return None


def authorities(worker: Worker) -> tuple[PublicationAuthority, ...]:
    from .machine_publication import registrations

    options = worker.options
    path = getattr(options, "hub_access_path", None)
    hubs = registrations(path)[0] if path else options.hubs
    default = options.publication_authority
    return (() if default is None else (default,)) + hubs


def hub_key(worker: Worker, hub: str) -> str:
    """A run's Hub as this machine keys what it reads there: "" is its default Hub. A Hub
    it is not registered at is refused."""
    default = worker.options.publication_authority
    if not hub or (default is not None and same_origin(default.origin, hub)):
        return ""
    held = registration(worker, hub)
    if held is None:
        raise ExecutionWorkspaceRefusal(
            "hub_access_absent", f"this machine has no execution access for {hub[:256]}"
        )
    return held.origin


def own(worker: Worker, package: str = "", hub: str = "") -> Catalog:
    """The run's Hub (empty: this machine's default), as its registration there names it:
    where it reads packages and Models, its owner's unpublished ones included. For a
    package, what it already read of it there."""
    key = hub_key(worker, hub)
    authority = registration(worker, key)
    if authority is None:
        raise WorkspaceRefusal("this machine has no Hub grant to read the catalog at")
    try:
        headers = authority.headers()
    except PublicationRefusal as exc:
        raise ExecutionWorkspaceRefusal(
            "hub_access_expired",
            "this machine's execution access has expired; renew it at the controller",
        ) from exc
    return Catalog(
        authority.origin,
        authority.context(),
        headers,
        worker.resolutions.of(package, key) if package else None,
    )


def _version(row: Mapping[str, Any]) -> tuple[int, Version | str]:
    try:
        return 1, Version(str(row.get("release", "")))
    except InvalidVersion:
        return 0, str(row.get("release", ""))


def _code(raw: bytes) -> str:
    """The Hub's typed error code, e.g. worker.unauthorized or worker.ended, else ""."""
    try:
        error = json.loads(raw).get("error", {})
        return str(error.get("code", ""))[:128] if isinstance(error, dict) else ""
    except (ValueError, AttributeError):
        return ""


def _ladder(
    catalog: Catalog,
    package: str,
    declaration: Mapping[str, Any],
    bound: bool = True,
    owner: str = "",
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """The owner's binding, else the authored default: (model, release, rungs). An org-relative
    default names the package's org, or for unpublished code (local/) its owner's."""
    if bound and package and not package.startswith("local/"):
        row = catalog.binding(package, str(declaration["path"]))
        if row is not None:
            rungs = row.get("ladder")
            if not isinstance(rungs, list) or not rungs:
                raise WorkspaceRefusal(f"the owner's binding of {declaration['path']} has no lanes")
            return str(row.get("model", "")), str(row.get("release", "")), rungs
    authored = declaration.get("default_ladder")
    if not authored:
        return None
    try:
        rows = default_ladder(authored)
        model, release, _ = reference(rows[0][2])
    except ConformanceError as exc:
        raise WorkspaceRefusal(f"{declaration['path']} has an invalid authored default") from exc
    if "/" not in model:
        org = package.partition("/")[0]
        org = owner if org == "local" else org
        if not org:
            raise WorkspaceRefusal(f"{declaration['path']} names its owner's model with no owner")
        model = org + "/" + model
    return (
        model,
        release,
        [{"gpu": gpu, "gpus": gpus, "lane": reference(lane)[2]} for gpu, gpus, lane in rows],
    )


def slot(
    catalog: Catalog,
    package: str,
    declaration: Mapping[str, Any],
    parameter: str,
    choice: Mapping[str, Any] | None,
    fits: Fits,
    owner: str = "",
    note: Note | None = None,
) -> dict[str, Any]:
    """One slot's exact checkpoint for this machine, in the captured default row shape."""
    base = {"parameter": parameter, "public_origin": catalog.origin}
    choice = choice or {}
    manifest = choice.get("manifest") or {}
    if manifest.get("digest"):
        digest, repository = str(manifest["digest"]), str(choice.get("repository", ""))
        documents.raw(digest)
        pinned: dict[str, Any] = {"digest": digest, "length": int(manifest.get("length", 0))}
        if pinned["length"] <= 0:
            pinned = catalog.resolve(repository, digest, "")["manifest"]
        return {
            **base,
            "gpu": "*",
            "gpus": 0,
            "repository": repository,
            "manifest": pinned,
            "release": choice.get("release", ""),
            "lane": choice.get("lane", ""),
        }
    degrees = set(declaration.get("sequence_parallel", {}).get("degrees", []))

    def usable(rung: object) -> bool:
        return (
            isinstance(rung, dict)
            and int(rung.get("gpus", 0) or 0) in {0, 1, *degrees}
            and fits(rung)
        )

    # An explicit repository is the caller's choice: the owner's binding is not read for it.
    ladder = _ladder(catalog, package, declaration, not choice.get("repository"), owner)
    have: dict[str, Row] = {}
    if choice.get("repository"):
        model, release = str(choice["repository"]), str(choice.get("release", ""))
        if choice.get("lane"):
            rungs = [{"gpu": "*", "gpus": 0, "lane": choice["lane"]}]
        else:
            # No lane named: the repository's own, as the slot's ladder ranks them here, else
            # full precision, else the smallest.
            release, have = catalog.lanes(model, release)
            if not have:
                raise WorkspaceRefusal(f"{model}@{release} has no lane to serve {parameter}")
            rungs = [
                r for r in (ladder[2] if ladder else []) if usable(r) and r.get("lane") in have
            ]
            if not rungs:
                lane = next((name for name in _FULL_PRECISION if name in have), None) or min(
                    have, key=lambda name: (int(have[name].get("bytes") or 0), name)
                )
                rungs = [{"gpu": "*", "gpus": 0, "lane": lane}]
        ladder = model, release, rungs
    if ladder is None:
        raise WorkspaceRefusal(f"{parameter} has no owner binding or authored default")
    model, release, rungs = ladder
    if _REPOSITORY.fullmatch(model) is None or model.startswith("local/"):
        raise WorkspaceRefusal(f"{parameter} names no public catalog repository")
    fitting = [rung for rung in rungs if usable(rung)]
    if not fitting:
        raise WorkspaceRefusal(f"{parameter} has no rung for this accelerator")
    # The widest rung this machine holds: the width itself is the declared degree.
    rung = max(fitting, key=lambda candidate: int(candidate.get("gpus", 0) or 0))
    if have and note is not None:
        note(
            f"{parameter}: {model}@{release} lane {rung['lane']} of {', '.join(sorted(have))}", 0, 0
        )
    resolved = catalog.resolve(model, release, str(rung["lane"]))
    return {
        **base,
        "gpu": str(rung.get("gpu", "*")),
        "gpus": int(rung.get("gpus", 0) or 0),
        "repository": model,
        **resolved,
    }
