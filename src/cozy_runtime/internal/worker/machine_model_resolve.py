"""Model slots this machine resolves itself: the caller's explicit choice, else the owner's
binding, else the callee's authored default. The rung is chosen for this machine's devices
before anything is read, so a slot costs one public catalog resolution of that rung's lane,
once per package: a warm or restarted run reads nothing, whatever release or snapshot of the
package it runs. The owner's rebinding, which this machine cannot see, reaches it as
`Resolutions.forget` from the controller that made it."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import ssl
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import quote, urlencode, urlsplit

import msgspec
from packaging.version import InvalidVersion, Version

from cozy_runtime.author import ConformanceError
from cozy_runtime.author._model_defaults import default_ladder, reference
from cozy_runtime.internal import egress, fill
from cozy_runtime.protocol import documents

from .machine_publication import PublicationRefusal, hub_error, origin_key
from .workspace import WorkspaceRefusal
from .workspace_executions import ExecutionWorkspaceRefusal

if TYPE_CHECKING:
    from .machine_publication import PublicationAuthority
    from .session import Worker

MAX_METADATA = 1 << 20
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

type Fits = Callable[[Mapping[str, object]], bool]
type Note = Callable[[str, int, int], None]
#: A repository named without a lane serves full precision first when the slot ranks none.
_FULL_PRECISION = ("bf16", "fp16", "fp32")


class Manifest(TypedDict):
    digest: str
    length: int


class Checkpoint(TypedDict):
    repository: str
    manifest: Manifest
    release: str
    lane: str


class Selection(Checkpoint):
    parameter: str
    public_origin: str
    gpu: str
    gpus: int


class _Binding(msgspec.Struct, frozen=True):
    slot: str
    model: str = ""
    release: str = ""
    ladder: tuple[object, ...] | None = None


class _Bindings(msgspec.Struct, frozen=True):
    bindings: tuple[object, ...] = ()


class _Rung(msgspec.Struct, frozen=True):
    lane: str
    gpu: str | msgspec.UnsetType = msgspec.UNSET
    gpus: int = 0


class _Lane(msgspec.Struct, frozen=True):
    lane: str = ""
    bytes: object = 0  # relevant only when no ranked or full-precision lane exists

    def size(self) -> int:
        value = self.bytes or 0
        if isinstance(value, (int, float, str)):
            try:
                return int(value)
            except (ValueError, OverflowError):
                pass
        raise WorkspaceRefusal("model catalog answered invalid lane size")


class _Release(msgspec.Struct, frozen=True):
    release: str = ""
    yanked: object = False
    lanes: object = None  # parsed only for the selected release


class _Model(msgspec.Struct, frozen=True):
    releases: tuple[object, ...] = ()


class _Resolved(msgspec.Struct, frozen=True):
    model: str = ""
    manifest_id: str = ""
    # Older catalogs omit the length; any unusable value triggers digest-verified measurement.
    manifest_length: object = None
    release: str | None = None
    lane: str | None = None


class _Parallel(msgspec.Struct, frozen=True):
    degrees: tuple[int, ...] = ()


class _Declaration(msgspec.Struct, frozen=True):
    path: str
    default_ladder: object = None
    sequence_parallel: _Parallel = _Parallel()


class _Pin(msgspec.Struct, frozen=True):
    digest: str = ""
    length: int = 0


class _Choice(msgspec.Struct, frozen=True):
    repository: str = ""
    release: str = ""
    lane: str = ""
    manifest: _Pin | None = None


def _read[T](value: object, into: type[T], what: str) -> T:
    """Read only consumed fields; newer catalog members remain advisory."""
    try:
        return msgspec.convert(value, into)
    except (msgspec.ValidationError, TypeError) as exc:
        raise WorkspaceRefusal(f"model catalog answered invalid {what}") from exc


KEPT = "cozy.runtime.KeptResolutions/1"


class _KeptLanes(msgspec.Struct, frozen=True):
    model: str
    release: str
    answered: str
    lanes: dict[str, _Lane]


class _KeptCheckpoint(msgspec.Struct, frozen=True):
    model: str
    release: str
    lane: str
    checkpoint: Checkpoint


class _Kept(msgspec.Struct, frozen=True):
    format: str
    hub: str
    package: str
    bindings: dict[str, dict[str, object]] = {}
    lanes: list[_KeptLanes] = []
    checkpoints: list[_KeptCheckpoint] = []
    newest: str = ""


@dataclass
class Resolved:
    """What one package's slots read at one Hub: the owner's bindings, each release's lanes,
    each lane's or digest's checkpoint and the package's newest release. Every release and
    installation of the package shares it, and it outlives the Runtime at `path`."""

    hub: str = ""
    package: str = ""
    path: Path | None = None
    bindings: dict[str, dict[str, object]] = field(default_factory=dict)
    lanes: dict[tuple[str, str], tuple[str, dict[str, _Lane]]] = field(default_factory=dict)
    checkpoints: dict[tuple[str, str, str], Checkpoint] = field(default_factory=dict)
    newest: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def keep(self, change: Callable[[Resolved], None]) -> None:
        """Apply one read's answer and keep the whole record; a failed write only costs a read."""
        with self.lock:
            change(self)
            if self.path is None:
                return
            raw = msgspec.json.encode(
                _Kept(
                    KEPT,
                    self.hub,
                    self.package,
                    self.bindings,
                    [_KeptLanes(m, r, a, rows) for (m, r), (a, rows) in self.lanes.items()],
                    [_KeptCheckpoint(m, r, n, c) for (m, r, n), c in self.checkpoints.items()],
                    self.newest,
                )
            )
            with contextlib.suppress(OSError):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_name(f".{uuid.uuid4().hex}")
                temporary.write_bytes(raw)
                os.replace(temporary, self.path)

    def drop_model(self, name: str) -> None:
        """Forget every lane and checkpoint read of the model `name` (casefolded)."""
        if any(m.casefold() == name for m, _ in self.lanes) or any(
            m.casefold() == name for m, _, _ in self.checkpoints
        ):
            self.keep(_dropping(name))


def _dropping(name: str) -> Callable[[Resolved], None]:
    def drop(held: Resolved) -> None:
        held.lanes = {k: v for k, v in held.lanes.items() if k[0].casefold() != name}
        held.checkpoints = {k: v for k, v in held.checkpoints.items() if k[0].casefold() != name}

    return drop


def _name(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _kept(path: Path) -> _Kept | None:
    try:
        kept = msgspec.json.decode(path.read_bytes(), type=_Kept)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError):
        return None  # absent, torn or foreign: a miss, replaced by the next read
    return kept if kept.format == KEPT else None


def supervised(directory: Path) -> Path:
    """`directory`'s subtree for the process supervising this Runtime, as (pid, start time).
    Reads kept there outlive a Runtime restart under that supervisor, never a new one: a
    controller tells only the machines it reaches that a package or model changed, and a
    stopped machine starts afresh. Other supervisors' subtrees are removed."""
    parent = os.getppid()
    try:
        started = Path(f"/proc/{parent}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        started = ""
    own = directory / f"{parent}-{started}"
    with contextlib.suppress(OSError):
        for other in directory.iterdir():
            if other != own:
                shutil.rmtree(other, ignore_errors=True)
    return own


class Resolutions:
    """What this machine read of each package at each Hub (`hub_key`), kept under `root` so a
    restarted Runtime reads nothing either, until `forget` names the package or a model."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.held: dict[tuple[str, str], Resolved] = {}

    def of(self, package: str, hub: str = "") -> Resolved:
        key = (hub, package.casefold())
        with self.lock:
            held = self.held.get(key)
            if held is None:
                path = None if self.root is None else self.root / _name(key[1]) / _name(hub)
                kept = None if path is None else _kept(path)
                held = self.held[key] = Resolved(hub, package, path)
                if kept is not None and (kept.hub, kept.package.casefold()) == key:
                    held.bindings = dict(kept.bindings)
                    held.lanes = {(r.model, r.release): (r.answered, r.lanes) for r in kept.lanes}
                    held.checkpoints = {
                        (r.model, r.release, r.lane): r.checkpoint for r in kept.checkpoints
                    }
                    held.newest = kept.newest
            return held

    def forget(self, name: str) -> None:
        """Drop what was read of a package at every Hub, and every lane and checkpoint read of a
        model by that name: the next run reads them once more."""
        key = name.casefold()
        with self.lock:
            for held in [held for held in self.held if held[1] == key]:
                del self.held[held]
        if self.root is not None:
            shutil.rmtree(self.root / _name(key), ignore_errors=True)
            for path in self.root.glob("*/*"):  # every other package read at every Hub
                if (kept := _kept(path)) is not None:
                    self.of(kept.package, kept.hub)
        with self.lock:
            others = list(self.held.values())
        for resolved in others:
            resolved.drop_model(key)


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

    def text(self, path: str, limit: int = MAX_METADATA) -> bytes | None:
        """One bounded document's bytes, or None when the catalog has none."""
        try:
            status, raw = egress.request_catalog_metadata(
                self.host, self.port, self.tls, path, limit=limit, credential=self.credential
            )
        except egress.EgressRefusal as exc:
            raise WorkspaceRefusal(f"catalog read failed: {exc.code}") from exc
        if status == 404 and hub_error(raw):
            return None  # the Hub's own answer: it has no such document
        if status == 404:
            seen = re.search(rb"ERR_NGROK_\d{1,6}", raw)
            raise WorkspaceRefusal(
                f"the Hub at {self.origin} is unreachable: a tunnel or proxy in front of it "
                f"answered 404 for {path[:200]}" + (f" ({seen[0].decode()})" if seen else "")
            )
        if status != 200:
            raise WorkspaceRefusal(f"catalog answered {status} {hub_error(raw)} for {path[:200]}")
        return raw

    def get(self, path: str, limit: int = MAX_METADATA) -> object:
        raw = self.text(path, limit)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise WorkspaceRefusal("catalog answered invalid JSON") from exc

    def binding(self, package: str, slot: str) -> _Binding | None:
        """The owner's current binding of one slot, or None."""
        if package not in self.resolved.bindings:
            org, _, name = package.partition("/")
            value = self.get(f"/v1/packages/{quote(org)}/{quote(name)}/bindings") or {}
            rows = _read(value, _Bindings, "bindings").bindings
            bindings: dict[str, object] = {}
            for row in rows:
                if isinstance(row, dict):
                    slot_name: object = row.get("slot")
                    if isinstance(slot_name, str):
                        bindings[slot_name] = row
            self.resolved.keep(lambda kept: kept.bindings.update({package: bindings}))
        selected = self.resolved.bindings[package].get(slot)
        if selected is None or isinstance(selected, _Binding):
            return selected
        binding = _read(selected, _Binding, "binding")
        self.resolved.bindings[package][slot] = binding  # decoded once; kept as its fields
        return binding

    def lanes(self, model: str, release: str) -> tuple[str, dict[str, _Lane]]:
        """A release's lanes by name: the named release, else the newest one not yanked."""
        held = self.resolved.lanes.get((model, release))
        if held is None:
            held = self._lanes(model, release)
            self.resolved.keep(lambda kept: kept.lanes.update({(model, release): held}))
        return held

    def _lanes(self, model: str, release: str) -> tuple[str, dict[str, _Lane]]:
        org, _, name = model.partition("/")
        card = self.get(f"/v1/models/{quote(org)}/{quote(name)}") or {}
        rows = [
            _read(row, _Release, "release")
            for row in _read(card, _Model, "model").releases
            if isinstance(row, dict)
        ]
        if release:
            rows = [row for row in rows if row.release.lower() == release.lower()]
        else:
            rows = sorted((row for row in rows if not row.yanked), key=_version, reverse=True)
        if not rows:
            raise WorkspaceRefusal(f"model catalog has no release {release or '(any)'} of {model}")
        lane_rows = rows[0].lanes or ()
        if not isinstance(lane_rows, (list, tuple)):
            raise WorkspaceRefusal("model catalog answered invalid lanes")
        lanes = [_read(row, _Lane, "lane") for row in lane_rows]
        return rows[0].release or release, {row.lane: row for row in lanes if row.lane}

    def resolve(self, model: str, release: str, lane: str) -> Checkpoint:
        """One lane's exact checkpoint: its manifest digest and length."""
        held = self.resolved.checkpoints.get((model, release, lane))
        if held is None:
            held = self._resolve(model, release, lane)
            self.resolved.keep(lambda kept: kept.checkpoints.update({(model, release, lane): held}))
        return held

    def _resolve(self, model: str, release: str, lane: str) -> Checkpoint:
        spec = model + ("@" + release if release else "")
        query = {"ref": spec, **({"lane": lane} if lane else {})}
        raw = self.get("/v1/models/resolve?" + urlencode(query))
        if not isinstance(raw, dict):
            raise WorkspaceRefusal(f"model catalog cannot resolve {spec}/{lane}")
        value = _read(raw, _Resolved, "resolution")
        # Names are case-insensitive: "Fidika/Anima" is the Hub's "fidika/anima".
        named = value.model
        if named.casefold() != model.casefold():
            raise WorkspaceRefusal(f"model catalog cannot resolve {spec}/{lane}")
        digest, length = value.manifest_id, value.manifest_length
        try:
            documents.raw(digest)
        except documents.DocumentError as exc:
            raise WorkspaceRefusal(f"model catalog returned no manifest for {spec}/{lane}") from exc
        if type(length) is not int or length <= 0:
            length = self.measure(named, digest)
        return {
            "repository": named,
            "manifest": {"digest": digest, "length": length},
            "release": value.release or release,
            "lane": value.lane or lane,
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
        # Empty never borrows one of a persistent machine's Hub grants (wire 69).
        raise WorkspaceRefusal(
            "this run names no Hub to read the model catalog at, and this machine has none "
            "of its own; run it from a client logged in to the Hub its models come from"
        )
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


def _version(row: _Release) -> tuple[int, Version | str]:
    try:
        return 1, Version(row.release)
    except InvalidVersion:
        return 0, row.release


def _ladder(
    catalog: Catalog,
    package: str,
    declaration: _Declaration,
    bound: bool = True,
    owner: str = "",
) -> tuple[str, str, list[_Rung]] | None:
    """The owner's binding, else the authored default: (model, release, rungs). An org-relative
    default names the package's org, or for unpublished code (local/) its owner's."""
    if bound and package and not package.startswith("local/"):
        row = catalog.binding(package, declaration.path)
        if row is not None:
            if not row.ladder:
                raise WorkspaceRefusal(f"the owner's binding of {declaration.path} has no lanes")
            rungs = []
            for raw in row.ladder:
                try:
                    rungs.append(msgspec.convert(raw, _Rung))
                except msgspec.ValidationError:
                    continue  # an unreadable rung is not selectable on this machine
            return row.model, row.release, rungs
    if not declaration.default_ladder:
        return None
    try:
        rows = default_ladder(declaration.default_ladder)
        model, release, _ = reference(rows[0][2])
    except ConformanceError as exc:
        raise WorkspaceRefusal(f"{declaration.path} has an invalid authored default") from exc
    if "/" not in model:
        org = package.partition("/")[0]
        org = owner if org == "local" else org
        if not org:
            raise WorkspaceRefusal(f"{declaration.path} names its owner's model with no owner")
        model = org + "/" + model
    return model, release, [_Rung(reference(lane)[2], gpu, gpus) for gpu, gpus, lane in rows]


def slot(
    catalog: Catalog,
    package: str,
    declaration: Mapping[str, object],
    parameter: str,
    choice: Mapping[str, object] | None,
    fits: Fits,
    owner: str = "",
    note: Note | None = None,
) -> Selection:
    """One slot's exact checkpoint for this machine, in the captured default row shape."""
    selected = _read(choice or {}, _Choice, "model choice")
    manifest = selected.manifest
    if manifest is not None and manifest.digest:
        digest, repository = manifest.digest, selected.repository
        documents.raw(digest)
        pinned: Manifest = {"digest": digest, "length": manifest.length}
        if pinned["length"] <= 0:
            pinned = catalog.resolve(repository, digest, "")["manifest"]
        return {
            "parameter": parameter,
            "public_origin": catalog.origin,
            "gpu": "*",
            "gpus": 0,
            "repository": repository,
            "manifest": pinned,
            "release": selected.release,
            "lane": selected.lane,
        }
    declared = _read(declaration, _Declaration, "slot declaration")
    degrees = set(declared.sequence_parallel.degrees)

    def usable(rung: _Rung) -> bool:
        selector: dict[str, object] = {"gpus": rung.gpus, "lane": rung.lane}
        if rung.gpu is not msgspec.UNSET:
            selector["gpu"] = rung.gpu
        return rung.gpus in {0, 1, *degrees} and fits(selector)

    # An explicit repository is the caller's choice: the owner's binding is not read for it.
    ladder = _ladder(catalog, package, declared, not selected.repository, owner)
    have: dict[str, _Lane] = {}
    if selected.repository:
        model, release = selected.repository, selected.release
        if selected.lane:
            rungs = [_Rung(selected.lane, "*")]
        else:
            # No lane named: the repository's own, as the slot's ladder ranks them here, else
            # full precision, else the smallest.
            release, have = catalog.lanes(model, release)
            if not have:
                raise WorkspaceRefusal(f"{model}@{release} has no lane to serve {parameter}")
            rungs = [r for r in (ladder[2] if ladder else []) if usable(r) and r.lane in have]
            if not rungs:
                lane = next((name for name in _FULL_PRECISION if name in have), None) or min(
                    have, key=lambda name: (have[name].size(), name)
                )
                rungs = [_Rung(lane, "*")]
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
    rung = max(fitting, key=lambda candidate: candidate.gpus)
    if have and note is not None:
        note(f"{parameter}: {model}@{release} lane {rung.lane} of {', '.join(sorted(have))}", 0, 0)
    resolved = catalog.resolve(model, release, rung.lane)
    return {
        "parameter": parameter,
        "public_origin": catalog.origin,
        "gpu": "*" if rung.gpu is msgspec.UNSET else rung.gpu,
        "gpus": rung.gpus,
        **resolved,
    }
