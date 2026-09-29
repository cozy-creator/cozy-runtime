"""Publication metadata for an accepted machine execution.

Runtime's call journal freezes intent and records sends. TensorFS retains and moves
the bytes. This adapter holds neither a second journal nor account credentials.
"""

from __future__ import annotations

import base64
import hashlib
import re
import ssl
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.author.publication import AssessmentRef, CheckpointRef, ReleaseReceipt
from cozy_runtime.internal import egress
from cozy_runtime.internal.config import (
    ConfigError,
    parse_object_storage_hosts,
    parse_tensorhub_origin,
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z")
_MAX_METADATA = 8 << 20
# Pushes in flight an upload starts with; measured progress grows it within memory.
PUBLICATION_UPLOADS = 16
# Hub's per-request object selection bound, and the progress-publication object bound.
_PUBLICATION_WINDOW = 128
_PUBLICATION_OBJECTS = 4096


class PublicationRefusal(Exception):
    def __init__(self, code: str, *, status: int = 409):
        super().__init__(code)
        self.code, self.status = code, status


class PublicationUncommitted(PublicationRefusal):
    """Hub durably answered that a sent commit did not land; nothing is left to reconcile."""


# Statuses that promise nothing about the request; retry. Other 4xx answers are final.
_RETRYABLE = frozenset({408, 425, 429})


class PublicationClient(Protocol):
    def request(self, method: str, path: str, body: bytes | None = None) -> bytes: ...


def origin_key(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value.rstrip("/"))
    return (
        parsed.scheme,
        (parsed.hostname or "").lower(),
        parsed.port or (443 if parsed.scheme == "https" else 80),
    )


def access_principal(token: str) -> tuple[str, ...]:
    """Credential continuity only; the Hub remains the JWT signature/authority verifier."""
    opaque = ("opaque", hashlib.sha256(token.encode()).hexdigest())
    parts = token.split(".")
    if len(token) > 32 << 10 or len(parts) != 3 or not parts[2]:
        return opaque
    try:
        header, claims = (
            msgspec.json.decode(
                base64.b64decode(part + "=" * (-len(part) % 4), altchars=b"-_", validate=True)
            )
            for part in parts[:2]
        )
        issuer, subject = claims.get("iss"), claims.get("delegated_sub")
        if (
            not isinstance(header.get("typ"), str)
            or header["typ"].strip().lower() != "delegated-access+jwt"
            or not isinstance(issuer, str)
            or not 0 < len(issuer.encode()) <= 2048
            or not isinstance(subject, str)
            or not 0 < len(subject.encode()) <= 256
            or claims.get("sub", "") != ""
            or claims.get("permissions") != ["cozy.execution-access"]
        ):
            return opaque
    except (ValueError, msgspec.DecodeError, AttributeError, TypeError):
        return opaque
    return "delegated", issuer, subject


@dataclass(frozen=True)
class PublicationAuthority:
    """Scoped catalog/publication access, independent of rental registration."""

    origin: str
    worker_id: str = ""
    worker_token: str = field(default="", repr=False)
    ca: Path | None = None  # the grant's private Hub CA, when its Hub uses one
    object_storage_hosts: tuple[str, ...] = ()  # where that Hub's presigned bytes live
    access_token: str = field(default="", repr=False)
    expires_at: int = 0
    access_path: Path | None = field(default=None, repr=False, compare=False)

    @property
    def principal(self) -> tuple[str, ...]:
        return (
            access_principal(self.access_token) if self.access_token else ("worker", self.worker_id)
        )

    def current(self) -> PublicationAuthority:
        if self.access_path is None:
            return self
        for authority in registrations(self.access_path)[0]:
            if origin_key(authority.origin) == origin_key(self.origin):
                if authority.principal != self.principal:
                    raise PublicationRefusal("publication.authority_principal_conflict", status=403)
                return authority
        raise PublicationRefusal("publication.authority_absent", status=403)

    def headers(self) -> dict[str, str]:
        if self.access_token:
            if self.expires_at <= time.time():
                raise PublicationRefusal("publication.authority_expired", status=403)
            return {"Authorization": "Bearer " + self.access_token}
        return {"X-Cozy-Worker-ID": self.worker_id, "X-Cozy-Worker-Token": self.worker_token}

    def context(self) -> ssl.SSLContext:
        return ssl.create_default_context(cafile=str(self.ca) if self.ca else None)

    def client(self, authorization_id: str) -> MachinePublicationClient:
        return MachinePublicationClient(
            self.origin,
            authorization_id,
            self.context(),
            worker_id=self.worker_id,
            worker_token=self.worker_token,
            access_token=self.access_token,
            credential=self.current,
        )


class _Hub(msgspec.Struct, frozen=True):
    origin: str
    worker_id: str = ""
    worker_token: str = ""
    ca: str = ""  # a file name beside the document
    object_storage_hosts: tuple[str, ...] = ()
    access_token: str = ""
    expires_at: int = 0


class _Hubs(msgspec.Struct, frozen=True):
    version: int
    hubs: tuple[msgspec.Raw, ...] = ()


def registrations(path: Path) -> tuple[tuple[PublicationAuthority, ...], list[str]]:
    """Read rental credentials or scoped owned-machine access without requiring a Hub row.

    Unreadable entries affect only their Hub, never machine boot or unrelated work.
    """
    try:
        with path.open("rb") as source:
            raw = source.read((256 << 10) + 1)
    except FileNotFoundError:
        return (), []
    try:
        if len(raw) > 256 << 10:
            raise ValueError("exceeds its bound")
        document = msgspec.json.decode(raw, type=_Hubs)
    except (ValueError, msgspec.ValidationError) as exc:
        return (), [f"{path.name} is unreadable: {exc}"[:512]]
    if document.version != 1:
        return (), [f"{path.name} version {document.version} is not one this Runtime reads"]
    held: list[PublicationAuthority] = []
    skipped: list[str] = []
    for entry in document.hubs[:64]:
        try:
            hub = msgspec.json.decode(entry, type=_Hub)
            parse_tensorhub_origin(hub.origin, allow_loopback_http=bool(hub.access_token))
            token = base64.urlsafe_b64decode(hub.worker_token + "=" * (-len(hub.worker_token) % 4))
            valid_access = (
                bool(hub.access_token)
                and len(hub.access_token) <= 16 << 10
                and all(33 <= ord(c) <= 126 for c in hub.access_token)
                and hub.expires_at > 0
                and not hub.worker_id
                and not hub.worker_token
            )
            valid_worker = (
                not hub.access_token
                and 0 < len(hub.worker_id) <= 128
                and hub.worker_id.isprintable()
                and len(token) == 32
                and base64.urlsafe_b64encode(token).decode().rstrip("=") == hub.worker_token
            )
            if (
                not (valid_access or valid_worker)
                or (hub.ca and (Path(hub.ca).name != hub.ca or hub.ca in {".", ".."}))
                or (hub.ca and not (path.parent / hub.ca).is_file())
            ):
                raise ValueError("invalid registration")
            hosts = parse_object_storage_hosts(",".join(sorted(set(hub.object_storage_hosts))))
        except (ValueError, msgspec.ValidationError, ConfigError) as exc:
            skipped.append(f"{path.name} entry skipped: {exc}"[:512])
            continue
        held.append(
            PublicationAuthority(
                hub.origin,
                hub.worker_id,
                hub.worker_token,
                path.parent / hub.ca if hub.ca else None,
                hosts,
                access_token=hub.access_token,
                expires_at=hub.expires_at,
                access_path=path,
            )
        )
    return tuple(held), skipped


class _Token(msgspec.Struct, frozen=True):
    token: str
    expires_at: str


class _Lane(msgspec.Struct, frozen=True):
    lane: str
    manifest_id: str = ""
    checkpoint_id: str = ""


class _Release(msgspec.Struct, frozen=True):
    """A model card's release row, and Hub's answer to a release update."""

    release: str
    revision: int = 0
    yanked: bool = False
    lanes: tuple[_Lane, ...] = ()


class _Card(msgspec.Struct, frozen=True):
    releases: tuple[_Release, ...]


class _Intent(msgspec.Struct, frozen=True):
    """The release intent a durable call row freezes before any write."""

    destination: str
    release: str
    set_lanes: dict[str, str]
    baseline_revision: int
    baseline: dict[str, str]
    desired: dict[str, str]


class _Member(msgspec.Struct, frozen=True):
    object_id: str
    length: int
    state: str


class _Publication(msgspec.Struct, frozen=True):
    objects: tuple[_Member, ...]
    operation: str = ""
    state: str = ""


class _Opened(msgspec.Struct, frozen=True):
    publication: _Publication


class _Grant(msgspec.Struct, frozen=True):
    object_id: str
    length: int
    url: str
    expires_at_unix: int
    required_headers: dict[str, str] = {}


class _Grants(msgspec.Struct, frozen=True):
    server_time_unix: int
    # Go's empty slices may be JSON null; they still represent zero rows.
    grants: tuple[_Grant, ...] | None = None
    held: tuple[_Member, ...] | None = None


class _Manifest(msgspec.Struct, frozen=True):
    sha256: str
    length: int


class _Checkpointed(msgspec.Struct, frozen=True):
    publish_id: str
    checkpoint_id: str
    manifest: _Manifest
    state: str
    objects: int
    bytes: int


class _Failure(msgspec.Struct, frozen=True):
    code: str = ""


class _Finalization(msgspec.Struct, frozen=True):
    """Hub's durable finalization job for one publication."""

    operation: str
    status_url: str
    state: str
    result: _Checkpointed | None = None
    error: _Failure | None = None


class _Refused(msgspec.Struct, frozen=True):
    """Hub's error answer."""

    error: _Failure = msgspec.field(default_factory=_Failure)


class _Report(msgspec.Struct, frozen=True):
    digest: str
    length: int


class _Assessment(msgspec.Struct, frozen=True):
    checkpoint_id: str
    scope: str
    report: _Report
    verdict: str = ""


class _Acknowledgement(msgspec.Struct, frozen=True):
    assessment: _Assessment
    publisher_reported_verdict: str = ""


def _decode[T](raw: bytes, into: type[T], code: str = "publication.response_invalid") -> T:
    try:
        return msgspec.json.decode(raw, type=into)
    except msgspec.DecodeError as exc:
        raise PublicationRefusal(code) from exc


def _require(condition: bool, code: str = "publication.response_changed") -> None:
    if not condition:
        raise PublicationRefusal(code)


def destination_path(destination: str) -> str:
    """The Hub path of an `org/name` repository, in any case: the Hub folds names."""
    parts = destination.strip().lower().split("/")
    _require(
        len(parts) == 2 and parts[0] != "local" and all(_NAME.fullmatch(p) for p in parts),
        "publication.destination_invalid",
    )
    return "/v1/models/" + "/".join(parts)


# A job root's weights output access spelled ``model://org/name`` publishes that output
# as a checkpoint of the repository, under the root's publication authorization.
DESTINATION_SCHEME = "model://"


def output_destination(urls: Iterable[str]) -> str:
    """The one repository a root's output grants name, or "" when none names one."""
    named = {
        url.removeprefix(DESTINATION_SCHEME).strip().lower()
        for url in urls
        if url.startswith(DESTINATION_SCHEME)
    }
    _require(len(named) <= 1, "publication.destination_ambiguous")
    if not named:
        return ""
    (destination,) = named
    destination_path(destination)
    return destination


class MachinePublicationClient:
    """One root's bounded AuthKit grant over verified public HTTPS.

    The independent worker capability authenticates renewal and binds publication
    to this worker. Python job executors never receive either credential.
    HTTP redirects are not followed and each connection is closed after use.
    """

    def __init__(
        self,
        origin: str,
        authorization_id: str,
        context: ssl.SSLContext,
        *,
        worker_id: str,
        worker_token: str,
        access_token: str = "",
        credential: Callable[[], PublicationAuthority] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        parsed = parse_tensorhub_origin(origin, allow_loopback_http=bool(access_token))
        try:
            valid = str(uuid.UUID(authorization_id)) == authorization_id
        except ValueError:
            valid = False
        _require(valid and uuid.UUID(authorization_id).int != 0, "publication.authority_invalid")
        _require(
            context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED,
            "publication.tls_verification_required",
        )
        _require(
            (
                bool(access_token)
                and not worker_id
                and not worker_token
                and len(access_token) <= 16 << 10
                and all(33 <= ord(c) <= 126 for c in access_token)
            )
            or (
                not access_token
                and bool(worker_id)
                and len(worker_id) <= 256
                and all(33 <= ord(c) <= 126 for c in worker_id)
                and re.fullmatch(r"[A-Za-z0-9_-]{43}", worker_token) is not None
            ),
            "publication.worker_authority_invalid",
        )
        self.worker_id, self.worker_token = worker_id, worker_token
        self.access_token, self.credential = access_token, credential
        self.origin = parsed
        self.authorization_id, self.context, self.clock = authorization_id, context, clock
        self._token, self._expires = "", 0.0
        self._lock = threading.Lock()

    def _exchange(
        self, method: str, path: str, body: bytes | None, token: str = ""
    ) -> tuple[int, bytes]:
        authority = self.credential() if self.credential else None
        access_token = authority.access_token if authority else self.access_token
        headers = {"Accept": "application/json"}
        if access_token:
            if authority is not None:
                authority.headers()  # check refreshed access expiry before sending
            if not token:
                headers["X-Cozy-Execution-Access"] = access_token
        else:
            headers.update(
                authority.headers()
                if authority
                else {
                    "X-Cozy-Worker-ID": self.worker_id,
                    "X-Cozy-Worker-Token": self.worker_token,
                }
            )
        if token:
            headers["Authorization"] = "Bearer " + token
        if body is not None:
            _require(len(body) <= _MAX_METADATA, "publication.metadata_bound")
            headers["Content-Type"] = "application/json"
        if method != "GET":
            headers["X-Tensorhub-Reason"] = "explicit private script publication"
        try:
            return egress.request_control_metadata(
                self.origin, self.context, method, path, body, headers, limit=_MAX_METADATA
            )
        except egress.EgressRefusal as exc:
            # URLs, signed headers and token strings never enter a retained error.
            status = 503 if exc.code == "control_transport_unavailable" else 409
            raise PublicationRefusal("publication." + exc.code, status=status) from exc

    def _renew(self) -> None:
        status, raw = self._exchange(
            "POST", f"/v1/worker/machine-authorizations/{self.authorization_id}/token", None
        )
        if status != 200:
            raise PublicationRefusal("publication.authority_refused", status=status)
        value = _decode(raw, _Token, "publication.authority_response_invalid")
        _require(0 < len(value.token) <= 16 << 10)
        try:
            stamp = datetime.fromisoformat(value.expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PublicationRefusal("publication.authority_response_invalid") from exc
        _require(stamp.tzinfo is not None, "publication.authority_response_invalid")
        deadline = stamp.timestamp()
        _require(deadline > self.clock(), "publication.authority_expired")
        self._token, self._expires = value.token, deadline

    def request(self, method: str, path: str, body: bytes | None = None) -> bytes:
        _require(
            method in {"GET", "POST", "PUT", "DELETE"}
            and path.startswith("/v1/models/")
            and (method != "DELETE" or "/publications/" in path)
            and not any(c in path for c in "\r\n?#"),
            "publication.route_forbidden",
        )
        with self._lock:
            if self._expires <= self.clock() + 30:
                self._renew()
            status, raw = self._exchange(method, path, body, self._token)
            if status == 401:
                # A rejected token did not authorize a write; renew once. Lost
                # write replies are reconciled by the frozen effect, not here.
                self._renew()
                status, raw = self._exchange(method, path, body, self._token)
        if not 200 <= status < 300:
            raise PublicationRefusal("publication.http_refused", status=status)
        return raw


def _read_release(
    client: PublicationClient, destination: str, release: str
) -> tuple[int, dict[str, str]]:
    # The model card is the existing public read surface, including all lanes.
    card = _decode(client.request("GET", destination_path(destination)), _Card)
    matches = [row for row in card.releases if row.release == release]
    _require(len(matches) <= 1)
    if not matches:
        return 0, {}
    (row,) = matches
    if not 1 <= row.revision < (1 << 53) - 1:
        raise PublicationRefusal("publication.response_invalid")
    _require(not row.yanked)
    lanes: dict[str, str] = {}
    for lane in row.lanes:
        _require(
            _LABEL.fullmatch(lane.lane) is not None
            and _DIGEST.fullmatch(lane.manifest_id) is not None
            and lane.lane not in lanes
        )
        lanes[lane.lane] = lane.manifest_id
    _require(bool(lanes))
    return row.revision, lanes


def prepare_release(
    client: PublicationClient,
    *,
    destination: str,
    release: str,
    lanes: Mapping[str, CheckpointRef],
    expected_revision: int | None,
) -> bytes:
    """Return the intent the existing durable call row must freeze before writes."""
    destination_path(destination)
    _require(_LABEL.fullmatch(release) is not None and 1 <= len(lanes) <= 32)
    if expected_revision is not None:
        _require(0 <= expected_revision < (1 << 53) - 1)
    requested: dict[str, str] = {}
    for name, checkpoint in lanes.items():
        _require(_LABEL.fullmatch(name) is not None)
        _require(
            checkpoint.destination.lower() == destination.lower()
            and checkpoint.checkpoint == checkpoint.manifest.digest
            and _DIGEST.fullmatch(checkpoint.checkpoint) is not None
            and checkpoint.manifest.length > 0
            and bool(checkpoint.publication)
        )
        requested[name] = checkpoint.checkpoint
    revision, baseline = _read_release(client, destination, release)
    _require(
        expected_revision is None or expected_revision == revision, "publication.release_conflict"
    )
    intent = _Intent(destination, release, requested, revision, baseline, baseline | requested)
    return canonical_json.encode(msgspec.to_builtins(intent))


def apply_release(
    client: PublicationClient, frozen: bytes, *, sent: bool, before_write: Callable[[], None]
) -> ReleaseReceipt:
    intent = _decode(frozen, _Intent, "publication.intent_changed")
    destination, release = intent.destination, intent.release
    baseline, desired = intent.baseline, intent.desired
    expected = intent.baseline_revision
    _require(baseline | intent.set_lanes == desired, "publication.intent_changed")
    revision, current = _read_release(client, destination, release)

    def receipt(observation: str) -> ReleaseReceipt:
        return ReleaseReceipt(destination, release, revision, current, observation)

    if revision == expected + 1 and current == desired:
        return receipt("observed_convergence")
    _require(
        revision == expected and current == baseline,
        "publication.outcome_unknown" if sent else "publication.release_conflict",
    )
    if current == desired:
        return receipt("observed_noop")
    before_write()
    updated = _decode(
        client.request(
            "POST",
            destination_path(destination) + "/releases/" + quote(release, safe=""),
            canonical_json.encode(
                {"expected_revision": expected, "set_lanes": intent.set_lanes, "remove_lanes": []}
            ),
        ),
        _Release,
        "publication.outcome_unknown",
    )
    actual = {row.lane: row.checkpoint_id for row in updated.lanes}
    _require(
        updated.release == release
        and updated.revision == expected + 1
        and not updated.yanked
        and len(actual) == len(updated.lanes)
        and actual == desired,
        "publication.outcome_unknown",
    )
    return ReleaseReceipt(destination, release, expected + 1, desired, "acknowledged")


def _checkpoint_inputs(
    operation: str, checkpoint: CheckpointRef, objects: Mapping[str, int]
) -> tuple[str, ObjectRef]:
    manifest = checkpoint.manifest
    _require(
        checkpoint.checkpoint == manifest.digest and objects.get(manifest.digest) == manifest.length
    )
    _require(0 < len(objects) <= 65536 and bool(operation))
    _require(all(_DIGEST.fullmatch(key) and size > 0 for key, size in objects.items()))
    path = destination_path(checkpoint.destination) + "/publications/" + quote(operation, safe="")
    return path, manifest


def _checkpointed(
    result: _Checkpointed | None,
    operation: str,
    checkpoint: CheckpointRef,
    objects: Mapping[str, int],
) -> CheckpointRef:
    manifest = checkpoint.manifest
    _require(
        result is not None
        and result.publish_id == operation
        and result.checkpoint_id == manifest.digest
        and result.manifest == _Manifest(manifest.digest[7:], manifest.length)
        and result.state == "checkpointed"
        and result.objects == len(objects) - 1
        and result.bytes == sum(objects.values()) - manifest.length,
        "publication.checkpoint_changed",
    )
    return CheckpointRef(
        checkpoint.destination, manifest.digest, manifest, operation, "acknowledged"
    )


def _settle(
    view: _Finalization,
    path: str,
    operation: str,
    checkpoint: CheckpointRef,
    objects: Mapping[str, int],
) -> CheckpointRef:
    """Hub's durable finalization job is the commit's only authority."""
    _require(
        view.operation == operation and view.status_url == path + "/finalization",
        "publication.checkpoint_changed",
    )
    if view.state in {"queued", "running"}:
        raise PublicationRefusal("publication.finalization_pending", status=503)
    if view.state == "failed":
        code = view.error.code if view.error is not None else ""
        raise PublicationUncommitted(
            code if _LABEL.fullmatch(code) else "publication.finalization_failed"
        )
    _require(view.state == "completed", "publication.checkpoint_changed")
    return _checkpointed(view.result, operation, checkpoint, objects)


def settle_checkpoint(
    body: bytes,
    *,
    operation: str,
    checkpoint: CheckpointRef,
    objects: Mapping[str, int],
) -> CheckpointRef:
    """Settle a sent finalize from the finalization status another reader fetched."""
    path, _ = _checkpoint_inputs(operation, checkpoint, objects)
    return _settle(_decode(body, _Finalization), path, operation, checkpoint, objects)


def finalization_absent(body: bytes) -> bool:
    """Whether another reader's 404 finalization read says no finalize ever reached Hub."""
    code = _decode(body, _Refused).error.code
    return code in {"publication.not_found", "publication.finalization_absent"}


def reconcile_checkpoint(
    client: PublicationClient,
    *,
    operation: str,
    checkpoint: CheckpointRef,
    objects: Mapping[str, int],
) -> CheckpointRef | None:
    """Settle a sent finalize from Hub's finalization status; None if it never arrived.

    Pending raises a transient refusal, a failed job raises ``PublicationUncommitted``.
    The public checkpoint read cannot decide this: a private checkpoint answers 404.
    """
    path, _ = _checkpoint_inputs(operation, checkpoint, objects)
    try:
        view = _decode(client.request("GET", path + "/finalization"), _Finalization)
    except PublicationRefusal as exc:
        if exc.status != 404:
            raise
        return None
    return _settle(view, path, operation, checkpoint, objects)


def upload_checkpoint(
    client: PublicationClient,
    *,
    operation: str,
    checkpoint: CheckpointRef,
    objects: Mapping[str, int],
    before_stage: Callable[[], None],
    before_write: Callable[[], None],
    push: Callable[[str, int, str], None],
    landed: Callable[[int, int], None],
    streams: int,
) -> CheckpointRef:
    """Stage one exact retained closure and commit it; native TensorFS performs each push.

    The caller freezes the closure and keeps it retained before invoking this function,
    and bounds ``streams``, the pushes its memory holds in flight. ``push`` receives only
    an admitted member and its signed grant.
    Opening, granting and pushing only stage bytes under the frozen operation, so
    ``before_stage`` fences them without a send marker; retrying them re-grants only
    objects the Hub does not hold. ``before_write`` records the one commit, finalize.
    ``landed`` receives the count and bytes of closure objects the Hub is known to hold.
    """
    path, manifest = _checkpoint_inputs(operation, checkpoint, objects)
    try:
        raw = client.request(
            "GET", destination_path(checkpoint.destination) + "/checkpoints/" + manifest.digest
        )
    except PublicationRefusal as exc:
        if exc.status != 404:
            raise
    else:
        # Another operation already made this exact checkpoint public.
        _require(
            len(raw) == manifest.length
            and "sha256:" + hashlib.sha256(raw).hexdigest() == manifest.digest
        )
        return CheckpointRef(
            checkpoint.destination, manifest.digest, manifest, operation, "observed_convergence"
        )
    before_stage()
    try:
        opened = _decode(
            client.request(
                "PUT",
                path,
                canonical_json.encode(
                    {
                        "objects": [
                            {"object_id": key, "length": size}
                            for key, size in sorted(objects.items())
                        ]
                    }
                ),
            ),
            _Opened,
        ).publication
    except PublicationRefusal as exc:
        # A finalize this operation sent may have closed the publication.
        if exc.status == 409:
            settled = reconcile_checkpoint(
                client, operation=operation, checkpoint=checkpoint, objects=objects
            )
            if settled is not None:
                return settled
        raise
    _require(opened.operation == operation and opened.state in {"open", "checkpointed"})
    if len(opened.objects) != len(objects):
        raise PublicationRefusal("publication.response_invalid")
    observed: set[str] = set()
    pending: list[str] = []
    for row in opened.objects:
        key = row.object_id
        _require(key in objects and key not in observed and row.length == objects[key])
        _require(row.state in {"accepted", "verifying", "claimed", "transferring"})
        observed.add(key)
        if row.state not in {"accepted", "verifying"}:
            pending.append(key)
    stage_objects(client, path, objects, pending, before_stage, push, landed, streams)
    before_write()
    try:
        answer = _decode(
            client.request(
                "POST",
                path + "/finalize",
                canonical_json.encode(
                    {"manifest_id": manifest.digest, "manifest_length": manifest.length}
                ),
            ),
            _Finalization,
        )
    except PublicationRefusal as exc:
        if exc.status in _RETRYABLE or exc.status >= 500:
            raise
        # Hub refused this commit request. Only an earlier durable job can still land.
        settled = reconcile_checkpoint(
            client, operation=operation, checkpoint=checkpoint, objects=objects
        )
        if settled is None:
            raise PublicationUncommitted(exc.code, status=exc.status) from exc
        return settled
    return _settle(answer, path, operation, checkpoint, objects)


class _Pace:
    """How many pushes run: from ``first``, doubled while the rate they land at improves.

    A level's rate is the pushes measured in flight on average since it began, times its
    landed pushes' bytes per second of push, once that many pushes started at the level
    have landed. The best level measured then holds, never above ``most``, the pushes the
    machine's memory holds.
    """

    def __init__(self, first: int, most: int) -> None:
        self.level, self.most = max(1, min(first, most)), most
        self.best = (0.0, self.level)
        self.settled = self.level >= most
        self.running = 0
        self.at = time.monotonic()
        self._begin()

    def _begin(self) -> None:
        self.since, self.busy = self.at, 0.0
        self.moved = self.spent = 0.0
        self.count = 0

    def _tick(self, change: int) -> None:
        now = time.monotonic()
        self.busy += self.running * (now - self.at)
        self.at, self.running = now, self.running + change

    def started(self) -> None:
        self._tick(1)

    def ended(self, level: int, landed: tuple[int, float] | None) -> None:
        """One push ended; ``landed`` is its bytes and seconds when it succeeded."""
        self._tick(-1)
        if self.settled or level != self.level or landed is None:
            return
        self.moved += landed[0]
        self.spent += landed[1]
        self.count += 1
        if self.count < level or not self.spent or self.at <= self.since:
            return
        rate = self.busy / (self.at - self.since) * self.moved / self.spent
        if rate > self.best[0]:
            self.best = (rate, level)
            if level < self.most:
                self.level = min(2 * level, self.most)
                self._begin()
                return
        self.level, self.settled = self.best[1], True


def stage_objects(
    client: PublicationClient,
    path: str,
    objects: Mapping[str, int],
    pending: list[str],
    before_stage: Callable[[], None],
    push: Callable[[str, int, str], None],
    landed: Callable[[int, int], None],
    streams: int,
) -> None:
    """Grant, push and verify the claimed members of one open publication the Hub lacks.

    ``landed`` receives the count and bytes of ``objects`` the Hub is known to hold;
    ``streams`` bounds the pushes in flight, and measured progress picks how many run.
    """
    pending = sorted(pending)
    held_count = len(objects) - len(pending)
    held_bytes = sum(objects.values()) - sum(objects[key] for key in pending)
    count_lock = threading.Lock()
    landed(held_count, held_bytes)

    def pushed(key: str, length: int, grant: str) -> float:
        nonlocal held_count, held_bytes
        started = time.monotonic()
        push(key, length, grant)
        seconds = time.monotonic() - started
        with count_lock:
            held_count, held_bytes = held_count + 1, held_bytes + length
            landed(held_count, held_bytes)
        return seconds

    def granted(selected: list[str]) -> list[tuple[str, int, str, float]]:
        """Validate one complete grants answer before spending any of its grants.

        Each grant carries the local time it goes cold: its lifetime on the Hub's clock,
        counted from before the request, less the second that clock truncates.
        """
        nonlocal held_count, held_bytes
        before_stage()
        asked = time.monotonic()
        body = canonical_json.encode({"object_ids": selected})
        answer = _decode(client.request("POST", path + "/grants", body), _Grants)
        grants, held = answer.grants or (), answer.held or ()
        rows = [(row.object_id, row.length) for row in grants]
        rows += [(row.object_id, row.length) for row in held]
        _require(
            len(rows) == len(selected)
            and {key for key, _ in rows} == set(selected)
            and all(length == objects[key] for key, length in rows)
        )
        _require(all(member.state in {"accepted", "verifying"} for member in held))
        uploads: list[tuple[str, int, str, float]] = []
        for grant in grants:
            key = grant.object_id
            required = {
                "if-none-match": "*",
                "x-amz-checksum-sha256": base64.b64encode(bytes.fromhex(key[7:])).decode("ascii"),
            }
            _require(grant.required_headers == required)
            expiry = asked + grant.expires_at_unix - answer.server_time_unix - 1
            _require(expiry > time.monotonic(), "publication.grant_expired")
            headers = (f"{name}: {value}" for name, value in required.items())
            uploads.append((key, objects[key], "\n".join([grant.url, *headers]), expiry))
        if held:
            with count_lock:
                held_count += len(held)
                held_bytes += sum(objects[member.object_id] for member in held)
                landed(held_count, held_bytes)
        return uploads

    def verified(selected: list[str]) -> list[tuple[str, int, str, float]]:
        """Hub verifies landed objects while pushes run, so its commit need not."""
        before_stage()
        body = canonical_json.encode({"object_ids": selected})
        _decode(client.request("POST", path + "/verify", body), _Opened)
        return []

    # A sliding window of pushes. One Hub thread fetches grants two windows ahead and
    # verifies landed objects, so a push never waits on the Hub. A grant gone cold before
    # its push starts is asked for again. A failed push stops new pushes; the Hub re-grants
    # only what is still missing when the same operation is retried. Every push finishes
    # before this scope exits, so the caller's retention outlives it.
    pace = _Pace(PUBLICATION_UPLOADS, streams)
    queued: deque[tuple[str, int, str, float]] = deque()
    cold: list[str] = []
    unverified: list[str] = []
    running: dict[Future[float], tuple[str, int, int]] = {}
    # The one Hub call in flight: grants, answering uploads, or a verification, none.
    asking: Future[list[tuple[str, int, str, float]]] | None = None
    failure: BaseException | None = None
    offset = 0
    with ThreadPoolExecutor(max_workers=streams) as pool, ThreadPoolExecutor(1) as hub:
        while True:
            if failure is None:
                unpushed = bool(cold or offset < len(pending))
                if asking is None and len(queued) < pace.level and unpushed:
                    batch = min(2 * pace.level, _PUBLICATION_WINDOW)
                    selected, cold = cold[:batch], cold[batch:]
                    fresh = pending[offset : offset + batch - len(selected)]
                    offset += len(fresh)
                    asking = hub.submit(granted, selected + fresh)
                elif (
                    asking is None
                    and unverified
                    and (
                        len(unverified) >= _PUBLICATION_WINDOW
                        or not (unpushed or queued or running)
                    )
                ):
                    selected = unverified[:_PUBLICATION_WINDOW]
                    del unverified[:_PUBLICATION_WINDOW]
                    asking = hub.submit(verified, selected)
                while queued and len(running) < pace.level:
                    key, length, grant, expiry = queued.popleft()
                    if time.monotonic() >= expiry:
                        cold.append(key)
                        continue
                    before_stage()
                    running[pool.submit(pushed, key, length, grant)] = key, length, pace.level
                    pace.started()
            # `Future` is invariant, so one wait set over pushes and Hub calls needs `Any`.
            waiting: set[Future[Any]] = set(running)
            if asking is not None:
                waiting.add(asking)
            if not waiting:
                if failure is not None:
                    raise failure
                if cold or queued or unverified or offset < len(pending):
                    continue
                break
            done, _ = wait(waiting, return_when=FIRST_COMPLETED)
            for future in done:
                error = future.exception()
                if future is asking:
                    asking = None
                    if error is None:
                        queued.extend(future.result())
                else:
                    key, length, level = running.pop(future)
                    pace.ended(level, None if error else (length, future.result()))
                    if error is None:
                        unverified.append(key)
                if failure is None:
                    failure = error


def retain_objects(
    client: PublicationClient,
    *,
    destination: str,
    operation: str,
    objects: Mapping[str, int],
    before_stage: Callable[[], None],
    push: Callable[[str, int, str], None],
    streams: int,
) -> None:
    """Put exact objects in Hub custody under an open publication that never finalizes.

    Returns once Hub has verified every one. A later checkpoint naming them claims them
    without their bytes; ``abandon_publication`` releases this hold after it commits.
    """
    _require(0 < len(objects) <= _PUBLICATION_OBJECTS and bool(operation))
    _require(all(_DIGEST.fullmatch(key) and size > 0 for key, size in objects.items()))
    path = destination_path(destination) + "/publications/" + quote(operation, safe="")
    before_stage()
    opened = _decode(
        client.request(
            "PUT",
            path,
            canonical_json.encode(
                {
                    "objects": [
                        {"object_id": key, "length": size} for key, size in sorted(objects.items())
                    ]
                }
            ),
        ),
        _Opened,
    ).publication
    _require(
        opened.operation == operation
        and opened.state == "open"
        and len(opened.objects) == len(objects),
        "publication.response_invalid",
    )
    pending: list[str] = []
    for row in opened.objects:
        _require(row.object_id in objects and row.length == objects[row.object_id])
        _require(row.state in {"accepted", "verifying", "claimed", "transferring"})
        if row.state not in {"accepted", "verifying"}:
            pending.append(row.object_id)
    stage_objects(client, path, objects, pending, before_stage, push, lambda _n, _b: None, streams)
    keys = sorted(objects)
    for offset in range(0, len(keys), _PUBLICATION_WINDOW):
        selected = keys[offset : offset + _PUBLICATION_WINDOW]
        before_stage()
        verified = _decode(
            client.request(
                "POST", path + "/verify", canonical_json.encode({"object_ids": selected})
            ),
            _Opened,
        ).publication
        accepted = {row.object_id for row in verified.objects if row.state == "accepted"}
        _require(accepted >= set(selected), "publication.custody_unverified")


def abandon_publication(client: PublicationClient, *, destination: str, operation: str) -> None:
    """Release an open publication's holds; one already gone is already released."""
    try:
        client.request(
            "DELETE",
            destination_path(destination) + "/publications/" + quote(operation, safe=""),
        )
    except PublicationRefusal as exc:
        if exc.status != 404:
            raise


def attach_assessment(
    client: PublicationClient,
    checkpoint: CheckpointRef,
    report: bytes,
    verdict: str,
    before_write: Callable[[], None],
) -> AssessmentRef:
    """Reconcile the exact report after the machine has verified its associations."""
    _require(0 < len(report) <= _MAX_METADATA and verdict in {"pass", "fail", "indeterminate"})
    digest = "sha256:" + hashlib.sha256(report).hexdigest()
    path = (
        destination_path(checkpoint.destination)
        + "/checkpoints/"
        + checkpoint.checkpoint
        + "/assessments/"
        + digest
    )

    def result(observation: str) -> AssessmentRef:
        return AssessmentRef(
            checkpoint.destination,
            checkpoint.checkpoint,
            ObjectRef(digest, len(report)),
            verdict,
            observation,
        )

    try:
        existing = client.request("GET", path)
    except PublicationRefusal as exc:
        if exc.status != 404:
            raise
    else:
        _require(existing == report, "assessment.readback_changed")
        return result("observed_convergence")
    before_write()
    acknowledgement = _decode(
        client.request("PUT", path, report), _Acknowledgement, "assessment.acknowledgement_changed"
    )
    value = acknowledgement.assessment
    _require(
        value.checkpoint_id == checkpoint.checkpoint
        and value.scope == "publisher_assessment"
        and value.report == _Report(digest, len(report))
        and value.verdict == verdict
        and acknowledgement.publisher_reported_verdict == verdict,
        "assessment.acknowledgement_changed",
    )
    _require(client.request("GET", path) == report, "assessment.readback_changed")
    return result("acknowledged")
