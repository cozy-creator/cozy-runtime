"""Publication metadata for an accepted machine execution.

Runtime's call journal freezes intent and records sends. TensorFS retains and moves
the bytes. This adapter holds neither a second journal nor account credentials.
"""

from __future__ import annotations

import base64
import hashlib
import json
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
from typing import Any, Protocol
from urllib.parse import quote

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.author.publication import AssessmentRef, CheckpointRef, ReleaseReceipt
from cozy_runtime.internal import egress
from cozy_runtime.internal.config import parse_tensorhub_origin

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


@dataclass(frozen=True)
class PublicationAuthority:
    origin: str
    worker_id: str
    worker_token: str = field(repr=False)

    def client(self, authorization_id: str) -> MachinePublicationClient:
        return MachinePublicationClient(
            self.origin,
            authorization_id,
            ssl.create_default_context(),
            worker_id=self.worker_id,
            worker_token=self.worker_token,
        )


def _object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise PublicationRefusal("publication.response_invalid") from exc
    if not isinstance(value, dict):
        raise PublicationRefusal("publication.response_invalid")
    return value


def _require(condition: bool, code: str = "publication.response_changed") -> None:
    if not condition:
        raise PublicationRefusal(code)


def destination_path(destination: str) -> str:
    parts = destination.split("/")
    _require(
        len(parts) == 2 and parts[0] != "local" and all(_NAME.fullmatch(p) for p in parts),
        "publication.destination_invalid",
    )
    return "/v1/models/" + destination


# A job root's weights output access spelled ``model://org/name`` publishes that output
# as a checkpoint of the repository, under the root's publication authorization.
DESTINATION_SCHEME = "model://"


def output_destination(urls: Iterable[str]) -> str:
    """The one repository a root's output grants name, or "" when none names one."""
    named = {
        url.removeprefix(DESTINATION_SCHEME) for url in urls if url.startswith(DESTINATION_SCHEME)
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
        clock: Callable[[], float] = time.time,
    ) -> None:
        parsed = parse_tensorhub_origin(origin)
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
            bool(worker_id)
            and len(worker_id) <= 256
            and all(33 <= ord(c) <= 126 for c in worker_id)
            and re.fullmatch(r"[A-Za-z0-9_-]{43}", worker_token) is not None,
            "publication.worker_authority_invalid",
        )
        self.worker_id, self.worker_token = worker_id, worker_token
        self.origin = parsed
        self.authorization_id, self.context, self.clock = authorization_id, context, clock
        self._token, self._expires = "", 0.0
        self._lock = threading.Lock()

    def _exchange(
        self, method: str, path: str, body: bytes | None, token: str = ""
    ) -> tuple[int, bytes]:
        headers = {
            "Accept": "application/json",
            "X-Cozy-Worker-ID": self.worker_id,
            "X-Cozy-Worker-Token": self.worker_token,
        }
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
        value = _object(raw)
        token, expires = value.get("token"), value.get("expires_at")
        if not isinstance(token, str) or not isinstance(expires, str):
            raise PublicationRefusal("publication.authority_response_invalid")
        _require(0 < len(token) <= 16 << 10)
        try:
            stamp = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            _require(stamp.tzinfo is not None, "publication.authority_response_invalid")
            deadline = stamp.timestamp()
        except (ValueError, TypeError) as exc:
            raise PublicationRefusal("publication.authority_response_invalid") from exc
        _require(deadline > self.clock(), "publication.authority_expired")
        self._token, self._expires = token, deadline

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
    card = _object(client.request("GET", destination_path(destination)))
    rows = card.get("releases")
    if not isinstance(rows, list):
        raise PublicationRefusal("publication.response_invalid")
    matches = [row for row in rows if isinstance(row, dict) and row.get("release") == release]
    _require(len(matches) <= 1)
    if not matches:
        return 0, {}
    row = matches[0]
    revision = row.get("revision")
    if type(revision) is not int or not 1 <= revision < (1 << 53) - 1:
        raise PublicationRefusal("publication.response_invalid")
    _require(not row.get("yanked", False))
    lanes: dict[str, str] = {}
    _require(isinstance(row.get("lanes"), list))
    for lane in row["lanes"]:
        _require(isinstance(lane, dict))
        name, manifest = lane.get("lane"), lane.get("manifest_id")
        _require(isinstance(name, str) and _LABEL.fullmatch(name) is not None)
        _require(isinstance(manifest, str) and _DIGEST.fullmatch(manifest) is not None)
        _require(name not in lanes)
        lanes[name] = manifest
    _require(bool(lanes))
    return revision, lanes


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
        _require(type(expected_revision) is int and 0 <= expected_revision < (1 << 53) - 1)
    requested: dict[str, str] = {}
    for name, checkpoint in lanes.items():
        _require(_LABEL.fullmatch(name) is not None)
        _require(
            checkpoint.destination == destination
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
    return canonical_json.encode(
        {
            "destination": destination,
            "release": release,
            "set_lanes": requested,
            "baseline_revision": revision,
            "baseline": baseline,
            "desired": baseline | requested,
        }
    )


def apply_release(
    client: PublicationClient, frozen: bytes, *, sent: bool, before_write: Callable[[], None]
) -> ReleaseReceipt:
    intent = _object(frozen)
    destination, release = intent["destination"], intent["release"]
    baseline, desired = intent["baseline"], intent["desired"]
    expected = intent["baseline_revision"]
    _require(baseline | intent["set_lanes"] == desired, "publication.intent_changed")
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
    updated = _object(
        client.request(
            "POST",
            destination_path(destination) + "/releases/" + quote(release, safe=""),
            canonical_json.encode(
                {
                    "expected_revision": expected,
                    "set_lanes": intent["set_lanes"],
                    "remove_lanes": [],
                }
            ),
        )
    )
    _require(
        updated.get("release") == release
        and updated.get("revision") == expected + 1
        and not updated.get("yanked", False),
        "publication.outcome_unknown",
    )
    lane_rows = updated.get("lanes")
    if not isinstance(lane_rows, list) or not all(isinstance(row, dict) for row in lane_rows):
        raise PublicationRefusal("publication.outcome_unknown")
    actual = {row.get("lane"): row.get("checkpoint_id") for row in lane_rows}
    _require(len(actual) == len(lane_rows) and actual == desired, "publication.outcome_unknown")
    return ReleaseReceipt(destination, release, expected + 1, desired, "acknowledged")


def _checkpoint_inputs(
    operation: str, checkpoint: CheckpointRef, objects: Mapping[str, int]
) -> tuple[str, ObjectRef]:
    manifest = checkpoint.manifest
    _require(
        checkpoint.checkpoint == manifest.digest and objects.get(manifest.digest) == manifest.length
    )
    _require(0 < len(objects) <= 65536 and bool(operation))
    _require(
        all(
            _DIGEST.fullmatch(key) and type(size) is int and size > 0
            for key, size in objects.items()
        )
    )
    path = destination_path(checkpoint.destination) + "/publications/" + quote(operation, safe="")
    return path, manifest


def _checkpointed(
    result: object, operation: str, checkpoint: CheckpointRef, objects: Mapping[str, int]
) -> CheckpointRef:
    manifest = checkpoint.manifest
    _require(
        isinstance(result, dict)
        and result.get("publish_id") == operation
        and result.get("checkpoint_id") == manifest.digest
        and result.get("manifest") == {"sha256": manifest.digest[7:], "length": manifest.length}
        and result.get("state") == "checkpointed"
        and result.get("objects") == len(objects) - 1
        and result.get("bytes") == sum(objects.values()) - manifest.length,
        "publication.checkpoint_changed",
    )
    return CheckpointRef(
        checkpoint.destination, manifest.digest, manifest, operation, "acknowledged"
    )


def _settle(
    view: dict[str, Any],
    path: str,
    operation: str,
    checkpoint: CheckpointRef,
    objects: Mapping[str, int],
) -> CheckpointRef:
    """Hub's durable finalization job is the commit's only authority."""
    _require(
        view.get("operation") == operation and view.get("status_url") == path + "/finalization",
        "publication.checkpoint_changed",
    )
    state = view.get("state")
    if state in {"queued", "running"}:
        raise PublicationRefusal("publication.finalization_pending", status=503)
    if state == "failed":
        error = view.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        raise PublicationUncommitted(
            code
            if isinstance(code, str) and _LABEL.fullmatch(code)
            else "publication.finalization_failed"
        )
    _require(state == "completed", "publication.checkpoint_changed")
    return _checkpointed(view.get("result"), operation, checkpoint, objects)


#: Hub codes of the owner's finalization read meaning no finalize ever reached it.
FINALIZATION_ABSENT = frozenset({"publication.not_found", "publication.finalization_absent"})


def settle_checkpoint(
    view: Mapping[str, Any],
    *,
    operation: str,
    checkpoint: CheckpointRef,
    objects: Mapping[str, int],
) -> CheckpointRef:
    """Settle a sent finalize from a finalization document another reader fetched."""
    path, _ = _checkpoint_inputs(operation, checkpoint, objects)
    return _settle(dict(view), path, operation, checkpoint, objects)


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
        view = _object(client.request("GET", path + "/finalization"))
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
        opened = _object(
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
            )
        )
    except PublicationRefusal as exc:
        # A finalize this operation sent may have closed the publication.
        if exc.status == 409:
            settled = reconcile_checkpoint(
                client, operation=operation, checkpoint=checkpoint, objects=objects
            )
            if settled is not None:
                return settled
        raise
    publication = opened.get("publication")
    if not isinstance(publication, dict):
        raise PublicationRefusal("publication.response_invalid")
    _require(
        publication.get("operation") == operation
        and publication.get("state") in {"open", "checkpointed"}
    )
    rows = publication.get("objects")
    if not isinstance(rows, list) or len(rows) != len(objects):
        raise PublicationRefusal("publication.response_invalid")
    observed: dict[str, int] = {}
    pending: list[str] = []
    for row in rows:
        key, length, state = row.get("object_id"), row.get("length"), row.get("state")
        _require(key in objects and key not in observed and length == objects[key])
        _require(state in {"accepted", "verifying", "claimed", "transferring"})
        observed[key] = length
        if state not in {"accepted", "verifying"}:
            pending.append(key)
    stage_objects(client, path, objects, pending, before_stage, push, landed, streams)
    before_write()
    try:
        answer = _object(
            client.request(
                "POST",
                path + "/finalize",
                canonical_json.encode(
                    {"manifest_id": manifest.digest, "manifest_length": manifest.length}
                ),
            )
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
    if "operation" in answer:
        return _settle(answer, path, operation, checkpoint, objects)
    return _checkpointed(answer, operation, checkpoint, objects)


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
        answer = _object(client.request("POST", path + "/grants", body))
        grants, held = answer.get("grants"), answer.get("held")
        # Go's empty slices may be JSON null; they still represent zero rows.
        grants = [] if grants is None else grants
        held = [] if held is None else held
        if not isinstance(grants, list) or not isinstance(held, list):
            raise PublicationRefusal("publication.response_invalid")
        seen: set[str] = set()
        for item in grants + held:
            _require(isinstance(item, dict))
            key = item.get("object_id")
            _require(isinstance(key, str) and key in selected and key not in seen)
            _require(item.get("length") == objects[key])
            seen.add(key)
        _require(seen == set(selected))
        for item in held:
            _require(item.get("state") in {"accepted", "verifying"})
        uploads: list[tuple[str, int, str, float]] = []
        for item in grants:
            key = item["object_id"]
            required = {
                "if-none-match": "*",
                "x-amz-checksum-sha256": base64.b64encode(bytes.fromhex(key[7:])).decode("ascii"),
            }
            _require(item.get("required_headers") == required and isinstance(item.get("url"), str))
            now, expires = answer.get("server_time_unix"), item.get("expires_at_unix")
            _require(type(now) is int and type(expires) is int)
            expiry = asked + expires - now - 1
            _require(expiry > time.monotonic(), "publication.grant_expired")
            headers = (f"{name}: {value}" for name, value in required.items())
            uploads.append((key, objects[key], "\n".join([item["url"], *headers]), expiry))
        if held:
            with count_lock:
                held_count += len(held)
                held_bytes += sum(objects[item["object_id"]] for item in held)
                landed(held_count, held_bytes)
        return uploads

    def verified(selected: list[str]) -> list[tuple[str, int, str, float]]:
        """Hub verifies landed objects while pushes run, so its commit need not."""
        before_stage()
        body = canonical_json.encode({"object_ids": selected})
        answer = _object(client.request("POST", path + "/verify", body)).get("publication")
        if not isinstance(answer, dict):
            raise PublicationRefusal("publication.response_invalid")
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
    _require(
        all(
            _DIGEST.fullmatch(key) and type(size) is int and size > 0
            for key, size in objects.items()
        )
    )
    path = destination_path(destination) + "/publications/" + quote(operation, safe="")
    before_stage()
    opened = _object(
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
        )
    )
    publication = opened.get("publication")
    if not isinstance(publication, dict) or not isinstance(publication.get("objects"), list):
        raise PublicationRefusal("publication.response_invalid")
    _require(
        publication.get("operation") == operation
        and publication.get("state") == "open"
        and len(publication["objects"]) == len(objects),
        "publication.response_invalid",
    )
    pending: list[str] = []
    for row in publication["objects"]:
        _require(isinstance(row, dict), "publication.response_invalid")
        key, state = row.get("object_id"), row.get("state")
        _require(key in objects and row.get("length") == objects[key])
        _require(state in {"accepted", "verifying", "claimed", "transferring"})
        if state not in {"accepted", "verifying"}:
            pending.append(key)
    stage_objects(client, path, objects, pending, before_stage, push, lambda _n, _b: None, streams)
    keys = sorted(objects)
    for offset in range(0, len(keys), _PUBLICATION_WINDOW):
        selected = keys[offset : offset + _PUBLICATION_WINDOW]
        before_stage()
        verified = _object(
            client.request(
                "POST", path + "/verify", canonical_json.encode({"object_ids": selected})
            )
        ).get("publication")
        if not isinstance(verified, dict) or not isinstance(verified.get("objects"), list):
            raise PublicationRefusal("publication.response_invalid")
        accepted = {
            row.get("object_id")
            for row in verified["objects"]
            if isinstance(row, dict) and row.get("state") == "accepted"
        }
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
    acknowledgement = _object(client.request("PUT", path, report))
    value = acknowledgement.get("assessment", {})
    _require(
        value.get("checkpoint_id") == checkpoint.checkpoint
        and value.get("scope") == "publisher_assessment"
        and value.get("report") == {"digest": digest, "length": len(report)}
        and value.get("verdict") == verdict
        and acknowledgement.get("publisher_reported_verdict") == verdict,
        "assessment.acknowledgement_changed",
    )
    _require(client.request("GET", path) == report, "assessment.readback_changed")
    return result("acknowledged")
