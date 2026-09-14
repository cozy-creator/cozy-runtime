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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


class PublicationRefusal(Exception):
    def __init__(self, code: str, *, status: int = 409):
        super().__init__(code)
        self.code, self.status = code, status


class PublicationClient(Protocol):
    def request(self, method: str, path: str, body: bytes | None = None) -> bytes: ...


@dataclass(frozen=True)
class PublicationTLS:
    origin: str
    certificate: Path
    private_key: Path
    ca_der: bytes = b""

    def client(self, authorization_id: str) -> MachinePublicationClient:
        context = ssl.create_default_context()
        if self.ca_der:
            context.load_verify_locations(cadata=self.ca_der)
        context.load_cert_chain(self.certificate, self.private_key)
        return MachinePublicationClient(self.origin, authorization_id, context)


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


class MachinePublicationClient:
    """One root's bounded AuthKit grant on the operator-configured mTLS edge.

    The SSL context contains only the existing worker certificate. Python job
    executors never receive this instance, its token, or its certificate key.
    HTTP redirects are not followed and each connection is closed after use.
    """

    def __init__(
        self,
        origin: str,
        authorization_id: str,
        context: ssl.SSLContext,
        *,
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
        self.origin = parsed
        self.authorization_id, self.context, self.clock = authorization_id, context, clock
        self._token, self._expires = "", 0.0
        self._lock = threading.Lock()

    def _exchange(
        self, method: str, path: str, body: bytes | None, token: str = ""
    ) -> tuple[int, bytes]:
        headers = {"Accept": "application/json"}
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
            method in {"GET", "POST", "PUT"}
            and path.startswith("/v1/models/")
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


def upload_checkpoint(
    client: PublicationClient,
    *,
    operation: str,
    checkpoint: CheckpointRef,
    objects: Mapping[str, int],
    before_write: Callable[[], None],
    push: Callable[[str, int, str], None],
) -> CheckpointRef:
    """Reconcile one exact retained closure; native TensorFS performs each push.

    The caller freezes the closure and holds its native lease before invoking
    this function. ``push`` receives only an admitted member and its signed grant.
    """
    destination = destination_path(checkpoint.destination)
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
    try:
        raw = client.request("GET", destination + "/checkpoints/" + manifest.digest)
    except PublicationRefusal as exc:
        if exc.status != 404:
            raise
    else:
        _require(
            len(raw) == manifest.length
            and "sha256:" + hashlib.sha256(raw).hexdigest() == manifest.digest
        )
        return CheckpointRef(
            checkpoint.destination, manifest.digest, manifest, operation, "observed_convergence"
        )

    path = destination + "/publications/" + quote(operation, safe="")
    before_write()
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
    for key in sorted(pending):
        before_write()
        granted = _object(
            client.request("POST", path + "/grants", canonical_json.encode({"object_ids": [key]}))
        )
        grants, held = granted.get("grants"), granted.get("held")
        # Go's empty slices may be JSON null; they still represent zero rows.
        grants = [] if grants is None else grants
        held = [] if held is None else held
        if not isinstance(grants, list) or not isinstance(held, list):
            raise PublicationRefusal("publication.response_invalid")
        _require(len(grants) + len(held) == 1)
        item = (grants + held)[0]
        _require(item.get("object_id") == key and item.get("length") == objects[key])
        if held:
            _require(item.get("state") in {"accepted", "verifying"})
            continue
        headers = item.get("required_headers")
        required = {
            "if-none-match": "*",
            "x-amz-checksum-sha256": base64.b64encode(bytes.fromhex(key[7:])).decode("ascii"),
        }
        _require(headers == required and isinstance(item.get("url"), str))
        now, expires = granted.get("server_time_unix"), item.get("expires_at_unix")
        _require(
            type(now) is int and type(expires) is int and expires > now, "publication.grant_expired"
        )
        before_write()
        push(
            key,
            objects[key],
            "\n".join([item["url"], *(f"{name}: {value}" for name, value in required.items())]),
        )
    before_write()
    result = _object(
        client.request(
            "POST",
            path + "/finalize",
            canonical_json.encode(
                {"manifest_id": manifest.digest, "manifest_length": manifest.length}
            ),
        )
    )
    _require(
        result.get("publish_id") == operation
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
