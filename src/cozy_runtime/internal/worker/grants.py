"""DeliveryGrant reads and writes, and the CLOSED hub-call authorization table.

The grant is refreshable AUTHORITY, never meaning: it is outside `exec_spec_digest`, and a
refresh may re-derive exactly `{credential, expires_at_unix, urls}` and nothing else
(worker-protocol/01 §5). This module is where that authority is SPENT — the only place the
supervisor reads an input or writes an output — and every spend goes through
`authorize()` first.

**The table is closed** (§3.1). Method + full-path regex + enumerated params, and a refusal
is a SECURITY EVENT, not a 404: it is counted, journaled by the caller and reported as a
`Fault`, because a worker asking for something it was never granted is a fact an operator
must see. Nothing widens the table at runtime; a new call site is a new reviewed row.

Transport at launch is `file://` only, and that is a statement rather than a placeholder:
the executor has no outbound network at all, and the supervisor's HTTPS fetch path (egress
allowlist, SSRF fail-closed per redirect hop, presign re-plan) is cr-012's. An `https://`
grant URL here refuses typed instead of being half-implemented.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cozy_runtime.protocol import worker_pb2 as pb


class GrantRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class HubCall:
    """One reviewed row of the closed table."""

    name: str
    method: str
    path: str
    """A full-path regex — never a prefix, never a host-only rule."""
    params: tuple[str, ...]
    why: str


#: The COMPLETE set of calls this supervisor may make. Read it as the security boundary it
#: is: anything not spelled here cannot be dialled, whoever asks.
HUB_CALLS: tuple[HubCall, ...] = (
    HubCall(
        "grant_input_read",
        "GET",
        r"^file://(?P<path>/[\w./-]+)$",
        ("input_id",),
        "read one granted input asset (the payload rides the grant as an input)",
    ),
    HubCall(
        "grant_output_write",
        "PUT",
        r"^file://(?P<path>/[\w./-]+)$",
        ("output_id", "length"),
        "write one granted output destination under an exact output grant",
    ),
)

CALLS: dict[str, HubCall] = {c.name: c for c in HUB_CALLS}


@dataclass(slots=True)
class Authorizer:
    """The gate every grant spend passes. Counts refusals — they are security events."""

    refusals: list[dict[str, str]] = field(default_factory=list)
    allowed: int = 0

    def authorize(self, name: str, method: str, url: str, params: dict[str, Any]) -> None:
        row = CALLS.get(name)
        if row is None:
            self._refuse(name, method, url, f"{name!r} is not in the closed hub-call table")
        assert row is not None
        if method != row.method:
            self._refuse(name, method, url, f"{name} is {row.method}, not {method}")
        if re.match(row.path, url) is None:
            self._refuse(name, method, url, f"{url!r} does not match {row.path}")
        extra = sorted(set(params) - set(row.params))
        if extra:
            self._refuse(name, method, url, f"unenumerated params {', '.join(extra)}")
        self.allowed += 1

    def _refuse(self, name: str, method: str, url: str, why: str) -> None:
        event = {"call": name, "method": method, "url": url[:200], "why": why}
        self.refusals.append(event)
        raise GrantRefusal(
            "hub_call_unauthorized",
            f"SECURITY EVENT — {method} {url[:120]} as {name!r}: {why}. The supervisor's "
            "hub-call table is closed; a refusal is not a 404",
        )


# --------------------------------------------------------------------------- grant I/O


def _local(url: str, base: str) -> Path:
    absolute = url if "://" in url else base.rstrip("/") + "/" + url.lstrip("/")
    parsed = urlparse(absolute)
    if parsed.scheme != "file":
        raise GrantRefusal(
            "grant_transport",
            f"{parsed.scheme or 'relative'}:// grant URLs need the supervisor HTTPS fetch "
            "path (egress allowlist, SSRF fail-closed per hop, presign re-plan), which is "
            "cr-012 — at launch this worker spends grants over file:// only",
        )
    return Path(parsed.path)


def expired(grant: pb.DeliveryGrant, now: float | None = None) -> bool:
    at = time.time() if now is None else now
    return bool(grant.expires_at_unix) and grant.expires_at_unix < at


def read_input(
    grant: pb.DeliveryGrant, input_id: str, auth: Authorizer
) -> bytes:
    """One granted input, verified against the grant's own digest and length."""
    if expired(grant):
        raise GrantRefusal(
            "grant_expired",
            f"the delivery grant expired at {grant.expires_at_unix}; a resolver fallback is "
            "a NEW attempt, never a patch to this one",
        )
    for entry in grant.inputs:
        if entry.input_id != input_id:
            continue
        url = entry.url if "://" in entry.url else grant.file_base_url.rstrip("/") + "/" + entry.url
        auth.authorize("grant_input_read", "GET", url, {"input_id": input_id})
        data = _local(entry.url, grant.file_base_url).read_bytes()
        if entry.length and len(data) != entry.length:
            raise GrantRefusal(
                "input_length_mismatch",
                f"{input_id}: grant declares {entry.length} B, read {len(data)} B",
            )
        if entry.digest and hashlib.sha256(data).digest() != entry.digest:
            raise GrantRefusal(
                "input_digest_mismatch",
                f"{input_id}: content digest disagrees with the grant — a signed grant never "
                "bypasses the local bytes check",
            )
        return data
    raise GrantRefusal(
        "input_absent",
        f"the grant carries no input {input_id!r} (it has "
        f"{', '.join(e.input_id for e in grant.inputs) or 'none'})",
    )


def write_output(
    grant: pb.DeliveryGrant, output_id: str, data: bytes, auth: Authorizer
) -> pb.OutputEntry:
    """Write ONE output under the EXACT destination the grant names.

    A stale or wrong grant cannot write: the destination must be in this grant, the grant
    must be live, and the bytes must fit the destination's declared cap. Writing produces a
    manifest entry and NOTHING else — the runtime may write under an exact output grant and
    cannot authorize readers, publish media, or decide a winning attempt.
    """
    if expired(grant):
        raise GrantRefusal("grant_expired", "the output grant expired before the write")
    for destination in grant.outputs:
        if destination.output_id != output_id:
            continue
        url = (
            destination.url
            if "://" in destination.url
            else grant.file_base_url.rstrip("/") + "/" + destination.url
        )
        auth.authorize(
            "grant_output_write", "PUT", url, {"output_id": output_id, "length": len(data)}
        )
        if destination.max_bytes and len(data) > destination.max_bytes:
            raise GrantRefusal(
                "output_too_large",
                f"{output_id}: {len(data)} B over the granted {destination.max_bytes} B",
            )
        path = _local(destination.url, grant.file_base_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return pb.OutputEntry(
            output_id=output_id,
            digest=hashlib.sha256(data).digest(),
            length=len(data),
            mime_type="application/octet-stream",
        )
    raise GrantRefusal(
        "destination_absent",
        f"no granted destination {output_id!r} (the grant names "
        f"{', '.join(d.output_id for d in grant.outputs) or 'none'}) — an output the "
        "coordinator did not grant is unwritable, however the attempt produced it",
    )


def abort_outputs(spool: Path) -> int:
    """Abort an attempt's uncommitted outputs. Bytes written under the grant are the
    coordinator's problem only once a terminal claims them; the spool is ours."""
    removed = 0
    for path in sorted(spool.rglob("*")):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed
