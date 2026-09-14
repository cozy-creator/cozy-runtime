"""cr-012 — the ONE bounded stream reader, and the fail-closed egress boundary.

The hazard this module exists against is not "the SSRF check was wrong", it is "there were
two downloaders and only one had the check". cr-090 resolved it by subtraction: this
runtime downloads no artifact bytes — the transport and its address predicate live in
tensorfs (`tensorfs-core/src/transport/policy.rs`, the port of `blocked()` below: same six
classes, IPv6 fail-closed, every redirect hop re-checked, connects to the checked
addresses). What remains here is the UPLOAD half and its check: `put_from` has exactly one
caller, the worker's control-plane-minted output grant, and it moves to tensorfs under
th-132, taking `blocked()` with it. The `one-address-predicate` fence in
`checks/architecture.py` reddens if any other urllib/http.client chain appears beside this
module — a second network byte mover in this runtime is the defect above, restated.
The bounded machine-publication metadata request also lives here: it uses the
operator-configured HTTPS control origin and mTLS context, and never follows redirects.

**A stream may not exceed its declared size.** `copy_bounded` stops at the declared length
and refuses before writing the over-bound chunk; it never buffers the body in memory. A body that
UNDER-runs is equally a refusal — a truncated read that returns short bytes is how a digest
check gets skipped by an empty file.

**The allowlist is a deployment-wide DECLARATION and the resolution is fail-closed.** A
host absent from the allowlist refuses. A host that resolves to a private, loopback,
link-local, multicast or reserved address refuses — every address it resolves to, not the
first, because a name with one public and one 169.254.169.254 answer is the metadata-service
bypass with an extra step.

**A redirect never replaces the exact granted URL.** `put_from` refuses one outright —
after validating the target, so a metadata/private-address redirect is a precise security
refusal rather than a generic one. Following a redirect inside an HTTP library would put
the second request outside this module's front door, which is the whole bug.

**The DNS-rebinding limit is stated, not papered over.** Validation happens at resolve time
and the connection is made by hostname, so a name that answers publicly during the check and
privately during the connect is not closed by this design. Every answer is checked and the
window is one connect; closing it needs connect-to-pinned-IP with SNI/Host preserved, which
is named here as the work it is rather than implied by silence.

**An inbound `local_path` is caller-controlled wire data.** There is no field on any type
here that could carry one: `asset_dec_hook` accepts a bare ref STRING and nothing else, so
a request cannot hand itself a path. That is stronger than nulling one and is why nothing
below nulls anything.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import ssl
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import BinaryIO
from urllib.parse import urljoin, urlparse

from cozy_runtime.internal import liveness
from cozy_runtime.internal.config import TensorhubOrigin

#: How much of a body one read() pulls. Small enough that a cap refusal happens promptly.
CHUNK = 1 << 20

#: The silence bound on ONE socket operation — a read or a write that moves nothing — and
#: never on a transfer: a 99 GB PUT keeps its socket busy for as long as the bytes take.
#: DERIVED, not chosen: it is the runtime's one noise floor (`liveness.noise_floor`), the
#: smallest silence any transport ledger acts on (tensorfs's included), so an egress socket
#: is given exactly the patience a pull stream is given before either is called stalled.
SILENCE_SECONDS = liveness.noise_floor()

#: How many times one object is re-sent. The PUT is idempotent by construction — one
#: immutable key, `If-None-Match: *` — so a retry costs one object's bytes and never its
#: identity: a 412 on a re-send of a body the first send completed is the first send landing.
#: The same shape as the transport's per-object re-ask budget, for the same reason.
PUT_ATTEMPTS = 4

#: The store answers a re-send is worth attempting for. Everything else is a fact about the
#: request or the grant, and asking again about those is a lie about time.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: An `If-None-Match: *` write that lands on an existing immutable key answers with this.
HTTP_PRECONDITION_FAILED = 412


#: The address classes nothing in this runtime connects to. ONE list, one remaining caller
#: (`EgressPolicy.check_addresses`, the upload lane): the download half of this predicate
#: moved to tensorfs `transport/policy.rs` with the transport (cr-090), and the
#: `one-address-predicate` fence in `checks/architecture.py` reddens if a second chain
#: appears here.
_CLASSES: tuple[tuple[str, str], ...] = (
    ("private", "is_private"),
    ("loopback", "is_loopback"),
    ("link-local", "is_link_local"),
    ("multicast", "is_multicast"),
    ("reserved", "is_reserved"),
    ("unspecified", "is_unspecified"),
)


def blocked(address: str) -> str:
    """Why `address` is not fetchable, or `""` when it is a public routable address."""
    parsed = ipaddress.ip_address(address)
    named = [name for name, attribute in _CLASSES if getattr(parsed, attribute)]
    return ", ".join(named)


def resolve(host: str, port: int | None = None) -> list[str]:
    """EVERY answer for `host`. One resolver call for every caller that then asks
    `blocked()` about what came back — a caller that resolves for itself can check a
    different set of answers than it connects over."""
    answers = socket.getaddrinfo(host, port or None, proto=socket.IPPROTO_TCP)
    return [str(answer[4][0]) for answer in answers]


class EgressRefusal(Exception):
    """A typed fetch refusal. Every one fires before the bytes are used for anything."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def request_control_metadata(
    origin: TensorhubOrigin,
    context: ssl.SSLContext,
    method: str,
    path: str,
    body: bytes | None,
    headers: Mapping[str, str],
    *,
    limit: int,
) -> tuple[int, bytes]:
    """One bounded request to the operator's fixed HTTPS control authority.

    The origin comes from the host configuration, never a Python job or a
    redirect. Unlike a presigned data URL, an operator may intentionally host
    this authenticated control service on a private address. Model and artifact
    bytes still move through TensorFS; this function accepts only small metadata.
    It retries nothing: the execution journal must reconcile ambiguous writes.
    """
    import http.client

    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise EgressRefusal("control_tls_required", "control TLS verification is required")
    if not 0 < limit <= 8 << 20 or (body is not None and len(body) > limit):
        raise EgressRefusal("control_metadata_bound", "control metadata exceeds its bound")
    if method not in {"GET", "POST", "PUT"} or not path.startswith("/v1/"):
        raise EgressRefusal("control_route_invalid", "control route is invalid")
    connection = http.client.HTTPSConnection(
        origin.host, origin.port, context=context, timeout=SILENCE_SECONDS
    )
    try:
        connection.request(method, path, body=body, headers=dict(headers))
        response = connection.getresponse()
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise EgressRefusal("control_response_bound", "control response exceeds its bound")
        return response.status, raw
    except (OSError, http.client.HTTPException) as exc:
        raise EgressRefusal(
            "control_transport_unavailable", "control transport unavailable"
        ) from exc
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class StreamReceipt:
    """Facts measured while one stream was copied to its destination."""

    length: int
    sha256: bytes
    prefix: bytes
    #: The store's own verdict on a PUT. 0 for a copy that never spoke HTTP.
    http_status: int = 0
    etag: str = ""


def copy_bounded(
    chunks: Iterable[bytes],
    sink: BinaryIO | None,
    *,
    expected_length: int | None,
    cap: int,
    prefix_bytes: int,
    what: str,
) -> StreamReceipt:
    """Verify one stream; write it only when a destination is needed."""
    if expected_length is not None and expected_length < 0:
        raise EgressRefusal("stream_invalid_length", f"{what}: negative expected length")
    limit = min(expected_length, cap) if expected_length is not None else cap
    seen = 0
    digest = hashlib.sha256()
    prefix = bytearray()
    for chunk in chunks:
        next_seen = seen + len(chunk)
        if next_seen > limit:
            raise EgressRefusal(
                "stream_over_declared_size",
                f"{what}: the stream passed {limit} B (declared {expected_length} B, "
                f"deployment cap {cap} B) and was cut there. A stream may not exceed its declared "
                "size, and finding out afterwards is not a bound",
            )
        if sink is not None:
            sink.write(chunk)
        digest.update(chunk)
        if len(prefix) < prefix_bytes:
            prefix.extend(chunk[: prefix_bytes - len(prefix)])
        seen = next_seen
    if expected_length is not None and seen != expected_length:
        raise EgressRefusal(
            "stream_under_declared_size",
            f"{what}: the stream ended at {seen} B and the grant declares "
            f"{expected_length} B. "
            "A short read is a refusal: a truncated body that is never compared against its "
            "declared length is how an empty file passes for content",
        )
    return StreamReceipt(seen, digest.digest(), bytes(prefix))


@dataclass(frozen=True, slots=True)
class EgressPolicy:
    """The deployment's declared egress bound. Empty allowlist means NO egress at all."""

    allowed_hosts: tuple[str, ...] = ()
    allow_private: bool = False
    """Loopback and private ranges, admitted ONLY for a local deployment that declares it.
    It is a separate switch from the allowlist because "I trust this host" and "I accept a
    private address" are different statements, and conflating them is how a localhost dev
    setting becomes a production metadata read."""

    def check_host(self, host: str) -> None:
        if not host:
            raise EgressRefusal("egress_no_host", "the URL names no host")
        if host.lower() not in {h.lower() for h in self.allowed_hosts}:
            raise EgressRefusal(
                "egress_host_not_allowed",
                f"{host!r} is not in this deployment's declared allowed-hosts "
                f"({list(self.allowed_hosts) or 'none — egress is off'}). The allowlist is "
                "a deployment-wide declaration; nothing widens it at request time",
            )

    def check_addresses(self, host: str, port: int) -> list[str]:
        """Resolve and check EVERY answer. Returns them so a caller can log what it saw."""
        try:
            answers = resolve(host, port)
        except OSError as exc:
            raise EgressRefusal(
                "egress_unresolvable",
                f"{host!r} does not resolve ({exc}); an address this runtime cannot check "
                "is not an address it will connect to",
            ) from exc
        seen: list[str] = []
        for address in answers:
            seen.append(address)
            if self.allow_private:
                continue
            why = blocked(address)
            if why:
                raise EgressRefusal(
                    "egress_blocked_address",
                    f"{host!r} resolves to {address}, which is {why}. EVERY answer is "
                    "checked, not the first: a name with one public answer and one "
                    "169.254.169.254 is the metadata-service bypass with an extra step",
                )
        if not seen:
            raise EgressRefusal("egress_unresolvable", f"{host!r} resolved to nothing")
        return seen


def _validate(url: str, policy: EgressPolicy) -> tuple[str, list[str]]:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise EgressRefusal("egress_url", f"the URL is malformed ({exc})") from exc
    if parsed.username is not None or parsed.password is not None:
        raise EgressRefusal(
            "egress_url", "the URL carries user-info; credentials belong in no host"
        )
    if parsed.fragment:
        raise EgressRefusal(
            "egress_url", "the URL carries a fragment, which is not sent to a server"
        )
    if parsed.scheme != "https" and not (policy.allow_private and parsed.scheme == "http"):
        raise EgressRefusal(
            "egress_scheme",
            f"{parsed.scheme or 'relative'}:// is not fetchable: egress is https, and plain "
            "http only for a deployment that declared allow_private",
        )
    policy.check_host(host)
    return host, policy.check_addresses(host, port)


class _MeasuredReader:
    """A file-like HTTP body that measures and bounds the bytes while they leave."""

    def __init__(self, source: BinaryIO, expected_length: int, what: str) -> None:
        self.source = source
        self.expected_length = expected_length
        self.what = what
        self.length = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)
        next_length = self.length + len(chunk)
        if next_length > self.expected_length:
            raise EgressRefusal(
                "stream_over_declared_size",
                f"{self.what}: the upload passed its declared {self.expected_length} B",
            )
        self.length = next_length
        self.digest.update(chunk)
        return chunk

    def receipt(self, *, http_status: int = 0, etag: str = "", exact: bool = True) -> StreamReceipt:
        # `exact=False` is for the one outcome where a short body is the CORRECT result: the
        # store refused the write on a precondition, so it stopped reading. Every other path
        # still treats a truncated upload as a refusal.
        if exact and self.length != self.expected_length:
            raise EgressRefusal(
                "stream_under_declared_size",
                f"{self.what}: the upload ended at {self.length} B and the grant declares "
                f"{self.expected_length} B",
            )
        return StreamReceipt(self.length, self.digest.digest(), b"", http_status, etag)


def put_from(
    url: str,
    policy: EgressPolicy,
    source: BinaryIO,
    *,
    expected_length: int,
    media_type: str,
    what: str = "output",
    required_headers: Mapping[str, str] | None = None,
    accept_precondition_failed: bool = False,
) -> StreamReceipt:
    """Stream one immutable object to an exact presigned URL.

    The URL is already the control plane's per-attempt capability. Responses cannot choose a
    replacement destination: a redirect target is fully validated, then refused. This is both
    simpler and stricter than replaying a PUT body to an address absent from the grant.

    `required_headers` are the grant's own signed headers, sent verbatim. The caller minting
    the grant decides what they are; this module does not invent or reinterpret them, because
    a header this function altered would be a header the signature no longer covers.

    `accept_precondition_failed` admits HTTP 412 as an OUTCOME rather than an error. Under an
    `If-None-Match: *` grant that status means the immutable key already holds these exact
    bytes, which is a successful end state for a content-addressed write, not a failure.

    The object is RE-SENT when the socket goes silent or the store answers a retryable
    status, `PUT_ATTEMPTS` times, exactly as the transport re-asks for one object: the
    socket's silence bound is observed per operation and a re-send costs this object's bytes,
    never the attempt. A source that cannot be rewound gets one send.
    """
    import http.client

    if expected_length < 0:
        raise EgressRefusal("stream_invalid_length", f"{what}: negative expected length")
    host, _ = _validate(url, policy)
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    headers = {
        "accept-encoding": "identity",
        "content-length": str(expected_length),
        "content-type": media_type or "application/octet-stream",
        "if-none-match": "*",
    }
    # HTTP names are case-insensitive. A signed grant replaces the same default
    # header, rather than adding a second differently-cased field on the wire.
    headers.update((name.lower(), value) for name, value in (required_headers or {}).items())
    #: The one send that streamed the WHOLE body. Its digest is the object's, whatever the
    #: store then said; a 412 on a later re-send is that send having landed.
    complete: _MeasuredReader | None = None
    attempts = PUT_ATTEMPTS if source.seekable() else 1
    for attempt in range(1, attempts + 1):
        last = attempt == attempts
        if attempt > 1:
            source.seek(0)
        measured = _MeasuredReader(source, expected_length, what)
        connection = connection_type(host, port, timeout=SILENCE_SECONDS)
        try:
            connection.request("PUT", path, body=measured, headers=headers)
            response = connection.getresponse()
            try:
                location = response.getheader("Location")
                if 300 <= response.status < 400 and location:
                    redirected = urljoin(url, location)
                    # Validate the whole redirect even though we never follow it. This makes
                    # a metadata/private-address redirect a precise security refusal, while a
                    # benign same-host redirect still cannot replace the exact destination the
                    # grant named.
                    _validate(redirected, policy)
                    raise EgressRefusal(
                        "egress_redirect_refused",
                        f"{what}: HTTP {response.status} tried to replace the exact granted URL",
                    )
                body = response.read(64 * 1024 + 1)
                if len(body) > 64 * 1024:
                    raise EgressRefusal(
                        "egress_response_over_bound", f"{what}: the PUT response exceeded 64 KiB"
                    )
                status = response.status
                etag = (response.getheader("ETag") or "").strip()
            finally:
                response.close()
        except EgressRefusal:
            raise
        except (http.client.HTTPException, OSError) as exc:
            if measured.length == expected_length:
                complete = measured
            if not last:
                continue
            raise EgressRefusal(
                "egress_unreachable", f"{what}: {exc} (after {attempts} send(s))"
            ) from exc
        finally:
            connection.close()
        if status in RETRYABLE_STATUS and not last:
            if measured.length == expected_length:
                complete = measured
            continue
        if status == HTTP_PRECONDITION_FAILED and complete is not None:
            # An earlier send streamed every byte and then lost the answer; the key now
            # holding bytes IS that answer, and the digest is the one that send measured.
            return complete.receipt(http_status=status, etag=etag, exact=True)
        if not (
            200 <= status < 300
            or (accept_precondition_failed and status == HTTP_PRECONDITION_FAILED)
        ):
            raise EgressRefusal("egress_http_error", f"{what}: HTTP {status} from {host}")
        return measured.receipt(
            http_status=status, etag=etag, exact=status != HTTP_PRECONDITION_FAILED
        )
    raise AssertionError("unreachable: the send loop returns or raises on its last attempt")
