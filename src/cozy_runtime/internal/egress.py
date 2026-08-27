"""cr-012 — the ONE bounded stream reader, and the fail-closed egress boundary.

Everything that pulls bytes into this runtime from an address someone else minted comes
through here. That is the point: the failure mode this module exists against is not "the
SSRF check was wrong", it is "there were two downloaders and only one had the check".

**A stream may not exceed its declared size.** `copy_bounded` stops at the declared length
and refuses before writing the over-bound chunk; it never buffers the body in memory. A body that
UNDER-runs is equally a refusal — a truncated read that returns short bytes is how a digest
check gets skipped by an empty file.

**The allowlist is a deployment-wide DECLARATION and the resolution is fail-closed.** A
host absent from the allowlist refuses. A host that resolves to a private, loopback,
link-local, multicast or reserved address refuses — every address it resolves to, not the
first, because a name with one public and one 169.254.169.254 answer is the metadata-service
bypass with an extra step.

**Every redirect hop is a NEW request and gets the whole check again.** Hop-0-only
validation IS the metadata-service bypass: `https://allowed.example/x` answering `302
http://169.254.169.254/latest/meta-data/` passes a hop-0 check and fetches the credential.
So redirects are never followed by the HTTP library — `_NoRedirect` turns them back into
responses, and this module re-enters its own front door with the new URL.

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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import BinaryIO
from urllib.parse import urlparse

#: How much of a body one read() pulls. Small enough that a cap refusal happens promptly.
CHUNK = 1 << 20

#: The redirect budget. A chain longer than this is a loop or an evasion, either way no.
MAX_REDIRECTS = 4


class EgressRefusal(Exception):
    """A typed fetch refusal. Every one fires before the bytes are used for anything."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class StreamReceipt:
    """Facts measured while one stream was copied to its destination."""

    length: int
    sha256: bytes
    prefix: bytes


def copy_bounded(
    chunks: Iterable[bytes],
    sink: BinaryIO,
    *,
    expected_length: int | None,
    cap: int,
    prefix_bytes: int,
    what: str,
) -> StreamReceipt:
    """Copy once while enforcing size and measuring identity plus a sniff prefix."""
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
    max_redirects: int = MAX_REDIRECTS
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
            answers = socket.getaddrinfo(host, port or None, proto=socket.IPPROTO_TCP)
        except OSError as exc:
            raise EgressRefusal(
                "egress_unresolvable",
                f"{host!r} does not resolve ({exc}); an address this runtime cannot check "
                "is not an address it will connect to",
            ) from exc
        seen: list[str] = []
        for answer in answers:
            address = str(answer[4][0])
            seen.append(address)
            if self.allow_private:
                continue
            parsed = ipaddress.ip_address(address)
            if (
                parsed.is_private
                or parsed.is_loopback
                or parsed.is_link_local
                or parsed.is_multicast
                or parsed.is_reserved
                or parsed.is_unspecified
            ):
                raise EgressRefusal(
                    "egress_blocked_address",
                    f"{host!r} resolves to {address}, which is private, loopback, "
                    "link-local, multicast or reserved. EVERY answer is checked, not the "
                    "first: a name with one public answer and one 169.254.169.254 is the "
                    "metadata-service bypass with an extra step",
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
    if parsed.scheme != "https" and not (policy.allow_private and parsed.scheme == "http"):
        raise EgressRefusal(
            "egress_scheme",
            f"{parsed.scheme or 'relative'}:// is not fetchable: egress is https, and plain "
            "http only for a deployment that declared allow_private",
        )
    policy.check_host(host)
    return host, policy.check_addresses(host, port)


def fetch_into(
    url: str,
    policy: EgressPolicy,
    sink: BinaryIO,
    *,
    expected_length: int | None,
    cap: int,
    prefix_bytes: int,
    what: str = "input",
) -> StreamReceipt:
    """Fetch one object into a sink, re-validating at every redirect hop."""
    import http.client
    import urllib.error
    import urllib.parse
    import urllib.request

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        """Redirects are RETURNED, not followed. Following one inside the library would put
        the second request outside this module's front door, which is the whole bug."""

        def redirect_request(self, *args: object, **kwargs: object) -> None:
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    hops: list[str] = []
    current = url
    for hop in range(policy.max_redirects + 1):
        host, addresses = _validate(current, policy)
        hops.append(f"{host} {addresses}")
        try:
            request = urllib.request.Request(
                current, method="GET", headers={"Accept-Encoding": "identity"}
            )
            response = opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            location = error.headers.get("Location") if error.headers else None
            if error.code in (301, 302, 303, 307, 308) and location:
                if hop == policy.max_redirects:
                    raise EgressRefusal(
                        "egress_redirect_budget",
                        f"{what}: more than {policy.max_redirects} redirects "
                        f"({' -> '.join(hops)}); a longer chain is a loop or an evasion",
                    ) from error
                try:
                    current = urllib.parse.urljoin(current, location)
                except ValueError as exc:
                    raise EgressRefusal(
                        "egress_url", f"{what}: a redirect names a malformed URL ({exc})"
                    ) from exc
                continue
            raise EgressRefusal("egress_http_error", f"{what}: {error.code} from {host}") from error
        except ValueError as exc:
            raise EgressRefusal("egress_url", f"{what}: the URL is malformed ({exc})") from exc
        except (http.client.HTTPException, OSError) as exc:
            raise EgressRefusal("egress_unreachable", f"{what}: {exc}") from exc
        try:
            with response as body:
                return copy_bounded(
                    iter(lambda body=body: body.read(CHUNK), b""),  # type: ignore[misc]
                    sink,
                    expected_length=expected_length,
                    cap=cap,
                    prefix_bytes=prefix_bytes,
                    what=what,
                )
        except EgressRefusal:
            raise
        except (http.client.HTTPException, OSError) as exc:
            raise EgressRefusal(
                "egress_body_error",
                f"{what}: the response body could not be read completely ({type(exc).__name__})",
            ) from exc
    raise EgressRefusal("egress_redirect_budget", f"{what}: redirect budget exhausted")
