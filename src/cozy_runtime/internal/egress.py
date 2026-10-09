"""cr-012 — the Runtime's one fail-closed egress door: public, pinned wheel downloads.

The hazard this module exists against is "there were two downloaders and only one had the
check". Artifact bytes move through tensorfs (`tensorfs-core/src/transport/policy.rs` holds
the port of `blocked()` below); publication moves through the machine (th-241). What stays
here is `fetch_public_wheel`: a public wheel a lock pins by URL and sha256, fetched anonymously
— no credential, no Hub authority, nothing naming the run. The `one-address-predicate` fence
in `checks/architecture.py` reddens if any other urllib/http.client chain appears.

**The resolution is fail-closed.** A host that resolves to a private, loopback, link-local,
multicast or reserved address refuses — every address it resolves to, not the first, because
a name with one public and one 169.254.169.254 answer is the metadata-service bypass with an
extra step. Every redirect hop is re-checked.

**The DNS-rebinding limit is stated, not papered over.** Validation happens at resolve time
and the connection is made by hostname, so a name that answers publicly during the check and
privately during the connect is not closed by this design.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from cozy_runtime.internal import liveness

#: How much of a body one read() pulls.
CHUNK = 1 << 20

#: The silence bound on ONE socket operation, never on a transfer: the runtime's one noise
#: floor (`liveness.noise_floor`), the patience a tensorfs pull stream is given too.
SILENCE_SECONDS = liveness.noise_floor()


#: The address classes nothing in this runtime connects to. ONE list, checked by
#: `EgressPolicy.check_addresses`; tensorfs `transport/policy.rs` holds its port (cr-090).
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


def fetch_public_wheel(
    path: Path, url: str, expected_digest: str, ceiling: int
) -> tuple[Path, int]:
    """One public wheel, pinned by URL and sha256, fetched anonymously through this door."""
    import os
    import uuid
    from urllib.request import HTTPRedirectHandler, Request, build_opener

    from cozy_runtime.internal import storage_admission
    from cozy_runtime.internal.package_environment import (
        EnvironmentRefusal,
        _require_public_wheel_url,
    )

    class PublicRedirect(HTTPRedirectHandler):
        def redirect_request(
            self, req: object, fp: object, code: int, msg: str, headers: object, newurl: str
        ) -> Request | None:
            check(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]

    def check(target: str) -> None:
        _require_public_wheel_url(target)
        host = urlparse(target).hostname or ""
        _validate(
            target, EgressPolicy(allowed_hosts=(host,), allow_private=host in {"127.0.0.1", "::1"})
        )

    check(url)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with build_opener(PublicRedirect()).open(url, timeout=SILENCE_SECONDS) as response:
            header = response.headers.get("Content-Length")
            bound = int(header) if header is not None else ceiling
            if not 0 < bound <= ceiling:
                raise EnvironmentRefusal("package_environment_wheel_invalid", "wheel size bound")
            with storage_admission.admit(storage_admission.Write(path.parent, bound, 16)):
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                digest = hashlib.sha256()
                length = 0
                with temporary.open("xb") as target:
                    while block := response.read(CHUNK):
                        length += len(block)
                        if length > bound:
                            raise EnvironmentRefusal(
                                "package_environment_wheel_invalid", "wheel size bound"
                            )
                        digest.update(block)
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                if (
                    header is not None and length != bound
                ) or "sha256:" + digest.hexdigest() != expected_digest:
                    raise EnvironmentRefusal(
                        "package_environment_wheel_invalid", "wheel hash or length changed"
                    )
                temporary.chmod(0o444)
                os.replace(temporary, path)
        return path, length
    except (OSError, ValueError, EgressRefusal) as exc:
        raise EnvironmentRefusal(
            "package_environment_wheel_fetch_failed", type(exc).__name__
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
