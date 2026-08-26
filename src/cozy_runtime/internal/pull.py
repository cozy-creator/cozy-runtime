"""cr-016's `pull` — the TRANSPORT half of the artifact pull (tfs-003 row 220).

boundaries.md §9 places `HTTP/R2 fetch with attempt authority` here, in a cozy-runtime
transport adapter, and §2 says why: *"The network adapter is outside TensorFS core.
cozy-runtime uses DeliveryGrant to obtain bytes and asks TensorFS to verify/admit them.
TensorFS never sees a presigned URL or worker credential."* So this module owns exactly four
things — the hub's read routes, the SSRF guard, the credential, the deadline — and owns
nothing about what the bytes ARE.

**Every byte crosses one door and it is not in this file.** `tensorfs.fetch_admit` hashes
what arrives, checks it against the plan's digest AND length, and commits at the digest it
computed or commits nothing. This module never writes into the CAS, never decides an object
is already present, and never computes a digest: `tensorfs.fetch_plan` answers residency and
`tensorfs.fetch_complete` answers "did it land". A transport that answered any of those
itself would be a second opinion about identity, and the two would disagree exactly once,
silently, on the artifact that mattered.

**What resume means here.** The pull is resumable PER OBJECT, which is what the plan makes
free: a dropped connection loses the partial temp file of one object and nothing else, and
the next invocation re-plans against the store and asks only for what is still missing. There
is no byte-range resume inside a single object — the artifact is already cut on the object
grid, so the recovery unit is an object.

**No magic timeouts.** A deadline arrives from the caller or there is none; the socket
timeout is whatever is LEFT of that deadline. Nothing here invents a number of seconds.

**The wanted list is walked by N streams, not one.** An artifact is already cut on the
object grid, so the objects are independent and the serial loop was leaving the whole link
on the table — measured, not supposed: at 92.42 GiB on a rented pod the serial loop moved
49.8 MB/s against a 141 MB/s transport ceiling on the same pods and the same presigned GETs,
and a 16-way walk through this same door landed 117.6 MB/s. Concurrency changes the ORDER
objects are asked for and nothing else: the plan is still `fetch_plan`'s, every byte still
enters at `fetch_admit`, and the proof is still `fetch_complete`.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

#: How many objects are in flight by default. It is a MEASURED number, not a taste: #566b
#: swept the same pods, the same bucket and the same presigned GETs and found the per-stream
#: rate flat and the aggregate rate saturating by 16 (49.8 MB/s at 1, 117.6 MB/s at 16,
#: against a 141 MB/s ceiling the pod's own transport probe reached). It is a flag because
#: the right number is a property of the LINK, and this file has no way to measure the
#: caller's.
STREAMS = 16

#: The closed refusal registry for this leg. Every code here has a producer below, and the
#: import-time check at the bottom of this module is the #503/#504 invariant: a code with no
#: producer is a standing falsehood about the surface, and an unregistered producer is worse.
REFUSALS: tuple[str, ...] = (
    "hub_unreachable",
    "hub_refused",
    "ref_not_found",
    "credential_required",
    "source_not_allowed",
    "transfer_failed",
    "deadline_exceeded",
    "artifact_incomplete",
    "border_refused",
)


class PullRefusal(Exception):
    """A typed refusal from the transport leg. `cli/pull.py` maps it to an exit code.

    It carries the CODE, what happened, and the remedy — never a bare transport traceback:
    a worker that cannot pull needs to be told which credential to add or which coordinate
    was wrong, not which socket call raised.
    """

    def __init__(self, code: str, message: str, remedy: str, *, cause: str = "") -> None:
        super().__init__(message)
        if code not in REFUSALS:
            raise AssertionError(f"{code!r} is not in this leg's closed refusal registry")
        self.code = code
        self.message = message
        self.remedy = remedy
        self.cause = cause


# ---------------------------------------------------------------- deadline


@dataclass(frozen=True, slots=True)
class Deadline:
    """A caller-supplied wall deadline, or none at all.

    `remaining()` is what every socket gets, so a slow hub and a slow object share one
    budget instead of each inventing its own. `None` means the caller set no deadline and
    the pull blocks until the transport itself gives up — which is honest: a runtime that
    substituted a number here would be deciding, on the user's behalf, that a large artifact
    on a slow link is a failure.
    """

    at: float | None = None

    @staticmethod
    def after(seconds: float | None) -> Deadline:
        return Deadline(None if seconds is None else time.monotonic() + seconds)

    def remaining(self) -> float | None:
        if self.at is None:
            return None
        left = self.at - time.monotonic()
        if left <= 0:
            raise PullRefusal(
                "deadline_exceeded",
                "the pull deadline passed with objects still to move",
                "raise --timeout, or re-run: a pull resumes per object and re-fetches only "
                "what is still missing",
            )
        return left


# ---------------------------------------------------------------- the SSRF guard


def guard(url: str, *, allow_local: bool) -> str:
    """One allowlisted, SSRF-guarded door for every URL this leg opens (tfs-003).

    A presigned URL arrives from the hub, and the hub is not automatically trusted to name a
    destination: an object store URL that resolved to a link-local address would make the
    worker a proxy into its own metadata service. `allow_local` is the DEV carve-out and it
    is explicit — it is true only when the hub base itself is a loopback address, so a
    developer's localhost hub works and a production hub can never hand a worker one.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise PullRefusal(
            "source_not_allowed",
            f"{parts.scheme or '(none)'}: not an allowed transport scheme",
            "artifact bytes travel over https (http only against a loopback dev hub)",
        )
    host = parts.hostname
    if not host:
        raise PullRefusal(
            "source_not_allowed", f"{url[:120]}: no host", "supply an absolute http(s) URL"
        )
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError as exc:
        raise PullRefusal(
            "hub_unreachable",
            f"{host}: name resolution failed",
            "check the host name and this machine's DNS",
            cause=str(exc),
        ) from exc
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        local = addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast
        if local and not allow_local:
            raise PullRefusal(
                "source_not_allowed",
                f"{host} resolves to {addr}, which is not a public address",
                "a worker does not fetch artifact bytes from its own network; if this is a "
                "development hub, point --hub at it directly and it is admitted explicitly",
            )
    if parts.scheme == "http" and not allow_local:
        raise PullRefusal(
            "source_not_allowed",
            f"{host}: plaintext http to a non-loopback host",
            "use https — a presigned URL carries its own authorization in the query string",
        )
    return url


def local_hub(base: str) -> bool:
    """Whether the HUB ITSELF is loopback. The only thing that opens the http/private door."""
    host = urllib.parse.urlsplit(base).hostname or ""
    try:
        return all(
            ipaddress.ip_address(i[4][0]).is_loopback for i in socket.getaddrinfo(host, None)
        )
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- the hub's read routes


def _open(req: urllib.request.Request, deadline: Deadline) -> Any:
    """One opener. Redirects are refused rather than followed: a presigned GET has no
    business redirecting, and following one would re-open a URL the guard never saw."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_: Any, **__: Any) -> None:
            return None

    opener = urllib.request.build_opener(NoRedirect)
    return opener.open(req, timeout=deadline.remaining())


def _json(base: str, path: str, deadline: Deadline, *, token: str = "", body: Any = None) -> Any:
    url = base.rstrip("/") + path
    guard(url, allow_local=local_hub(base))
    data = None
    req = urllib.request.Request(url, method="POST" if body is not None else "GET")
    if body is not None:
        data = json.dumps(body).encode()
        req.data = data
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with _open(req, deadline) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise _hub_error(exc, path) from exc
    except urllib.error.URLError as exc:
        raise PullRefusal(
            "hub_unreachable",
            f"the hub at {base} did not answer {path}",
            "check --hub and that the hub is serving; artifacts already in the local CAS "
            "are usable offline",
            cause=str(exc),
        ) from exc


def _hub_error(exc: urllib.error.HTTPError, path: str) -> PullRefusal:
    """Render the hub's own typed envelope. Its code and remedy are BETTER than anything
    invented here — the hub knows why it refused — so they travel verbatim."""
    try:
        env = json.loads(exc.read() or b"{}").get("error", {})
    except Exception:
        env = {}
    code = str(env.get("code", "")) or f"http.{exc.code}"
    message = str(env.get("message", "")) or exc.reason
    remedy = str(env.get("remedy", "")) or "see the hub's error envelope"
    if exc.code == 404:
        return PullRefusal("ref_not_found", f"{code}: {message}", remedy)
    if exc.code in (401, 403):
        return PullRefusal(
            "credential_required",
            f"{code}: {message}",
            remedy or "set TENSORHUB_TOKEN and re-run — this is never a prompt",
        )
    return PullRefusal("hub_refused", f"{code}: {message}", remedy, cause=f"{path} -> {exc.code}")


@dataclass(frozen=True, slots=True)
class Resolved:
    """One exact checkpoint. A ref plus a lane selector went in; a digest came out."""

    org: str
    name: str
    release: str
    #: BARE 64-hex. The digest is the identity, and every document that carries one carries
    #: it prefixed — so the two spellings are kept apart deliberately rather than trusted to
    #: agree: `snapshot` is what an ObjectRef holds, `path_id` is what the hub's routes take.
    snapshot: str
    lane: str = ""

    @property
    def path_id(self) -> str:
        return "sha256:" + self.snapshot


def resolve(
    base: str, ref: str, deadline: Deadline, *, lane: str = "", token: str = ""
) -> Resolved:
    """`org/repo[@release|@sha256:...]` -> one exact checkpoint, decided by the HUB.

    Resolution is not re-implemented here on purpose: the hub owns the release/lane
    vocabulary, and a client that resolved locally would need a copy of it that goes stale.
    """
    q = urllib.parse.urlencode({"ref": ref, **({"lane": lane} if lane else {})})
    doc = _json(base, f"/v1/resolve?{q}", deadline, token=token)
    # The hub's own spelling for "the exact bytes this ref names". `checkpoint_id` is what
    # `/v1/resolve` returns and `snapshot_id` is what the checkpoint listing returns; both
    # are read rather than one of them being assumed, because a client that guessed would
    # fail on whichever route it did not test against.
    snapshot = str(
        doc.get("checkpoint_id")
        or doc.get("snapshot_id")
        or doc.get("checkpoint", {}).get("snapshot_id")
        or ""
    ).strip()
    if not snapshot:
        raise PullRefusal(
            "ref_not_found",
            f"{ref}: the hub resolved it to no checkpoint",
            "check the ref and the lane selector; `--lane` narrows a release with several",
            cause=json.dumps(doc)[:400],
        )
    repo = str(doc.get("repo") or doc.get("ref") or ref).split("@")[0]
    org, _, name = repo.partition("/")
    return Resolved(
        org=org,
        name=name,
        release=str(doc.get("release") or ""),
        snapshot=snapshot.removeprefix("sha256:"),
        lane=lane,
    )


@dataclass(frozen=True, slots=True)
class Read:
    """One presigned GET the hub authorized, and the length it claims for it."""

    object_id: str
    length: int
    url: str


def reads(base: str, at: Resolved, deadline: Deadline, *, token: str = "") -> list[Read]:
    """The hub's read authorizations — the mirror of a publish grant.

    This is a CLAIM about what the checkpoint reaches. Nothing is trusted about it: the
    lengths and digests are handed to the border as the EXPECTATION each object must meet,
    and an object that arrives as anything else is refused, not accepted with a warning.
    """
    path = f"/v1/repos/{at.org}/{at.name}/checkpoints/{at.path_id}/reads"
    doc = _json(base, path, deadline, token=token, body={})
    out = [Read(str(r["object_id"]), int(r["length"]), str(r["url"])) for r in doc.get("reads", [])]
    if not out:
        raise PullRefusal(
            "ref_not_found",
            f"sha256:{at.snapshot}: the hub authorized no reads for this checkpoint",
            "the checkpoint exists in the catalog but reaches no custodied object; the hub's "
            "own envelope names the repair",
        )
    return out


def manifest(base: str, at: Resolved, deadline: Deadline) -> bytes:
    """The snapshot manifest's exact bytes, served verbatim by the hub.

    Deliberately read whole: a manifest is a bounded canonical DOCUMENT, not the artifact.
    Its bytes still cross the same verifying door as every tensor object — the border checks
    them against the digest the ref resolved to, so a hub that served the wrong manifest is
    caught at the door rather than believed.
    """
    url = base.rstrip("/") + f"/v1/repos/{at.org}/{at.name}/checkpoints/{at.path_id}/manifest"
    guard(url, allow_local=local_hub(base))
    try:
        with _open(urllib.request.Request(url), deadline) as resp:
            return bytes(resp.read())
    except urllib.error.HTTPError as exc:
        raise _hub_error(exc, "manifest") from exc
    except urllib.error.URLError as exc:
        raise PullRefusal(
            "hub_unreachable",
            f"the hub at {base} did not serve the manifest for sha256:{at.snapshot}",
            "check --hub and that the hub is serving",
            cause=str(exc),
        ) from exc


# ---------------------------------------------------------------- the pull


@dataclass(slots=True)
class PullResult:
    """What a pull actually did — counted, never estimated."""

    ref: str
    snapshot: str
    store: str
    #: The release the HUB resolved this ref to, or "" when the ref pinned a digest and the
    #: hub therefore named none. It is carried because the local index row is keyed by a ref
    #: and a row that dropped the release names a different artifact than the one pulled.
    release: str = ""
    streams: int = 1
    declared: int = 0
    held: int = 0
    fetched: int = 0
    bytes_moved: int = 0
    bytes_held: int = 0
    bytes_total: int = 0
    presence: dict[str, str] = field(default_factory=dict)
    objects: list[str] = field(default_factory=list)

    def document(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "snapshot": "sha256:" + self.snapshot,
            "store": self.store,
            "declared": self.declared,
            "held": self.held,
            "fetched": self.fetched,
            "bytes_moved": self.bytes_moved,
            "bytes_held": self.bytes_held,
            "bytes_total": self.bytes_total,
            "streams": self.streams,
        }


class _Reader:
    """The HTTP body, handed to the border as a plain `.read(n)`.

    It exists for one reason: to make the response's own read raise a PullRefusal rather than
    a socket error, so a connection that dies mid-object is reported as `transfer_failed`
    naming the object — with nothing committed, because the border removes its partial temp
    file before it returns.
    """

    def __init__(self, resp: Any, object_id: str) -> None:
        self._resp = resp
        self._id = object_id

    def read(self, n: int) -> bytes:
        try:
            return bytes(self._resp.read(n))
        except (OSError, urllib.error.URLError) as exc:
            raise PullRefusal(
                "transfer_failed",
                f"{self._id}: the connection failed mid-object",
                "re-run the pull: it resumes per object and re-fetches only what is missing",
                cause=str(exc),
            ) from exc


def pull(
    tensorfs: Any,
    store: Any,
    base: str,
    ref: str,
    *,
    lane: str = "",
    token: str = "",
    timeout: float | None = None,
    session: str = "",
    streams: int = STREAMS,
    on_object: Callable[[str, int, bool], None] | None = None,
) -> PullResult:
    """Pull one hub checkpoint into this machine's local CAS.

    `tensorfs` and `store` are injected rather than imported so the caller owns the facade
    lookup and its typed absent-reader refusal (`fill.tensorfs_module`), and so this module
    stays testable against a real store with no hub in sight.

    `streams` objects are in flight at once. One is the serial walk this leg used to be and
    it stays reachable, because a link where concurrency does not help is a real link.
    """
    streams = max(1, int(streams))
    deadline = Deadline.after(timeout)
    at = resolve(base, ref, deadline, lane=lane, token=token)
    declared = reads(base, at, deadline, token=token)
    doc = manifest(base, at, deadline)

    session = session or ("pull-" + at.snapshot[:12])
    plan = _plan(
        tensorfs,
        store,
        session,
        at.snapshot,
        len(doc),
        [(r.object_id, r.length) for r in declared],
    )
    result = PullResult(
        ref=ref,
        snapshot=at.snapshot,
        store=str(store.root),
        release=at.release,
        streams=streams,
        declared=len(plan["held"]) + len(plan["wanted"]),
        held=len(plan["held"]),
        bytes_held=int(plan["held_bytes"]),
        bytes_total=int(plan["held_bytes"]) + int(plan["wanted_bytes"]),
        presence=dict(plan["presence"]),
    )
    urls = {r.object_id: r.url for r in declared}
    allow_local = local_hub(base)
    ledger = threading.Lock()

    def one(item: tuple[str, int]) -> None:
        object_id, length = item
        if object_id == "sha256:" + at.snapshot and object_id not in urls:
            # The root travels on its own route. Its bytes are already in hand and still go
            # through the door, so a hub that served a manifest for a different checkpoint
            # is an OBJECT_ID_MISMATCH here rather than a wrong artifact later.
            source: Any = _Bytes(doc)
            _admit(tensorfs, store, plan["document"], object_id, source)
        else:
            url = urls.get(object_id)
            if url is None:
                raise PullRefusal(
                    "artifact_incomplete",
                    f"{object_id}: the checkpoint reaches it and the hub authorized no read",
                    "ask the hub for the object's custody state; a reachable uncustodied "
                    "object is a catalog defect, not a client error",
                )
            _fetch(tensorfs, store, plan["document"], object_id, url, deadline, allow_local)
        # The counters are the RESULT, so they are taken once under one lock rather than
        # incremented from N threads — a byte total that raced would be the one number in
        # this document nobody could check.
        with ledger:
            result.fetched += 1
            result.bytes_moved += length
            result.objects.append(object_id)
            if on_object is not None:
                on_object(object_id, length, True)

    wanted = [(str(o), int(n)) for o, n in plan["wanted"]]
    with ThreadPoolExecutor(max_workers=streams, thread_name_prefix="pull") as pool:
        futures = [pool.submit(one, item) for item in wanted]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        failed = next((f for f in done if f.exception() is not None), None)
        if failed is not None:
            # STOP ASKING. Every object already in flight still finishes at the door (a
            # thread cannot be interrupted mid-admit and a half-admitted object is exactly
            # what the border refuses to leave behind), but nothing new is started and the
            # first refusal is the one the caller sees.
            for f in pending:
                f.cancel()
            for f in futures:
                if not f.cancelled():
                    f.exception()
            first = failed.exception()
            assert first is not None
            raise first

    # The PROOF, and it is not "every GET returned 200": every declared object is installed
    # and this store's own verification record vouches for it.
    try:
        tensorfs.fetch_complete(store, plan["document"])
    except Exception as exc:
        raise _border(exc, "artifact_incomplete") from exc
    return result


class _Bytes:
    """An in-hand document, presented to the border as a reader."""

    def __init__(self, data: bytes) -> None:
        self._data, self._at = data, 0

    def read(self, n: int) -> bytes:
        chunk = self._data[self._at : self._at + n]
        self._at += len(chunk)
        return chunk


def _fetch(
    tensorfs: Any,
    store: Any,
    plan_doc: bytes,
    object_id: str,
    url: str,
    deadline: Deadline,
    allow_local: bool,
) -> None:
    guard(url, allow_local=allow_local)
    try:
        resp = _open(urllib.request.Request(url), deadline)
    except urllib.error.HTTPError as exc:
        raise PullRefusal(
            "transfer_failed",
            f"{object_id}: the object store answered {exc.code}",
            "a presigned read expires; re-run the pull to be re-authorized",
            cause=str(exc.reason),
        ) from exc
    except urllib.error.URLError as exc:
        raise PullRefusal(
            "transfer_failed",
            f"{object_id}: the object store did not answer",
            "re-run the pull: it resumes per object",
            cause=str(exc),
        ) from exc
    with resp:
        _admit(tensorfs, store, plan_doc, object_id, _Reader(resp, object_id))


def _admit(tensorfs: Any, store: Any, plan_doc: bytes, object_id: str, source: Any) -> None:
    try:
        tensorfs.fetch_admit(store, plan_doc, object_id, source)
    except PullRefusal:
        raise
    except Exception as exc:
        raise _border(exc, "border_refused", object_id) from exc


def _plan(
    tensorfs: Any,
    store: Any,
    session: str,
    snapshot: str,
    snapshot_length: int,
    declared: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    try:
        return dict(tensorfs.fetch_plan(store, session, snapshot, snapshot_length, list(declared)))
    except Exception as exc:
        raise _border(exc, "border_refused") from exc


def _border(exc: BaseException, code: str, subject: str = "") -> PullRefusal:
    """A TensorFS refusal, translated once, at the one boundary that crosses it.

    #511d's lesson: a border refusal that escapes UNTRANSLATED across a one-refusal-table
    boundary is the same defect class as an unregistered producer. The border's own code and
    detail travel verbatim inside the message — they are more precise than anything this
    layer could restate.
    """
    detail = f"{type(exc).__name__}: {exc}"
    if code == "artifact_incomplete":
        return PullRefusal(
            "artifact_incomplete",
            f"the pull did not land every declared object — {detail}",
            "re-run the pull: it re-plans against the store and asks only for what is "
            "still missing",
        )
    return PullRefusal(
        "border_refused",
        f"{subject or 'the artifact'} was refused at the TensorFS border — {detail}",
        "the bytes that arrived are not the bytes the catalog names; nothing was installed. "
        "Re-run to re-fetch, and if it repeats the hub's custody of that object is suspect",
    )


def _producers() -> Iterable[str]:
    """Every code this module can actually raise, read off the source rather than listed by
    hand — a hand-maintained list is the thing that goes stale."""
    import pathlib
    import re

    text = pathlib.Path(__file__).read_text()
    return set(re.findall(r'PullRefusal\(\s*"([a-z_]+)"', text)) | set(
        re.findall(r'return PullRefusal\(\s*\n?\s*"([a-z_]+)"', text)
    )


#: #504's symmetry, checked at IMPORT so the class is unrepresentable rather than patched:
#: the registry lists exactly the producible codes, in both directions.
_produced = set(_producers())
if _produced - set(REFUSALS):
    raise AssertionError(f"unregistered pull refusals: {sorted(_produced - set(REFUSALS))}")
if set(REFUSALS) - _produced:
    raise AssertionError(f"pull refusals with no producer: {sorted(set(REFUSALS) - _produced)}")
