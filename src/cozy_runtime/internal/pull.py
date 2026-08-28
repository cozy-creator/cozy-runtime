"""cr-016's `pull` — the TRANSPORT half of the artifact pull (tfs-003 row 220).

boundaries.md §9 places `HTTP/R2 fetch with attempt authority` here, in a cozy-runtime
transport adapter, and §2 says why: *"The network adapter is outside TensorFS core.
cozy-runtime uses DeliveryGrant to obtain bytes and asks TensorFS to verify/admit them.
TensorFS never sees a presigned URL or worker credential."* So this module owns exactly four
things — the hub's read routes, the SSRF guard, the credential, the deadline — and owns
nothing about what the bytes ARE.

**Every byte crosses a TensorFS-owned door.** Network bodies use `tensorfs.fetch_admit`;
cache-to-local and verified local-to-cache copies use `tensorfs.transfer_object`, whose
destination is the same digest/length/no-clobber Store admission. This module never resolves
Store layout, writes into a CAS, or computes a digest: `tensorfs.fetch_plan` answers local
residency and `tensorfs.fetch_complete` answers "did it land". A transport that answered any
of those itself would be a second opinion about identity, and the two would disagree exactly
once, silently, on the artifact that mattered.

**What resume means here.** The pull is resumable PER OBJECT, which is what the plan makes
free: a dropped connection loses the partial temp file of one object and nothing else, and
the next invocation re-plans against the store and asks only for what is still missing. There
is no byte-range resume inside a single object — the artifact is already cut on the object
grid, so the recovery unit is an object.

**No magic timeouts, and no unbounded silence either.** A deadline arrives from the caller
or there is none, and nothing here invents a number of seconds to fail on. But the socket
timeout used to be whatever was LEFT of that deadline, which made ONE black-holed TCP stream
cost the whole budget: the read blocked, the pool waited on it, and a 96 GiB closure that
moves at 470 MB/s died `readiness.deadline_exceeded` with 15 streams idle. The kill rule is
cl-025's and it is RELATIVE: a stream that moved no bytes while its peers moved theirs is
stalled, its socket is closed, and its object is re-fetched. When NO stream advances the link
itself is down and nothing is singled out — the caller's deadline is the only authority then.
The sampling interval is resolution, not a bound.

**The wanted list is walked by N streams, not one.** An artifact is already cut on the
object grid, so the objects are independent and the serial loop was leaving the whole link
on the table — measured, not supposed: at 92.42 GiB on a rented pod the serial loop moved
49.8 MB/s against a 141 MB/s transport ceiling on the same pods and the same presigned GETs,
and a 16-way walk through this same door landed 117.6 MB/s. Concurrency changes the ORDER
objects are asked for and nothing else: the plan is still `fetch_plan`'s, every byte still
enters at `fetch_admit`, and the proof is still `fetch_complete`.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import egress

#: How many objects are in flight by default. It is a MEASURED number, not a taste: #566b
#: swept the same pods, the same bucket and the same presigned GETs and found the per-stream
#: rate flat and the aggregate rate saturating by 16 (49.8 MB/s at 1, 117.6 MB/s at 16,
#: against a 141 MB/s ceiling the pod's own transport probe reached). It is a flag because
#: the right number is a property of the LINK, and this file has no way to measure the
#: caller's.
STREAMS = 16

#: How often the stream watchdog SAMPLES the per-stream byte counters. Resolution, not a
#: verdict: nothing is killed because this many seconds passed.
STALL_SAMPLE_SECONDS = 5.0

#: The NOISE FLOOR, in samples, below which silence is not evidence of anything. A pull whose
#: chunks came back in milliseconds must not condemn a stream that is merely a second slow.
STALL_SAMPLES = 6

#: How many times the pull's OWN worst measured chunk gap a stream may be silent for before it
#: is stalled. This is the term that carries the verdict, and it is measured: a link that has
#: shown 30 s between chunks earns 30 s of patience, and one that has never paused earns the
#: floor. Both conditions must hold, so neither a slow link nor a fast one is judged by the
#: other's clock.
STALL_FACTOR = 8.0

#: How many times one object may be re-fetched. A grant is finite and the plan is per-object
#: resumable, so a retry costs one object, not the pull. A stall or a transient object-store
#: answer that survives this many attempts is the link, not a blip, and is reported as one.
FETCH_ATTEMPTS = 4

#: The object-store answers a re-fetch is worth attempting for. Everything else — an expired
#: presign, an absent key — is a fact about authorization, and retrying it is a lie about time.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

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


def open_store(tensorfs: Any, root: Path) -> Any:
    """Open or initialize one TensorFS-owned store root without resolving its layout."""

    root.mkdir(parents=True, exist_ok=True)
    try:
        return (
            tensorfs.Store.open(str(root))
            if (root / "store.json").is_file()
            else tensorfs.Store.init(str(root))
        )
    except Exception as exc:
        raise PullRefusal(
            "border_refused",
            f"{root}: not a usable TensorFS store — {type(exc).__name__}: {exc}",
            "repair the backing filesystem or launch against an empty store directory",
        ) from exc


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

    WHICH addresses are unfetchable is `egress.blocked`'s and only its (cr-042). This
    function once carried its own four-class chain; the two lists differed and agreed only
    by CPython accident. What stays here is what is genuinely this leg's: the scheme rule,
    the plaintext-http rule, and the typed `PullRefusal` vocabulary.
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
        answers = egress.resolve(host, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError as exc:
        raise PullRefusal(
            "hub_unreachable",
            f"{host}: name resolution failed",
            "check the host name and this machine's DNS",
            cause=str(exc),
        ) from exc
    for address in answers:
        why = egress.blocked(address)
        if why and not allow_local:
            raise PullRefusal(
                "source_not_allowed",
                f"{host} resolves to {address}, which is {why} and not a public address",
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
        answers = egress.resolve(host)
    except (OSError, ValueError):
        return False
    return bool(answers) and all("loopback" in egress.blocked(a) for a in answers)


# ---------------------------------------------------------------- the hub's read routes


def _open(req: urllib.request.Request, deadline: Deadline, *, no_proxy: bool = False) -> Any:
    """One opener. Redirects are refused rather than followed: a presigned GET has no
    business redirecting, and following one would re-open a URL the guard never saw."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_: Any, **__: Any) -> None:
            return None

    handlers: list[Any] = [NoRedirect]
    if no_proxy:
        handlers.insert(0, urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    return opener.open(req, timeout=deadline.remaining())


def open_guarded(
    url: str,
    deadline: Deadline,
    *,
    headers: Mapping[str, str] | None = None,
    allow_local: bool = False,
    no_proxy: bool = False,
) -> Any:
    """Open one already-authorized URL through the shared guarded transport."""

    guard(url, allow_local=allow_local)
    return _open(
        urllib.request.Request(url, headers=dict(headers or {})), deadline, no_proxy=no_proxy
    )


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
    stamps: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """THE RESOLVED LANE'S STAMPED-FACT SET (#567d) — the third coordinate of the contract
    the hub just matched, beside `structure` and `encoding`. It comes down with the
    resolution because it is a fact about the checkpoint the ref named, and the pod that
    installs those bytes is the machine whose model classes have to join against it. A hub
    that answers no stamps leaves this empty, and an unstamped artifact fails every stamped
    class closed — which is the honest outcome, not a default."""

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
        stamps=_stamps(doc.get("stamps")),
    )


def _stamps(raw: object) -> dict[str, tuple[str, ...]]:
    """The hub's `key -> [values]` stamp map, read defensively and never invented.

    Anything that is not a map of string keys to string lists is DROPPED rather than
    coerced: a half-read stamp set would fail a class closed for the wrong reason, and an
    absent one already fails it closed for the right one.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for key, values in raw.items():
        if isinstance(values, str):
            out[str(key)] = (values,)
        elif isinstance(values, list):
            out[str(key)] = tuple(str(v) for v in values if isinstance(v, str))
    return {k: v for k, v in out.items() if v}


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
    #: The resolved lane's stamped-fact set, carried into the local index row (#567d).
    stamps: dict[str, tuple[str, ...]] = field(default_factory=dict)
    streams: int = 1
    declared: int = 0
    held: int = 0
    fetched: int = 0
    bytes_moved: int = 0
    bytes_held: int = 0
    bytes_total: int = 0
    presence: dict[str, str] = field(default_factory=dict)
    objects: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    cache_degradations: set[str] = field(default_factory=set)

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

    It exists for two reasons: to make the response's own read raise a PullRefusal rather than
    a socket error, so a connection that dies mid-object is reported as `transfer_failed`
    naming the object — with nothing committed, because the border removes its partial temp
    file before it returns — and to be the one place every arriving byte is COUNTED, which is
    what makes "this stream is not moving" a measurement instead of an opinion.
    """

    def __init__(self, resp: Any, object_id: str, streams: _Streams | None = None) -> None:
        self._resp = resp
        self._id = object_id
        self._streams = streams
        self._key = threading.get_ident()

    def read(self, n: int) -> bytes:
        try:
            chunk = bytes(self._resp.read(n))
        except (OSError, urllib.error.URLError) as exc:
            raise PullRefusal(
                "transfer_failed",
                f"{self._id}: the connection failed mid-object",
                "re-run the pull: it resumes per object and re-fetches only what is missing",
                cause=str(exc),
            ) from exc
        if chunk and self._streams is not None:
            self._streams.moved(self._key, len(chunk))
        return chunk


class _Streams:
    """The pull's own liveness ledger, and the one thing allowed to end a stream early.

    A presigned GET that black-holes is indistinguishable from a slow one to the socket that
    is waiting on it, and the socket's timeout is what is left of the caller's deadline — so
    before this existed, one dead TCP stream held a pool slot for the entire budget and the
    pull died on a deadline with the link mostly idle.

    The rule is cl-025's, and it is MEASURED rather than invented: this ledger watches the gap
    between one stream's chunks and remembers the longest gap the pull has actually shown. A
    stream silent for longer than `STALL_FACTOR` times that measured worst case — and longer
    than the sampling noise floor, so a millisecond link cannot condemn a one-second pause —
    is stalled. Its socket is shut down so the blocked read raises, and `_fetch` asks for that
    one object again. A genuinely slow link teaches this ledger its own pace and is left alone;
    only a stream that has stopped, against the evidence of the same pull, is ended.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._advanced: dict[int, float] = {}
        self._socket: dict[int, Any] = {}
        self._stalled: set[int] = set()
        self._worst_gap = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- the streams' side
    def open(self, key: int) -> None:
        with self._lock:
            self._advanced[key] = time.monotonic()
            self._socket.pop(key, None)
            self._stalled.discard(key)

    def bind(self, key: int, sock: Any) -> None:
        with self._lock:
            if sock is not None:
                self._socket[key] = sock

    def moved(self, key: int, count: int) -> None:
        """One chunk arrived. Its GAP is the measurement the stall rule is judged against.

        A stream this ledger has ALREADY named teaches it nothing. Its next `read` returns the
        moment the socket is shut down, and counting that silence as the link's honest pace
        would let every stall raise the patience that the next stall is judged by — the ledger
        would talk itself out of its own rule, one object at a time.
        """
        now = time.monotonic()
        with self._lock:
            previous = self._advanced.get(key)
            if previous is None:
                return
            self._advanced[key] = now
            if key not in self._stalled:
                self._worst_gap = max(self._worst_gap, now - previous)

    def stalled(self, key: int) -> bool:
        with self._lock:
            return key in self._stalled

    def done(self, key: int) -> None:
        with self._lock:
            self._advanced.pop(key, None)
            self._socket.pop(key, None)
            self._stalled.discard(key)

    # ---- the watchdog's side
    def __enter__(self) -> _Streams:
        self._thread = threading.Thread(target=self._watch, name="pull-liveness", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _watch(self) -> None:
        while not self._stop.wait(STALL_SAMPLE_SECONDS):
            for sock in self._sample():
                # SHUT_RDWR rather than close(): the blocked read is woken without the file
                # descriptor being freed under a thread that is still inside a syscall on it.
                with contextlib.suppress(OSError, ValueError):
                    sock.shutdown(socket.SHUT_RDWR)

    def _sample(self) -> list[Any]:
        woken: list[Any] = []
        now = time.monotonic()
        with self._lock:
            patience = max(STALL_FACTOR * self._worst_gap, STALL_SAMPLES * STALL_SAMPLE_SECONDS)
            for key, advanced in self._advanced.items():
                if now - advanced <= patience or key in self._stalled:
                    continue
                self._stalled.add(key)
                sock = self._socket.get(key)
                if sock is not None:
                    woken.append(sock)
        return woken


def _socket_of(resp: Any) -> Any:
    """The live socket under one urllib response, or None when the transport hides it.

    Reaching for it is deliberate and best-effort: without it a stalled stream can only be
    counted, not woken, and the leg degrades to exactly the behaviour it had before.
    """
    with contextlib.suppress(AttributeError):
        return resp.fp.raw._sock
    return None


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
        stamps=dict(at.stamps),
        streams=streams,
        declared=len(plan["held"]) + len(plan["wanted"]),
        held=len(plan["held"]),
        bytes_held=int(plan["held_bytes"]),
        bytes_total=int(plan["held_bytes"]) + int(plan["wanted_bytes"]),
        presence=dict(plan["presence"]),
        sources={str(object_id): "local" for object_id, _ in plan["held"]},
    )
    _execute_plan(
        tensorfs,
        store,
        plan,
        result,
        urls={r.object_id: r.url for r in declared},
        in_hand={"sha256:" + at.snapshot: doc},
        deadline=deadline,
        allow_local=local_hub(base),
        streams=streams,
        on_object=on_object,
    )
    return result


def pull_granted_snapshot(
    tensorfs: Any,
    store: Any,
    snapshot: str,
    snapshot_length: int,
    granted: Sequence[Read],
    *,
    timeout: float | None = None,
    session: str = "",
    streams: int = STREAMS,
    on_object: Callable[[str, int, bool], None] | None = None,
    cache_store: Any | None = None,
    allow_local: bool = False,
) -> PullResult:
    """Admit one exact SnapshotManifest closure from closed direct HTTPS grants.

    Rental provisioning already carries delivery authority, so resolving a mutable hub ref
    would add a second meaning clock.  This door instead takes one already-selected root and
    the exact per-root object rows which TensorFS must verify.  The caller partitions a
    release-wide ObjectSet by SnapshotManifest reachability; this function deliberately does
    not guess that a union of several roots is one root's closure.

    Every object, including the root manifest, travels through ``fetch_admit``.  No local
    provision copy or human-ref artifact index is written, and HTTPS grants never use an
    inherited proxy or the loopback development carve-out.
    """

    root = snapshot.removeprefix("sha256:")
    root_id = "sha256:" + root
    if (
        len(root) != 64
        or any(c not in "0123456789abcdef" for c in root)
        or not isinstance(snapshot_length, int)
        or isinstance(snapshot_length, bool)
        or snapshot_length <= 0
    ):
        raise PullRefusal(
            "artifact_incomplete",
            f"{snapshot}: not one bounded SnapshotManifest reference",
            "supply the exact sha256 digest and positive stored-byte length from the binding",
        )
    rows = list(granted)
    if not rows or [row.object_id for row in rows] != sorted(row.object_id for row in rows):
        raise PullRefusal(
            "artifact_incomplete",
            f"{root_id}: direct grants are absent or not sorted by object id",
            "derive one sorted, unique grant list for this SnapshotManifest closure",
        )
    by_id: dict[str, Read] = {}
    for row in rows:
        location = urllib.parse.urlsplit(row.url)
        if (
            row.object_id in by_id
            or not row.object_id.startswith("sha256:")
            or len(row.object_id) != 71
            or any(c not in "0123456789abcdef" for c in row.object_id[7:])
            or not isinstance(row.length, int)
            or isinstance(row.length, bool)
            or row.length <= 0
            or location.scheme != "https"
            or not location.netloc
            or location.username is not None
            or location.password is not None
            or bool(location.fragment)
        ):
            raise PullRefusal(
                "artifact_incomplete",
                f"{row.object_id}: invalid or duplicate direct grant identity",
                "author one positive-length HTTPS grant for every exact object id",
            )
        by_id[row.object_id] = row
    root_row = by_id.get(root_id)
    if root_row is None or root_row.length != snapshot_length:
        raise PullRefusal(
            "artifact_incomplete",
            f"{root_id}: root grant length does not equal {snapshot_length}",
            "grant the exact SnapshotManifest bytes named by the resolved binding",
        )

    streams = max(1, int(streams))
    deadline = Deadline.after(timeout)
    plan = _plan(
        tensorfs,
        store,
        session or ("provision-" + root[:12]),
        root,
        snapshot_length,
        [(row.object_id, row.length) for row in rows if row.object_id != root_id],
    )
    result = PullResult(
        ref=root_id,
        snapshot=root,
        store=str(store.root),
        streams=streams,
        declared=len(plan["held"]) + len(plan["wanted"]),
        held=len(plan["held"]),
        bytes_held=int(plan["held_bytes"]),
        bytes_total=int(plan["held_bytes"]) + int(plan["wanted_bytes"]),
        presence=dict(plan["presence"]),
        sources={str(object_id): "local" for object_id, _ in plan["held"]},
    )
    _execute_plan(
        tensorfs,
        store,
        plan,
        result,
        urls={object_id: row.url for object_id, row in by_id.items()},
        in_hand={},
        deadline=deadline,
        allow_local=allow_local,
        streams=streams,
        on_object=on_object,
        no_proxy=True,
        cache_store=cache_store,
    )
    return result


def _execute_plan(
    tensorfs: Any,
    store: Any,
    plan: Mapping[str, Any],
    result: PullResult,
    *,
    urls: Mapping[str, str],
    in_hand: Mapping[str, bytes],
    deadline: Deadline,
    allow_local: bool,
    streams: int,
    on_object: Callable[[str, int, bool], None] | None,
    no_proxy: bool = False,
    cache_store: Any | None = None,
) -> None:
    """Run one TensorFS-authored plan and require its own completion proof."""

    ledger = threading.Lock()
    liveness = _Streams()

    def degraded(code: str) -> None:
        with ledger:
            result.cache_degradations.add(code)

    def one(item: tuple[str, int]) -> None:
        object_id, length = item
        raw = in_hand.get(object_id)
        if cache_store is not None and _admit_cached(
            tensorfs, cache_store, store, object_id, length, degraded
        ):
            source = "network_cache"
        elif raw is not None:
            _admit(tensorfs, store, plan["document"], object_id, _Bytes(raw))
            _backfill(tensorfs, store, cache_store, object_id, length, degraded)
            source = "cloud"
        else:
            url = urls.get(object_id)
            if url is None:
                raise PullRefusal(
                    "artifact_incomplete",
                    f"{object_id}: the SnapshotManifest reaches it and no exact grant exists",
                    "repair the closed grant plane; Runtime never resolves an ungranted object",
                )
            _fetch(
                tensorfs,
                store,
                plan["document"],
                object_id,
                url,
                deadline,
                allow_local,
                no_proxy=no_proxy,
                streams=liveness,
            )
            _backfill(tensorfs, store, cache_store, object_id, length, degraded)
            source = "cloud"
        with ledger:
            result.fetched += 1
            result.bytes_moved += length
            result.objects.append(object_id)
            result.sources[object_id] = source
            if on_object is not None:
                on_object(object_id, length, True)

    wanted = [(str(o), int(n)) for o, n in plan["wanted"]]
    with liveness, ThreadPoolExecutor(max_workers=streams, thread_name_prefix="pull") as pool:
        futures = [pool.submit(one, item) for item in wanted]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        failed = next((f for f in done if f.exception() is not None), None)
        if failed is not None:
            for f in pending:
                f.cancel()
            for f in futures:
                if not f.cancelled():
                    f.exception()
            first = failed.exception()
            assert first is not None
            raise first
    try:
        tensorfs.fetch_complete(store, plan["document"])
    except Exception as exc:
        raise _border(exc, "artifact_incomplete") from exc


class _Bytes:
    """An in-hand document, presented to the border as a reader."""

    def __init__(self, data: bytes) -> None:
        self._data, self._at = data, 0

    def read(self, n: int) -> bytes:
        chunk = self._data[self._at : self._at + n]
        self._at += len(chunk)
        return chunk


def _admit_cached(
    tensorfs: Any,
    cache_store: Any,
    local_store: Any,
    object_id: str,
    length: int,
    degraded: Callable[[str], None],
) -> bool:
    transfer = tensorfs.transfer_object
    try:
        transfer(cache_store, local_store, object_id, length)
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        if code == "OBJECT_ABSENT":
            return False
        degraded("corrupt" if code in {"LENGTH_MISMATCH", "OBJECT_CORRUPT"} else "read_unavailable")
        return False
    return True


def _backfill(
    tensorfs: Any,
    local_store: Any,
    cache_store: Any | None,
    object_id: str,
    length: int,
    degraded: Callable[[str], None],
) -> None:
    if cache_store is None:
        return
    transfer = tensorfs.transfer_object
    try:
        transfer(local_store, cache_store, object_id, length)
    except Exception:
        degraded("backfill_unavailable")


def _fetch(
    tensorfs: Any,
    store: Any,
    plan_doc: bytes,
    object_id: str,
    url: str,
    deadline: Deadline,
    allow_local: bool,
    *,
    no_proxy: bool,
    streams: _Streams | None = None,
) -> None:
    """Move ONE object across the door, re-asking for it while re-asking is honest.

    The plan already made a retry cost one object rather than the pull, so the only question
    is which failures deserve one. A stall the ledger NAMED, a connection that died, and the
    object store's own retryable answers do; an expired presign and an absent key do not —
    those are facts about authorization, and asking again is a lie about time.
    """
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        last = attempt == FETCH_ATTEMPTS
        key = threading.get_ident()
        if streams is not None:
            streams.open(key)
        try:
            try:
                resp = open_guarded(url, deadline, allow_local=allow_local, no_proxy=no_proxy)
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_STATUS and not last:
                    continue
                raise PullRefusal(
                    "transfer_failed",
                    f"{object_id}: the object store answered {exc.code}",
                    "a presigned read expires; re-run the pull to be re-authorized",
                    cause=str(exc.reason),
                ) from exc
            except urllib.error.URLError as exc:
                if not last:
                    continue
                raise PullRefusal(
                    "transfer_failed",
                    f"{object_id}: the object store did not answer",
                    "re-run the pull: it resumes per object",
                    cause=str(exc),
                ) from exc
            try:
                if streams is not None:
                    streams.bind(key, _socket_of(resp))
                with resp:
                    _admit(tensorfs, store, plan_doc, object_id, _Reader(resp, object_id, streams))
                return
            except PullRefusal as exc:
                stalled = streams is not None and streams.stalled(key)
                if not last and (stalled or exc.code == "transfer_failed"):
                    continue
                if stalled:
                    raise PullRefusal(
                        "transfer_failed",
                        f"{object_id}: the stream moved no bytes while its peers moved theirs, "
                        f"{FETCH_ATTEMPTS} times",
                        "this link drops streams rather than slowing them; re-run the pull, "
                        "and if it repeats the route between this worker and the object store "
                        "is the subject, not the artifact",
                        cause=exc.code,
                    ) from exc
                raise
        finally:
            if streams is not None:
                streams.done(key)
    raise AssertionError("unreachable: the retry loop returns or raises on its last attempt")


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
    cause = str(getattr(exc, "code", ""))
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
        cause=cause,
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
