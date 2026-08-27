"""DeliveryGrant reads and writes, and the CLOSED hub-call authorization table.

The grant is refreshable ACCESS, never meaning (worker-protocol/01, #439): input
identities, lengths, media kinds and output contracts live INSIDE the InvocationSpec's
digested document, and the grant carries exactly `{invocation_digest, credential, urls,
expiry}`. `bind()` is the JOIN — one BoundGrant per attempt, spec identity zipped to grant
access, refused typed when the two disagree — and a refresh may re-derive access and
nothing else: there is no field left for it to substitute meaning through. This module is
where that authority is SPENT — the only place the worker reads an input or writes an
output — and every spend goes through `authorize()` first.

**The table is closed** (§3.1). Method + full-path regex + enumerated params, and a refusal
is a SECURITY EVENT, not a 404: it is counted, journaled by the caller and reported as a
`Fault`, because a worker asking for something it was never granted is a fact an operator
must see. Nothing widens the table at runtime; a new call site is a new reviewed row.

Two transports, one reader (cr-012). `file://` is the local deployment's own spool or CAS;
`https://` goes out through `internal/egress.py`, which holds a declared allowlist, resolves
fail-closed and re-applies the whole check on EVERY redirect hop. Both land in one bounded
reader. The EXECUTOR still has no outbound network at all — every fetch happens here, in the
worker, before acceptance, and the executor is handed verified files in the attempt
spool.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping
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


#: The COMPLETE set of calls this worker may make. Read it as the security boundary it
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
        "grant_input_fetch",
        "GET",
        r"^https://[\w.-]+(:\d+)?/[\w./?=&%~-]*$",
        ("input_id",),
        "fetch one granted input asset over the egress boundary (cr-012): the allowlist, "
        "the per-hop SSRF check and the bounded reader all apply BELOW this row — this "
        "table says the call may be made at all, not that the address is safe",
    ),
    HubCall(
        "grant_tree_read",
        "GET",
        r"^file://(?P<path>/[\w./-]+)$",
        ("input_id",),
        "open one granted MATERIALIZED INPUT TREE (a job's typed model/dataset input)",
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
    roots: tuple[str, ...] = ()
    """The DECLARED grant roots: every directory tree a `file://` grant may address on this
    machine, stated at launch (`serve --grant-root`) and never widened at request time.

    It closes the gap the remote leg made visible. A `file://` URL is an ADDRESS ON SOME
    MACHINE, and a worker handed one has no way to tell "the owner's attempt directory,
    which I share" from "a path that exists only on the owner's disk" — a pod handed the
    second would happily CREATE it, write the outputs into a directory nobody can read, and
    declare them in a true-looking manifest the owner then cannot mirror. Refusing it here
    turns that into an observed refusal at the named gate, before a byte moves.

    EMPTY MEANS NO FENCE, and that is the honest default for a launcher that declared none:
    a one-shot in-memory worker (§8) grants into directories it makes as it goes, and a
    fence over roots nobody stated would be a fence over a guess. Every launcher that knows
    its own root states it — cozy-creator passes its local root, a pod passes its media
    subtree — so the lanes that HAVE a boundary all carry one.
    """

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
        self._within_roots(name, method, url)
        self.allowed += 1

    def _within_roots(self, name: str, method: str, url: str) -> None:
        """A `file://` grant addresses a declared root or it addresses nothing here."""
        if not self.roots or not url.startswith("file://"):
            return
        target = Path(urlparse(url).path)
        for root in self.roots:
            try:
                target.relative_to(Path(root))
            except ValueError:
                continue
            return
        event = {"call": name, "method": method, "url": url[:200], "why": "outside every root"}
        self.refusals.append(event)
        raise GrantRefusal(
            "grant_path_outside_roots",
            f"{name}: the grant addresses {str(target)[:120]} and this worker's declared "
            f"grant roots are {list(self.roots)}. A `file://` URL is an address on SOME "
            "machine and a worker cannot tell a shared directory from one that exists only "
            "on the owner's disk — so the roots are declared at launch and a grant outside "
            "every one of them is refused here, before a byte moves, rather than becoming a "
            "directory this worker created and nobody else can read",
        )

    def _refuse(self, name: str, method: str, url: str, why: str) -> None:
        event = {"call": name, "method": method, "url": url[:200], "why": why}
        self.refusals.append(event)
        raise GrantRefusal(
            "hub_call_unauthorized",
            f"SECURITY EVENT — {method} {url[:120]} as {name!r}: {why}. The worker's "
            "hub-call table is closed; a refusal is not a 404",
        )


# --------------------------------------------------------------------------- the join


@dataclass(frozen=True, slots=True)
class BoundInput:
    """One input: identity from the SPEC (inside the digest), access from the GRANT."""

    input_id: str
    url: str
    digest: bytes  # raw 32-byte sha256, or b"" when the spec declares none (a tree)
    length: int
    kind_mime: str
    order: int


@dataclass(frozen=True, slots=True)
class BoundOutput:
    """One output destination: contract from the SPEC, presigned/granted URL from the GRANT."""

    output_id: str
    url: str
    max_bytes: int
    mime_type: str


@dataclass(slots=True)
class BoundGrant:
    """The attempt's one working view of spec identity + grant access."""

    file_base_url: str = ""
    expires_at_unix: int = 0
    credential_epoch: int = 0
    inputs: dict[str, BoundInput] = field(default_factory=dict)
    outputs: dict[str, BoundOutput] = field(default_factory=dict)


def bind(spec: Mapping[str, Any], grant: pb.DeliveryGrant, invocation_digest: bytes) -> BoundGrant:
    """JOIN the InvocationSpec document's identities to the grant's access, typed refusals.

    The grant serves exactly one invocation: its `invocation_spec_digest` must name this
    attempt's. The id SETS must agree exactly — an input the spec declares with no access
    is unfetchable, and access to an id the spec never declared is authority nobody's
    digest covers.
    """
    if grant.invocation_spec_digest and bytes(grant.invocation_spec_digest) != invocation_digest:
        raise GrantRefusal(
            "grant_subject_mismatch",
            "the delivery grant names a different invocation_spec_digest than this attempt; a "
            "grant serves exactly one invocation and a refresh can change access only",
        )
    spec_inputs = {str(row["input_id"]): row for row in spec.get("inputs", ())}
    grant_inputs = {entry.input_id: entry.url for entry in grant.inputs}
    if set(spec_inputs) != set(grant_inputs):
        raise GrantRefusal(
            "grant_binding_mismatch",
            f"the spec declares inputs {sorted(spec_inputs) or 'none'} and the grant "
            f"carries access for {sorted(grant_inputs) or 'none'}; the sets must agree "
            "exactly",
        )
    spec_outputs = {str(row["output_id"]): row for row in spec.get("outputs", ())}
    grant_outputs = {entry.output_id: entry.url for entry in grant.outputs}
    if set(spec_outputs) != set(grant_outputs):
        raise GrantRefusal(
            "grant_binding_mismatch",
            f"the spec declares outputs {sorted(spec_outputs) or 'none'} and the grant "
            f"carries access for {sorted(grant_outputs) or 'none'}; the sets must agree "
            "exactly",
        )
    bound = BoundGrant(
        file_base_url=grant.file_base_url,
        expires_at_unix=int(grant.expires_at_unix),
        credential_epoch=int(grant.credential.epoch),
    )
    for order, (input_id, row) in enumerate(sorted(spec_inputs.items())):
        spelled = str(row.get("digest", ""))
        bound.inputs[input_id] = BoundInput(
            input_id=input_id,
            url=grant_inputs[input_id],
            digest=bytes.fromhex(spelled[7:]) if spelled.startswith("sha256:") else b"",
            length=int(row.get("length", 0)),
            kind_mime=str(row.get("kind_mime", "")),
            order=int(row.get("order", order)),
        )
    for output_id, row in spec_outputs.items():
        bound.outputs[output_id] = BoundOutput(
            output_id=output_id,
            url=grant_outputs[output_id],
            max_bytes=int(row.get("max_bytes", 0)),
            mime_type=str(row.get("mime_type", "")),
        )
    return bound


# --------------------------------------------------------------------------- grant I/O


def _local(url: str, base: str) -> Path:
    absolute = url if "://" in url else base.rstrip("/") + "/" + url.lstrip("/")
    parsed = urlparse(absolute)
    if parsed.scheme != "file":
        raise GrantRefusal(
            "grant_transport",
            f"{parsed.scheme or 'relative'}:// is not a local grant URL; an https grant is "
            "fetched through the egress boundary (`internal/egress.py`), never opened here",
        )
    return Path(parsed.path)


def _read_granted(url: str, base: str, *, declared: int, policy: Any, what: str) -> bytes:
    """The ONE way a granted object's bytes enter this process (cr-012).

    Two transports and one reader. `file://` is the local deployment's own CAS or spool;
    `https://` goes through the egress boundary, which re-validates every redirect hop.
    Both land in `read_bounded`, which holds the stream to the smaller of the grant's
    declared length and the deployment cap — there is deliberately no second downloader,
    because "two downloaders and only one had the check" is the failure this shape exists
    against.
    """
    from cozy_runtime.internal import egress

    absolute = url if "://" in url else base.rstrip("/") + "/" + url.lstrip("/")
    if urlparse(absolute).scheme == "file":
        path = _local(url, base)
        with open(path, "rb") as handle:
            return egress.read_bounded(
                iter(lambda: handle.read(egress.CHUNK), b""),
                declared=declared,
                cap=policy.max_bytes,
                what=what,
            )
    if policy is None or not policy.allowed_hosts:
        raise GrantRefusal(
            "grant_transport",
            f"{what}: the grant names {absolute[:80]} and this deployment declares no "
            "allowed egress hosts. The allowlist is a deployment-wide declaration and "
            "nothing widens it at request time, so a remote grant is unfetchable here",
        )
    return egress.fetch(absolute, policy, declared=declared, what=what)


def expired(grant: BoundGrant, now: float | None = None) -> bool:
    at = time.time() if now is None else now
    return bool(grant.expires_at_unix) and grant.expires_at_unix < at


def read_input(grant: BoundGrant, input_id: str, auth: Authorizer, policy: Any = None) -> bytes:
    """One bound input, verified against the SPEC's digest and length (the grant carries
    access only, #439)."""
    from cozy_runtime.internal import egress

    if expired(grant):
        raise GrantRefusal(
            "grant_expired",
            f"the delivery grant expired at {grant.expires_at_unix}; a resolver fallback is "
            "a NEW attempt, never a patch to this one",
        )
    policy = policy or egress.EgressPolicy()
    entry = grant.inputs.get(input_id)
    if entry is None:
        raise GrantRefusal(
            "input_absent",
            f"no bound input {input_id!r} (this attempt has "
            f"{', '.join(sorted(grant.inputs)) or 'none'})",
        )
    url = entry.url if "://" in entry.url else grant.file_base_url.rstrip("/") + "/" + entry.url
    auth.authorize(
        "grant_input_read" if url.startswith("file://") else "grant_input_fetch",
        "GET",
        url,
        {"input_id": input_id},
    )
    try:
        data = _read_granted(
            entry.url,
            grant.file_base_url,
            declared=int(entry.length),
            policy=policy,
            what=f"input {input_id!r}",
        )
    except egress.EgressRefusal as exc:
        raise GrantRefusal(exc.code, exc.detail) from exc
    if entry.length and len(data) != entry.length:
        raise GrantRefusal(
            "input_length_mismatch",
            f"{input_id}: the spec declares {entry.length} B, read {len(data)} B",
        )
    if entry.digest and hashlib.sha256(data).digest() != entry.digest:
        raise GrantRefusal(
            "input_digest_mismatch",
            f"{input_id}: content digest disagrees with the invocation spec — a grant never "
            "bypasses the local bytes check",
        )
    return data


#: How a granted input TREE is spelled on a DeliveryGrant. The `input_id` carries the ref
#: the job's typed field names, so the mapping from field value to grant is the ref itself.
TREE_PREFIX = "tree:"

#: The one input_id that is not an asset: the request DOCUMENT itself rides the grant as an
#: input, which is why the payload and the assets share one authorization table and one
#: bounded reader. It is excluded from asset hydration by name because it is the only
#: reserved id there is (§1.3's payload/asset split, cr-009's 64 KiB seam law: control goes
#: through the frame, DATA goes through the spool, and the seam is what makes them different
#: things rather than one field that got large).
PAYLOAD_INPUT = "payload"

#: A conservative name for a spooled input. The input_id is a FIELD PATH the caller can
#: influence through list indices, so it never reaches the filesystem uninspected.
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def hydrate_inputs(
    grant: BoundGrant,
    auth: Authorizer,
    *,
    spool: Path,
    policy: Any = None,
    max_bytes: int = 64 << 20,
    max_total_bytes: int = 256 << 20,
) -> dict[str, Any]:
    """Fetch, verify, sniff and spool EVERY input asset the grant carries (cr-012).

    THE LAUNCH CONTRACT IS EAGER (§1.3), and this is where the eagerness lives: the whole
    present asset set is materialized here, before acceptance and before any lease, so a
    slow URL can never pin staged weights. Branch-selective lazy hydration is a deferred
    optimization behind the same author-facing type; the ordering law it would have to
    keep is exactly this one.

    Every size check happens on the grant's DECLARED length first — so an oversized input
    refuses before a byte moves — and again on the bytes, because a declaration is a claim.
    """
    from cozy_runtime.author._assets import GrantedInput
    from cozy_runtime.author._media import SNIFF_BYTES, normalize, sniff

    if expired(grant):
        raise GrantRefusal("grant_expired", "the delivery grant expired before hydration")
    entries = [
        e
        for e in sorted(grant.inputs.values(), key=lambda e: (e.order, e.input_id))
        if e.input_id != PAYLOAD_INPUT and not e.input_id.startswith(TREE_PREFIX)
    ]
    declared_total = sum(int(e.length) for e in entries)
    if declared_total > max_total_bytes:
        raise GrantRefusal(
            "inputs_over_total_cap",
            f"this attempt's {len(entries)} input asset(s) declare {declared_total} B and "
            f"this deployment admits {max_total_bytes} B per attempt. Input retention is "
            "its own bounded story: inputs are not weights, they take no CAS lease, and "
            "they live exactly as long as the attempt spool",
        )
    out: dict[str, Any] = {}
    root = spool / "inputs"
    for order, entry in enumerate(entries):
        if entry.length and int(entry.length) > max_bytes:
            raise GrantRefusal(
                "input_over_cap",
                f"input {entry.input_id!r} declares {entry.length} B and this deployment "
                f"admits {max_bytes} B per input. The DECLARED length refuses before a byte "
                "moves; the bytes are held to the same bound on the way in",
            )
        data = read_input(grant, entry.input_id, auth, policy)
        declared = normalize(entry.kind_mime)
        actual = sniff(data[:SNIFF_BYTES])
        if declared and actual and declared != actual:
            raise GrantRefusal(
                "input_media_type",
                f"input {entry.input_id!r} is declared {declared!r} and its bytes are "
                f"{actual!r}. The declaration is a claim by whoever minted the grant; the "
                "signature is what the object is, and a mislabelled input refuses rather "
                "than reaching a decoder under a type nobody agreed to",
            )
        root.mkdir(parents=True, exist_ok=True)
        local = root / f"{order:03d}-{_SAFE.sub('_', entry.input_id)[:96]}"
        local.write_bytes(data)
        out[entry.input_id] = GrantedInput(
            input_id=entry.input_id,
            local=local,
            media_type=actual or declared,
            digest="sha256:" + entry.digest.hex() if entry.digest else "",
            length=len(data),
            order=int(entry.order) or order,
        )
    return out


def release_inputs(spool: Path) -> int:
    """Drop this attempt's hydrated inputs. Returns how many files went.

    Input retention is its OWN bounded story, deliberately outside the weights CAS: an
    input takes no lease, participates in no eviction ladder, and is not content-addressed
    storage — it is one attempt's spool and it ends with the attempt.
    """
    root = spool / "inputs"
    removed = 0
    for path in sorted(root.glob("*")) if root.is_dir() else ():
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def read_trees(grant: BoundGrant, auth: Authorizer) -> dict[str, tuple[Path, str]]:
    """Every granted input tree, as `ref -> (root, declared digest)` (cr-009).

    The BYTES of a tree are not re-hashed here and that is a layering statement, not a gap:
    a tree is a materialized TensorFS snapshot, and its verification is the store's own —
    the job reads it back through the verified read path, which re-checks every object it
    actually touches (law 18, tfs-007). What this function enforces is the part it owns:
    the grant must NAME the tree, the grant must be live, the call must be authorized, and
    the address must actually be a directory. An ungranted ref never reaches a path at all.
    """
    if expired(grant):
        raise GrantRefusal("grant_expired", "the delivery grant expired before the tree read")
    out: dict[str, tuple[Path, str]] = {}
    for entry in sorted(grant.inputs.values(), key=lambda e: e.input_id):
        if not entry.input_id.startswith(TREE_PREFIX):
            continue
        ref = entry.input_id[len(TREE_PREFIX) :]
        url = entry.url if "://" in entry.url else grant.file_base_url.rstrip("/") + "/" + entry.url
        auth.authorize("grant_tree_read", "GET", url, {"input_id": entry.input_id})
        root = _local(entry.url, grant.file_base_url)
        if not root.is_dir():
            raise GrantRefusal(
                "tree_not_a_directory",
                f"{ref}: the grant names {root} as a materialized tree and it is not a "
                "directory; the RecordOwner minted this address, so the worker substitutes none",
            )
        out[ref] = (root, "sha256:" + entry.digest.hex() if entry.digest else "")
    return out


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """ONE canonical bundle snapshot per publication (cr-009).

    What an attempt actually wrote, as a document with its own identity: the scope it wrote
    under, whose build wrote it, which attempt, and every entry with its digest and length.
    A RecordOwner validates it by recomputing this document from what it already holds —
    the publication contract it minted and the manifest on the terminal — and roots and
    projects on the digest. Before this there was no single typed subject per publication,
    so the Creator side was reading the mid-attempt CHECKPOINT lane to infer one, which is
    a durability lane for a job's own progress and not a record of what it published
    (cl-004).

    The identity travels as `OutputManifest.manifest_id`. That field used to be a label
    (`man-<request>-<attempt>`) which named nothing a reader could check.
    """

    mode: str
    """`job` or `serving`. One rule for both, so there is one spelling of the field."""
    grant_id: str
    """The publication contract's scope — the scratch repo a job's writes are rewritten
    into. Empty on the serving lane, which publishes into granted destinations only."""
    build_id: str
    subject_id: str
    """`job_descriptor_id` in job mode, `entrypoint_binding_plan_id` in serving mode."""
    request_id: str
    attempt: int
    outputs: tuple[pb.OutputEntry, ...]

    def document(self) -> dict[str, Any]:
        rows = sorted(self.outputs, key=lambda entry: entry.output_id)
        return {
            "format": "cozy.runtime.PublicationReceipt/1",
            "mode": self.mode,
            "grant_id": self.grant_id,
            "build_id": self.build_id,
            "subject_id": self.subject_id,
            "request_id": self.request_id,
            "attempt": self.attempt,
            "outputs": [
                {
                    "output_id": entry.output_id,
                    "digest": "sha256:" + entry.digest.hex(),
                    "length": int(entry.length),
                    "media_type": entry.mime_type,
                }
                for entry in rows
            ],
            "total_bytes": sum(int(entry.length) for entry in rows),
        }

    def digest(self) -> str:
        from cozy_runtime.internal import canonical

        return canonical.digest(self.document())


def write_output(
    grant: BoundGrant,
    output_id: str,
    data: bytes,
    auth: Authorizer,
    media_type: str = "",
) -> pb.OutputEntry:
    """Write ONE output under the EXACT destination the grant names.

    A stale or wrong grant cannot write: the destination must be in this grant, the grant
    must be live, and the bytes must fit the destination's declared cap. Writing produces a
    manifest entry and NOTHING else — the runtime may write under an exact output grant and
    cannot authorize readers, publish media, or decide a winning attempt.
    """
    if expired(grant):
        raise GrantRefusal("grant_expired", "the output grant expired before the write")
    destination = grant.outputs.get(output_id)
    if destination is None:
        raise GrantRefusal(
            "destination_absent",
            f"no granted destination {output_id!r} (this attempt names "
            f"{', '.join(sorted(grant.outputs)) or 'none'}) — an output the owner did not "
            "grant is unwritable, however the attempt produced it",
        )
    url = (
        destination.url
        if "://" in destination.url
        else grant.file_base_url.rstrip("/") + "/" + destination.url
    )
    auth.authorize("grant_output_write", "PUT", url, {"output_id": output_id, "length": len(data)})
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
        # The media type the AUTHOR declared when the asset was saved.
        # `application/octet-stream` remains the honest answer when nothing declared one.
        mime_type=media_type or "application/octet-stream",
    )


def abort_outputs(spool: Path) -> int:
    """Abort an attempt's uncommitted outputs. Bytes written under the grant are the
    RecordOwner's problem only once a terminal claims them; the spool is ours."""
    removed = 0
    for path in sorted(spool.rglob("*")):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed
