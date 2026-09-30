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
is a SECURITY EVENT, not a 404: it is counted, recorded by the caller and reported as a
`Fault`, because a worker asking for something it was never granted is a fact an operator
must see. Nothing widens the table at runtime; a new call site is a new reviewed row.

**EVERY DESTINATION IS CONTROL-PLANE-MINTED, never named by request content.** The grant is
minted for one invocation and spent here, so nothing the package produced — and nothing a
request body can influence — chooses where bytes land: a destination the table has no row
for refuses typed. `scripts/worker-live.py grants` plants a presigned-looking metadata-address
destination and observes `egress_blocked_address` through the real terminal. An input row is
`file://`, or — for the payload only — the document itself as a `data:` URL: the grant IS
the bytes, so a record owner on the same host stages nothing on disk to hand them over. The
output row admits the exact `file://` path or `https://` presigned URL already carried by
`OutputAccess.url` (Tensorhub mints it, R2 sits behind it, and the worker never holds bucket
credentials), or a `file://` DIRECTORY (trailing slash): the worker names the file there by
its own content digest, `<sha256>.<ext>`, so the granted directory is the durable store
itself and a regenerated file lands on the one already there. `internal.egress` validates
every resolved address and refuses a redirect-selected destination. That is what keeps the
property; a `file://`-only table never did — `scripts/lifecycle-live.py --only egress`
watches a `ctypes` connect(2) LAND from inside the serving executor, so a hostile package
already holds a socket. The rung is the fence's own, below: a careless
package and a malicious GRANT, never a malicious package.

cr-012 shipped an `https://` INPUT row and an `EgressPolicy` allowlist for it. Both are gone
(cr-042): `DeclaredBinding.egress_allowed_hosts` never had a writer, so the row refused every
`https://` grant unconditionally, and its per-binding shape was also unioned across
deployments, which decision #451 forbids. `internal/egress.py` survives — it owns the ONE
bounded reader `copy_bounded`, the ONE address predicate both downloaders call, and the
fetch boundary th-008's hub-minted input URLs and cr-012's author `Egress` capability will
enter through. Whoever lands that row lands its allowlist writer with it.

The executor is handed verified files at their granted paths and never an address, and it
installs `sandbox.refuse_network` so a dial of its own refuses (cr-042) — a Python-level
bound on a buggy package, not a kernel bound on a hostile one.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping, Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast
from urllib.parse import ParseResult, unquote, urlparse

import msgspec

from cozy_runtime.author._assets import GrantedInput, file_state
from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.author._media import SNIFF_BYTES, extension, normalize, reconcile, sniff
from cozy_runtime.internal import canonical, egress
from cozy_runtime.internal.canonical import Json
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.workspace import Workspace


class GrantRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


#: How a granted input TREE is spelled on a DeliveryGrant. The `input_id` carries the ref
#: the job's typed field names, so the mapping from field value to grant is the ref itself.
TREE_PREFIX = "tree:"

#: A job Model parameter is one exact TensorFS Manifest identity. Its grant access is an
#: opaque host-local capability, never a filesystem path or URL the executor can spend.
MODEL_PREFIX = "model:"
MODEL_MIME = "application/vnd.cozy.model-manifest"
_MODEL_PARAM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MODEL_SOURCE_PARAM = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]{0,127}\.(?:adapter\.(?:0|[1-9][0-9]{0,18})|composed)$"
)

#: The one input_id that is not an asset: the request DOCUMENT itself rides the grant as an
#: input, which is why the payload and the assets share one authorization table and reader.
PAYLOAD_INPUT = "payload"


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
#:
#: Input rows are `file://`, plus the inline `data:` form the payload may take. The output row
#: admits the local path, the local directory, or the exact HTTPS capability the control plane
#: minted. The path/query never appears in a refusal event because a presigned query is
#: structurally secret, and a `data:` body never does because it is the request. Armed by
#: `scripts/worker-live.py grants` and `scripts/assets-live.py egress`.
HUB_CALLS: tuple[HubCall, ...] = (
    HubCall(
        "grant_input_read",
        "GET",
        r"^(?:file://(?P<path>/(?:[\w./~-]|%[0-9A-Fa-f]{2})+)|data:application/json;base64,[A-Za-z0-9+/]*={0,2})$",
        ("input_id",),
        "read one granted input asset (the payload rides the grant as an input, inline or a file)",
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
        r"^(?:file:///[\w./-]+|https://[A-Za-z0-9.-]+(?::[0-9]{1,5})?/[^\s#]+)$",
        ("output_id", "length"),
        "write one granted output destination: an exact path or URL, or a directory the "
        "file is named in by its content digest",
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

    def authorize(self, name: str, method: str, url: str, params: Mapping[str, str | int]) -> None:
        row = CALLS.get(name)
        if row is None:
            self._refuse(name, method, url, f"{name!r} is not in the closed hub-call table")
        assert row is not None
        if method != row.method:
            self._refuse(name, method, url, f"{name} is {row.method}, not {method}")
        if re.match(row.path, url) is None:
            self._refuse(name, method, url, f"{_safe_url(url)!r} does not match {row.path}")
        extra = sorted(set(params) - set(row.params))
        if extra:
            self._refuse(name, method, url, f"unenumerated params {', '.join(extra)}")
        self._within_roots(name, method, url)
        self.allowed += 1

    def _within_roots(self, name: str, method: str, url: str) -> None:
        """A `file://` grant addresses a declared root or it addresses nothing here."""
        if not self.roots or not url.startswith("file://"):
            return
        target = _local(url, "")
        if _select_root(target, self.roots) is not None:
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
        safe_url = _safe_url(url)
        event = {"call": name, "method": method, "url": safe_url, "why": why}
        self.refusals.append(event)
        raise GrantRefusal(
            "hub_call_unauthorized",
            f"SECURITY EVENT — {method} {safe_url} as {name!r}: {why}. The worker's "
            "hub-call table is closed; a refusal is not a 404",
        )


def _safe_url(url: str) -> str:
    """One credential-free URL spelling for faults and events."""
    if url.startswith("data:"):
        head, _, body = url.partition(",")
        return f"{head},<{len(body)} chars>"
    try:
        parsed = urlparse(url)
    except ValueError:
        return "<malformed-url>"
    if parsed.scheme not in ("http", "https"):
        return url[:200]
    host = parsed.hostname or "<missing-host>"
    try:
        port = parsed.port
    except ValueError:
        return "<malformed-url>"
    if port is not None:
        host += f":{port}"
    return f"{parsed.scheme}://{host}{parsed.path[:120]}"


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
    native_tree: pb.NativeByteRetentionRequest | None = None
    catalog_repository: str = ""


def _tree_input(entry: BoundInput) -> bool:
    from cozy_runtime.internal.worker.workspace_byte_outputs import TREE_MIME

    return entry.input_id.startswith(TREE_PREFIX) or (
        entry.native_tree is not None and entry.kind_mime == TREE_MIME
    )


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


class _SpecInput(msgspec.Struct, frozen=True, kw_only=True):
    input_id: str = ""
    digest: str = ""
    length: int = 0
    kind_mime: str = ""
    order: int = 0


class _SpecOutput(msgspec.Struct, frozen=True, kw_only=True):
    output_id: str = ""
    mime_type: str = ""
    max_bytes: int = 0


class _Present(msgspec.Struct, frozen=True):
    """A section whose presence is the fact read here."""


class _SpecBindings(msgspec.Struct, frozen=True, kw_only=True):
    """The InvocationSpec document's binding identities; absent fields are proto3 defaults."""

    inputs: tuple[_SpecInput, ...] = ()
    outputs: tuple[_SpecOutput, ...] = ()
    serving: _Present | None = None
    job: _Present | None = None


def bind(
    spec: Mapping[str, object], grant: pb.DeliveryGrant, invocation_digest: bytes
) -> BoundGrant:
    """JOIN the InvocationSpec document's identities to the grant's access, typed refusals.

    The grant serves exactly one invocation: its `invocation_spec_digest` must name this
    attempt's. The id SETS must agree exactly — an input the spec declares with no access
    is unfetchable, and access to an id the spec never declared is authority nobody's
    digest covers.
    """
    subject = bytes(grant.invocation_spec_digest)
    if (
        len(subject) != hashlib.sha256().digest_size
        or len(invocation_digest) != hashlib.sha256().digest_size
        or subject != invocation_digest
    ):
        raise GrantRefusal(
            "grant_subject_mismatch",
            "the delivery grant must name this attempt's exact 32-byte "
            "invocation_spec_digest; a grant serves exactly one invocation and a refresh "
            "can change access only",
        )
    try:
        declared = msgspec.convert(spec, _SpecBindings)
    except msgspec.ValidationError as exc:
        raise GrantRefusal(
            "grant_spec_malformed", f"the invocation spec's bindings: {exc}"
        ) from exc
    spec_inputs = _unique(
        {row.input_id: row for row in declared.inputs}, declared.inputs, "input_id", "spec inputs"
    )
    grant_inputs = _unique(
        {row.input_id: row.url for row in grant.inputs}, grant.inputs, "input_id", "grant inputs"
    )
    if set(spec_inputs) != set(grant_inputs):
        raise GrantRefusal(
            "grant_binding_mismatch",
            f"the spec declares inputs {sorted(spec_inputs) or 'none'} and the grant "
            f"carries access for {sorted(grant_inputs) or 'none'}; the sets must agree "
            "exactly",
        )
    spec_outputs = _unique(
        {row.output_id: row for row in declared.outputs},
        declared.outputs,
        "output_id",
        "spec outputs",
    )
    grant_outputs = _unique(
        {row.output_id: row.url for row in grant.outputs},
        grant.outputs,
        "output_id",
        "grant outputs",
    )
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
        credential_epoch=int(grant.credential.credential_epoch),
    )
    native_inputs = {
        row.input_id: row.native_tree for row in grant.inputs if row.HasField("native_tree")
    }
    catalog_inputs = {
        row.input_id: row.catalog_model.repository
        for row in grant.inputs
        if row.HasField("catalog_model")
    }
    for input_id, row in spec_inputs.items():
        spelled, carried_order, carried_length = row.digest, row.order, row.length
        from cozy_runtime.internal.worker.workspace_byte_outputs import TREE_MIME

        native = native_inputs.get(input_id)
        is_tree = input_id.startswith(TREE_PREFIX) or (
            native is not None and row.kind_mime == TREE_MIME
        )
        is_model = input_id.startswith(MODEL_PREFIX)
        catalog_repository = catalog_inputs.get(input_id, "")
        if input_id in catalog_inputs and (
            not is_model
            or native is not None
            or re.fullmatch(r"[A-Za-z0-9._-]{1,128}/[A-Za-z0-9._-]{1,128}", catalog_repository)
            is None
            or any(part in (".", "..") for part in catalog_repository.split("/"))
        ):
            raise GrantRefusal(
                "grant_catalog_access", "catalog source requires one exact Model repository grant"
            )
        if not is_tree and (
            not spelled.startswith("sha256:")
            or len(spelled) != len("sha256:") + hashlib.sha256().digest_size * 2
        ):
            raise GrantRefusal(
                "grant_input_identity",
                f"input {input_id!r} does not carry one exact sha256 identity",
            )
        if not is_tree and carried_length < 0:
            raise GrantRefusal(
                "grant_input_identity",
                f"input {input_id!r} does not carry one exact nonnegative length",
            )
        try:
            digest = bytes.fromhex(spelled[7:]) if spelled.startswith("sha256:") else b""
        except ValueError as exc:
            raise GrantRefusal(
                "grant_input_identity", f"input {input_id!r} has a malformed sha256 digest"
            ) from exc
        if is_model:
            parameter = input_id[len(MODEL_PREFIX) :]
            if declared.job is None and not (declared.serving is not None and catalog_repository):
                raise GrantRefusal(
                    "grant_model_serving_refused",
                    f"serving InvocationSpec cannot carry job Model input {input_id!r}",
                )
            # Generated adapter/view capabilities are independent catalog roots, not
            # authored Model parameters. Admit only their closed serving namespace.
            auxiliary_source = (
                declared.serving is not None
                and declared.job is None
                and bool(catalog_repository)
                and _MODEL_SOURCE_PARAM.fullmatch(parameter) is not None
            )
            if (
                (_MODEL_PARAM.fullmatch(parameter) is None and not auxiliary_source)
                or row.kind_mime != MODEL_MIME
                or carried_order != 0
                or carried_length <= 0
                or spelled != "sha256:" + digest.hex()
                or grant_inputs[input_id] != f"model://{spelled}"
            ):
                raise GrantRefusal(
                    "grant_model_identity",
                    f"input {input_id!r} is not one exact host-local Model Manifest capability",
                )
        if native is not None:
            from cozy_runtime.internal.worker.byte_inputs import validate_source

            try:
                validate_source(native.source, native.retention_id)
            except ValueError as exc:
                raise GrantRefusal("grant_native_identity", str(exc)) from exc
            if grant_inputs[input_id] or is_model or input_id == PAYLOAD_INPUT:
                raise GrantRefusal(
                    "grant_native_access", "native tree and URL access are mutually exclusive"
                )
            if is_tree and (
                digest != native.source.manifest.digest
                or carried_length != native.source.manifest.length
                or (
                    input_id.startswith(TREE_PREFIX)
                    and input_id[len(TREE_PREFIX) :] != "sha256:" + digest.hex()
                )
            ):
                raise GrantRefusal(
                    "grant_native_identity", "native tree input differs from its manifest"
                )
            copied = pb.NativeByteRetentionRequest()
            copied.CopyFrom(native)
            native = copied
        elif not grant_inputs[input_id]:
            raise GrantRefusal("grant_input_access", "input has neither native tree nor URL access")
        bound.inputs[input_id] = BoundInput(
            input_id=input_id,
            url=grant_inputs[input_id],
            native_tree=native,
            catalog_repository=catalog_repository,
            digest=digest,
            length=carried_length,
            kind_mime=row.kind_mime,
            order=carried_order,
        )
    for output_id, contract in spec_outputs.items():
        bound.outputs[output_id] = BoundOutput(
            output_id=output_id,
            url=grant_outputs[output_id],
            max_bytes=contract.max_bytes,
            mime_type=contract.mime_type,
        )
    return bound


def model_inputs(grant: BoundGrant) -> dict[str, BoundInput]:
    """Descriptor parameter -> exact Manifest identity, with no access address exposed."""
    return {
        input_id[len(MODEL_PREFIX) :]: entry
        for input_id, entry in grant.inputs.items()
        if input_id.startswith(MODEL_PREFIX)
    }


def _unique[T](indexed: dict[str, T], rows: Sized, field_name: str, what: str) -> dict[str, T]:
    """One binding set, refused when an identity is empty or repeated: never last-wins."""
    if "" in indexed or len(indexed) != len(rows):
        problem = "empty" if "" in indexed else "repeated"
        raise GrantRefusal(
            "grant_binding_ambiguous",
            f"{what} carry an {problem} {field_name}; every binding identity must be "
            "non-empty and unique",
        )
    return indexed


# --------------------------------------------------------------------------- grant I/O


def _parse_grant_url(url: str) -> ParseResult:
    """Parse untrusted grant text without leaking parser exceptions into admission."""
    try:
        return urlparse(url)
    except ValueError as exc:
        raise GrantRefusal("grant_transport", f"the grant URL is malformed ({exc})") from exc


def _local(url: str, base: str) -> Path:
    """Resolve the exact local target used by both authorization and I/O.

    `Path.relative_to` on a lexical path admits `root/../secret`, and opening the raw path
    after checking a resolved one would restore the same escape. Returning only the
    canonical target makes traversal and existing symlink escapes visible to the root
    fence and makes the later open spend that same address.
    """
    absolute = url if "://" in url else base.rstrip("/") + "/" + url.lstrip("/")
    parsed = _parse_grant_url(absolute)
    if parsed.scheme != "file":
        raise GrantRefusal(
            "grant_transport",
            f"{parsed.scheme or 'relative'}:// is not a local grant URL; an https grant is "
            "fetched through the egress boundary (`internal/egress.py`), never opened here",
        )
    if parsed.netloc or parsed.query or parsed.fragment:
        raise GrantRefusal("grant_transport", "local grants require an exact file path")
    return Path(unquote(parsed.path)).resolve()


def _select_root(path: Path, roots: tuple[str, ...]) -> tuple[Path, Path] | None:
    """Return one declared root and the path relative to it, if any."""
    for declared in roots:
        root = Path(declared).resolve()
        try:
            return root, path.relative_to(root)
        except ValueError:
            continue
    return None


def _open_local_grant(path: Path, auth: Authorizer, absolute: str, input_id: str) -> int:
    """Authorize and open one local grant beneath a held declared-root descriptor.

    The root fd is opened before authorization.  A parent renamed to a symlink after the
    authorization check therefore cannot redirect the later read: every component is
    walked relative to that held directory with no-follow semantics.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not auth.roots:
        auth.authorize("grant_input_read", "GET", absolute, {"input_id": input_id})
        return os.open(path, os.O_RDONLY | nofollow)

    selected = _select_root(path, auth.roots)
    if selected is None:
        # Preserve the closed-table refusal event and its stable typed code.
        auth.authorize("grant_input_read", "GET", absolute, {"input_id": input_id})
        raise GrantRefusal(
            "grant_path_outside_roots", f"input {input_id!r} is outside every declared root"
        )

    root, relative = selected
    held: list[int] = [os.open(root, os.O_RDONLY | os.O_DIRECTORY | nofollow)]
    try:
        auth.authorize("grant_input_read", "GET", absolute, {"input_id": input_id})
        for position, component in enumerate(relative.parts):
            flags = os.O_RDONLY | nofollow
            if position < len(relative.parts) - 1:
                flags |= os.O_DIRECTORY
            held.append(os.open(component, flags, dir_fd=held[-1]))
        return held.pop()
    finally:
        for descriptor in reversed(held):
            os.close(descriptor)


def _transfer_granted(
    grant: BoundGrant,
    entry: BoundInput,
    auth: Authorizer,
    sink: BinaryIO | None,
    *,
    cap: int,
    prefix_bytes: int,
) -> egress.StreamReceipt:
    """Authorize, stream, measure and verify one exact bound input.

    The payload may ride the grant inline (`data:`); every other input is `file://`, and
    `_local` refuses any other scheme before this reads a byte, which is the whole transport
    decision: the worker's grant table has no network row (cr-042).
    """
    if expired(grant):
        raise GrantRefusal(
            "grant_expired",
            f"the delivery grant expired at {grant.expires_at_unix}; a resolver fallback is "
            "a NEW attempt, never a patch to this one",
        )
    if entry.url.startswith("data:"):
        return _transfer_inline(entry, auth, sink, cap=cap, prefix_bytes=prefix_bytes)
    local_path = _local(entry.url, grant.file_base_url)
    absolute = local_path.as_uri()
    try:
        try:
            descriptor = _open_local_grant(local_path, auth, absolute, entry.input_id)
        except OSError as exc:
            raise GrantRefusal(
                "input_not_a_file",
                f"{entry.input_id}: the grant does not name an openable regular file",
            ) from exc
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise GrantRefusal(
                    "input_not_a_file",
                    f"{entry.input_id}: the grant does not name a regular file",
                )
            receipt = egress.copy_bounded(
                iter(lambda: handle.read(egress.CHUNK), b""),
                sink,
                expected_length=entry.length,
                cap=cap,
                prefix_bytes=prefix_bytes,
                what=f"input {entry.input_id!r}",
            )
    except egress.EgressRefusal as exc:
        raise GrantRefusal(exc.code, exc.detail) from exc
    if receipt.sha256 != entry.digest:
        raise GrantRefusal(
            "input_digest_mismatch",
            f"{entry.input_id}: content digest disagrees with the invocation spec — a "
            "grant never bypasses the local bytes check",
        )
    return receipt


def _transfer_inline(
    entry: BoundInput, auth: Authorizer, sink: BinaryIO | None, *, cap: int, prefix_bytes: int
) -> egress.StreamReceipt:
    """The payload as the grant carries it: decoded, bounded and digest-checked exactly as a
    file would be. Only the payload is admitted inline — an asset is bytes the record owner
    holds in its own store and grants by address."""
    if entry.input_id != PAYLOAD_INPUT:
        raise GrantRefusal(
            "grant_transport",
            f"{entry.input_id}: only the payload rides the grant inline; an asset is granted "
            "by address",
        )
    auth.authorize("grant_input_read", "GET", entry.url, {"input_id": entry.input_id})
    try:
        data = base64.b64decode(entry.url.partition(",")[2], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GrantRefusal(
            "grant_transport", f"{entry.input_id}: the inline payload is not base64"
        ) from exc
    try:
        receipt = egress.copy_bounded(
            iter((data,)),
            sink,
            expected_length=entry.length,
            cap=cap,
            prefix_bytes=prefix_bytes,
            what=f"input {entry.input_id!r}",
        )
    except egress.EgressRefusal as exc:
        raise GrantRefusal(exc.code, exc.detail) from exc
    if receipt.sha256 != entry.digest:
        raise GrantRefusal(
            "input_digest_mismatch",
            f"{entry.input_id}: content digest disagrees with the invocation spec — a "
            "grant never bypasses the bytes check",
        )
    return receipt


def expired(grant: BoundGrant, now: float | None = None) -> bool:
    at = time.time() if now is None else now
    return bool(grant.expires_at_unix) and grant.expires_at_unix < at


def read_input(
    grant: BoundGrant,
    input_id: str,
    auth: Authorizer,
    *,
    cap: int,
) -> bytes:
    """One bound input, verified against the SPEC's digest and length (the grant carries
    access only, #439)."""
    entry = grant.inputs.get(input_id)
    if entry is None:
        raise GrantRefusal(
            "input_absent",
            f"no bound input {input_id!r} (this attempt has "
            f"{', '.join(sorted(grant.inputs)) or 'none'})",
        )
    sink = io.BytesIO()
    _transfer_granted(grant, entry, auth, sink, cap=cap, prefix_bytes=0)
    return sink.getvalue()


#: A conservative name for a spooled input. The input_id is a FIELD PATH the caller can
#: influence through list indices, so it never reaches the filesystem uninspected.
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _cause(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def hydrate_inputs(
    grant: BoundGrant,
    auth: Authorizer,
    *,
    spool: Path,
    max_bytes: int = 64 << 20,
    max_total_bytes: int = 256 << 20,
    workspace: Workspace | None = None,
    owner: str = "",
) -> dict[str, GrantedInput]:
    """Verify every local input in place; materialize only native retained bytes.

    THE LAUNCH CONTRACT IS EAGER (§1.3), and this is where the eagerness lives: the whole
    present asset set is materialized here, before acceptance and before any lease, so a
    slow URL can never pin staged weights. Branch-selective lazy hydration is a deferred
    optimization behind the same author-facing type; the ordering law it would have to
    keep is exactly this one.

    Every size check happens on the grant's DECLARED length first — so an oversized input
    refuses before a byte moves — and again on the bytes, because a declaration is a claim.
    """

    if expired(grant):
        raise GrantRefusal("grant_expired", "the delivery grant expired before hydration")
    entries = [
        e
        for e in sorted(grant.inputs.values(), key=lambda e: (e.order, e.input_id))
        if e.input_id != PAYLOAD_INPUT
        and not _tree_input(e)
        and not e.input_id.startswith(MODEL_PREFIX)
    ]
    declared_total = sum(int(e.length) for e in entries) + sum(
        e.native_tree.source.content_bytes
        for e in grant.inputs.values()
        if _tree_input(e) and e.native_tree is not None
    )
    if declared_total > max_total_bytes:
        raise GrantRefusal(
            "inputs_over_total_cap",
            f"this attempt's {len(entries)} input asset(s) declare {declared_total} B and "
            f"this deployment admits {max_total_bytes} B per attempt",
        )
    out: dict[str, GrantedInput] = {}
    root = spool / "inputs"
    published: list[Path] = []
    complete = False
    try:
        for position, entry in enumerate(entries):
            if int(entry.length) > max_bytes:
                raise GrantRefusal(
                    "input_over_cap",
                    f"input {entry.input_id!r} declares {entry.length} B and this deployment "
                    f"admits {max_bytes} B per input. The DECLARED length refuses before a "
                    "byte moves; the bytes are held to the same bound on the way in",
                )
            temporary: Path | None = None
            source_state = None
            try:
                if entry.native_tree is None:
                    local = _local(entry.url, grant.file_base_url)
                    source_state = file_state(local)
                    receipt = _transfer_granted(
                        grant, entry, auth, None, cap=max_bytes, prefix_bytes=SNIFF_BYTES
                    )
                    if file_state(local) != source_state:
                        raise GrantRefusal(
                            "input_changed", "input file changed during verification"
                        )
                else:
                    root.mkdir(parents=True, exist_ok=True)
                    local = root / f"{position:03d}-{_SAFE.sub('_', entry.input_id)[:96]}"
                    with tempfile.NamedTemporaryFile(
                        mode="w+b", dir=root, prefix=f".{position:03d}-", delete=False
                    ) as sink:
                        temporary = Path(sink.name)
                        if workspace is None:
                            raise GrantRefusal(
                                "grant_native_unavailable", "native input has no workspace"
                            )
                        from cozy_runtime.internal.worker import byte_inputs
                        from cozy_runtime.internal.worker.workspace_byte_outputs import leased

                        try:
                            with leased(workspace, owner, entry.native_tree) as (
                                _store,
                                lease,
                                members,
                            ):
                                blob = byte_inputs.file_member(entry.native_tree, members)
                                if (blob.sha256, blob.length) != (
                                    entry.digest.hex(),
                                    entry.length,
                                ):
                                    raise GrantRefusal(
                                        "input_digest_mismatch",
                                        "native input differs from its binding",
                                    )
                                receipt = egress.copy_bounded(
                                    byte_inputs.chunks(
                                        lease, "sha256:" + blob.sha256, entry.length
                                    ),
                                    cast(BinaryIO, sink),
                                    expected_length=entry.length,
                                    cap=max_bytes,
                                    prefix_bytes=SNIFF_BYTES,
                                    what="native input",
                                )
                                if receipt.sha256 != entry.digest:
                                    raise GrantRefusal(
                                        "input_digest_mismatch", "native input bytes changed"
                                    )
                        except GrantRefusal:
                            raise
                        except Exception as exc:
                            raise GrantRefusal(
                                "grant_native_unavailable",
                                f"native input bytes are unavailable: {_cause(exc)}",
                            ) from exc
                actual = sniff(receipt.prefix)
                media_type = reconcile(entry.kind_mime, actual)
                if media_type is None:
                    raise GrantRefusal(
                        "input_media_type",
                        f"input {entry.input_id!r} is declared {normalize(entry.kind_mime)!r} "
                        f"and its bytes are {actual!r}, another kind of media",
                    )
                if temporary is not None:
                    temporary.chmod(0o444)
                    os.link(temporary, local)
                    published.append(local)
                    temporary.unlink()
                    temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            out[entry.input_id] = GrantedInput(
                input_id=entry.input_id,
                local=local,
                file_state=source_state,
                media_type=media_type,
                digest="sha256:" + receipt.sha256.hex(),
                length=receipt.length,
                order=int(entry.order),
            )
        complete = True
    except RuntimeFailure as exc:
        raise GrantRefusal(exc.code, exc.message) from exc
    except OSError as exc:
        raise GrantRefusal("input_spool_io", f"input spool publication failed: {exc}") from exc
    finally:
        if not complete:
            for path in published:
                path.unlink(missing_ok=True)
    return out


def release_inputs(spool: Path) -> int:
    """Remove only this attempt's native materializations, never borrowed media files."""
    from cozy_runtime.internal.worker.byte_inputs import remove_checkout

    remove_checkout(spool / "native-input-trees")
    root = spool / "inputs"
    removed = 0
    for path in sorted(root.glob("*")) if root.is_dir() else ():
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def read_trees(
    grant: BoundGrant,
    auth: Authorizer,
    *,
    workspace: Workspace | None = None,
    owner: str = "",
    spool: Path | None = None,
    max_bytes: int = 256 << 20,
) -> dict[str, tuple[Path, str]]:
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
    for position, entry in enumerate(sorted(grant.inputs.values(), key=lambda e: e.input_id)):
        if not _tree_input(entry):
            continue
        ref = (
            entry.input_id[len(TREE_PREFIX) :]
            if entry.input_id.startswith(TREE_PREFIX)
            else "sha256:" + entry.digest.hex()
        )
        if entry.native_tree is not None:
            if workspace is None or spool is None:
                raise GrantRefusal("grant_native_unavailable", "native tree input has no workspace")
            from cozy_runtime.internal.worker import byte_inputs

            root = spool / "native-input-trees" / str(position)
            try:
                byte_inputs.copy_retained(
                    workspace, owner, entry.native_tree, root, max_bytes=max_bytes
                )
            except Exception as exc:
                byte_inputs.remove_checkout(root)
                raise GrantRefusal(
                    "grant_native_unavailable", f"native tree bytes are unavailable: {_cause(exc)}"
                ) from exc
            out[ref] = (root, "sha256:" + entry.digest.hex())
            continue
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


class PublishedOutput(msgspec.Struct, frozen=True, kw_only=True):
    output_id: str
    digest: str
    length: int
    media_type: str


class Publication(msgspec.Struct, frozen=True, kw_only=True):
    """A publication receipt's document, as written to and read back from the records."""

    mode: str
    grant_id: str
    installation_id: str
    subject_id: str
    request_id: str
    attempt: int
    outputs: tuple[PublishedOutput, ...]
    total_bytes: int


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """One typed process record per publication (cr-009).

    What an attempt actually wrote, as a closed record with its own identity: the scope it wrote
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
    """The publication contract's scope — the scratch model a job's writes are rewritten
    into. Empty on the serving lane, which publishes into granted destinations only."""
    installation_id: str
    subject_id: str
    """`job_descriptor_id` in job mode, `entrypoint_binding_digest` in serving mode."""
    request_id: str
    attempt: int
    outputs: tuple[pb.OutputEntry, ...]

    def record(self) -> Publication:
        rows = sorted(self.outputs, key=lambda entry: entry.output_id)
        return Publication(
            mode=self.mode,
            grant_id=self.grant_id,
            installation_id=self.installation_id,
            subject_id=self.subject_id,
            request_id=self.request_id,
            attempt=self.attempt,
            outputs=tuple(
                PublishedOutput(
                    output_id=entry.output_id,
                    digest="sha256:" + entry.digest.hex(),
                    length=entry.length,
                    media_type=entry.mime_type,
                )
                for entry in rows
            ),
            total_bytes=sum(entry.length for entry in rows),
        )

    def document(self) -> dict[str, Json]:
        document: dict[str, Json] = msgspec.to_builtins(self.record())
        return document

    def digest(self) -> str:
        encoded = canonical.write(self.document())
        return (
            "sha256:" + hashlib.sha256(b"cozy.runtime.publication-receipt\0" + encoded).hexdigest()
        )


def write_output(
    grant: BoundGrant,
    output_id: str,
    source: Path,
    auth: Authorizer,
    media_type: str = "",
) -> pb.OutputEntry:
    """Write ONE output under the EXACT destination the grant names.

    A stale or wrong grant cannot write: the destination must be in this grant, the grant
    must be live, and the bytes must fit the destination's declared cap. Writing produces a
    manifest entry and NOTHING else — the runtime may write under an exact output grant and
    cannot authorize readers, publish media, or decide a winning attempt.

    A `file://` destination ending in `/` is a DIRECTORY: the file is named there by its own
    content digest and the closed extension table, `<sha256>.<ext>`, so the same bytes land
    on the same name and an existing identical file is a replay, not a conflict. The record
    owner recomputes that name from the manifest entry and reads exactly it.
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
    try:
        source_info = source.stat()
    except OSError as exc:
        raise GrantRefusal(
            "output_spool_io", f"{output_id}: output spool bytes are unreadable ({exc})"
        ) from exc
    if not stat.S_ISREG(source_info.st_mode):
        raise GrantRefusal("output_spool_io", f"{output_id}: output spool is not a regular file")
    length = int(source_info.st_size)
    if destination.max_bytes and length > destination.max_bytes:
        raise GrantRefusal(
            "output_too_large",
            f"{output_id}: {length} B over the granted {destination.max_bytes} B",
        )
    selected_media_type = _output_media_type(output_id, destination.mime_type, media_type)
    auth.authorize("grant_output_write", "PUT", url, {"output_id": output_id, "length": length})
    parsed = _parse_grant_url(url)
    if parsed.scheme == "https":
        host = parsed.hostname or ""
        try:
            with source.open("rb") as opened:
                if not stat.S_ISREG(os.fstat(opened.fileno()).st_mode):
                    raise GrantRefusal(
                        "output_spool_io", f"{output_id}: output spool is not a regular file"
                    )
                receipt = egress.put_from(
                    url,
                    egress.EgressPolicy(allowed_hosts=(host,)),
                    opened,
                    expected_length=length,
                    media_type=selected_media_type,
                    what=f"output {output_id!r}",
                )
        except egress.EgressRefusal as exc:
            raise GrantRefusal(exc.code, exc.detail) from exc
        except GrantRefusal:
            raise
        except OSError as exc:
            raise GrantRefusal(
                "output_spool_io", f"{output_id}: streaming publication failed ({exc})"
            ) from exc
        return pb.OutputEntry(
            output_id=output_id,
            digest=receipt.sha256,
            length=receipt.length,
            mime_type=selected_media_type,
        )

    named_by_digest = url.endswith("/")
    path = _local(destination.url, grant.file_base_url)
    directory = path if named_by_digest else path.parent
    directory.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with (
            source.open("rb") as opened,
            tempfile.NamedTemporaryFile(
                mode="w+b", dir=directory, prefix=".cozy-output-", delete=False
            ) as sink,
        ):
            temporary = Path(sink.name)
            receipt = egress.copy_bounded(
                iter(lambda: opened.read(egress.CHUNK), b""),
                cast(BinaryIO, sink),
                expected_length=length,
                cap=destination.max_bytes or length,
                prefix_bytes=0,
                what=f"output {output_id!r}",
            )
            sink.flush()
            os.fsync(sink.fileno())
            # The destination is READ by whoever granted it, and on a pod that is the media
            # plane's own uid, not this worker's: a 0600 spool file behind a granted path is
            # an output nobody can fetch. Content, never a credential; readable is right.
            os.fchmod(sink.fileno(), 0o644)
        if named_by_digest:
            path = directory / (receipt.sha256.hex() + extension(selected_media_type))
        try:
            os.link(temporary, path)
            _fsync_directory(directory)
        except FileExistsError as exc:
            if not named_by_digest:
                raise GrantRefusal(
                    "output_destination_exists",
                    f"{output_id}: the granted destination already exists; outputs never overwrite",
                ) from exc
            # The name IS the digest: an existing file is the same bytes, or the store holds
            # a file under a name it does not hash to, which is a conflict and never a
            # silent overwrite.
            if _sha256_of(path) != receipt.sha256:
                raise GrantRefusal(
                    "output_destination_conflict",
                    f"{output_id}: {path.name} already exists in the granted directory with "
                    "different bytes",
                ) from exc
    except egress.EgressRefusal as exc:
        raise GrantRefusal(exc.code, exc.detail) from exc
    except GrantRefusal:
        raise
    except OSError as exc:
        raise GrantRefusal(
            "output_spool_io", f"{output_id}: streaming publication failed ({exc})"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return pb.OutputEntry(
        output_id=output_id,
        digest=receipt.sha256,
        length=receipt.length,
        mime_type=selected_media_type,
    )


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_of(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(egress.CHUNK), b""):
            digest.update(chunk)
    return digest.digest()


def _output_media_type(output_id: str, bound: str, produced: str) -> str:
    """Join the invocation's MIME contract to the type observed from the author result."""
    exact = normalize(bound)
    observed = normalize(produced)
    if exact and observed != exact:
        raise GrantRefusal(
            "output_media_type_mismatch",
            f"{output_id}: the invocation binds {exact!r}, but the produced output declares "
            f"{observed or 'no media type'!r}",
        )
    return exact or observed or "application/octet-stream"


def abort_outputs(spool: Path) -> int:
    """Abort an attempt's uncommitted outputs. Bytes written under the grant are the
    RecordOwner's problem only once a terminal claims them; the spool is ours."""
    from cozy_runtime.internal.worker.byte_inputs import remove_checkout

    return remove_checkout(spool, keep_root=True)
