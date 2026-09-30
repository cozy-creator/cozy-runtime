"""Checkpoints in a TensorFS store, as the runtime reads them: headers, plan rows, leases.

The weight plane (`weights.py`, `tensorfs-plane`) moves the bytes. What stays here is the
store side every consumer shares: opening a store, a checkpoint's header projected into the
contract's rows (`PlanRow`, with each tensor's stored roles), verified read leases keyed by
snapshot, the servability answer derived from plan resolution, and the one refusal
vocabulary (`FillRefusal`). A snapshot is verified when its lease is taken, so a corrupt
artifact refuses before construction.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import tensorfs

from cozy_runtime.author._loader import (
    TensorSpec,
    canonical_dtype,
)
from cozy_runtime.internal.derive import torch_module
from cozy_runtime.internal.encoding import (
    TORCH_DTYPES,
    Capabilities,
    DeviceFacts,
    Encoded,
    Provider,
    RolePart,
    aliases,
    measure_runtime,
)
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.resolution import (
    PlanRefusal,
    ResolvedModelPlan,
    script_plan,
)

# Every `Any` below is a torch module, tensor, event or device: torch is absent from the
# check venv.

if TYPE_CHECKING:
    from tensorfs import ReadLease as ReadLease
    from tensorfs import ReadPlan, Tensor
    from tensorfs import Store as Store

# ------------------------------------------------------------------------- refusals


class FillRefusal(Exception):
    """A typed fill refusal. Every one of these fires BEFORE any byte reaches a
    destination, or aborts the generation that was going to hold it."""

    def __init__(self, message: str, *, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(fields)


#: The complete refusal vocabulary of this plane, and EXACTLY the producible codes. Named
#: as a frozen set so a new code cannot appear without touching this line — and a code
#: leaves with its last producer, because an entry nothing can raise makes the registry lie
#: in the same way an unregistered producer does, just in the other direction (#504).
REFUSALS = frozenset(
    {
        "missing_object",  # the CAS has no object file for a declared segment
        "digest_mismatch",  # object bytes do not hash to the object id
        "length_mismatch",  # segment/destination/object lengths disagree
        "dtype_mismatch",  # stored dtype is not the destination's (decode is cr-006)
        "destination_absent",  # a fill for a key the contract does not name
        "incomplete_fill",  # commit with a DestinationSet key never enqueued
        "lease_live",  # a staging slot reused before its completion event
        "derived_unmaterialized",  # a non-persistent buffer the fill plane cannot supply
        # cr-006's three, spoken in the SAME vocabulary because the worker, the
        # worker records and the CLI all read one refusal table. `encoding.py` owns their
        # meaning; this line is what keeps a new code from appearing unnoticed.
        "unknown_encoding",  # the header cites a spec digest no provider claims
        "encoding_unqualified",  # a claimed encoding, unqualified on THIS device
        "role_mismatch",  # the stored roles are not the roles the decoder accounts for
        # The store's own lifecycle and header refusals, spoken here because
        # `STORE_REFUSALS` already routes TensorFS codes to them. They were being spoken
        # WITHOUT being declared, so `_refuse`'s guard turned every one of them into an
        # AssertionError abort — the guard catching exactly what it exists to catch.
        "lease_revoked",  # the read lease covering these bytes was revoked mid-fill
        "checkpoint_unreadable",  # the snapshot's header/document bytes do not parse
        "store_unusable",  # a caller-prepared Store root that TensorFS refuses to open
        # The stored carrier does not match the encoding spec it CITES: a role set that is
        # not the spec's exact set, or a role whose shape is not what the spec's relation
        # derives. Distinct from `role_mismatch` on purpose — that one says THIS RUNTIME's
        # decoder does not account for the roles (a runtime gap, READER side), and this one
        # says the stored bytes are not the encoding they claim (DATA side, rebuild the
        # artifact). Collapsing them would print the wrong remedy, which is the exact cost
        # `worker/refusal.py` exists to avoid (#503b). It becomes producible with mxfp8: a
        # block scale is the first role whose geometry is DERIVED (`ceil_div(cols, 32)`)
        # rather than fixed, so "the right roles at the wrong shape" is a state that can
        # exist for the first time.
        "carrier_geometry",
        # #549.4 and #549.5, spoken here for the same reason cr-006's other codes are:
        # one refusal table. `spec_unreviewed` says the registry knows this exact spec
        # digest and no provider in this build was ever read against it (remedy: review a
        # provider, or stop citing the spec) — the alias-keyed launch table could not
        # produce it, because it handed the digest a sibling's decoder. `geometry_unsupported`
        # says every implementation refuses this tensor's own SHAPE (remedy: the artifact's
        # geometry, or a provider with declared padding semantics); the device and the bytes
        # are both fine, which is why it is not `encoding_unqualified`.
        "spec_unreviewed",
        "geometry_unsupported",
        # RETIRED with #549.3: `objective_unranked`. It fired when two providers qualified
        # and the deployment's objective named no per-record axis — which was `latency`, the
        # DEFAULT, so the default path refused as soon as a second implementation existed.
        # The axis was never missing; it lived one layer up, in the measured envelopes of
        # whole plans, and the chooser that held them never ran. `resolution.py` makes the
        # joint decision now and CONFESSES uncalibrated instead of refusing.
        # The `encoded_gemm` route's own two. Distinct on purpose, because they print
        # different remedies: `leaf_unconsented` says the PACKAGE must declare that its
        # code survives a replaced linear op (an author edit), while `leaf_schema` says
        # the module tree cannot take a leaf where this destination is, or did not end up
        # holding what the substitution declared (a runtime/artifact disagreement).
        "leaf_unconsented",
        "leaf_schema",
        # cr-025's ONE-SELECTION-AUTHORITY law. `plan_divergence`: the resolved plan this
        # generation was priced and identified under names a tensor, an implementation, a
        # route, a geometry or a dtype that is not what this checkpoint holds or this build
        # and device can execute. The plane never re-selects to paper over it — the
        # planner-A/executor-B split is exactly what the digest in generation identity
        # forbids. `plan_unresolved`: no executable plan resolves for these bytes here,
        # spoken in this vocabulary by the honesty surface (`servability`).
        "plan_divergence",
        "plan_unresolved",
    }
)


def _refuse(code: str, message: str, fields: Sequence[str] = ()) -> FillRefusal:
    if code not in REFUSALS:
        raise AssertionError(f"{code} is not a declared fill refusal")
    return FillRefusal(message, code=code, fields=fields)


# ------------------------------------------------------------------- admission arithmetic


# ------------------------------------------------------------------- the TensorFS reader


def tensorfs_module() -> ModuleType:
    """TensorFS for `internal/worker` modules, which do not import it themselves (cr-067)."""
    return tensorfs


def capabilities() -> frozenset[str]:
    """The features this process's TensorFS build names (`tensorfs.CAPABILITIES`, e.g.
    "ensure/1", "deliver/1"); an older build names none. Detected by name, never by version."""
    return frozenset(getattr(tensorfs, "CAPABILITIES", ()))


_stores: dict[tuple[int, str], Store] = {}
_stores_lock = threading.Lock()


def store(root: str | Path) -> Store:
    """The process's ONE TensorFS Store handle for `root`.

    TensorFS keeps its trust index and catalog connection alive only while a handle to the
    root lives, so a fresh `Store.open` per call re-validated and re-loaded the catalog every
    time. Keyed by pid: a forked child never shares its parent's SQLite connection.
    """
    key = (os.getpid(), os.path.abspath(root))
    with _stores_lock:
        held = _stores.get(key)
        if held is None:
            held = _stores[key] = tensorfs.Store.open(key[1])
        return held


def ensure_store(root: str | Path) -> Store:
    """`store(root)`, first creating the Store when its catalog is absent."""
    path = os.path.abspath(root)
    if not os.path.isfile(os.path.join(path, "tensorfs.sqlite")):
        created: Store = tensorfs.Store.ensure(path)
        with _stores_lock:
            _stores[(os.getpid(), path)] = created
        return created
    return store(path)


def open_store(root: Path) -> Store:
    """The process's Store handle for one caller-prepared root, refusals in fill vocabulary."""

    try:
        return store(root)
    except Exception as exc:
        raise _refuse(
            "store_unusable",
            f"{root}: not a usable TensorFS store — {type(exc).__name__}: {exc}; "
            "have the Store owner ensure it before launching Runtime",
        ) from exc


#: How a TensorFS refusal is spoken in THIS plane's vocabulary. Closed and explicit: a code
#: this table does not name is re-raised UNCHANGED, because a plane that renames a refusal
#: it does not understand is worse than one that admits it did not.
STORE_REFUSALS: dict[str, str] = {
    "OBJECT_CORRUPT": "digest_mismatch",
    "OBJECT_ID_MISMATCH": "digest_mismatch",
    "MALFORMED_DIGEST": "digest_mismatch",
    "OBJECT_ABSENT": "missing_object",
    "LEASE_NOT_COVERED": "missing_object",
    "SHORT_READ": "length_mismatch",
    "RANGE_BOUNDS": "length_mismatch",
    "BUFFER_SIZE": "length_mismatch",
    "LENGTH_MISMATCH": "length_mismatch",
    "BYTE_LENGTH_MISMATCH": "length_mismatch",
    "DTYPE_MISMATCH": "dtype_mismatch",
    "DTYPE_UNKNOWN": "dtype_mismatch",
    "MISSING_TENSOR": "destination_absent",
    "TRAVERSAL_INCOMPLETE": "incomplete_fill",
    "SLOT_STARVED": "lease_live",
    "LEASE_REVOKED": "lease_revoked",
    "NONCANONICAL_ENCODING": "checkpoint_unreadable",
    "MALFORMED_JSON": "checkpoint_unreadable",
    "UNKNOWN_FORMAT": "checkpoint_unreadable",
    "MISSING_FIELD": "checkpoint_unreadable",
    # The header re-check against its own encoding closure (`Checkpoint.__init__`), which is
    # law 18 made mechanical: a RecordOwner's authority is not worker safety, so the worker
    # asks TensorFS whether every tensor's stored roles ARE the roles its cited spec
    # declares. Both codes were escaping this plane UNTRANSLATED — a raw
    # `tensorfs.errors.Refusal` crossing a boundary whose whole contract is that the
    # worker records and the CLI read ONE refusal table. Same class as #496a, one
    # layer up: the check fired correctly and the answer had nowhere to go.
    "ROLE_SET_MISMATCH": "carrier_geometry",
    "SHAPE_MISMATCH": "carrier_geometry",
}

#: The table's codomain is CHECKED against the vocabulary at import, not at the moment a
#: store refusal happens to fire. A translation to an undeclared code is a defect in this
#: file, and finding it on the failure path means finding it during an outage.
_undeclared = sorted(set(STORE_REFUSALS.values()) - REFUSALS)
if _undeclared:
    raise AssertionError(f"STORE_REFUSALS translates to undeclared codes: {_undeclared}")


@contextlib.contextmanager
def store_refusals(what: str) -> Iterator[None]:
    """Speak a TensorFS refusal in this plane's vocabulary, or let it through untranslated."""
    try:
        yield
    except tensorfs.errors.Refusal as exc:
        code = STORE_REFUSALS.get(getattr(exc, "code", ""))
        if code is None:
            raise
        raise _refuse(
            code,
            f"{what}: TensorFS refused {exc.code} - {getattr(exc, 'detail', exc)}",
        ) from exc


def block_read_bytes() -> int:
    """Bytes THIS process has actually pulled from the block layer, or -1 (cr-100).

    The one cheap way to tell a warm page-cache read from a cold one: page faults on the
    mapped CAS objects are charged here, cache hits are not. Observation only, and it is
    allowed to be unavailable -- every non-Linux host reports -1 rather than a zero that
    would read as "warm".
    """
    try:
        with open("/proc/self/io", "rb") as rows:
            for row in rows:
                if row.startswith(b"read_bytes:"):
                    return int(row.split(b":")[1])
    except (OSError, ValueError):
        return -1
    return -1


@dataclass(frozen=True, slots=True)
class PlanRow:
    """One checkpoint header row: what the store holds for one logical tensor.

    `what` is TensorFS's own name for the destination (`<component>/<key>#<part>`), carried
    verbatim so a batch item can be routed back to its tensor without this plane re-deriving
    a naming rule the store already owns.
    """

    key: str
    """The CONTRACT's key: component-qualified, e.g. `unet.conv_in.weight`."""
    name: str
    """The HEADER's key for the same tensor, unqualified, e.g. `conv_in.weight`."""
    component: str
    dtype: str
    """The LOGICAL dtype: what the destination is and what a decode targets."""
    shape: tuple[int, ...]
    nbytes: int
    """The LOGICAL byte count. For an encoded tensor this is NOT what is stored."""
    encoded: Encoded
    """The per-tensor encoding assignment, from the header's own citation (cr-006)."""

    def role_what(self, role: str) -> str:
        """TensorFS's own name for one stored role, `<component>/<name>#<role>`."""
        return f"{self.component}/{self.name}#{role}"

    @property
    def stored_nbytes(self) -> int:
        return self.encoded.stored_nbytes


class Checkpoint:
    """One model manifest in a TensorFS store, opened through the tfs-007 facade.

    This class is the ONLY place in the runtime that speaks to TensorFS, and it is
    deliberately thin. Object verification, GC-safe holds, plan ordering, dtype names, the
    reader pool and every refusal are the store's; what this plane adds is the projection
    from a header into the contract's rows and the H2D transaction on top of the stream.

    It replaces cr-005's interim `CasStore` + `read_plan` outright (decisions #264). The
    interim reader knew CAS layout, cached a file descriptor per (thread, object) and
    verified per object in Python; none of that survives, and none of it needs to.
    """

    def __init__(self, root: str | Path, manifest_id: str) -> None:
        self.root = str(root)
        self.manifest_id = manifest_id
        with store_refusals(f"opening manifest {manifest_id[:23]}"):
            self.store = store(self.root)
            header = self.store.manifest(manifest_id)["header"]
            if header is None:
                raise _refuse("checkpoint_unreadable", f"{manifest_id}: no cozytensors header")
            self.header_bytes = header
            # Encoding definitions are nested directly in cozytensors/1. TensorFS parses
            # and validates the header once and exposes each definition with the digest
            # that tensor rows cite; there is no second closure-object lookup.
            self.header = tensorfs.parse_header(self.header_bytes)
            self.specs = tuple(row["id"] for row in self.header["encodings"])
        self._aliases = aliases()

    def has(self, component: str) -> bool:
        """Does the header carry this component? A constructed component it does not carry
        is the fit's `component_missing`, so nothing here refuses it first."""
        return component in self.header["components"]

    def rows(self, component: str) -> list[PlanRow]:
        """The header's lookup rows for one component.

        Their iteration order has no execution authority. Plan resolution surveys each row
        before construction; the later construction census supplies the traversal that
        TensorFS validates and executes.
        """
        table = self.header["components"].get(component)
        if table is None:
            raise _refuse(
                "destination_absent",
                f"this checkpoint has no component {component!r} (it has "
                f"{', '.join(sorted(self.header['components'])) or 'none'})",
            )
        out: list[PlanRow] = []
        for key, entry in table.items():
            logical = entry["logical"]
            shape = tuple(logical["shape"])
            dtype = logical["logical_dtype"]
            width = tensorfs.DTYPES.get(dtype)
            if width is None:
                raise _refuse(
                    "dtype_mismatch",
                    f"{key}: the header declares dtype {dtype!r}, which this build of "
                    "TensorFS does not know",
                    [key],
                )
            if dtype not in TORCH_DTYPES:
                # The store knows this dtype and this runtime has no torch spelling for it.
                # Saying so is the point of keeping the map CHECKED rather than trusted: the
                # alternative is passing the raw name through and comparing it against a
                # torch dtype it can never equal, which refuses too — with a message about
                # the wrong thing.
                raise _refuse(
                    "dtype_mismatch",
                    f"{key}: TensorFS supplies dtype {dtype!r} and this runtime has no "
                    "torch binding for it; the map lives in the runtime because torch is "
                    "not TensorFS's business, and a missing row is a refusal, not a guess",
                    [key],
                )
            nbytes = width
            for extent in shape:
                nbytes *= extent
            out.append(
                PlanRow(
                    # The contract's key is component-qualified (`unet.conv_in.weight`);
                    # the header's is not, and TensorFS's `what` is a third spelling.
                    # Carrying all three is what keeps the plane from re-deriving any of
                    # them — a naming rule guessed in two places is a wrong copy waiting.
                    key=f"{component}.{key}",
                    name=key,
                    component=component,
                    dtype=dtype,
                    shape=shape,
                    nbytes=nbytes,
                    encoded=self._encoded(f"{component}.{key}", entry),
                )
            )
        return out

    def _encoded(self, key: str, entry: Tensor) -> Encoded:
        """The tensor's encoding assignment and stored roles, verbatim from the header.

        `plain/1` is a real entry here, not an absent form: the selection path has one
        branch and the verbatim provider is what it selects for unencoded bytes.
        """
        parts: list[RolePart] = []
        for role, part in entry["parts"].items():
            shape, dtype = tuple(part["shape"]), part["dtype"]
            width = tensorfs.DTYPES.get(dtype)
            if width is None or dtype not in TORCH_DTYPES:
                # BOTH directions, for the same reason the logical row checks both: a
                # carrier dtype TensorFS knows and this runtime has no torch spelling for
                # would otherwise reach the scratch allocation as a KeyError, which is an
                # implementation detail escaping where a refusal belongs.
                why = (
                    "this build of TensorFS does not know"
                    if width is None
                    else "TensorFS knows and this runtime has no torch binding for"
                )
                raise _refuse(
                    "dtype_mismatch",
                    f"{key}: role {role!r} is stored as {dtype!r}, which {why}",
                    [key],
                )
            nbytes = width
            for extent in shape:
                nbytes *= extent
            parts.append(RolePart(role, dtype, shape, nbytes))
        digest = entry["encoding"]
        return Encoded(digest, self._aliases.get(digest, "unregistered"), tuple(parts))

    def read_plan(
        self,
        traversal: Sequence[tuple[str, str]],
        window_bytes: int,
        components: Sequence[str],
    ) -> ReadPlan:
        """The store's own read plan over the traversal THIS PLANE asks for.

        cr-005's rule survives the swap and is now the store's to keep: read order is the
        STORE's, write order is the CONTRACT's. The runtime hands over TensorRequirements
        traversal and TensorFS decides how to touch the disk for it.

        `components` DECLARES what this plan is for (#570b), and the fill plane always knows
        it: this loop is already grouped BY COMPONENT, so the scope is the group's own name.
        Completeness is then checked against those components' tensors rather than the whole
        header — which is what makes an N-ary artifact servable at all. Under the old
        whole-snapshot rule H3's Fl2VA class named 917 of 3445 tensors and was refused for
        the 2528 belonging to a sibling transformer its contract forbids it to touch.
        """
        with store_refusals("planning the read"):
            return tensorfs.plan(
                self.header_bytes, list(traversal), list(components), window=window_bytes
            )

    def acquire(self) -> ReadLease:
        """A verified read lease over this snapshot's CozyTensors runtime closure."""
        with store_refusals(f"acquiring a lease on {self.manifest_id[:23]}"):
            lease = self.store.acquire_cozytensors(self.manifest_id)
        return lease

    def model_assets(self) -> dict[str, bytes]:
        """Read the header-declared model files through one verified closure lease."""
        names = sorted(self.header["assets"])
        if not names:
            return {}
        with (
            store_refusals(f"reading model assets from {self.manifest_id[:23]}"),
            self.acquire() as lease,
        ):
            assets = {
                name: bytes(lease.read_asset(self.header_bytes, name, max_bytes=64 << 20))
                for name in names
            }
        return assets


def logical_weight_bytes(root: str | Path, snapshots: Mapping[str, str]) -> dict[str, int]:
    """Each constructed component's LOGICAL bytes, from the header of the snapshot it fills
    from: what its destinations hold. An encoded checkpoint stores fewer bytes than that
    (fp8 SDXL: 4.71 GB stored, 6.94 GB logical), so the stored closure never prices one. A
    component its header lacks has no entry; model-code-fit judges that mismatch."""
    checkpoints = {snapshot: Checkpoint(root, snapshot) for snapshot in set(snapshots.values())}
    return {
        component: sum(row.nbytes for row in checkpoints[snapshot].rows(component))
        for component, snapshot in snapshots.items()
        if checkpoints[snapshot].has(component)
    }


@dataclass(frozen=True, slots=True)
class Servability:
    """This runtime's OWN answer to "can I serve these bytes", given before any of them move.

    #501f's runtime twin. That ruling makes a PRODUCER declare, in its artifact's manifest,
    that its encoding has no runtime provider. This is the other half, and it is the half
    that cannot be gamed: the runtime never reads an encoding CLAIM. It reads the per-tensor
    spec DIGESTS the header cites, looks each one up in the provider set this build was
    compiled with, and asks the capability table — whose records exist only where measured
    numerics minted them — whether this device serves it. A manifest that says "servable"
    changes nothing here, and neither does a manifest that says nothing at all.

    `serves` is therefore never a claim about the artifact; it is a claim about THIS worker
    standing on THIS card, which is the only thing a worker is entitled to claim.
    """

    serves: bool
    by_alias: dict[str, int]
    """Every encoding alias the header actually cites, and how many tensors cite it."""
    routes: dict[str, int]
    refusal: str
    code: str

    def line(self) -> str:
        cited = ", ".join(f"{alias} x{n}" for alias, n in sorted(self.by_alias.items()))
        if self.serves:
            served = ", ".join(f"{route} x{n}" for route, n in sorted(self.routes.items()))
            return f"serves {cited} via {served}"
        return f"CANNOT serve {cited}: [{self.code}] {self.refusal}"

    def require(self) -> None:
        """Raise the refusal, in the fill plane's vocabulary. A caller that wants to CHOOSE
        another variant reads `serves` instead — which is what the delivery lane does."""
        if not self.serves:
            raise _refuse(self.code, self.refusal)


def servability(rows: Sequence[PlanRow], answer: ResolvedModelPlan | PlanRefusal) -> Servability:
    """The honesty answer, DERIVED from plan resolution — never from a second selector.

    `answer` is what `resolution.resolve()` (or `script_plan()`) produced for these rows: a
    plan, in which case every route it names is executable here by construction, or the
    `PlanRefusal` it raised. A refusal is spoken through the walk's first typed verdict
    (`encoding_unqualified`, `role_mismatch`, ...) so the answer names the ENCODING and
    the remedy rather than the resolver's summary; a plan that refused for a reason the
    walk does not type is `plan_unresolved`.
    """
    by_alias: dict[str, int] = {}
    for row in rows:
        by_alias[row.encoded.alias] = by_alias.get(row.encoded.alias, 0) + 1
    if isinstance(answer, ResolvedModelPlan):
        return Servability(True, by_alias, answer.route_census(), "", "")
    summary = {"no_rows", "unconsented", "unscored", "eligible", "chosen", "chosen_uncalibrated"}
    for step in answer.steps:
        if step.verdict in summary:
            continue
        code = "encoding_unqualified" if step.verdict == "unqualified" else step.verdict
        if code in REFUSALS:
            return Servability(False, by_alias, {}, step.why, code)
    return Servability(False, by_alias, {}, answer.detail, "plan_unresolved")


def servability_for_script(
    rows: Sequence[PlanRow],
    *,
    release: str,
    providers: Mapping[str, tuple[Provider, ...]],
    capabilities: Capabilities,
    device: DeviceFacts,
    store: str = "synthetic",
    snapshot: str = "synthetic",
    objective: str = "latency",
    encoded_leaves: str = "accept",
) -> Servability:
    """The honesty answer for a DRIVER SCRIPT: resolve a `script_plan`, then derive.

    Header work only — no tensor bytes move — through the same `resolve()` the executor
    uses, so what a script prints as servable is what a worker would execute.
    """
    try:
        answer: ResolvedModelPlan | PlanRefusal = script_plan(
            rows=rows,
            components=sorted({row.component for row in rows}),
            release=release,
            store=store,
            snapshot=snapshot,
            providers=providers,
            capabilities=capabilities,
            device=device,
            runtime=measure_runtime(torch_module(), release),
            facts=PlanFacts(),
            objective=objective,
            encoded_leaves=encoded_leaves,
            dtype_name=dtype_name,
        )
    except PlanRefusal as exc:
        answer = exc
    return servability(rows, answer)


def tensor_schema_of(rows: Sequence[PlanRow]) -> dict[str, TensorSpec]:
    """The checkpoint tensor schema the loader matches the code against.

    The spec's `encoding` carries the tensor's real ALIAS rather than a constant `plain`:
    the match itself is over logical shape (an encoding route changes no tensor schema), but
    a table that lied about the
    encoding would be the only place in the runtime that did.
    """
    return {
        row.key: TensorSpec(row.shape, dtype_name(row.dtype), row.encoded.alias, row.nbytes)
        for row in rows
    }


# -------------------------------------------------------------------------- backend


class ReadLeases:
    """Every verified read lease one generation holds, keyed by WHAT A LEASE COVERS.

    A lease is a hold on a SNAPSHOT. The fill plane asks for one per COMPONENT, and the
    components of a single-snapshot model all read from the same snapshot — so a map keyed
    by component name made the identical hold be taken once per component. Measured on
    sdxl: four acquisitions of the same 2,602-object lease, 1091/1104/1160/1257 ms, and
    10,408 pinned file descriptors for 2,602 distinct objects — against the FD_HEADROOM
    admission `acquire_cozytensors` performs on its way in (cr-102/cr-103).

    It lives here, apart from the backend, because the backend's constructor measures a
    device and runs the qualification suite: lease bookkeeping that could only be exercised
    behind those could only be proved by a fill on a card, and this is arithmetic over a
    real store that a test can drive directly. A component-per-snapshot binding (cr-008b)
    still takes one lease per distinct snapshot, which is the point — the key is the
    snapshot, not the model.
    """

    def __init__(self, checkpoints: Mapping[str, Checkpoint]) -> None:
        self.checkpoints = checkpoints
        self.held: dict[str, ReadLease] = {}
        #: How many times this generation went to the store. The EFFECT the keying has to
        #: show, reported rather than asserted about the code: one per distinct manifest.
        self.acquisitions = 0
        #: HIGH-WATER, like `pinned`: what was held at the peak, not what is held now. A
        #: full-resident generation releases at its first commit, so a live count read
        #: from the prepare facts is always zero and says nothing about the descriptors
        #: this fill actually spent against FD_HEADROOM.
        self.peak = {"objects": 0, "bytes": 0}

    def acquire(self, component: str) -> ReadLease:
        """The lease covering this component's manifest, taken once and reused."""
        checkpoint = self.checkpoints.get(component)
        if checkpoint is None:
            raise _refuse(
                "destination_absent",
                f"no checkpoint is bound for component {component!r} (bound: "
                f"{', '.join(sorted(self.checkpoints)) or 'none'})",
            )
        manifest = checkpoint.manifest_id
        lease = self.held.get(manifest)
        if lease is None or not lease.live:
            lease = self.held[manifest] = checkpoint.acquire()
            self.acquisitions += 1
            self.peak = {
                "objects": max(self.peak["objects"], self._objects()),
                "bytes": max(self.peak["bytes"], self._bytes()),
            }
        return lease

    def take(self, component: str) -> ReadLease:
        """The lease for this component's manifest, handed over: a plane weight set consumes
        its lease, so the next component of the same manifest takes a fresh one."""
        lease = self.acquire(component)
        del self.held[self.checkpoints[component].manifest_id]
        return lease

    def _objects(self) -> int:
        return sum(len(one.objects) for one in self.held.values())

    def _bytes(self) -> int:
        return sum(int(one.bytes) for one in self.held.values())

    def release(self) -> None:
        """Give every hold back. Idempotent: a poison and a commit may both reach here."""
        held, self.held = self.held, {}
        for lease in held.values():
            if lease.live:
                lease.release()

    def document(self) -> dict[str, int]:
        """What this generation leased, in the units the FD_HEADROOM admission is spent in.

        `leases` is what is held right now; `objects` and `bytes` are the peak, because a
        commit releases and a report of the after state would say a fill leased nothing.
        """
        return {
            "leases": len(self.held),
            "acquisitions": self.acquisitions,
            "objects": max(self.peak["objects"], self._objects()),
            "bytes": max(self.peak["bytes"], self._bytes()),
        }


# ------------------------------------------------------------------------- helpers


#: TensorFS dtype name -> torch dtype name, imported from the FORMAT seam rather than kept
#: here (#549.10): `probe.py` needs the same map to build role tensors out of the vendored
#: spec vectors, and two copies of one dtype table is the #497-class duplication.
#: `Checkpoint.rows` still CHECKS it against `tensorfs.DTYPES` in both directions, and every
#: consumer imports it from `cozy_runtime.internal.encoding` now — one definition, one door.


def dtype_name(dtype: object) -> str:
    """The one dtype spelling. The store writes TensorFS names (`f16`), torch prints its
    own (`torch.float16`), and a fill that guesses which is which is a silent wrong copy."""
    name = str(getattr(dtype, "name", dtype)).removeprefix("torch.")
    return TORCH_DTYPES.get(name, name)


def tensorfs_requirement_dtype(dtype: object | None) -> str | None:
    """Constraint spelling for TensorFS fit; ordinary bf16/f16 remain convertible.

    `canonical_dtype` also covers a census derived by an OLDER runtime inside the package
    environment, whose wire rows still spell torch names — the boundary canonicalizes."""

    if dtype is None:
        return None
    name = canonical_dtype(dtype)
    return None if name in ("bf16", "f16") else name
