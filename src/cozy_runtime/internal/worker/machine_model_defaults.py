"""Frozen public Model selections; only omitted inference arguments consume them."""

from __future__ import annotations

import ipaddress
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack, nullcontext
from functools import partial
from typing import TYPE_CHECKING, Annotated, Protocol
from urllib.parse import urlsplit

import msgspec

from cozy_runtime.internal import fill
from cozy_runtime.internal.canonical import Json
from cozy_runtime.protocol import worker_pb2 as pb

from . import grants, machine_model_resolve
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .machine_child_target import Target
    from .session import Worker
    from .workspace_calls import Call


class Cancellation(Protocol):
    """TensorFS's pull cancellation token, as a preparation checks it between checkpoints."""

    @property
    def cancelled(self) -> bool: ...


class Manifest(msgspec.Struct, frozen=True):
    digest: Annotated[str, msgspec.Meta(pattern=r"^sha256:[0-9a-f]{64}$")]
    length: Annotated[int, msgspec.Meta(gt=0)]


class Selector(msgspec.Struct, frozen=True):
    """A ladder rung's accelerator selector: GPU name tokens and the count it needs."""

    gpu: str = ""
    gpus: int = 0


class Rung(Selector, frozen=True, kw_only=True, omit_defaults=True):
    repository: str = ""
    manifest: Manifest


class Captured(msgspec.Struct, frozen=True):
    """One `MachineModelDefault` as the accepted capture document spells it."""

    entrypoint: str = ""
    parameter: str = ""
    public_origin: str = ""
    rungs: tuple[Rung, ...] = ()
    unavailable_code: str = ""
    callee_installation_id: str = ""
    callee_deferred_key: str = ""


class _Capture(msgspec.Struct, frozen=True):
    model_defaults: tuple[Captured, ...] = ()


# One model's pull: the bytes landed of its total.
Sample = Callable[[int, int], None]


class CheckpointSource(msgspec.Struct, frozen=True, kw_only=True):
    model: str
    manifest: str
    manifest_length: int
    release: str = ""
    lane: str = ""


class AdapterSource(CheckpointSource, frozen=True, kw_only=True):
    component: str
    source_component: str = "adapter"
    scale: str = "1"


class Selected(msgspec.Struct, frozen=True, kw_only=True):
    """One slot's exact checkpoint for this machine, as a preparation lands and binds it."""

    parameter: str
    repository: str
    manifest: Manifest
    public_origin: str = ""
    gpus: int = 0
    release: str = ""
    lane: str = ""
    native: bool = False
    adapters: tuple[AdapterSource, ...] = ()
    composed: CheckpointSource | None = None


UNAVAILABLE = frozenset(
    {"model_default_unavailable", "model_default_unbound", "model_default_origin_unsupported"}
)
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def local_origin(origin: str) -> bool:
    try:
        host = urlsplit(origin).hostname or ""
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def local_development(worker: Worker) -> bool:
    """A machine registered at a Hub on loopback, as its grants name them: a development
    machine, which may reach the private addresses that Hub hands it (checkpoint pulls,
    object uploads). A production Hub is never loopback, and a job never names a grant."""
    return any(
        local_origin(authority.origin) for authority in machine_model_resolve.authorities(worker)
    )


def hub_ca(worker: Worker, origin: str) -> str | None:
    """The private CA of a Hub this machine is registered at, when it uses one."""
    authority = machine_model_resolve.registration(worker, origin)
    return None if authority is None or authority.ca is None else str(authority.ca)


def credential(worker: Worker, origin: str) -> str:
    """This machine's worker capability at a Hub it is registered at: there it reads its
    owner's unpublished checkpoints. Any other origin gets nothing, and a refusal is never
    retried anonymously."""
    authority = machine_model_resolve.registration(worker, origin)
    if authority is None:
        return ""
    if authority.access_token:
        authority.headers()  # an expired local grant is an operation failure, never anonymous
        return "bearer " + authority.access_token
    return f"worker {authority.worker_id} {authority.worker_token}"


def storage_hosts(worker: Worker, origin: str) -> tuple[str, ...]:
    """Where a Hub this machine is registered at keeps its bytes; else the default's."""
    authority = machine_model_resolve.registration(worker, origin)
    if authority is None or authority is worker.options.publication_authority:
        return worker.config.object_storage_hosts
    return authority.object_storage_hosts


def valid_origin(origin: str, *, local: bool) -> bool:
    try:
        value = urlsplit(origin)
        return bool(
            0 < len(origin) <= 512
            and origin == origin.strip()
            and not any(ord(c) <= 32 or ord(c) == 127 for c in origin)
            and value.hostname
            and value.port != 0
            and not value.username
            and not value.password
            and not value.path
            and not value.query
            and not value.fragment
            and (
                value.scheme == "https"
                or (local and value.scheme == "http" and local_origin(origin))
            )
        )
    except ValueError:
        return False


def verify(worker: Worker, capture: pb.MachineExecutionCapture) -> None:
    if not capture.model_defaults:
        return
    keys = []
    for row in capture.model_defaults:
        if not callee(row) or not row.entrypoint or not row.parameter:
            raise WorkspaceRefusal("captured Model default has no exact slot identity")
        keys.append((callee(row), row.entrypoint, row.parameter))
        if row.unavailable_code:
            if row.unavailable_code not in UNAVAILABLE or row.rungs or row.public_origin:
                raise WorkspaceRefusal("captured Model default has conflicting availability")
            continue
        if not 1 <= len(row.rungs) <= 32 or not valid_origin(
            row.public_origin, local=local_development(worker)
        ):
            raise WorkspaceRefusal("captured Model default has no exact public origin or ladder")
        for rung in row.rungs:
            if (
                not 0 < len(rung.gpu) <= 128
                or any(ord(c) < 32 for c in rung.gpu)
                or _REPOSITORY.fullmatch(rung.repository) is None
                or rung.repository.startswith("local/")
                or len(rung.manifest.digest) != 32
                or rung.manifest.length <= 0
            ):
                raise WorkspaceRefusal("captured Model default has invalid exact rung facts")
        selectors = [(rung.gpu.lower(), rung.gpus) for rung in row.rungs]
        if len(set(selectors)) != len(selectors):
            raise WorkspaceRefusal("captured Model default repeats a GPU and count selector")
    # A slot the capture left open resolves at this machine's Hub; one for a slot this
    # interface lacks is never read. Only two defaults for one slot are ambiguous.
    if len(keys) > 4096 or len(set(keys)) != len(keys):
        raise WorkspaceRefusal("captured Model default inventory is ambiguous")


def callee(row: pb.MachineCallableBinding | pb.MachineModelDefault | Captured) -> str:
    """A binding or default row's callee: its installation, or a deferred release key."""
    return row.callee_installation_id or row.callee_deferred_key


def matches(selector: str, observed: str) -> bool:
    """The existing Creator GPU ladder's case-insensitive token subsequence rule."""
    if selector == "*":
        return True
    wanted = re.findall(r"[^\W_]+", selector.lower())
    actual = iter(re.findall(r"[^\W_]+", observed.lower()))
    return bool(wanted) and all(any(token == candidate for candidate in actual) for token in wanted)


def fitter(worker: Worker) -> Callable[[Selector], bool]:
    """Whether a rung's GPU selector and count fit this machine's granted devices."""
    facts = worker.host_facts()
    if "gpu_name" in facts.unreadable:
        raise WorkspaceRefusal("captured Model default cannot select an unreadable accelerator")
    granted = len(worker.lanes.entries)

    def fits(rung: Selector) -> bool:
        return matches(rung.gpu, facts.gpu_name) and (not rung.gpus or rung.gpus <= granted)

    return fits


def catalog_fits(fits: Callable[[Selector], bool]) -> Callable[[Mapping[str, object]], bool]:
    """`fits` over a catalog ladder's JSON rungs."""

    def check(rung: Mapping[str, object]) -> bool:
        try:
            return fits(msgspec.convert(rung, Selector))
        except msgspec.ValidationError:
            return False  # a rung this machine cannot read is a rung it does not take

    return check


def select(
    worker: Worker,
    owner: str,
    call: Call,
    target: Target,
    arguments: Mapping[str, object],
    *,
    model_choices: Mapping[str, pb.ModelChoice] | None = None,
) -> list[dict[str, Json]]:
    """Each omitted Model argument's checkpoint here: its captured default's widest fitting
    rung, or this machine's own Hub's answer for a slot the capture left open."""
    declared = {
        model["path"].rpartition(".")[2]: model for model in target.declaration.get("models", [])
    }
    from . import machine_model_overrides

    if model_choices is None:
        model_choices = machine_model_overrides.captured(worker, owner, call, target)
    for parameter, explicit in model_choices.items():
        if arguments.get(parameter) is not None and machine_model_overrides.has_base(explicit):
            raise WorkspaceRefusal(
                f"model_override_conflict: {parameter} already receives an owned ModelArtifact; "
                "use an adapter-only override to retain that base"
            )
    missing = sorted(parameter for parameter in declared if arguments.get(parameter) is None)
    if not missing:
        return []
    assert worker.executions is not None
    root = worker.executions.capture_root(owner, call.parent_request)
    try:
        captured = msgspec.convert(worker.executions.capture(owner, root), _Capture)
    except msgspec.ValidationError as exc:
        raise WorkspaceRefusal(f"accepted capture has malformed Model defaults: {exc}") from exc
    slot = (target.callee or target.installation_id, target.entrypoint)
    defaults = {
        row.parameter: row
        for row in captured.model_defaults
        if (callee(row), row.entrypoint) == slot
    }
    for parameter in missing:
        default = defaults.get(parameter)
        if (
            default is not None
            and default.unavailable_code
            and not (
                parameter in model_choices
                and machine_model_overrides.has_base(model_choices[parameter])
            )
        ):
            raise WorkspaceRefusal("captured Model default is unavailable for " + parameter)
    placement = target.prepared_installation.get("placement", {})
    package = str(
        (placement.get("package") or placement.get("development") or {}).get("package", "")
    )
    # A slot the capture left open is resolved here, at the Hub its run came from.
    open_slots = any(parameter not in defaults for parameter in missing)
    prepared = worker.executions.prepared(owner, root) if open_slots else None
    catalog = machine_model_resolve.own(worker, package, prepared.hub) if prepared else None
    fits = fitter(worker)
    selected: list[dict[str, Json]] = []
    for parameter in missing:
        choice = model_choices.get(parameter)
        if choice is not None and machine_model_overrides.has_base(choice):
            hub, account, credentials = machine_model_overrides.context(worker, owner, call)
            selected.append(
                machine_model_overrides.checkpoint(
                    worker,
                    choice,
                    declared[parameter],
                    package=package,
                    hub=hub,
                    owner=account,
                    credentials=credentials,
                    note=lambda *_: None,
                )
            )
            continue
        default = defaults.get(parameter)
        if default is None:
            assert catalog is not None and prepared is not None
            selected.append(
                dict(
                    machine_model_resolve.slot(
                        catalog,
                        package,
                        declared[parameter],
                        parameter,
                        None,
                        catalog_fits(fits),
                        prepared.account,
                    )
                )
            )
            continue
        # The widest rung this machine holds: the width itself is the declared degree.
        rung = max(
            (rung for rung in default.rungs if fits(rung)), key=lambda rung: rung.gpus, default=None
        )
        if rung is None:
            raise WorkspaceRefusal("captured Model default has no rung for this accelerator")
        selected.append(
            {"parameter": parameter, "public_origin": default.public_origin}
            | msgspec.to_builtins(rung)
        )
    return selected


def decode(rows: Iterable[Mapping[str, object]]) -> list[Selected]:
    """Selected rows a serving preparation carries as JSON, decoded once where read."""
    try:
        return [msgspec.convert(row, Selected) for row in rows]
    except msgspec.ValidationError as exc:
        raise WorkspaceRefusal(f"selected Model row is malformed: {exc}") from exc


def closure(rows: Iterable[Mapping[str, object]]) -> list[Selected]:
    """Every catalog checkpoint the selected component graph owns, with unique input IDs."""
    out: list[Selected] = []
    for row in decode(rows):
        if not row.native:
            out.append(msgspec.structs.replace(row, adapters=(), composed=None))
        sources: list[tuple[str, CheckpointSource]] = [
            (f"{row.parameter}.adapter.{index}", source)
            for index, source in enumerate(row.adapters)
        ]
        if row.composed is not None:
            sources.append((row.parameter + ".composed", row.composed))
        for parameter, source in sources:
            out.append(
                Selected(
                    parameter=parameter,
                    repository=source.model,
                    manifest=Manifest(source.manifest, source.manifest_length),
                    public_origin=row.public_origin,
                )
            )
    return out


def materialize(
    worker: Worker,
    rows: Iterable[Mapping[str, object]],
    access: ExitStack | None,
    *,
    downloading: Callable[[Selected], AbstractContextManager[Sample | None]] = (
        lambda _: nullcontext()
    ),
    cancellation: Cancellation | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Verify exact sources; lease bytes only when the preparer will read them.

    A reused placement consumes its registered binding document. Its fresh child
    admission still verifies and retains every checkpoint before submission. A cold
    preparation pulls its slots at once and reports their summed bytes: run 1510 pulled
    H3's 1.6 GB turbo adapter after its 100 GB base, 46 s more before the first segment.
    `downloading` brackets one model's pull and may take its own byte samples.
    """
    selected = closure(rows)
    if not selected:
        return
    assert worker.workspace is not None
    facade = fill.tensorfs_module()
    store = fill.store(worker.workspace.store_root)
    pulled: dict[str, tuple[int, int]] = {}
    lock = threading.Lock()
    keep = sorted({row.manifest.digest for row in selected})

    def report(digest: str, sample: Sample | None, available: int, total: int) -> None:
        if sample is not None:
            sample(available, total)
        if progress is None:
            return
        with lock:
            pulled[digest] = (available, total)
            landed = sum(done for done, _ in pulled.values())
            whole = sum(size for _, size in pulled.values())
        progress(landed, whole)

    def land(row: Selected) -> None:
        manifest = row.manifest
        if cancellation is not None and cancellation.cancelled:
            raise WorkspaceRefusal("model preparation canceled")
        # Existing cached custody is sufficient. A cold use fetches this exact
        # checkpoint; only this machine's own Hub sees its worker capability.
        with worker.model_transfer_lock:
            transfer = worker.model_transfers.setdefault(manifest.digest, threading.Lock())
        with transfer:
            if cancellation is not None and cancellation.cancelled:
                raise WorkspaceRefusal("model preparation canceled")
            try:
                store.verify_checkpoint_source(row.repository, manifest.digest, manifest.length)
            except facade.errors.Refusal:
                with downloading(row) as sample:
                    moved = (
                        None
                        if progress is None and sample is None
                        else partial(report, manifest.digest, sample)
                    )
                    if "ensure/1" in fill.capabilities():
                        # TensorFS owns the fetch: one flight per manifest, disk admission
                        # that never evicts this preparation's own models, the Hub CA.
                        facade.ensure(
                            store,
                            row.repository + "@" + manifest.digest,
                            hub=row.public_origin,
                            credential=credential(worker, row.public_origin),
                            ca_file=hub_ca(worker, row.public_origin),
                            allowed_hosts=storage_hosts(worker, row.public_origin),
                            allow_local=local_development(worker),
                            keep=keep,
                            cancellation=cancellation,
                            progress=None
                            if moved is None
                            else lambda event: moved(event["bytes_done"], event["bytes_total"]),
                        )
                    else:
                        facade.pull(
                            store,
                            row.public_origin,
                            row.repository + "@" + manifest.digest,
                            credential=credential(worker, row.public_origin),
                            allowed_hosts=storage_hosts(worker, row.public_origin),
                            allow_local=local_development(worker),
                            streams=8,
                            cancellation=cancellation,
                            progress=moved,
                        )
                store.verify_checkpoint_source(row.repository, manifest.digest, manifest.length)

    try:
        with ThreadPoolExecutor(len(selected), thread_name_prefix="model-pull") as pool:
            landing = [pool.submit(land, row) for row in selected]
        for future in landing:
            future.result()
        if access is not None:
            for row in selected:
                access.enter_context(store.acquire_cozytensors(row.manifest.digest))
    except facade.errors.Refusal as error:
        raise WorkspaceRefusal(
            "captured Model default materialization refused: " + str(error.code)
        ) from error


def inputs(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[pb.InputBinding], list[pb.InputAccess]]:
    bindings, accesses = [], []
    for row in closure(rows):
        input_id = grants.MODEL_PREFIX + row.parameter
        bindings.append(
            pb.InputBinding(
                input_id=input_id,
                digest=row.manifest.digest,
                length=row.manifest.length,
                kind_mime=grants.MODEL_MIME,
            )
        )
        accesses.append(
            pb.InputAccess(
                input_id=input_id,
                url="model://" + row.manifest.digest,
                catalog_model=pb.CatalogModelSource(repository=row.repository),
            )
        )
    return bindings, accesses
