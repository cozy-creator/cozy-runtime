"""Bounded-disk upload: a provider source streamed through conversion into a publication.

Every pass is admitted against the disk the machine observes free. It lands a window of
source members and converts what they unlock within its budget; the driver then puts the
new output in Hub custody and records that custody, so GC may drop the local copies.
Nothing is evicted until admission refuses the smallest pass that still makes progress:
custodied output goes first, then spent source, then the workspace cache. The TensorFS
journal and the Hub's verified objects are the durable progress, so an interrupted upload
resumes without converting or uploading committed work again, and a machine with room
keeps the source for its next conversion.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal import fill, storage_admission
from cozy_runtime.internal.source_interfaces import UploadCivitai
from cozy_runtime.internal.worker import store_gc, upload_plan
from cozy_runtime.internal.worker.machine_publication import (
    PublicationClient,
    PublicationRefusal,
    abandon_publication,
    retain_objects,
    upload_checkpoint,
)
from cozy_runtime.internal.worker.source_steps import (
    Access,
    Advance,
    Advanced,
    Answer,
    Facts,
    Heads,
    Launch,
    Member,
    Pin,
    Plan,
    Resolve,
    Resolved,
    Select,
    Selected,
    Selection,
    SourceRefusal,
    UploadStep,
    Written,
)

# One progress publication freezes at most this many objects.
BATCH_OBJECTS = 1024
# Consecutive publication attempts that land nothing before the upload gives up; Hub's
# own finalization job is exempt while it reports progress. Backoff doubles to a ceiling.
_STALLED_ATTEMPTS = 10
_MAX_BACKOFF_SECONDS = 60.0
_PENDING_BACKOFF_SECONDS = 8.0

# Source bytes whose conversion is journalled; upload overlaps it pass by pass.
STAGE = "Streaming source into the checkpoint"


class Canceled(Exception):
    """The call was canceled, or its parent stopped, between bounded steps."""


def _digest(*parts: object) -> str:
    return hashlib.sha256(canonical_json.encode(list(parts))).hexdigest()


def member_owner(native_owner: str, digest: str) -> str:
    return "sha256:" + _digest("upload-source", native_owner, digest)


def conversion_id(
    manifest: ObjectRef, slots: list[tuple[str, str]], registry: str, recipe: str
) -> str:
    """The TensorFS operation: the same conversion anywhere on this machine resumes it."""
    return "upload-" + _digest(msgspec.to_builtins(manifest), slots, registry, recipe)[:56]


class State(msgspec.Struct, frozen=True):
    """What an upload holds, written before its first byte lands so release finds it."""

    conversion: str
    native_owner: str
    owners: list[str]
    metadata_owner: str


def _state_path(directory: Path, service_id: str) -> Path:
    return directory / "uploads" / f"{service_id}.state"


def _write_state(path: Path, state: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        staged.write(canonical_json.encode(msgspec.to_builtins(state)))
        staged.flush()
        os.fsync(staged.fileno())
    os.replace(staged.name, path)


def _read_state(path: Path) -> State | None:
    try:
        return msgspec.json.decode(path.read_bytes(), type=State)
    except FileNotFoundError:
        return None
    except msgspec.DecodeError as exc:
        raise SourceRefusal("upload_state_malformed", f"{path.name}: {exc}") from exc


def _release_roots(store: fill.Store, owners: list[str]) -> None:
    for owner in owners:
        root = store.tree_root(owner)
        if root and not root["released"]:
            store.release_tree_root(owner)


def _release_composition(store: fill.Store, native_owner: str, *, dispose: bool = True) -> None:
    observed = store.derived_lookup(native_owner)
    if observed.get("state") == "committed" and dispose:
        store.derived_dispose(native_owner)
    elif observed.get("state") == "open":
        if epoch := observed.get("writer_session_id"):
            store.derived_fence(native_owner, epoch)
        store.derived_abandon(native_owner)


def _release_conversion(
    store: fill.Store, directory: Path, conversion: str, live: set[str]
) -> None:
    """The conversion session is shared by every upload of it; the last one drops it."""
    for path in (directory / "uploads").glob("*.state"):
        state = _read_state(path)
        if (
            state is not None
            and state.conversion == conversion
            and path.name.removesuffix(".state") in live
        ):
            return
    if conversion in store.model_source_operations():
        store.release_model_source(conversion)
    (directory / "uploads" / f"{conversion}.heads").unlink(missing_ok=True)


def retire(store: fill.Store, directory: Path, service_id: str, native_owner: str) -> bool:
    """The parent's final ACK. True when the upload keeps bytes a later call reuses.

    What a finished upload left on a disk with room (its source, and a composed model
    whose bytes are all resident) stays retired until pressure spends it, oldest first.
    """
    _release_composition(store, native_owner, dispose=False)
    state = _read_state(_state_path(directory, service_id))
    if state is None:
        return False
    return state.conversion in store.model_source_operations() or any(
        (root := store.tree_root(owner)) and not root["released"]
        for owner in [*state.owners, state.metadata_owner]
    )


def release(
    store: fill.Store, directory: Path, service_id: str, native_owner: str, live: set[str]
) -> None:
    """Spend a retired or finished upload: everything it holds becomes collectable."""
    path = _state_path(directory, service_id)
    state = _read_state(path)
    _release_composition(store, native_owner)
    if state is None:
        return
    _release_roots(store, [*state.owners, state.metadata_owner])
    _release_conversion(store, directory, state.conversion, live - {service_id})
    path.unlink(missing_ok=True)


class Child(Protocol):
    """One killable native step, answered as ``answer`` or refused by its launcher."""

    def __call__[A: Answer](self, launch: Launch[UploadStep], answer: type[A]) -> A: ...


@dataclass(frozen=True)
class Hooks:
    """What an upload asks of its machine."""

    child: Child
    canceled: Callable[[], bool]
    active: Callable[[], set[str]]  # native calls running now
    live: Callable[[], set[str]]  # uploads that may still run again
    evict_retired: Callable[[int], int]  # retired donors, oldest first; bytes reclaimed
    consume: Callable[[str], bool]  # spend a retired upload this one superseded
    progress: Callable[[str, int, int], None]  # stage, bytes done, bytes total


class Upload:
    """One upload call's bounded passes, custody, eviction and publication."""

    def __init__(
        self,
        *,
        store_root: Path,
        native_owner: str,
        service_id: str,
        request: upload_plan.Request,
        access: Access,
        registry: bytes | None,
        client: PublicationClient | None,
        hooks: Hooks,
        directory: Path,
    ) -> None:
        self.store_root, self.native_owner, self.service_id = store_root, native_owner, service_id
        self.request, self.access, self.registry = request, access, registry
        self.client, self.hooks, self.directory = client, hooks, directory
        self.destination = request.destination
        self.uri = upload_plan.source_uri(request)
        self.store = fill.store(store_root)
        self.tensorfs = fill.tensorfs_module()
        self.recipe = upload_plan.recipe_of(request)
        self.slots = upload_plan.slots(
            (self.recipe.profile,) if self.recipe is not None else request.profiles
        )
        self.passes = 0

    @property
    def publisher(self) -> PublicationClient:
        assert self.client is not None, "a local conversion publishes nothing"
        return self.client

    # ------------------------------------------------------------------ steps

    def check(self) -> None:
        if self.hooks.canceled():
            raise Canceled()

    def _step[A: Answer](
        self, step: UploadStep, answer: type[A], lease: storage_admission.Lease
    ) -> A:
        try:
            self.check()
            with lease.scope():
                return self.hooks.child(
                    Launch(store=str(self.store_root), parent_pid=os.getpid(), step=step), answer
                )
        finally:
            lease.close()

    def _metadata_lease(self) -> storage_admission.Lease:
        return storage_admission.acquire(storage_admission.native_write(self.store_root))

    def resolve(self) -> Selection:
        request, recipe = self.request, self.recipe
        if isinstance(request, UploadCivitai):
            carriers: list[str] | None = [request.file] if request.file else []
        elif recipe is not None:
            carriers = list(recipe.carriers)
        else:
            carriers = list(request.carriers) or None
        step = Resolve(
            uri=self.uri,
            carriers=carriers,
            profiles=list(request.profiles),
            metadata=sorted(recipe.metadata) if recipe is not None else [],
            diffusers=recipe is None and not isinstance(request, UploadCivitai),
            registry=self.registry,
            access=self.access,
        )
        return self._step(step, Resolved, self._metadata_lease()).selection

    def select(self, selection: Pin) -> None:
        """No profile named: this machine's TensorFS selects one from the pinned headers."""
        carriers = [row for row in selection.members if upload_plan.carrier(row[0])]
        step = Select(uri=self.uri, members=carriers, registry=self.registry, access=self.access)
        self.slots = [("model", self._step(step, Selected, self._metadata_lease()).profile)]

    # ------------------------------------------------------------------ the run

    def run(
        self, selection: Pin, manifest: ObjectRef, computation: str, epoch: int
    ) -> CheckpointRef:
        return self._publish(self.convert(selection, manifest, computation, epoch))

    def convert(
        self, selection: Pin, manifest: ObjectRef, computation: str, epoch: int
    ) -> ObjectRef:
        """The composed model. With no publication client it stays on this machine: only
        spent source is evicted, never converted output."""
        # A recipe names its metadata; otherwise every non-carrier (Diffusers configs) is.
        metadata_names = (
            set(self.recipe.metadata)
            if self.recipe is not None
            else {row[0] for row in selection.members if not upload_plan.carrier(row[0])}
        )
        self.carriers = [
            row
            for row in selection.members
            if row[0] not in metadata_names and upload_plan.carrier(row[0])
        ]
        self.metadata = [row for row in selection.members if row[0] in metadata_names]
        if not self.carriers:
            raise SourceRefusal(
                "model_source_profiles_invalid", "the selection names no tensor carrier"
            )
        registry = (
            "sha256:" + hashlib.sha256(self.registry).hexdigest() if self.registry else "builtin"
        )
        recipe = self.recipe.name if self.recipe is not None else ""
        self.conversion = conversion_id(manifest, self.slots, registry, recipe)
        self.owners = {row[0]: member_owner(self.native_owner, row[1]) for row in self.carriers}
        heads_path = self.directory / "uploads" / f"{self.conversion}.heads"
        metadata_owner = "sha256:" + _digest("upload-metadata", self.native_owner)
        # Written before the first byte lands, so release can always find what to drop.
        _write_state(
            _state_path(self.directory, self.service_id),
            State(self.conversion, self.native_owner, sorted(self.owners.values()), metadata_owner),
        )
        self.predecessors = self._predecessors()
        self.plan = Plan(
            native_owner=self.native_owner,
            pin=selection,
            conversion=self.conversion,
            source_selection_digest=manifest.digest,
            slots=self.slots,
            carriers=[row[0] for row in self.carriers],
            owners=self.owners,
            metadata=self.metadata,
            metadata_owner=metadata_owner,
            heads_path=str(heads_path),
            registry=self.registry,
            recipe=recipe,
            writer_epoch=epoch,
            computation_digest=computation,
            access=self.access,
        )
        if not heads_path.is_file():
            step = Heads(
                uri=self.uri, members=self.carriers, path=str(heads_path), access=self.access
            )
            self._step(step, Written, self._metadata_lease())
        # The first pass converts nothing: it takes custody of every member already on this
        # disk, measures the plan, and lands the recipe's small metadata beside it.
        present = [row for row in self.carriers if self._present(row)]
        answer = self._advance(present, 0, sum(row[2] for row in self.metadata))
        total = sum(row[2] for row in self.carriers)
        while (model := answer.model) is None:
            spent = set(answer.facts.spent)
            self.hooks.progress(
                STAGE, sum(row[2] for row in self.carriers if row[0] in spent), total
            )
            if self.client is not None:
                self._custody()
            answer = self._next(answer.facts)
        self.hooks.progress(STAGE, total, total)
        self.model = model
        return model

    def _advance(
        self,
        window: list[Member],
        budget: int | None,
        extra: int = 0,
        lease: storage_admission.Lease | None = None,
    ) -> Advanced:
        if lease is None:
            payload = self._downloads(window) + (budget or 0) + extra
            lease = storage_admission.acquire(
                storage_admission.native_write(self.store_root, payload, len(window) + 1)
            )
        self.passes += 1
        step = Advance(
            plan=self.plan,
            window=[row[0] for row in window],
            budget=budget,
            adopt={
                row[0]: [member_owner(owner, row[1]) for owner in self.predecessors.values()]
                for row in window
            },
        )
        return self._step(step, Advanced, lease)

    def _present(self, row: Member) -> bool:
        return self.store.contains(row[1].removeprefix("sha256:"))

    def _downloads(self, window: list[Member]) -> int:
        return sum(row[2] for row in window if not self._present(row))

    def _predecessors(self) -> dict[str, str]:
        """Stopped uploads of this conversion: their durable prefixes are this one's too."""
        active = self.hooks.active()
        owners = {}
        for path in sorted((self.directory / "uploads").glob("*.state")):
            service = path.name.removesuffix(".state")
            state = _read_state(path)
            if (
                service != self.service_id
                and service not in active
                and state is not None
                and state.conversion == self.conversion
            ):
                owners[service] = state.native_owner
        return owners

    # ------------------------------------------------------------------ windows

    def _next(self, facts: Facts) -> Advanced:
        """The largest pass the observed disk admits, evicting only when none fits.

        Eviction order: this upload's custodied output (the Hub holds it), then retired
        donors oldest first, then this upload's spent source, then the workspace cache.
        Source a pending op still needs is never evicted.
        """
        landed, spent = set(facts.landed), set(facts.spent)
        pending = [row for row in self.carriers if row[0] not in landed and row[0] not in spent]
        work = sum(row[2] for row in self.carriers if row[0] in landed and row[0] not in spent)
        required = facts.required_write_bytes
        # The smallest pass that still progresses: convert landed work, or land one member.
        smallest = ([], max(required, 1)) if work else (pending[:1], 0)
        measured: int | None = None
        for stage in ("observe", "custodied output", "retired donors", "spent source"):
            if stage == "custodied output":
                store_gc.collect(self.store_root)
            elif stage == "retired donors":
                need = self._downloads(smallest[0]) + smallest[1]
                self.hooks.evict_retired(need - (measured or 0))
            elif stage == "spent source":
                for name in sorted(landed & spent):
                    root = self.store.tree_root(self.owners[name])
                    if root and not root["released"]:
                        self.store.release_tree_root(self.owners[name])
                store_gc.collect(self.store_root)
            # Every stage measures afresh: an ideal request refused reports the free disk.
            free: int | None = None
            while (plan := self._window(pending, work, required, free)) is not None:
                window, budget = plan
                lease, measured = self._admit(window, budget, recover=False)
                if lease is not None:
                    return self._run_pass(facts, window, budget, lease)
                if measured == free:
                    break
                free = measured
        window, budget = smallest
        lease, measured = self._admit(window, budget, recover=True)
        if lease is not None:
            return self._run_pass(facts, window, budget, lease)
        raise storage_admission.StorageRefusal(
            "no upload pass fits the free disk after evicting custodied output, retired "
            "work, spent source and the workspace cache",
            {"available": measured or 0, "required": self._downloads(window) + budget},
        )

    def _run_pass(
        self, facts: Facts, window: list[Member], budget: int, lease: storage_admission.Lease
    ) -> Advanced:
        answer = self._advance(window, budget, lease=lease)
        self._progressed(facts, answer.facts)
        return answer

    @staticmethod
    def _window(
        pending: list[Member], work: int, required: int, free: int | None
    ) -> tuple[list[Member], int] | None:
        """Land members while their bodies and output fit beside the landed work.

        ``free`` is None until a refusal measures the disk: ask for everything first.
        """
        if free is None:
            return pending, work + sum(row[2] for row in pending)
        window: list[Member] = []
        downloads = 0
        for row in pending:
            if work + 2 * (downloads + row[2]) > free:
                break
            window.append(row)
            downloads += row[2]
        budget = min(work + downloads, free - downloads)
        if window:
            return window, budget
        if work and budget >= max(required, 1):
            return [], budget
        if not work and pending and pending[0][2] <= free:
            # Nothing is landed to convert: land one member now, convert it next pass.
            return [pending[0]], free - pending[0][2]
        return None

    def _admit(
        self, window: list[Member], budget: int, *, recover: bool
    ) -> tuple[storage_admission.Lease | None, int | None]:
        """A lease for this pass, or the payload bytes a refusal measured as free."""
        payload = self._downloads(window) + budget
        write = storage_admission.native_write(self.store_root, payload, len(window) + 1)
        try:
            return storage_admission.acquire(write, recover=recover), None
        except storage_admission.StorageRefusal as exc:
            facts = exc.facts
            if exc.code != "insufficient_storage" or "available" not in facts:
                raise
            if facts.get("required_inodes", 0) > facts.get("available_inodes", 0) - facts.get(
                "reserve_inodes", 0
            ) - facts.get("reserved_inodes", 0):
                return None, 0
            metadata = write.bytes - payload
            free = (
                facts["available"] - facts.get("reserve", 0) - facts.get("reserved", 0) - metadata
            )
            return None, max(free, 0)

    def _progressed(self, before: Facts, after: Facts) -> None:
        """A pass must land a member, convert an op, or measure the next op it could not."""
        if (
            len(after.landed) > len(before.landed)
            or len(after.spent) > len(before.spent)
            or after.converted_roles
            or after.complete
            or after.required_write_bytes != before.required_write_bytes
        ):
            return
        raise storage_admission.StorageRefusal(
            "an upload pass made no progress within its admitted disk",
            {"required": after.required_write_bytes},
        )

    # ------------------------------------------------------------------ custody

    def _push(self, manifest: str) -> Callable[[str, int, str], None]:
        def push(key: str, length: int, grant: str) -> None:
            self.check()
            self.store.checkpoint_push(
                key, length, key == manifest, grant, allow_local=self.access["allow_local"]
            )

        return push

    def _custody(self) -> None:
        """Put every resident journalled object without custody in Hub custody."""
        rows = self.tensorfs.model_source_objects(self.store, self.conversion)
        pending = sorted(
            (row["object_id"], row["length"])
            for row in rows
            if row["custody"] is None and row["resident"]
        )
        for offset in range(0, len(pending), BATCH_OBJECTS):
            batch = dict(pending[offset : offset + BATCH_OBJECTS])
            operation = "ingest-" + _digest(self.conversion, sorted(batch))[:48]
            self._retry(
                partial(
                    retain_objects,
                    self.publisher,
                    destination=self.destination,
                    operation=operation,
                    objects=batch,
                    before_stage=self.check,
                    push=self._push(""),
                    streams=self.tensorfs.transfer_streams(len(batch)),
                )
            )
            self.tensorfs.record_model_source_custody(
                self.store,
                self.conversion,
                sorted(batch.items()),
                f"{self.destination}#{operation}",
            )

    def _retry[T](self, attempt: Callable[[], T]) -> T:
        stalls, wake = 0, threading.Event()
        while True:
            self.check()
            try:
                return attempt()
            except PublicationRefusal as exc:
                pending = exc.code == "publication.finalization_pending"
                if not (exc.status in {408, 425, 429} or exc.status >= 500):
                    raise
                stalls = 0 if pending else stalls + 1
                if stalls >= _STALLED_ATTEMPTS:
                    raise
                wake.wait(
                    _PENDING_BACKOFF_SECONDS if pending else min(2.0**stalls, _MAX_BACKOFF_SECONDS)
                )

    # ------------------------------------------------------------------ publication

    def _publish(self, model: ObjectRef) -> CheckpointRef:
        objects = {model.digest: model.length}
        for row in self.store.walk(model.digest):
            objects[row["id"]] = row["length"]
        checkpoint = CheckpointRef(self.destination, model.digest, model, self.service_id, "")
        return self._retry(
            lambda: upload_checkpoint(
                self.publisher,
                operation=self.service_id,
                checkpoint=checkpoint,
                objects=objects,
                before_stage=self.check,
                before_write=self.check,
                push=self._push(model.digest),
                landed=lambda _n, _b: None,
                streams=self.tensorfs.transfer_streams(len(objects)),
            )
        )

    def finish(self) -> None:
        """After the checkpoint commits: drop progress holds and superseded work.

        The checkpoint retains every object, so progress publications hold nothing it
        does not. Nothing local is evicted without pressure: when every byte of the
        composed model is still resident it stays, with its conversion and source, for a
        later call on this machine. A model that pressure left partly remote is dropped:
        only the Hub can serve it now.
        """
        prefix = self.destination + "#"
        operations = sorted(
            {
                row["custody"].removeprefix(prefix)
                for row in self.tensorfs.model_source_objects(self.store, self.conversion)
                if isinstance(row["custody"], str) and row["custody"].startswith(prefix)
            }
        )
        for operation in operations:
            self._retry(
                partial(
                    abandon_publication,
                    self.publisher,
                    destination=self.destination,
                    operation=operation,
                )
            )
        for service in self.predecessors:
            self.hooks.consume(service)
        if self.resident():
            return
        _release_composition(self.store, self.native_owner)
        _release_conversion(
            self.store, self.directory, self.conversion, self.hooks.live() - {self.service_id}
        )

    def resident(self) -> bool:
        """Every object of the published model is on this disk."""
        return all(
            self.store.contains(row["id"].removeprefix("sha256:"))
            for row in self.store.walk(self.model.digest)
        )
