"""The run output log's products: what a run has made so far, as it makes it (wire 65).

A function publishes an asset into one of its declared outputs (an asset field of its result;
`list[...]` fields grow). Runtime commits the bytes into native custody, holds them for the
run's owner and journals one `product` entry on the root execution's log, in that order, so
every entry the owner reads can be fetched. Returning the result publishes what the fold does
not already show: the run's result is the fold of its products, whether it completes, fails or
is canceled. The owner reads bytes with ReadByteTreeObject from each entry's own hold and
acknowledges the terminal: delivered (wire 66). The products then stay until the store's GC
evicts them under pressure, delivered first, so any authorized client can fetch them later. A
single output's superseded product is released as soon as a newer one replaces it.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from cozy_runtime.author._executor_requests import Answer, Publish, Published, refuse
from cozy_runtime.internal import fill
from cozy_runtime.internal.executor_replies import OutputRow
from cozy_runtime.internal.worker import byte_outputs, machine_byte_results, machine_models
from cozy_runtime.internal.worker.workspace import NativeHold, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.attempts import AttemptRecord
    from cozy_runtime.internal.worker.calls import Calls
    from cozy_runtime.internal.worker.workspace_executions import Executions

SET, APPEND = pb.RUN_PRODUCT_OP_SET, pb.RUN_PRODUCT_OP_APPEND
MAX_PARTS = 128


class DeliveringStore(Protocol):
    """A TensorFS store that tells delivered tree roots from live ones ("deliver/1"): its GC
    may evict a delivered one under pressure, oldest delivery first."""

    def deliver_tree_root(self, owner: str) -> object: ...


def recipient(request: str) -> str:
    """The run log's own holder: separate from the execution's, so cancel keeps products."""
    return f"{request}/products"


@dataclass(slots=True)
class Fold:
    """The run's outputs as its log states them: last SET per single output, APPENDs in order."""

    single: dict[str, pb.RunProduct] = field(default_factory=dict)
    lists: dict[str, list[pb.RunProduct]] = field(default_factory=dict)

    def add(self, product: pb.RunProduct) -> None:
        if product.op == APPEND:
            self.lists.setdefault(product.output, []).append(product)
        else:
            self.single[product.output] = product

    def holds(self) -> set[str]:
        """The log's hold paths the fold still names: its products and their parts."""
        named = set()
        for product in [*self.single.values(), *(p for ps in self.lists.values() for p in ps)]:
            if product.HasField("source"):
                named.add(hold_path(product.content, product.source.source))
            named.update(hold_path(part.content, part.source.source) for part in product.parts)
        return named


def hold_path(content: pb.Ref, source: pb.NativeByteTreeRef) -> str:
    """One hold per byte string and native tree: the same bytes may come from two producers."""
    return f"{documents.spell(content.digest)} {source.producer_root_id}"


class Products:
    """Journals products on the root execution's log and keeps their bytes held."""

    def __init__(self, workspace: Workspace, executions: Executions, calls: Calls | None) -> None:
        self.workspace, self.executions, self.calls = workspace, executions, calls
        #: One publish or return at a time per run: an entry's index and its GC see one fold.
        self.lock = threading.Lock()

    def shown(self, owner: str, attempt: AttemptRecord) -> bool:
        """Only an execution's own root has an owner reading its log."""
        return (
            self.executions.owns(owner, attempt.request_id)
            and self.executions.scheduling_root(owner, attempt.request_id)[0] == attempt.request_id
        )

    def fold(self, owner: str, request: str) -> Fold:
        fold = Fold()
        for body in self.executions.product_bodies(owner, request):
            fold.add(documents.parse(body, pb.RunProduct))
        return fold

    # ------------------------------------------------------------------ publish

    def publish(self, owner: str, attempt: AttemptRecord, request: Publish) -> Answer:
        """`Outputs.publish`: commit, hold and journal one product while the run is running."""
        if not self.shown(owner, attempt):
            return Published(ok=True)
        slots = set(attempt.grant.outputs)
        if request.output in slots and not request.output.endswith(machine_byte_results.LIST):
            op, slot = SET, request.output
        elif request.output + machine_byte_results.LIST in slots:
            op, slot = APPEND, request.output + machine_byte_results.LIST
        else:
            return refuse(
                "output_undeclared",
                f"{request.output!r} is not an asset field of this function's result",
            )
        if len(request.parts) > MAX_PARTS:
            return refuse("output_parts", f"a product has at most {MAX_PARTS} parts")
        bound = attempt.grant.outputs[slot].max_bytes or (256 << 20)
        try:
            with self.lock:
                if request.parts:
                    product = self._composite(owner, attempt, request, bound)
                else:
                    product = self._single(owner, attempt, request, bound)
                product.output, product.op = request.output, op
                product.label, product.media_type = request.label, request.media_type
                sequence = self._journal(owner, attempt.request_id, product)
        except (WorkspaceRefusal, ValueError, OSError) as exc:
            return refuse("publish_refused", str(exc))
        return Published(
            ok=True,
            digest=documents.spell(product.content.digest),
            length=product.content.length,
            sequence=sequence,
        )

    def _single(
        self, owner: str, attempt: AttemptRecord, request: Publish, bound: int
    ) -> pb.RunProduct:
        if request.asset_ref.startswith("sha256:"):
            # A child call's result, already in this attempt's custody: hold the same tree.
            if self.calls is None:
                raise WorkspaceRefusal("a received asset needs the call lane")
            row = OutputRow(
                output_id=request.output,
                asset_ref=request.asset_ref,
                kind=request.asset_kind,
                media_type=request.media_type,
                size_bytes=request.size_bytes,
                digest=request.digest,
            )
            received = self.calls.received_output(attempt, row)
            if received is None:
                raise WorkspaceRefusal("the published asset is not one this attempt received")
            source = received[1].source
            content = pb.Ref(digest=documents.raw(request.asset_ref), length=request.size_bytes)
        else:
            path = self._spool_file(attempt, _spool_name(request.asset_ref))
            source, content = self._commit(owner, attempt, path, request.media_type, bound)
        return pb.RunProduct(content=content, source=self._hold(owner, attempt, content, source))

    def _composite(
        self, owner: str, attempt: AttemptRecord, request: Publish, bound: int
    ) -> pb.RunProduct:
        whole, length = hashlib.sha256(), 0
        product = pb.RunProduct()
        for part in request.parts:
            path = self._spool_file(attempt, part.local)
            with path.open("rb") as reader:
                while block := reader.read(1 << 20):
                    whole.update(block)
                    length += len(block)
            if length > bound:
                raise WorkspaceRefusal("the product exceeds its output's byte bound")
            source, content = self._commit(owner, attempt, path, "application/octet-stream", bound)
            product.parts.append(
                pb.RunProductPart(
                    content=content,
                    source=self._hold(owner, attempt, content, source),
                    duration_us=part.duration_us,
                )
            )
        product.content.CopyFrom(pb.Ref(digest=whole.digest(), length=length))
        return product

    @staticmethod
    def _spool_file(attempt: AttemptRecord, name: str) -> Path:
        """A regular file of the attempt's spool, at most one directory down (a join's parts)."""
        spool = attempt.spool
        steps = name.split("/")
        if (
            spool is None
            or not 1 <= len(steps) <= 2
            or any(not step or step in (".", "..") or step.startswith(".") for step in steps)
        ):
            raise WorkspaceRefusal("the published asset names no file in this attempt's spool")
        path = spool.joinpath(*steps)
        if len(steps) == 2 and (spool / steps[0]).is_symlink():
            raise WorkspaceRefusal("the published asset's directory is a symbolic link")
        if path.is_symlink() or not path.is_file():
            raise WorkspaceRefusal("the published asset has no bytes in this attempt's spool")
        return path

    def _commit(
        self, owner: str, attempt: AttemptRecord, path: Path, media_type: str, bound: int
    ) -> tuple[pb.NativeByteTreeRef, pb.Ref]:
        digest = hashlib.sha256()
        with path.open("rb") as reader:
            while block := reader.read(1 << 20):
                digest.update(block)
        entry = byte_outputs.commit(
            self.workspace,
            owner,
            attempt.request_id,
            attempt.attempt,
            attempt.digest,
            f"product.{digest.hexdigest()}",
            path,
            tree=False,
            max_bytes=bound,
            media_type=media_type or "application/octet-stream",
        )
        return entry.native_tree, pb.Ref(digest=entry.digest, length=entry.length)

    def _hold(
        self, owner: str, attempt: AttemptRecord, content: pb.Ref, source: pb.NativeByteTreeRef
    ) -> pb.NativeByteRetentionRequest:
        """The log's hold, keyed by content: one hold per distinct byte string of the run."""
        grant = machine_models.retain_bytes(
            self.workspace, owner, recipient(attempt.request_id), hold_path(content, source), source
        )
        return pb.NativeByteRetentionRequest(source=grant.source, retention_id=grant.retention_id)

    def _journal(self, owner: str, request: str, product: pb.RunProduct) -> int:
        fold = self.fold(owner, request)
        if product.op == SET:
            current = fold.single.get(product.output)
            if current is not None and current.content == product.content:
                return 0
        else:
            product.index = len(fold.lists.get(product.output, []))
        sequence = self.executions.notice(owner, request, "product", documents.document(product))
        fold.add(product)
        self._collect(owner, request, fold)
        return sequence

    def _collect(self, owner: str, request: str, fold: Fold) -> None:
        """Release the holds of bytes the fold no longer names: superseded SETs."""
        named = fold.holds()
        with self.workspace.locked() as db:
            rows = db.all(
                machine_models.ModelHold,
                "SELECT * FROM execution_model_holds WHERE owner=? AND recipient=?",
                (owner, recipient(request)),
            )
        for row in rows:
            if row.path not in named:
                _release(self.workspace, owner, row)

    # ------------------------------------------------------------------ return

    def finish(
        self, owner: str, attempt: AttemptRecord, entries: Sequence[pb.OutputEntry]
    ) -> str | None:
        """The returned result as the log's last products, or why it contradicts the log."""
        if not self.shown(owner, attempt):
            return None
        slots = set(attempt.grant.outputs)
        with self.lock:
            fold = self.fold(owner, attempt.request_id)
            returned: dict[str, list[pb.OutputEntry]] = {}
            fresh: list[tuple[pb.OutputEntry, int, int]] = []
            for entry in entries:
                slot = machine_byte_results.slot_of(entry.output_id, slots)
                if slot is None or not entry.HasField("native_tree"):
                    continue
                if slot.endswith(machine_byte_results.LIST):
                    returned.setdefault(slot.removesuffix(machine_byte_results.LIST), []).append(
                        entry
                    )
                    continue
                current = fold.single.get(entry.output_id)
                content = pb.Ref(digest=entry.digest, length=entry.length)
                if current is None or current.content != content:
                    fresh.append((entry, SET, 0))
            for output in set(fold.single) - {e.output_id for e in entries}:
                return f"output_changed: {output} was published and the result leaves it empty"
            for output in set(fold.lists) | set(returned):
                items = sorted(
                    returned.get(output, []), key=lambda e: int(e.output_id.rpartition(".")[2])
                )
                published = fold.lists.get(output, [])
                if len(items) < len(published) or any(
                    item.digest != shown.content.digest
                    for item, shown in zip(items, published, strict=False)
                ):
                    return (
                        f"output_changed: the returned {output} does not extend the "
                        f"{len(published)} products already published"
                    )
                for index in range(len(published), len(items)):
                    fresh.append((items[index], APPEND, index))
            for entry, op, index in fresh:
                # Publishing collects holds absent from the current fold. Acquire
                # each new hold only when its product is about to join that fold.
                self._journal(
                    owner, attempt.request_id, self._returned(owner, attempt, entry, op, index)
                )
        return None

    def _returned(
        self, owner: str, attempt: AttemptRecord, entry: pb.OutputEntry, op: int, index: int
    ) -> pb.RunProduct:
        content = pb.Ref(digest=entry.digest, length=entry.length)
        output = entry.output_id.rpartition(".")[0] if op == APPEND else entry.output_id
        return pb.RunProduct(
            output=output,
            op=op,  # type: ignore[arg-type]
            index=index,
            content=content,
            media_type=entry.mime_type,
            source=self._hold(owner, attempt, content, entry.native_tree),
        )

    # ------------------------------------------------------------------ acknowledgement

    def deliver(self, owner: str, request: str) -> None:
        """The owner holds the whole log. Its products stay until the store's GC evicts them;
        a store that cannot tell delivered bytes from live ones gets them released now."""
        if "deliver/1" not in fill.capabilities():
            machine_models.release(self.workspace, owner, recipient(request))
            return
        store = cast(DeliveringStore, fill.store(self.workspace.store_root))
        with self.workspace.locked() as db:
            rows = db.all(
                machine_models.ModelHold,
                "SELECT * FROM execution_model_holds WHERE owner=? AND recipient=?",
                (owner, recipient(request)),
            )
            for row in rows:
                root = pb.DerivedRetentionRequest.FromString(row.retention).retention_id
                try:
                    db.native(partial(store.deliver_tree_root, root))
                except fill.tensorfs_module().errors.Refusal as exc:
                    if exc.code != "ROOT_ABSENT":  # already evicted: a replayed ack
                        raise


def _spool_name(ref: str) -> str:
    """A saved asset's spool file, named as the post phase names it."""
    tail = ref.rsplit("/", 2)
    return f"{tail[-2]}-{tail[-1]}" if len(tail) == 3 else ""


def _release(workspace: Workspace, owner: str, row: machine_models.ModelHold) -> None:
    request = pb.DerivedRetentionRequest.FromString(row.retention)
    with workspace.locked() as db:
        held = db.one(NativeHold, "SELECT * FROM holds WHERE id=?", (request.retention_id,))
        if held is not None and held.state == "held":
            workspace._change_hold(db, owner, request, release=True, kind=held.kind)
