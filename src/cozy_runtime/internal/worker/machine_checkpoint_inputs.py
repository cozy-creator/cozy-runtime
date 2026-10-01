"""Execution input obligations backed by native catalog checkpoint roots, never receipts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Container, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import Literal

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from .grants import BoundInput
from .workspace import Journal, Workspace, WorkspaceRefusal

SCHEMA11 = """
CREATE TABLE execution_checkpoint_inputs (
 owner TEXT NOT NULL, recipient TEXT NOT NULL, input_id TEXT NOT NULL,
 spec BLOB NOT NULL, repository TEXT NOT NULL, manifest BLOB NOT NULL,
 manifest_length INTEGER NOT NULL, generation INTEGER NOT NULL,
 native_owner TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
 PRIMARY KEY(owner,recipient,input_id)
) STRICT;
"""


def _native[T](db: Journal, operation: Callable[[], T]) -> T:
    try:
        return db.native(operation)
    except fill.tensorfs_module().errors.Refusal as error:
        raise WorkspaceRefusal(
            f"catalog checkpoint input refused by TensorFS: {error.code}"
        ) from error


class CheckpointInput(msgspec.Struct, frozen=True):
    """One ``execution_checkpoint_inputs`` row."""

    owner: str
    recipient: str
    input_id: str
    spec: bytes
    repository: str
    manifest: bytes
    manifest_length: int
    generation: int
    native_owner: str
    state: Literal["retaining", "held", "releasing", "released"]

    def root(self) -> tuple[str, str, str, int]:
        """The native checkpoint root's owner, repository, manifest and length."""
        return (
            self.native_owner,
            self.repository,
            documents.spell(self.manifest),
            self.manifest_length,
        )


def _row(db: Journal, owner: str, recipient: str, input_id: str) -> CheckpointInput | None:
    return db.one(
        CheckpointInput,
        "SELECT * FROM execution_checkpoint_inputs WHERE owner=? AND recipient=? AND input_id=?",
        (owner, recipient, input_id),
    )


def already_accepted(workspace: Workspace, owner: str, offer: pb.AttemptOffer) -> bool:
    """A replay of an accepted submission does not acquire its collected inputs again."""
    with workspace.locked() as db:
        row = db.execute(
            "SELECT offer FROM executions WHERE owner=? AND request=?", (owner, offer.request_id)
        ).fetchone()
    if row is None:
        return False
    previous = pb.AttemptOffer.FromString(row["offer"])
    if previous.invocation_spec_digest != offer.invocation_spec_digest:
        raise WorkspaceRefusal("root Model inputs differ from the accepted invocation")
    # Executions.submit still checks the complete immutable submission identity.
    return True


def _identity(row: CheckpointInput, offer: pb.AttemptOffer, entry: BoundInput) -> None:
    if (row.spec, row.repository, row.manifest, row.manifest_length) != (
        offer.invocation_spec_digest,
        entry.catalog_repository,
        entry.digest,
        entry.length,
    ):
        raise WorkspaceRefusal("catalog Model input changed its exact accepted source")


def _native_identity(root: Mapping[str, object], row: CheckpointInput, *, released: bool) -> None:
    subject = (root["owner"], root["repository"], root["manifest_digest"], root["manifest_length"])
    if subject != row.root() or root["released"] is not released:
        raise WorkspaceRefusal("native checkpoint root differs from its input obligation")


def preflight(
    workspace: Workspace,
    owner: str,
    offer: pb.AttemptOffer,
    entries: Mapping[str, BoundInput],
    *,
    held: Container[str] = (),
) -> None:
    """Validate every new catalog source before any new recipient root is created.

    `held`: manifests this worker already holds verified complete (`HeldManifests`); only
    `retain`'s own native verification checks them again, and `admission` releases what a
    stale entry let through.
    """
    store = fill.store(workspace.store_root)
    for entry in entries.values():
        with workspace.locked() as db:
            row = _row(db, owner, offer.request_id, entry.input_id)
            if row is not None:
                _identity(row, offer, entry)
                if row.state == "releasing":
                    _release_row(workspace, db, row)
                    row = _row(db, owner, offer.request_id, entry.input_id)
                    assert row is not None
                if row.state in ("retaining", "held"):
                    root = _native(db, partial(store.checkpoint_root, row.native_owner))
                    if root is not None:
                        _native_identity(root, row, released=False)
                        # Existing custody is independent of the original repository.
                        _native(db, partial(store.retain_checkpoint_root, *row.root()))
                        continue
                    if row.state == "held":
                        raise WorkspaceRefusal("accepted catalog input lost its native root")
            if documents.spell(entry.digest) in held:
                continue
            _native(
                db,
                partial(
                    store.verify_checkpoint_source,
                    entry.catalog_repository,
                    documents.spell(entry.digest),
                    entry.length,
                ),
            )


def retain(workspace: Workspace, owner: str, offer: pb.AttemptOffer, entry: BoundInput) -> None:
    store = fill.store(workspace.store_root)
    with workspace.locked() as db:
        row = _row(db, owner, offer.request_id, entry.input_id)
        if row is not None:
            _identity(row, offer, entry)
        if row is None or row.state == "released":
            if db.execute(
                "SELECT 1 FROM executions WHERE owner=? AND request=?", (owner, offer.request_id)
            ).fetchone():
                raise WorkspaceRefusal("collected catalog inputs cannot be reacquired")
            generation = 1 if row is None else row.generation + 1
            if generation >= 1 << 53:
                raise WorkspaceRefusal("catalog input acquisition generation is exhausted")
            native_owner = documents.spell(
                hashlib.sha256(
                    canonical_json.encode(
                        [
                            "cozy.catalog-checkpoint-input/1",
                            owner,
                            offer.request_id,
                            entry.input_id,
                            generation,
                        ]
                    )
                ).digest()
            )
            db.execute(
                "INSERT INTO execution_checkpoint_inputs(owner,recipient,input_id,spec,repository,"
                "manifest,manifest_length,generation,native_owner,state) "
                "VALUES(?,?,?,?,?,?,?,?,?,'retaining') ON CONFLICT(owner,recipient,input_id) "
                "DO UPDATE SET generation=excluded.generation,native_owner=excluded.native_owner,"
                "state='retaining'",
                (
                    owner,
                    offer.request_id,
                    entry.input_id,
                    offer.invocation_spec_digest,
                    entry.catalog_repository,
                    entry.digest,
                    entry.length,
                    generation,
                    native_owner,
                ),
            )
            row = _row(db, owner, offer.request_id, entry.input_id)
            assert row is not None
        if row.state not in ("retaining", "held"):
            raise WorkspaceRefusal("catalog input release must finish before acquisition")
        native = _native(db, partial(store.retain_checkpoint_root, *row.root()))
        _native_identity(native, row, released=False)
        current = _row(db, owner, offer.request_id, entry.input_id)
        if (
            current is None
            or current.native_owner != row.native_owner
            or current.state not in ("retaining", "held")
        ):
            _native(db, partial(store.release_checkpoint_root, *row.root()))
            raise WorkspaceRefusal("catalog input was released during acquisition")
        db.execute(
            "UPDATE execution_checkpoint_inputs SET state='held' WHERE native_owner=?",
            (row.native_owner,),
        )


def check_submit(db: Journal, owner: str, offer: pb.AttemptOffer) -> None:
    wanted = {
        entry.input_id: entry.catalog_model.repository
        for entry in offer.grant.inputs
        if entry.HasField("catalog_model")
    }
    rows = db.all(
        CheckpointInput,
        "SELECT * FROM execution_checkpoint_inputs WHERE owner=? AND recipient=?",
        (owner, offer.request_id),
    )
    if {row.input_id for row in rows} != set(wanted) or any(
        row.state != "held"
        or row.spec != offer.invocation_spec_digest
        or row.repository != wanted[row.input_id]
        for row in rows
    ):
        raise WorkspaceRefusal("catalog Model input is not retained for submission")


def _release_row(workspace: Workspace, db: Journal, row: CheckpointInput) -> None:
    db.execute(
        "UPDATE execution_checkpoint_inputs SET state='releasing' WHERE native_owner=?",
        (row.native_owner,),
    )
    store = fill.store(workspace.store_root)
    native = _native(db, partial(store.release_checkpoint_root, *row.root()))
    _native_identity(native, row, released=True)
    db.execute(
        "UPDATE execution_checkpoint_inputs SET state='released' WHERE native_owner=?",
        (row.native_owner,),
    )


def release(
    workspace: Workspace, owner: str, recipient: str, *, unaccepted_only: bool = False
) -> None:
    with workspace.locked() as db:
        if (
            unaccepted_only
            and db.execute(
                "SELECT 1 FROM executions WHERE owner=? AND request=?", (owner, recipient)
            ).fetchone()
        ):
            return
        rows = db.all(
            CheckpointInput,
            "SELECT * FROM execution_checkpoint_inputs WHERE owner=? AND recipient=? "
            "AND state<>'released' ORDER BY input_id",
            (owner, recipient),
        )
        for row in rows:
            _release_row(workspace, db, row)


@contextmanager
def admission(workspace: Workspace, owner: str, recipient: str) -> Iterator[None]:
    try:
        yield
    except BaseException:
        release(workspace, owner, recipient, unaccepted_only=True)
        raise
