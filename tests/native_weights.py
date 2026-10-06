"""Real native TensorFS/descriptor bindings for component numerical proofs.

This fixture records native checkpoint/receipt facts, not a worker execution or a
Creator outcome. Rust-node/public-CLI lifecycle qualification is a separate test.
"""

from __future__ import annotations

import hashlib
import math
import os
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
from tensorfs.derived import OutputCapability

from cozy_runtime.author import Context, ModelArtifact, canonical_json
from cozy_runtime.author._executor_requests import (
    Answer,
    Reply,
    Request,
    WriterAdopt,
    WriterOutput,
    WriterSource,
    encode,
    respond,
)
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.weights_writer import (
    MODEL_PREFIX,
    ExecutionStorage,
    WriterAttempt,
    WriterBinding,
    WriterBroker,
)


class NativeExecution:
    def __init__(
        self,
        store: Any,
        root: Path,
        name: str,
        models: dict[str, ModelArtifact],
        outputs: dict[str, int],
        *,
        epoch: int = 1,
        after_checkpoint: Callable[[Any], None] | None = None,
    ) -> None:
        self.spool = root / f"{name}-{epoch}"
        self.spool.mkdir(parents=True, exist_ok=True)
        destinations = [
            {"output_id": slot, "max_bytes": maximum, "mime_type": "application/x-cozytensors"}
            for slot, maximum in outputs.items()
        ]
        spec = {
            "job": {"job_descriptor_id": "14" * 32},
            "payload_digest": "sha256:" + "12" * 32,
            "inputs": [
                {
                    "input_id": MODEL_PREFIX + parameter,
                    "digest": model.manifest.digest,
                    "length": model.manifest.length,
                }
                for parameter, model in models.items()
            ],
            "outputs": destinations,
        }
        raw = canonical_json.encode(spec)
        self.attempt = SimpleNamespace(
            request_id=name,
            attempt=epoch,
            digest=hashlib.sha256(raw).digest(),
            canceling="",
            state="running",
            spool=self.spool,
            spec=spec,
            weights_work_fingerprint="sha256:" + "16" * 32,
        )
        self.declarations: dict[str, bytes] = {}
        self.receipts: dict[str, Mapping[str, object]] = {}
        self.replayed_outputs: set[str] = set()
        self.checkpointed = False
        self._checkpoints = root / (hashlib.sha256(name.encode()).hexdigest() + ".checkpoints.json")
        self._heads: dict[str, tuple[str, int]] = (
            msgspec.json.decode(self._checkpoints.read_bytes(), type=dict[str, tuple[str, int]])
            if self._checkpoints.exists()
            else {}
        )

        def bind(attempt: WriterAttempt, slot: str, declaration: bytes) -> tuple[str, int]:
            subject = canonical_json.encode(
                {
                    "owner_authority_scope": "package-proof",
                    "request_id": attempt.request_id,
                    "invocation_spec_digest": "sha256:" + attempt.digest.hex(),
                    "output_slot": slot,
                }
            )
            transaction = (
                "sha256:"
                + hashlib.sha256(b"cozy.runtime.artifact-transaction\0" + subject).hexdigest()
            )
            self.declarations[transaction] = declaration
            self.broker.authorize(
                attempt,
                transaction,
                epoch,
                hashlib.sha256(declaration).digest(),
                slot,
                checkpoint=self._heads.get(transaction),
            )
            return transaction, epoch

        def checkpoint(
            attempt: WriterAttempt,
            transaction: str,
            binding: WriterBinding,
            facts: Mapping[str, object],
        ) -> None:
            head = msgspec.convert(facts["head"], str)
            length = msgspec.convert(facts["head_length"], int)
            store.validate_derived_checkpoint(
                transaction,
                self.declarations[transaction],
                head,
                length,
                operation_id=attempt.request_id,
                slot=binding.slot,
                plan_digest=msgspec.convert(facts["plan_digest"], str),
            )
            self._heads[transaction] = (head, length)
            staged = self._checkpoints.with_suffix(".pending")
            with staged.open("wb") as stream:
                stream.write(canonical_json.encode(self._heads))
                stream.flush()
                os.fsync(stream.fileno())
            staged.replace(self._checkpoints)
            self.checkpointed = True
            if after_checkpoint is not None:
                after_checkpoint(facts)

        def receipt(
            _attempt: WriterAttempt,
            transaction: str,
            _binding: WriterBinding,
            facts: Mapping[str, object],
        ) -> None:
            assert store.derived_lookup(transaction)["receipt"] == facts
            assert facts["declaration_digest"] == (
                "sha256:" + hashlib.sha256(self.declarations[transaction]).hexdigest()
            )
            self.receipts[transaction] = dict(facts)

        self.broker = WriterBroker(
            lambda: store,
            checkpoint=checkpoint,
            receipt=receipt,
            bind_output=bind,
        )
        self.client = ExecutionStorage(self.spool, self.exchange, outputs)

    def declaration(self, facts: Mapping[str, object]) -> dict[str, Any]:
        """The exact native declaration named by this receipt's digest."""
        raw = self.declarations[str(facts["transaction_id"])]
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == facts["declaration_digest"]
        return cast(dict[str, Any], canonical_json.decode(raw))

    def exchange[A: Answer](self, request: Request, into: type[A], /) -> A:
        """Use the production control framing and native descriptor transfer."""
        assert isinstance(request, WriterSource | WriterOutput | WriterAdopt)
        left, right = socket.socketpair()
        with left, right:
            executor, owner = Channel(left), Channel(right)
            executor.send({"event": "request", "seq": 1, **encode(request)})
            frame = owner.recv()
            assert frame is not None

            def handle(message: Request) -> Reply:
                assert isinstance(message, WriterSource | WriterOutput | WriterAdopt)
                return self.broker.handle(self.attempt, message)

            answer, handoff = respond(frame, handle)
            assert handoff is None or isinstance(handoff, socket.socket)
            try:
                owner.send(answer)
                if handoff is not None:
                    owner.send_descriptor(handoff)
            finally:
                if handoff is not None:
                    handoff.close()
            reply = executor.recv()
            assert reply is not None and reply["seq"] == 1
            if reply.get("descriptor") is True:
                reply["descriptor"] = executor.recv_descriptor()
            return msgspec.convert(reply, into)

    def context(self) -> Context:
        def open_output(slot: str, definition: Any) -> Any:
            transaction = self.client.open_output(slot, definition)
            if transaction.receipt is not None:
                self.replayed_outputs.add(slot)
            return transaction

        return Context(
            self.attempt.request_id,
            math.inf,
            _tensorfs_source=lambda model: self.client.source(model.checkpoint_ref),
            _tensorfs_output=lambda slot: OutputCapability(
                lambda definition: open_output(slot, definition)
            ),
            _tensorfs_adopt=self.client.adopt_model,
        )

    def __enter__(self) -> NativeExecution:
        return self

    def __exit__(self, *exc: object) -> None:
        self.broker.close()
