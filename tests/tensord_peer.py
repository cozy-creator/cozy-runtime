"""TensorD's half of an attempt's weights channels, over a real `tensorfs.Store`, for Runtime tests.

It answers `weights_writer` requests as `crates/tensord/src/weights.rs` does and serves the
source and derive channels as `crates/tensord/src/derive/{mod,source}.rs` document them, with
the bytes going through the real v2 store and `Writer`. What it does not do: pins, storage
admission, `set_order` (the Python `Writer` has none; tests write in natural order) and
renamed copies. The cross-process test against TensorD itself lives with TensorD.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tensorfs

from cozy_runtime.author import Context

from cozy_runtime.author._executor_requests import (
    Adopted,
    Answer,
    Opened,
    Request,
    WriterAdopt,
    WriterOutput,
    WriterSource,
    refuse,
)
from cozy_runtime.tensord_channels import Channel, Refused
from cozy_runtime.internal.weights_writer import ExecutionStorage, exchange_file

HEADER = "model.cozytensors"


def blob(store: tensorfs.Store, sha256: str) -> Path:
    return Path(store.root, "blobs", sha256[:2], sha256[2:4], sha256)


def stored(
    store: tensorfs.Store, ck: tensorfs.Checkpoint, part: tensorfs.Part, offset: int, length: int
) -> bytes:
    """`length` stored bytes of `part` from `offset`, across its chunks."""
    chunks = ck.chunks
    index, at = part.chunk, part.offset + offset
    while length and at >= chunks[index][1]:
        at -= chunks[index][1]
        index += 1
    out = bytearray()
    while len(out) < length:
        sha, size = chunks[index]
        take = min(length - len(out), size - at)
        with blob(store, sha).open("rb") as file:
            file.seek(at)
            out += file.read(take)
        index, at = index + 1, 0
    return bytes(out)


def location(ck: tensorfs.Checkpoint, component: str, key: str, role: str) -> tensorfs.Part:
    info = ck.tensors[component][key]
    return info.parts[role]


def ref(manifest: tensorfs.Manifest) -> dict[str, object]:
    return {"sha256": manifest.sha256, "length": manifest.length}


class _Session:
    """One served channel: frames, answers, refusals."""

    def __init__(self, sock: socket.socket) -> None:
        self.channel = Channel(sock.detach())

    def answer(self, result: object) -> None:
        self.channel.send({"result": result})

    def refuse(self, code: str, detail: str) -> None:
        self.channel.send({"error": code, "detail": detail[:1024]})

    def send(self, data: bytes) -> None:
        self.answer({"length": len(data)})
        self.channel.data(data)


def _read(
    session: _Session, store: tensorfs.Store, ck: tensorfs.Checkpoint, request: Mapping[str, Any]
) -> None:
    try:
        part = location(ck, request["component"], request["key"], request.get("role", ""))
    except KeyError as exc:
        session.refuse("absent", str(exc))
        return
    offset = int(request.get("offset", 0))
    length = int(request.get("length", part.length - offset))
    if offset + length > part.length:
        session.refuse("invalid_request", f"bytes {offset}+{length} of {part.length}")
        return
    session.send(stored(store, ck, part, offset, length))


def serve_source(store: tensorfs.Store, manifest: str, sock: socket.socket) -> None:
    session = _Session(sock)
    ck = store.checkpoint(manifest)
    try:
        while (request := session.channel.receive()) is not None:
            assert isinstance(request, dict)
            op = request.get("op")
            if op == "inspect":
                files = {path: ref_of(store, ck, path) for path in ck.files}
                assert ck.manifest is not None
                session.answer({"manifest": ref(ck.manifest), "files": files})
            elif op == "file":
                session.send(ck.file(request["path"]))
            elif op == "read":
                _read(session, store, ck, request)
            else:
                session.refuse("invalid_request", f"{op!r} is not a source operation")
    finally:
        session.channel.close()


def ref_of(store: tensorfs.Store, ck: tensorfs.Checkpoint, path: str) -> dict[str, object]:
    data = ck.file(path)
    return {"sha256": hashlib.sha256(data).hexdigest(), "length": len(data)}


class Peer:
    """`ExecutionStorage`'s exchange, answered as TensorD's weights grant does."""

    def __init__(
        self,
        store: tensorfs.Store,
        spool: Path,
        sources: Mapping[str, int],
        outputs: Mapping[str, int],
        request_id: str = "test",
    ) -> None:
        self.store, self.spool, self.request_id = store, spool, request_id
        self.sources, self.outputs = dict(sources), dict(outputs)
        spool.mkdir(parents=True, exist_ok=True)
        self.committed: dict[str, tuple[str, int]] = {}
        self.adopted: dict[str, tuple[str, int]] = {}
        #: channel ends a test may close to interrupt an output
        self.servers: list[socket.socket] = []
        self.client = ExecutionStorage(spool, self.exchange, outputs)

    def exchange[A: Answer](self, request: Request, into: type[A], /) -> A:
        if isinstance(request, WriterSource):
            answer: Answer = self._source(request)
        elif isinstance(request, WriterOutput):
            answer = self._output(request)
        elif isinstance(request, WriterAdopt):
            answer = self._adopt(request)
        else:
            answer = refuse("request_unsupported", type(request).__name__)
        assert isinstance(answer, into), answer
        return answer

    def _source(self, request: WriterSource) -> Answer:
        length = self.sources.get(request.manifest)
        if length is None:
            return refuse("weights_source_ungranted", "source is outside the invocation")
        client, server = socket.socketpair()
        threading.Thread(
            target=serve_source, args=(self.store, request.manifest, server), daemon=True
        ).start()
        return Opened(ok=True, length=length, descriptor=client.detach())

    def _output(self, request: WriterOutput) -> Answer:
        slot = request.output_slot
        raw = exchange_file(self.spool, slot, "derivation").read_bytes()
        derivation = json.loads(raw)
        if derivation["max_new_bytes"] != self.outputs.get(slot):
            return refuse("weights_writer_ungranted", "output exceeds invocation grant")
        granted = [r for r in [derivation["base"], *derivation["sources"]] if r is not None]
        if any(self.sources.get("sha256:" + r["sha256"]) != r["length"] for r in granted):
            return refuse("weights_source_ungranted", "source is outside invocation grant")
        transaction = "sha256:" + hashlib.sha256(f"{slot}\0".encode() + raw).hexdigest()
        client, server = socket.socketpair()
        self.servers.append(server)
        threading.Thread(
            target=self._serve_output, args=(transaction, derivation, server), daemon=True
        ).start()
        return Opened(ok=True, transaction=transaction, descriptor=client.detach())

    def _adopt(self, request: WriterAdopt) -> Answer:
        raw = exchange_file(self.spool, request.output_slot, "native-receipt").read_bytes()
        relayed = json.loads(raw)
        manifest = (relayed["manifest"]["sha256"], relayed["manifest"]["length"])
        if self.committed.get(request.transaction) != manifest:
            return refuse("weights_receipt_mismatch", "receipt differs from the store's commit")
        self.adopted[request.output_slot] = manifest
        return Adopted(
            ok=True,
            request_id=self.request_id,
            manifest="sha256:" + manifest[0],
            manifest_length=manifest[1],
            receipt_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
        )

    def _serve_output(
        self, transaction: str, derivation: Mapping[str, Any], sock: socket.socket
    ) -> None:
        session = _Session(sock)
        store = self.store
        granted = {
            "sha256:" + r["sha256"]: store.checkpoint("sha256:" + r["sha256"])
            for r in [derivation["base"], *derivation["sources"]]
            if r is not None
        }
        done = self.committed.get(transaction)
        base = derivation["base"]
        writer = (
            None
            if done
            else store.writer(
                base=None if base is None else "sha256:" + base["sha256"],
                key=transaction.removeprefix("sha256:"),
            )
        )
        written = 0
        try:
            while (request := session.channel.receive()) is not None:
                assert isinstance(request, dict)
                op = request.get("op")
                try:
                    if op == "has":
                        session.answer(
                            writer is None or writer.has(request["component"], request["key"])
                        )
                    elif op == "put":
                        parts = request.get("parts") or [
                            {
                                "role": "",
                                "dtype": request["dtype"],
                                "shape": request["shape"],
                                "length": request["length"],
                            }
                        ]
                        written += sum(p["length"] for p in parts)
                        if written > derivation["max_new_bytes"]:
                            raise Refused("bound_exceeded", f"{written} new bytes")
                        data = [self._bytes(session, granted, p["length"]) for p in parts]
                        if writer is not None:
                            if "encoding" in request:
                                encoding = request["encoding"]
                                writer.put_encoded(
                                    request["component"], request["key"], request["dtype"], request["shape"],
                                    encoding["scheme"],
                                    {p["role"]: (p["dtype"], p["shape"], d) for p, d in zip(parts, data, strict=True)},
                                    encoding.get("params"),
                                )  # fmt: skip
                            else:
                                writer.put(
                                    request["component"],
                                    request["key"],
                                    request["dtype"],
                                    request["shape"],
                                    data[0],
                                )
                        session.answer(None)
                    elif op == "copy":
                        if (
                            request.get("from_component", request["component"])
                            != request["component"]
                            or request.get("from_key", request["key"]) != request["key"]
                        ):
                            raise Refused("invalid_request", "the peer copies without renaming")
                        if writer is not None:
                            writer.copy(
                                request["component"], request["key"], granted[request["from"]]
                            )
                        session.answer(None)
                    elif op == "file":
                        blob = self._bytes(session, granted, request["length"])
                        written += len(blob)
                        if writer is not None:
                            writer.file(request["path"], blob)
                        session.answer(None)
                    elif op == "read":
                        _read(session, store, granted[request["from"]], request)
                    elif op == "commit":
                        if writer is not None:
                            manifest = writer.commit()
                            done = self.committed[transaction] = (manifest.sha256, manifest.length)
                        assert done is not None
                        session.answer({"manifest": {"sha256": done[0], "length": done[1]}})
                        return
                    else:
                        raise Refused("invalid_request", f"{op!r} is not a derive operation")
                except (Refused, tensorfs.Error) as exc:
                    session.refuse(getattr(exc, "code", "invalid"), str(exc))
                    return
        finally:
            session.channel.close()

    def _bytes(
        self, session: _Session, granted: Mapping[str, tensorfs.Checkpoint], length: int
    ) -> bytes:
        """`length` bytes in data frames; `read` requests between them are answered."""
        out = bytearray()
        while len(out) < length:
            frame = session.channel.receive()
            if isinstance(frame, bytes):
                out += frame
            elif isinstance(frame, dict) and frame.get("op") == "read":
                _read(session, self.store, granted[frame["from"]], frame)
            else:
                raise Refused("invalid_request", "a write's bytes come before the next operation")
        return bytes(out)


def checkpoint(
    store: tensorfs.Store,
    components: Mapping[str, Mapping[str, Any]],
    files: Mapping[str, bytes] = {},
) -> tensorfs.Manifest:
    """`components`' numpy arrays as one plain checkpoint, in insertion order per component."""
    writer = store.writer()
    for component, tensors in components.items():
        for key, value in tensors.items():
            dtype = {"float16": "F16", "float32": "F32", "uint8": "U8", "int64": "I64"}[
                str(value.dtype)
            ]
            writer.put(component, key, dtype, list(value.shape), value)
    for path, data in files.items():
        writer.file(path, data)
    return writer.commit()


def context(peer: Peer) -> Context:
    """An attempt's `Context` whose weights sources and outputs are `peer`'s channels."""
    return Context(
        peer.request_id,
        math.inf,
        _tensorfs_source=lambda weights: peer.client.source(weights.checkpoint_ref),
        _tensorfs_output=peer.client.open_output,
        _tensorfs_adopt=lambda out: peer.client.adopt_model(out.transaction, out.commit()),
    )


def lane(
    root: Path, components: Mapping[str, Mapping[str, Any]], bound: int = 64 << 20
) -> tuple[tensorfs.Store, str, Peer]:
    """A store holding `components` as one checkpoint, and a peer granting it with one
    weights output `model` of `bound` new bytes."""
    store = tensorfs.Store(root / "store")
    source = checkpoint(store, components)
    ref = f"sha256:{source.sha256}"
    return store, ref, Peer(store, root / "spool", {ref: source.length}, {"model": bound})


def output(store: tensorfs.Store, peer: Peer, slot: str = "model") -> tensorfs.Checkpoint:
    """The checkpoint `peer` adopted for `slot`."""
    return store.checkpoint(f"sha256:{peer.adopted[slot][0]}")
