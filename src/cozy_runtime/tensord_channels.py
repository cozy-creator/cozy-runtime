"""The executor's TensorD channels: fetch, source (one granted checkpoint) and derive (one new
checkpoint), each over one connected local stream TensorD hands the attempt.

Frames are TensorD's (`crates/tensord/src/frame.rs`): a kind byte (0: a JSON object, 1: at most
1 MiB of bytes) and a big-endian u32 length. The source and derive protocol is TensorD's
`derive/mod.rs`: a request is a JSON object with an `op`, an answer `{"result": …}` or
`{"error": code, "detail": text}`; checkpoints are `sha256:<hex>`, dtypes TensorFS's names, and a
plain tensor's part role is `""`. Headers are read only through the wheel
(`tensorfs.Checkpoint.parse`).
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import threading
from collections.abc import Buffer, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any

import tensorfs

if TYPE_CHECKING:
    from tensorfs import Info as Info

_HEAD = struct.Struct("!BI")
_JSON, _BYTES = 0, 1
CHUNK = 1 << 20
_JSON_MAX = 64 << 20
#: The header's path in a manifest.
HEADER = "model.cozytensors"


#: Bytes to write: a buffer, or a file read from its position to its end.
type Data = Buffer | IO[bytes]


def length(data: Data) -> int:
    """`data`'s bytes to write: a buffer's, or a file's (or `BytesIO`'s) from here to its end."""
    if isinstance(data, Buffer):
        return memoryview(data).nbytes
    here = data.tell()
    end = data.seek(0, os.SEEK_END)
    data.seek(here)
    return end - here


class Refused(Exception):
    """TensorD refused a request: its code and why."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class Channel:
    """One framed local stream, from a connected descriptor this process now owns."""

    def __init__(self, fd: int) -> None:
        sock = socket.socket(fileno=fd)
        if sock.family != socket.AF_UNIX or sock.type != socket.SOCK_STREAM:
            sock.close()
            raise Refused("invalid_channel", "a TensorD channel is a connected local stream")
        sock.getpeername()  # connected already: never dial
        sock.set_inheritable(False)
        self.sock = sock

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

    def send(self, value: Mapping[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self.sock.sendall(_HEAD.pack(_JSON, len(body)) + body)

    def data(self, value: Data) -> None:
        """`value`'s bytes as data frames: a buffer, or a file from its position to its end."""
        if isinstance(value, Buffer):
            view = memoryview(value).cast("B")
            for start in range(0, len(view), CHUNK):
                self._frame(view[start : start + CHUNK])
            return
        while piece := value.read(CHUNK):
            self._frame(piece)

    def _frame(self, piece: Buffer) -> None:
        self.sock.sendall(_HEAD.pack(_BYTES, len(memoryview(piece))))
        self.sock.sendall(piece)

    def _exact(self, n: int) -> bytes | None:
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            read = self.sock.recv_into(view[got:])
            if not read:
                if got == 0:
                    return None
                raise Refused("channel_closed", "TensorD closed the channel mid-frame")
            got += read
        return bytes(buf)

    def receive(self) -> dict[str, Any] | bytes | None:
        """The next frame; None when the peer closed between frames."""
        head = self._exact(_HEAD.size)
        if head is None:
            return None
        kind, length = _HEAD.unpack(head)
        if length > (_JSON_MAX if kind == _JSON else CHUNK) or kind not in (_JSON, _BYTES):
            raise Refused("invalid_frame", f"frame kind {kind}, {length} bytes")
        body = self._exact(length) if length else b""
        if body is None:
            raise Refused("channel_closed", "TensorD closed the channel mid-frame")
        if kind == _BYTES:
            return body
        value = json.loads(body)
        if not isinstance(value, dict):
            raise Refused("invalid_frame", "a JSON frame is an object")
        return value

    def answer(self) -> Any:
        """The next answer's result; a refusal raises."""
        value = self.receive()
        if not isinstance(value, dict):
            raise Refused("channel_closed", "TensorD ended the channel without an answer")
        if "error" in value:
            raise Refused(str(value["error"]), str(value.get("detail", "")))
        return value.get("result")

    def receive_into(self, into: memoryview) -> None:
        """Exactly `len(into)` bytes as data frames."""
        while into:
            block = self.receive()
            if not isinstance(block, bytes) or not block or len(block) > len(into):
                raise Refused("length_mismatch", "TensorD sent another byte count than declared")
            into[: len(block)] = block
            into = into[len(block) :]

    def call(self, request: Mapping[str, Any]) -> Any:
        self.send(request)
        return self.answer()


def _role(role: str) -> str:
    """The Runtime names a plain tensor's part `value`; the wire `""`."""
    return "" if role == "value" else role


class _Reads:
    """`read` requests answered `{"length"}` and then the bytes."""

    channel: Channel
    lock: threading.Lock

    def _read_into(self, request: dict[str, Any], into: Buffer) -> None:
        view = memoryview(into).cast("B")
        if view.readonly:
            raise Refused("invalid_buffer", "a read needs a writable buffer")
        with self.lock:
            result = self.channel.call({**request, "length": len(view)})
            if not isinstance(result, dict) or result.get("length") != len(view):
                raise Refused("length_mismatch", f"a read answered {result!r}")
            self.channel.receive_into(view)


class Source(_Reads):
    """One granted checkpoint: its header (parsed by the wheel), files and stored bytes."""

    def __init__(self, manifest: str, length: int, fd: int) -> None:
        self.manifest, self.length = manifest, length
        self.channel = Channel(fd)
        self.lock = threading.Lock()
        self._files: dict[str, tuple[str, int]] | None = None
        self._checkpoint: tensorfs.Checkpoint | None = None
        #: called before a read, `(component, key)`: the fetch look-ahead (`fetch_plan.source`)
        self.ahead: Callable[[str, str], None] | None = None

    def close(self) -> None:
        self.channel.close()

    def __enter__(self) -> Source:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def files(self) -> dict[str, tuple[str, int]]:
        """The manifest's files: path -> (sha256, length)."""
        if self._files is None:
            with self.lock:
                result = self.channel.call({"op": "inspect"})
            self._files = {
                path: (str(ref["sha256"]), int(ref["length"]))
                for path, ref in dict(result["files"]).items()
            }
        return self._files

    @property
    def checkpoint(self) -> tensorfs.Checkpoint:
        """The header's tensors, in construction order."""
        if self._checkpoint is None:
            self._checkpoint = tensorfs.Checkpoint.parse(self.file(HEADER))
        return self._checkpoint

    def file(self, path: str) -> bytes:
        with self.lock:
            result = self.channel.call({"op": "file", "path": path})
            buf = bytearray(int(result["length"]))
            self.channel.receive_into(memoryview(buf))
        return bytes(buf)

    def read_into(self, component: str, key: str, role: str, offset: int, into: Buffer) -> None:
        """Stored bytes of one part (`value`: a plain tensor's), from `offset`."""
        if self.ahead is not None:
            self.ahead(component, key)
        request = {"op": "read", "component": component, "key": key, "role": _role(role)}
        self._read_into({**request, "offset": offset}, into)


#: An encoded tensor's part to write: role, TensorFS dtype, shape, bytes.
type PartData = tuple[str, str, Sequence[int], Data]


class Output(_Reads):
    """One derive channel: writes one new checkpoint, resumably, then commits it."""

    def __init__(self, fd: int, transaction: str) -> None:
        self.channel = Channel(fd)
        self.transaction = transaction
        self.lock = threading.Lock()

    def close(self) -> None:
        self.channel.close()

    def __enter__(self) -> Output:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def has(self, component: str, key: str) -> bool:
        """Written already: by an interrupted attempt of this output, or a committed one."""
        with self.lock:
            return bool(self.channel.call({"op": "has", "component": component, "key": key}))

    def put(self, component: str, key: str, dtype: str, shape: Sequence[int], data: Data) -> None:
        """A plain tensor: exactly its bytes."""
        request = {"op": "put", "component": component, "key": key, "dtype": dtype,
                   "shape": list(shape), "length": length(data)}  # fmt: skip
        with self.lock:
            self.channel.send(request)
            self.channel.data(data)
            self.channel.answer()

    def put_encoded(
        self,
        component: str,
        key: str,
        dtype: str,
        shape: Sequence[int],
        scheme: str,
        params: Mapping[str, object],
        parts: Sequence[PartData],
    ) -> None:
        """An encoded tensor: its logical dtype and shape, scheme and params, then each part."""
        declared = [
            {"role": role, "dtype": part_dtype, "shape": list(part_shape), "length": length(data)}
            for role, part_dtype, part_shape, data in parts
        ]
        encoding: dict[str, object] = {"scheme": scheme}
        if params:
            encoding["params"] = dict(params)
        request = {"op": "put", "component": component, "key": key, "dtype": dtype,
                   "shape": list(shape), "encoding": encoding, "parts": declared}  # fmt: skip
        with self.lock:
            self.channel.send(request)
            for _, _, _, data in parts:
                self.channel.data(data)
            self.channel.answer()

    def copy(
        self,
        component: str,
        key: str,
        source: str,
        from_component: str | None = None,
        from_key: str | None = None,
    ) -> None:
        """A tensor of a granted checkpoint, renamed when asked; large parts keep their objects."""
        request: dict[str, object] = {"op": "copy", "component": component, "key": key,
                                      "from": source}  # fmt: skip
        if from_component is not None:
            request["from_component"] = from_component
        if from_key is not None:
            request["from_key"] = from_key
        with self.lock:
            self.channel.call(request)

    def file(self, path: str, data: Buffer) -> None:
        view = memoryview(data).cast("B")
        with self.lock:
            self.channel.send({"op": "file", "path": path, "length": len(view)})
            self.channel.data(view)
            self.channel.answer()

    def read_into(
        self, source: str, component: str, key: str, role: str, offset: int, into: Buffer
    ) -> None:
        """Stored bytes of a granted checkpoint's part, from `offset`."""
        request = {"op": "read", "from": source, "component": component, "key": key,
                   "role": _role(role), "offset": offset}  # fmt: skip
        self._read_into(request, into)

    def commit(self) -> tuple[str, int]:
        """The new checkpoint's manifest `(sha256, length)`; the channel ends."""
        with self.lock:
            result = self.channel.call({"op": "commit"})
        manifest = result["manifest"]
        return str(manifest["sha256"]), int(manifest["length"])


class _Call:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.answer: dict[str, Any] = {}


class Fetch:
    """One attempt's fetch channel: Runtime directs what TensorD fetches of its selected
    checkpoints (`origin/serve.rs`). Calls may come from any thread; a `wait` blocks only its
    caller, and answers arrive as their objects land."""

    def __init__(self, fd: int) -> None:
        self._channel = Channel(fd)
        self._sending = threading.Lock()
        self._lock = threading.Lock()
        self._next = 0
        self._calls: dict[int, _Call] = {}
        self._closed: BaseException | None = None
        threading.Thread(target=self._read, name="tensord-fetch", daemon=True).start()

    def schedule(
        self, manifest: str, objects: Sequence[tuple[str, int]], rank: int, behind: bool = False
    ) -> None:
        """Want `objects` of `manifest` before this run's later ranks, or behind everything."""
        self._call("schedule", manifest, objects, rank=rank, behind=behind)

    def wait(self, manifest: str, objects: Sequence[tuple[str, int]]) -> None:
        """Raise `objects` to now and return once they are present."""
        self._call("wait", manifest, objects)

    def present(self, manifest: str, objects: Sequence[tuple[str, int]]) -> list[bool]:
        answer = self._call("present", manifest, objects)
        if not isinstance(answer, list) or len(answer) != len(objects):
            raise Refused("invalid_answer", "invalid present answer")
        return [bool(value) for value in answer]

    def close(self) -> None:
        """End the channel: TensorD releases everything it wanted."""
        self._channel.close()

    def __enter__(self) -> Fetch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _call(
        self, op: str, manifest: str, objects: Sequence[tuple[str, int]], **fields: Any
    ) -> Any:
        call = _Call()
        with self._lock:
            if self._closed is not None:
                raise Refused("transfer_failed", f"fetch channel closed: {self._closed}")
            self._next += 1
            ident = self._next
            self._calls[ident] = call
        rows = [
            {"sha256": sha.removeprefix("sha256:"), "length": length} for sha, length in objects
        ]
        try:
            with self._sending:
                self._channel.send(
                    {"id": ident, "op": op, "manifest": manifest, "objects": rows, **fields}
                )
        except BaseException:
            with self._lock:
                self._calls.pop(ident, None)
            raise
        call.done.wait()
        if "error" in call.answer:
            raise Refused(str(call.answer["error"]), str(call.answer.get("detail", "")))
        return call.answer.get("result")

    def _read(self) -> None:
        try:
            while True:
                value = self._channel.receive()
                if value is None:
                    raise Refused("channel_closed", "TensorD closed the fetch channel")
                if not isinstance(value, dict) or type(value.get("id")) is not int:
                    raise Refused("invalid_answer", "invalid fetch answer")
                with self._lock:
                    call = self._calls.pop(value["id"], None)
                if call is not None:
                    call.answer = value
                    call.done.set()
        except BaseException as exc:  # the channel ended: every pending call fails with it
            with self._lock:
                self._closed = exc
                calls, self._calls = self._calls, {}
            for call in calls.values():
                call.answer = {"error": "transfer_failed", "detail": f"fetch channel closed: {exc}"}
                call.done.set()


def ref(manifest: str, length: int) -> dict[str, object]:
    """A checkpoint as the derivation file names it."""
    return {"sha256": manifest.removeprefix("sha256:"), "length": length}


@dataclass(frozen=True, slots=True)
class Derivation:
    """One output's derivation file: the checkpoint it starts from (every tensor and file not
    written again carries over), the checkpoints it may copy and read, and its tensors' order
    (construction order: header order of what it keeps, new tensors where they go)."""

    base: Source | None
    sources: Sequence[Source] = ()
    order: Sequence[tuple[str, str]] = field(default=())

    def document(self, max_new_bytes: int) -> dict[str, object]:
        return {
            "base": None if self.base is None else ref(self.base.manifest, self.base.length),
            "sources": [ref(s.manifest, s.length) for s in self.sources],
            "order": [[component, key] for component, key in self.order],
            "max_new_bytes": max_new_bytes,
        }
