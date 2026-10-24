"""The Runtime's weights in the machine's TensorFS store, read and written with TensorFS itself.

A checkpoint opens by its header ref (`sha256:<hex>`) from the shared store (the executor's
`TENSORFS_HOME`). A derived checkpoint is written with TensorFS's `Writer` straight into the
store, only its new chunks written. Its replay key names the work (who asked, what code and
request, which output, from what): each replay key has one local repository, so the same work
in any process reopens its latest commit and an interrupted one resumes its keyed writer.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import mmap
import os
import threading
from collections.abc import Buffer, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import tensorfs
from tensorfs import Checkpoint as Checkpoint
from tensorfs import TensorMetadata as TensorMetadata
from tensorfs import raw

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._errors import CapabilityError

#: The store's location when the machine names none (TensorFS's default, owner D3).
DEFAULT_HOME = "~/.cache/tensorfs"
#: A fresh output component's class: unknown (an empty library), as TensorFS spells it.
UNKNOWN_CLASS = ("", "", "")

#: Bytes to write: a buffer, or a file read from its position to its end.
type Data = Buffer | IO[bytes]
#: An encoded tensor's part to write: role, TensorFS dtype, shape, bytes.
type PartData = tuple[str, str, Sequence[int], Data]


class Bearers:
    """Each signer's newest run capability for each endpoint, as TensorD gives them (a Load's or
    Invoke's own, or a `capability` frame): it only ever goes to the endpoint it was given for.
    Of two, the one that expires later stands; an unknown expiry (0) always replaces."""

    def __init__(self) -> None:
        self._latest: dict[tuple[str, str], tuple[str, int]] = {}
        self._lock = threading.Lock()

    def note(self, actor: str, endpoint: str, token: str, expires_unix_ms: int) -> None:
        with self._lock:
            held = self._latest.get((actor, endpoint))
            if held is None or expires_unix_ms == 0 or held[1] <= expires_unix_ms:
                self._latest[(actor, endpoint)] = (token, expires_unix_ms)

    def offer(self, actor: str, endpoint: str, token: str, expires_unix_ms: int) -> None:
        """`token` if none came for its key: a call's own bearer never replaces a newer one."""
        with self._lock:
            self._latest.setdefault((actor, endpoint), (token, expires_unix_ms))

    def latest(self, actor: str, endpoint: str) -> str:
        held = self._latest.get((actor, endpoint))
        return "" if held is None else held[0]


@dataclass(frozen=True, slots=True)
class Bearer:
    """One call's TensorFS `token=` provider: its signer's newest run capability for its endpoint,
    presented as a bearer token and asked for every request's header, so a call that outlives one
    keeps working once TensorD sends a newer one. A 401 for one replaced while it was in flight is
    asked once more."""

    bearers: Bearers
    actor: str
    endpoint: str
    own: str = field(repr=False)

    def _current(self) -> str:
        return f"Bearer {self.bearers.latest(self.actor, self.endpoint) or self.own}"

    def headers(
        self, method: str, url: str, headers: Mapping[str, str], payload: str | None
    ) -> dict[str, str]:
        return {"authorization": self._current()}

    def retry(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: str | None,
        status: int,
        response_headers: Mapping[str, str],
    ) -> bool:
        sent = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
        return status == 401 and sent != self._current()


@dataclass(frozen=True, slots=True)
class Connection:
    """The request's TensorFS endpoint and its run capability, when it was supplied one."""

    endpoint: str | None = None
    token: str | Bearer | None = field(default=None, repr=False)


def open_checkpoint(store: Path, ref: str, connection: Connection | None = None) -> Checkpoint:
    """Open a checkpoint through the public TensorFS API in the shared cache."""
    options = connection or Connection()
    return tensorfs.open(ref, cache_dir=store, endpoint=options.endpoint, token=options.token)


def read_into(
    ck: Checkpoint, component: str, key: str, role: str, offset: int, into: Buffer
) -> None:
    """Stored bytes of one part (`""`: a plain tensor's), read by TensorFS."""
    try:
        tensors = ck.components[component].tensors
        tensors[key]  # Validate the logical tensor before requesting raw carrier metadata.
        part = raw.parts(ck, component, key)[role]
    except KeyError:
        raise CapabilityError("model tensor part is absent", code="weights_reader_tensor") from None
    try:
        view = memoryview(into)
    except TypeError:
        raise CapabilityError(
            "model read needs a writable buffer", code="weights_reader_buffer"
        ) from None
    if view.readonly or not view.c_contiguous or not 0 < view.nbytes <= 64 << 20:
        raise CapabilityError(
            "model read buffer is readonly, noncontiguous, empty or oversized",
            code="weights_reader_buffer",
        )
    if type(offset) is not int or offset < 0 or offset + view.nbytes > part.length:
        raise CapabilityError(
            f"{component}/{key}#{role}: bytes {offset}+{view.nbytes} of {part.length}",
            code="weights_reader_bounds",
        )
    tensors.read_into(key, view, role=role, offset=offset)


@dataclass(frozen=True, slots=True)
class Derivation:
    """What one output is written from: its base, whose components and tensors not written
    again carry over; the other checkpoints it copies or reads; its tensors' order
    (construction order: the base's header order for what it keeps, new tensors where they
    go)."""

    base: Checkpoint | None
    sources: Sequence[Checkpoint] = ()
    order: Sequence[tuple[str, str]] = ()
    #: The header's `provenance`: the recipe that writes it (`derive.recipes`).
    provenance: Mapping[str, object] | None = None

    def work(self, max_new_bytes: int) -> dict[str, object]:
        """The derivation as its replay key reads it."""
        work: dict[str, object] = {
            "base": None if self.base is None else self.base.id,
            "sources": [source.id for source in self.sources],
            "order": [[component, key] for component, key in self.order],
            "max_new_bytes": max_new_bytes,
        }
        if self.provenance is not None:
            work["provenance"] = dict(self.provenance)
        return work


def replay_key(actor: str, code: str, slot: str, work: Mapping[str, object]) -> str:
    """An output's transaction: who asked, the code and request that derive it, the output,
    and what it derives from. The same work replays its finished output; anything else
    derives apart."""
    body = hashlib.sha256(canonical_json.encode(dict(work))).hexdigest()
    return "sha256:" + hashlib.sha256(f"{actor}\0{code}\0{slot}\0{body}".encode()).hexdigest()


def repository(transaction: str) -> str:
    """One immutable derivation's local commit history, keyed by its existing replay identity."""
    return "local/derived-" + transaction.removeprefix("sha256:")


def finished(store: Path, transaction: str, slot: str) -> Checkpoint | None:
    """The finished commit for this replay key and output lane, if the store has one."""
    try:
        return tensorfs.open(
            repository(transaction),
            lane=slot,
            cache_dir=store,
            local_files_only=True,
        )
    except tensorfs.NotFoundError:
        return None


class Output:
    """One derived checkpoint, output `slot` and the lane of that name (a quantizer's `int8`
    output is the `int8` lane), written into `store` under its replay key `transaction`. A
    finished one replays (`done`): every tensor is there and `commit` answers it. An interrupted
    one resumes: what an earlier attempt wrote `has`. At most `max_new_bytes` are written."""

    def __init__(
        self,
        store: Path,
        slot: str,
        transaction: str,
        derivation: Derivation,
        max_new_bytes: int,
    ) -> None:
        self.store, self.slot, self.transaction = store, slot, transaction
        self.max_new_bytes, self.written = max_new_bytes, 0
        self.order = tuple(derivation.order)
        self._fresh = derivation.base is None and not derivation.sources
        #: each component a write may name, declared as its base (else first source) declares it
        froms = (*reversed(derivation.sources), *([derivation.base] if derivation.base else []))
        self.declared = {name: c for ck in froms for name, c in ck.components.items()}
        self.done = finished(store, transaction, slot)
        self._writer = (
            None
            if self.done is not None
            else tensorfs.Writer(
                base=derivation.base,
                repo_id=repository(transaction),
                lane=slot,
                resume=transaction.removeprefix("sha256:"),
                cache_dir=store,
            )
        )
        if self._writer is not None and derivation.provenance is not None:
            self._writer.metadata["provenance"] = dict(derivation.provenance)
        self._components: dict[str, tensorfs.ComponentWriter] = {}

    def close(self) -> None:
        """Let the writer go; what it wrote stays for an attempt that resumes it."""
        self._writer = None
        self._components.clear()

    def __enter__(self) -> Output:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def component(self, name: str) -> tensorfs.ComponentWriter:
        """Component `name`, with inherited class/config or empty metadata for a fresh output."""
        writer = self._components.get(name)
        return writer if writer is not None else self._open(name, None)

    def config(self, component: str, config: Mapping[str, object]) -> None:
        """`component`'s own config in place of the one it inherits, or a new component's:
        set before anything is written to it."""
        if self.done is not None:
            return
        if component in self._components:
            raise CapabilityError(
                f"component {component!r} is already open; set its config first",
                code="weights_output_config",
            )
        self._open(component, config)

    def _open(self, name: str, config: Mapping[str, object] | None) -> tensorfs.ComponentWriter:
        if self._writer is None:
            raise CapabilityError("the output is committed or closed", code="weights_writer_closed")
        declared = self.declared.get(name)
        if declared is None and not self._fresh and config is None:
            raise CapabilityError(
                f"no checkpoint this output derives from has a component {name!r}",
                code="weights_output_component",
            )
        writer = self._writer.component(
            name,
            cls=UNKNOWN_CLASS if declared is None else declared.cls,
            config=config if config is not None else ({} if declared is None else declared.config),
        )
        self._components[name] = writer
        return writer

    def inherit(self, source: Checkpoint) -> None:
        """`source`'s pipeline and each component's class, config and assets, no tensor: what a
        derivation without a base keeps of its source."""
        if self.done is not None:
            return
        assert self._writer is not None
        if source.pipeline is not None:
            self._writer.pipeline(source.pipeline.cls, source.pipeline.config)
        for name, component in source.components.items():
            writer = self.component(name)
            for asset in component.assets:
                writer.asset(asset, component.asset(asset))

    def has(self, component: str, key: str) -> bool:
        """Written already: by an interrupted attempt of this output, or a finished one."""
        if self.done is not None:
            held = self.done.components.get(component)
            return held is not None and key in held.tensors
        if self._writer is None:
            raise CapabilityError("the output is closed", code="weights_writer_closed")
        return self._writer.has(component, key)

    def put(self, component: str, key: str, dtype: str, shape: Sequence[int], data: Data) -> None:
        """A plain tensor: exactly its bytes."""
        if self.done is not None:
            return
        with _view(data) as view:
            self._spend(view.nbytes)
            self.component(component).put(key, view, dtype=dtype, shape=list(shape))

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
        """An encoded tensor: its logical dtype and shape, scheme and params, and each part."""
        if self.done is not None:
            return
        with contextlib.ExitStack() as stack:
            views = {
                role: (part_dtype, list(part_shape), stack.enter_context(_view(data)))
                for role, part_dtype, part_shape, data in parts
            }
            self._spend(sum(view.nbytes for _, _, view in views.values()))
            writer = self.component(component)
            writer.put_encoded(
                key, dtype=dtype, shape=list(shape), scheme=scheme, parts=views, params=dict(params)
            )

    def copy(
        self,
        component: str,
        key: str,
        source: Checkpoint,
        from_component: str | None = None,
        from_key: str | None = None,
    ) -> None:
        """A tensor of `source`, renamed when asked: its stored parts are referenced, never
        written again."""
        if self.done is not None:
            return
        self.component(component).copy(
            from_key or key, source.components[from_component or component], as_key=key
        )

    def commit(self) -> Checkpoint:
        """The finished checkpoint, kept in its replay key's local repository: once committed,
        every later commit of the same work answers this one."""
        if self.done is None:
            if self._writer is None:
                raise CapabilityError("the output is closed", code="weights_writer_closed")
            keys: dict[str, list[str]] = {}
            for component, key in self.order:
                keys.setdefault(component, []).append(key)
            for component, ordered in keys.items():
                self.component(component).order(ordered)
            self.done = self._writer.commit()
            self.close()
        return self.done

    def _spend(self, nbytes: int) -> None:
        self.written += nbytes
        if self.written > self.max_new_bytes:
            raise CapabilityError(
                f"the output writes over its {self.max_new_bytes} granted new bytes",
                code="weights_output_bound",
            )


class Outputs:
    """One attempt's weights outputs in `store`. Each opens under its replay key; adopting one
    commits it and shows it to the run (`show(slot, repo_id, commit)`). TensorD owns uploads."""

    def __init__(
        self,
        store: Path,
        *,
        request_id: str,
        actor: str,
        code: str,
        bounds: Mapping[str, int],
        show: Callable[[str, str, str], None],
    ) -> None:
        self.store, self.request_id, self.actor, self.code = store, request_id, actor, code
        self.bounds, self.show = dict(bounds), show

    def open(self, slot: str, derivation: Derivation) -> Output:
        bound = self.bounds.get(slot)
        if bound is None:
            raise CapabilityError("output is not declared", code="weights_output_ungranted")
        key = replay_key(self.actor, self.code, slot, derivation.work(bound))
        return Output(self.store, slot, key, derivation, bound)

    def adopt(self, output: Output) -> ModelArtifact:
        ck = output.commit()
        assert ck.tree is not None
        self.show(output.slot, repository(output.transaction), ck.tree)
        return ModelArtifact(self.request_id, output.slot, ck.id, output.transaction)


@contextlib.contextmanager
def _view(data: Data) -> Iterator[memoryview]:
    """`data`'s bytes: a buffer's, or a file's from its position to its end (mapped when it is
    one on disk)."""
    if isinstance(data, Buffer):
        yield memoryview(data).cast("B")
        return
    try:
        data.fileno()
    except (OSError, io.UnsupportedOperation):
        yield memoryview(data.read())
        return
    start = data.tell()
    end = data.seek(0, os.SEEK_END)
    data.seek(start)
    if end == start:
        yield memoryview(b"")
        return
    with mmap.mmap(data.fileno(), end, access=mmap.ACCESS_READ) as mapped:
        view = memoryview(mapped)[start:]
        try:
            yield view
        finally:
            view.release()
