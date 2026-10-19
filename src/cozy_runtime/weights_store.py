"""The Runtime's weights in the machine's TensorFS store, read and written with TensorFS itself.

A checkpoint opens by its header ref (`sha256:<hex>`) from the shared store (the executor's
`TENSORFS_HOME`). A derived checkpoint is written with TensorFS's `Writer` straight into the
store, only its new chunks written. Its replay key names the work (who asked, what code and
request, which output, from what): a finished derivation is the `local/derived` release of
that key, so the same work in any process replays it, and an interrupted one resumes its
keyed writer.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import mmap
import os
from collections.abc import Buffer, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import msgspec
import tensorfs
from tensorfs import Checkpoint as Checkpoint
from tensorfs import Info as Info

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._errors import CapabilityError, RuntimeFailure

#: The store's location when the machine names none (TensorFS's default, owner D3).
DEFAULT_HOME = "~/.cache/tensorfs"
#: The local repo of finished derivations, one release per replay key.
DERIVED = "local/derived"

#: Bytes to write: a buffer, or a file read from its position to its end.
type Data = Buffer | IO[bytes]
#: An encoded tensor's part to write: role, TensorFS dtype, shape, bytes.
type PartData = tuple[str, str, Sequence[int], Data]


class Hub(msgspec.Struct, frozen=True, kw_only=True):
    """Only this request's Hub capability and private P-256 key, sent by TensorD."""

    endpoint: str
    capability: str
    key: str
    token_endpoint: str
    #: An optional base64 DER root, for the same Hub's private TLS authority.
    ca_der: str | None = None
    #: A token already traded by this run's metadata helper, bound to the same key.
    access_token: str | None = None

    def __repr__(self) -> str:
        return (
            f"Hub(endpoint={self.endpoint!r}, capability=<redacted>, "
            "key=<redacted>, access_token=<redacted>)"
        )


class Access:
    """One request's lazy TensorFS provider, shared by its metadata, loads and uploads."""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.ca_der = None if hub.ca_der is None else base64.b64decode(hub.ca_der, validate=True)
        try:
            payload = hub.capability.split(".")[1]
            audience = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))[
                "aud"
            ]
            if not isinstance(audience, str) or not audience:
                raise ValueError("capability audience is absent")
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            raise CapabilityError(
                "Hub capability audience is invalid", code="capability_refused"
            ) from exc
        self.token = (
            tensorfs.DPoP.token(hub.key, hub.access_token, resource=audience, endpoint=hub.endpoint)
            if hub.access_token is not None
            else tensorfs.DPoP.grant(
                hub.key,
                token_endpoint=hub.token_endpoint,
                client_id="tensord",
                resource=audience,
                claims={"capability": hub.capability},
                endpoint=hub.endpoint,
                ca_der=self.ca_der,
            )
        )


def open_checkpoint(store: Path, ref: str, access: Access | None = None) -> Checkpoint:
    """A header in the shared store, fetching missing bytes only under this request's provider."""
    if access is None:
        return tensorfs.open(ref, cache_dir=store)
    return tensorfs.open(
        ref, cache_dir=store, endpoint=access.hub.endpoint,
        token=access.token, ca_der=access.ca_der,
    )  # fmt: skip


def upload(store: Path, ck: Checkpoint, repo_id: str, access: Access | None) -> str:
    """Append this request's checkpoint; local bytes stay on an upload failure."""
    if access is None:
        raise RuntimeFailure(f"no Hub capability to upload to {repo_id}", code="upload_failed")
    try:
        return tensorfs.upload_folder(
            repo_id=repo_id, folder_path=ck.path, cache_dir=store,
            endpoint=access.hub.endpoint, token=access.token, ca_der=access.ca_der,
        )  # fmt: skip
    except tensorfs.Error as exc:
        raise RuntimeFailure(f"uploading to {repo_id}: {exc}"[:900], code="upload_failed") from exc


def read_into(
    ck: Checkpoint, component: str, key: str, role: str, offset: int, into: Buffer
) -> None:
    """Stored bytes of one part (`""`: a plain tensor's), read by TensorFS."""
    try:
        tensors = ck.components[component].tensors
        part = tensors[key].parts[role]
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

    def work(self, max_new_bytes: int) -> dict[str, object]:
        """The derivation as its replay key reads it."""
        return {
            "base": None if self.base is None else self.base.id,
            "sources": [source.id for source in self.sources],
            "order": [[component, key] for component, key in self.order],
            "max_new_bytes": max_new_bytes,
        }


def replay_key(actor: str, code: str, slot: str, work: Mapping[str, object]) -> str:
    """An output's transaction: who asked, the code and request that derive it, the output,
    and what it derives from. The same work replays its finished output; anything else
    derives apart."""
    body = hashlib.sha256(canonical_json.encode(dict(work))).hexdigest()
    return "sha256:" + hashlib.sha256(f"{actor}\0{code}\0{slot}\0{body}".encode()).hexdigest()


def finished(store: Path, transaction: str) -> Checkpoint | None:
    """The output a finished derivation of `transaction` committed, if `store` has one."""
    try:
        return tensorfs.open(
            DERIVED, revision=_release(transaction), cache_dir=store, local_files_only=True
        )
    except tensorfs.NotFoundError:
        return None


def _release(transaction: str) -> str:
    return transaction.removeprefix("sha256:")


class Output:
    """One derived checkpoint, output `slot`, written into `store` under its replay key
    `transaction`. A finished one replays (`done`): every tensor is there and `commit`
    answers it. An interrupted one resumes: what an earlier attempt wrote `has`. At most
    `max_new_bytes` are written."""

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
        #: each component a write may name, declared as its base (else first source) declares it
        froms = (*reversed(derivation.sources), *([derivation.base] if derivation.base else []))
        self.declared = {name: c for ck in froms for name, c in ck.components.items()}
        self.done = finished(store, transaction)
        self._writer = (
            None
            if self.done is not None
            else tensorfs.Writer(
                base=derivation.base, resume=_release(transaction), cache_dir=store
            )
        )
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
        """The writer of component `name`, declared (class, config) as the checkpoint it comes
        from declares it."""
        writer = self._components.get(name)
        if writer is not None:
            return writer
        if self._writer is None:
            raise CapabilityError("the output is committed or closed", code="weights_writer_closed")
        declared = self.declared.get(name)
        if declared is None:
            raise CapabilityError(
                f"no checkpoint this output derives from has a component {name!r}",
                code="weights_output_component",
            )
        writer = self._writer.component(
            name, cls=declared.cls, config=declared.config, config_file=declared.config_file
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
        """The finished checkpoint, kept as `local/derived` at its replay key: once committed,
        every later commit of the same work answers this one."""
        if self.done is None:
            if self._writer is None:
                raise CapabilityError("the output is closed", code="weights_writer_closed")
            keys: dict[str, list[str]] = {}
            for component, key in self.order:
                keys.setdefault(component, []).append(key)
            for component, ordered in keys.items():
                self.component(component).order(ordered)
            committed = self._writer.commit()
            tensorfs.upload_folder(
                repo_id=DERIVED, revision=_release(self.transaction), folder_path=committed.path,
                cache_dir=self.store, endpoint="local",
            )  # fmt: skip
            self.done = tensorfs.open(
                DERIVED,
                revision=_release(self.transaction),
                cache_dir=self.store,
                local_files_only=True,
            )
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
    commits it and shows it to the run (`show(slot, repo_id, revision, variant)`)."""

    def __init__(
        self,
        store: Path,
        *,
        request_id: str,
        actor: str,
        code: str,
        bounds: Mapping[str, int],
        show: Callable[[str, str, str, str], None],
        destination: str = "",
        access: Access | None = None,
    ) -> None:
        self.store, self.request_id, self.actor, self.code = store, request_id, actor, code
        self.bounds, self.show = dict(bounds), show
        self.destination, self.access = destination, access

    def open(self, slot: str, derivation: Derivation) -> Output:
        bound = self.bounds.get(slot)
        if bound is None:
            raise CapabilityError("output is not declared", code="weights_output_ungranted")
        key = replay_key(self.actor, self.code, slot, derivation.work(bound))
        return Output(self.store, slot, key, derivation, bound)

    def adopt(self, output: Output) -> ModelArtifact:
        ck = output.commit()
        assert ck.id is not None and ck.tree is not None
        self.show(output.slot, DERIVED, ck.tree, ck.variant or "")
        if self.destination:
            revision = upload(self.store, ck, self.destination, self.access)
            self.show(output.slot, self.destination, revision, ck.variant or "")
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
