"""Native CPU process that writes one immutable adapter view from already-owned sources."""

from __future__ import annotations

import io
import time
from collections.abc import Buffer, Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager

import msgspec
from tensorfs.derived import SourceCapability, SourceInspection

from cozy_runtime.author import ObjectRef
from cozy_runtime.internal import fill, lora_composition
from cozy_runtime.internal.worker.source_steps import (
    AdapterViewReady,
    Answer,
    Launch,
    PrepareAdapterView,
    report,
    serve,
)
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal


class _File(msgspec.Struct, frozen=True):
    sha256: str
    length: int


class _Receipt(msgspec.Struct, frozen=True):
    manifest: _File


class _Existing(msgspec.Struct, frozen=True):
    state: str
    receipt: _Receipt | None = None
    writer_session_id: int | None = None


@contextmanager
def _source(
    store: fill.Store,
    manifest: ObjectRef,
    check: Callable[[], None],
    progress: Callable[[int], None],
) -> Iterator[SourceCapability]:
    check()
    lease = store.acquire_cozytensors(manifest.digest)
    try:
        header = store.manifest(manifest.digest)["header"]
        if header is None:
            raise WorkspaceRefusal("adapter source has no CozyTensors header")

        structure = fill.tensorfs_module().parse_header(header)

        def inspect(components: Sequence[str], configs: Sequence[str]) -> SourceInspection:
            check()
            return SourceInspection.from_native(
                store.inspect_derived_source(
                    manifest.digest,
                    manifest.length,
                    components or tuple(structure["components"]),
                    configs or tuple(structure["configs"]),
                )
            )

        def read(component: str, key: str, role: str, offset: int, into: object) -> None:
            check()
            lease.read_part_into(header, component, key, role, offset, into)
            if isinstance(into, Buffer):
                progress(memoryview(into).nbytes)

        yield SourceCapability(
            manifest.digest, manifest.length, inspect, _read_part=read, _check=check
        )
    finally:
        lease.release()


def execute(launch: Launch[PrepareAdapterView]) -> Answer:
    store, step = fill.store(launch.store), launch.step
    moved = 0

    def progress(count: int) -> None:
        nonlocal moved
        moved += count
        report("convert", moved, 0)

    previous = msgspec.convert(store.derived_lookup(step.identity), _Existing)
    if previous.state == "committed" and previous.receipt is not None:
        receipt = previous.receipt
    else:
        with ExitStack() as leases:
            base = leases.enter_context(_source(store, step.base, lambda: None, progress))
            inputs = [
                lora_composition.Selection(
                    leases.enter_context(_source(store, row.manifest, lambda: None, progress)),
                    row.component,
                    row.source_component,
                    float(row.scale),
                )
                for row in step.adapters
            ]
            prepared = lora_composition.prepare(base, inputs)
            data = prepared.config
            if previous.state == "open":
                if previous.writer_session_id is not None:
                    store.derived_fence(step.identity, previous.writer_session_id)
                store.derived_abandon(step.identity)
            writer = store.begin_derived(
                step.identity,
                time.time_ns() // 1_000_000,
                *prepared.declaration.native_arguments(len(data)),
                work_fingerprint=step.identity,
            )
            try:
                writer.add_config(lora_composition.GRAPH_CONFIG, io.BytesIO(data))
                receipt = msgspec.convert(writer.commit(), _Receipt)
            except BaseException:
                writer.fence()
                store.derived_abandon(step.identity)
                raise
    return AdapterViewReady(
        manifest=ObjectRef("sha256:" + receipt.manifest.sha256, receipt.manifest.length)
    )


if __name__ == "__main__":
    serve(Launch[PrepareAdapterView], execute)
