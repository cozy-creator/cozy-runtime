"""cr-009's live job: a STRUCTURAL DERIVATION over a real cozytensors store.

This is a job in the shape jobs.md §3 describes and job-002 will write for real: it takes a
digest-verified materialized input tree, reads canonical headers through the one TensorFS
border, derives a projection nobody stored, checkpoints its progress durably, and returns a
result. It is CPU-class, it writes nothing into the store, and it holds no model residency.

Why THIS derivation: the projection below is `topology_digest`-shaped — component -> key ->
logical shape, in canonical order — which is exactly the index `tfs-009` freezes and every
adapter/endpoint admissibility check runs over. Deriving it from real headers over 6.9 GB of
real objects is the honest smallest version of "a job produces facts about CozyTensors".

Nothing here names an artifact: the store arrives as `payload.store` (a `Tree`, whose PATH
rides the field value), and the snapshots to read arrive as request data. A job that named a
ref in code would be selecting its own input, which is a binding's job and never code's.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import msgspec

from cozy_runtime.author import (
    App,
    Budget,
    Checkpoints,
    Context,
    FileAsset,
    Outputs,
    Scratch,
    Telemetry,
    Tree,
)

app = App()


class CensusInput(msgspec.Struct, forbid_unknown_fields=True):
    store: Tree
    """The MATERIALIZED store tree. Its path rides this value; there is no store accessor."""
    snapshots: dict[str, str]
    """component -> snapshot digest. Request data, resolved by whoever submitted the job."""
    dwell_ms: int = 0
    """Per-component dwell, so a `kill -9` lands reliably INSIDE the loop rather than around
    it. A harness knob, and it is a request field because fault injection belongs to the
    harness and never to a plant switch inside the runtime (decisions #249)."""


class CensusResult(msgspec.Struct):
    census: FileAsset
    components: int
    tensors: int
    logical_bytes: int
    topology_digest: str
    resumed_from: int
    receipts: list[str]


def _component_rows(store_root: str, snapshot: str, component: str) -> list[dict[str, Any]]:
    """One component's canonical table, read through the TensorFS facade (tfs-007).

    The store verifies what it hands back; this job re-derives nothing about CAS layout and
    hashes no object. `parse_header` over the snapshot's exact canonical bytes is the whole
    read surface a structural job needs.
    """
    import tensorfs

    opened = tensorfs.Store.open(store_root)
    header = tensorfs.parse_header(opened.snapshot(snapshot)["header"])
    table = header["components"].get(component)
    if table is None:
        raise ValueError(
            f"snapshot {snapshot[:23]}… carries no component {component!r} "
            f"(it has {', '.join(sorted(header['components'])) or 'none'})"
        )
    rows = []
    for key, entry in table.items():
        logical = entry["logical"]
        shape = [int(n) for n in logical["shape"]]
        dtype = str(logical["dtype"])
        width = int(tensorfs.DTYPES[dtype])
        nbytes = width
        for extent in shape:
            nbytes *= extent
        rows.append({"key": key, "dtype": dtype, "shape": shape, "logical_bytes": nbytes})
    return sorted(rows, key=lambda row: str(row["key"]))


@app.job(publishes=True)
def census(
    ctx: Context,
    payload: CensusInput,
    scratch: Scratch,
    ckpt: Checkpoints,
    budget: Budget,
    out: Outputs,
    tel: Telemetry,
) -> CensusResult:
    """Derive the structural census, one component at a time, resumably.

    The loop is the shape a real conversion job has: check cancellation, check spend, do one
    unit of work, make it durable, repeat. A killed attempt leaves the completed units in the
    RUN's scratch, and the next attempt reads them back instead of re-reading the headers.
    """
    state = scratch.checkpoint_dir(key="components")
    receipts: list[str] = []
    done: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(payload.snapshots):
        cached = state / f"{name}.json"
        if cached.is_file():
            done[name] = json.loads(cached.read_text())
    resumed_from = len(done)

    for index, name in enumerate(sorted(payload.snapshots)):
        ctx.raise_if_cancelled()
        budget.raise_if_exhausted()
        if name in done:
            tel.log(f"component {name} resumed from scratch", level="info")
            tel.progress((index + 1) / len(payload.snapshots), stage="census")
            continue
        with tel.stage(f"read:{name}"):
            rows = _component_rows(str(payload.store.path), payload.snapshots[name], name)
        (state / f"{name}.json").write_text(json.dumps(rows))
        done[name] = rows
        tel.metric(f"{name}.tensors", float(len(rows)), unit="count")
        tel.progress((index + 1) / len(payload.snapshots), stage="census")
        if payload.dwell_ms:
            time.sleep(payload.dwell_ms / 1000.0)

    projection = {name: [[row["key"], row["shape"]] for row in done[name]] for name in sorted(done)}
    body = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    topology = "sha256:" + hashlib.sha256(body).hexdigest()
    receipts.append(ckpt.save("topology", body, operation_key="census/topology").receipt_id)

    document = {
        "format": "cozy.jobs.StructuralCensus/1",
        "store": payload.store.ref,
        "topology_digest": topology,
        "components": {
            name: {
                "snapshot": payload.snapshots[name],
                "tensors": len(rows),
                "logical_bytes": sum(int(r["logical_bytes"]) for r in rows),
                "dtypes": sorted({str(r["dtype"]) for r in rows}),
            }
            for name, rows in sorted(done.items())
        },
    }
    return CensusResult(
        census=out.save_bytes(
            json.dumps(document, indent=2, sort_keys=True).encode(), media_type="application/json"
        ),
        components=len(done),
        tensors=sum(len(rows) for rows in done.values()),
        logical_bytes=sum(int(r["logical_bytes"]) for rows in done.values() for r in rows),
        topology_digest=topology,
        resumed_from=resumed_from,
        receipts=receipts,
    )
