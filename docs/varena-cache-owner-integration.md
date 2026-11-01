# Coherent Cache owner integration

This development-only branch combines the tested native retention/Plan/consumer baseline
with the weak native Cache owner bridge. It starts from Runtime0.36.23 and preserves
its host-tier admission change. Prior qualified artifacts and source branches remain
immutable. GPU qualification belongs to each exact native/Runtime/fixture cohort.

The Python retention window must remain absent. Native owner callbacks are not reclaimed
bytes: only physical release permits a bounded real allocator retry. Python owners claim
their own lifecycle nonblocking; an unavailable GIL or owner may leave demand exhausted.

The separate CUDA `Memory.reclaim` hard cut is an audit/preparation task here. No deletion
is enabled until real native-owner recovery and lifetime fixtures pass. MPS recovery and
explicit lifecycle disposal are separate contracts.

## Source composition

Fresh base: `cb6c66bf343d15637eaeb4af35731a37fc56d61c` (`origin/master`, Runtime0.36.23).
Clean merges: tested coherent `7adc1e82c39e451a311ec67eb0b640cf1e4d7bca`, sparse producer
`9b820d88f795939727394da673ed4566995bdfd5`, owner bridge/DerivedCache
`918957e1903a300c065ed6a1756d074e9da39ca4`. Integration checkpoint: `5865d7ba7`.
No native kernel source or `_kernels` interface differs from7ad. No streaming-limit writer,
`window_bytes`, `_plan_phase`, `_replan`, `_split_budget` or `note_split` remains in residency.
The0.36.23 host-tier `_worth`/`_admit` changes remain separate host-RAM policy; they do not
reserve device bytes or restore the deleted device window.

The matching TensorFS cohort must combine the tested native Plan/retention baseline with
the797 Cache bridge. The earlier c3-based Cache wheel alone does not qualify this complete
Runtime integration. Exact release version floors are not invented by this development PR.

## CUDA recovery hard-cut audit

Line numbers below refer to integration checkpoint5865d7ba7. This table is the proposed
cutover, not a claim that the synchronous paths have already been removed.

| Current source seam | CUDA cutover after owner proof | Contract retained |
| --- | --- | --- |
| `Binding.load_slot`, residency.py:946 | Remove the Python `budget_bytes`/`room()` pre-reclaim and early capacity refusal. The installed native managed limit admits the actual grouped Regions. | Ready-Region pins, exact input layouts, source/Fill lifetimes and loader completion. Prove actual native grouped admission under the same managed cap. |
| `Memory._prepare_native`, residency.py:2622 | Remove the catch→`Memory.reclaim`→second `Region.prepare` path. Native Region allocation owns its finite recovery. | No source read or transform is retried. The existing free-only/capture-preparation boundary still refuses before execution; do not silently turn idle preparation into reclaim. |
| `Memory._empty`, residency.py:2459 | CUDA calls the ordinary Torch allocator once; remove Python cache selection and the second allocation. | MPS staging headroom and its existing classified allocation-only retry remain. CPU behavior is separate. |
| `Request.reclaim`, residency.py:3514, reached from `execution/catch.py:208,226` | Stop CUDA from performing additional byte selection after the native allocator has already completed its bounded recovery. Do not queue estimated `need()` bytes as native Cache credit. | The legacy adaptive split mechanism is a separate P2 migration. This cut alone must not be reported as removal of all package operation retries. Closed native consumers already refuse those retries explicitly. |
| `Memory.reclaim`, residency.py:2312 | Remove CUDA's Python sorting of holds, blocks, models and Regions, repeated `room()` reads, and measured-byte retry decision. | Non-CUDA recovery remains separate. Callback success is never equivalent to freed capacity; only native physical-release counters permit an actual allocator retry. |
| `Memory.trim`, residency.py:2421 | Separate explicit idle disposal from pressure recovery. Close eligible graph/cache owners, discard idle Python views, reclaim native idle backing and trim the working cache as an explicit lifecycle request. | Host-tier disposal and offload semantics; active/escaped leases stay authoritative. No byte-target victim algorithm is needed for an explicit discard-all request. |

`Model.hold(free_only=True)` already returns before its `room()`/`release_cached` logic
on native CUDA backing. That host/MPS path is not a CUDA cache policy to delete. Actual
`Model.close`, representation/overlay changes and explicit graph `forget`/`retire` remain
owner lifecycle operations. Their checked joins are not moved onto an allocating caller.

The bridge covers Loop Programs, Recorder caches, idle LM Engine caches and DerivedCache
Regions. The last was added at918 and needs its own real lifecycle cell. Do not claim
LM/DerivedCache consumer qualification from the generic Cache or Loop fixture. Unsupported
native capability cannot silently revive the deleted policy; deployment must establish the
actual compatible SDK capability before this later cut is enabled.

## Exact fixture preparation

Frozen Creator controller81ba58656 contains four separate parent-extension cells:
`owner_allocator_progress`, `owner_held_gil`, `owner_capture_close` and
`graph_owner_retirement`. Its native/helper inputs are byte-identical to797; the
`source-with-797-helper.json` receipt records that provenance explicitly. The earlier9b98
`native_owners`/`arena_concurrency` child cases execute their own statically linked native
core and cannot qualify the parent Python bridge.

Root packages this exact Runtime source and the integrated TensorFS source. Run the frozen
controller's `prepare.py` with those actual artifacts, not a hand-written expected version:

```text
--source-receipt <81ba source-with-797-helper.json>
--binary <exact797 varena-cache-pages>
--native-extension-sha256 <integrated-wheel extension hash>
--runtime-wheel <exact integrated Runtime wheel>
--gil-gate <manylinux library built from frozen81ba gil_resize.c>
--gpu-uuid <actual rented GPU UUID, or UNBOUND before allocation>
--output <new immutable staging directory>
```

Preparation validates source/archive/binary hashes and ELF/GLIBC, extracts the actual
Runtime version and participating module hashes from the wheel, and records the C GIL
gate hash. Each call checks the loaded Runtime files, native extension, GPU UUID and real
Request. The parent Python cells use that loaded extension; helper provenance remains
separately labelled. Root's package receipt must additionally verify the full Python source
copy and exact580 reused native files/285 Triton keys before the candidate is installed.

Source syntax and clean-merge checks pass. Portable packaging, the strict bound GPU input
and actual owner/lifetime results are pending. No new CUDA recovery deletion is enabled.
