# Small production-boundary H3 warm reproduction (h3a-093)

This fixture creates **3,697,560 bytes of synthetic weights** for a real tiny
upstream H3 transformer and an independent eight-byte overlay. It uses the same
mixed precision policy as Runtime H3. Both Models are prepared, preserving the
multi-model boundary without a large model download or an unverified overlay hook.

The base Model runs the unchanged `OfficialH3Pipeline.warm_dit` method with the
fixture's schedule values. That produces the actual warm geometry: eight video,
eight text and eight audio rows, with an eight-row replicated text refiner. It then
runs 1,029- and 1,032-token cases. There are 56 attention heads of dimension 128.
At degree four the actual 24-token warm has six local rows (~84 KiB per Q/K/V);
the larger cases have up to 258 local rows (~3.53 MiB per Q/K/V). Both allocation
regimes matter; the fixture does not assume expandable allocation caused the failure.

`../h3_group_warm_fixture.py` uses `run_slice`, the real LocalRecordOwner, Worker,
ExecutorSupervision/trampoline, TensorFS fills, rank creation and warm path. Its
RecordOwner authors the ordinary device pin. It sets the real pod executor
UID/GID 65533 and the existing allocator option. No CUDA transport, accountancy,
collective, result, or process-protection behavior is mocked. This is a Runtime
local-door diagnostic, not a claim that a Creator/Hub deployment passed.

Prepare synthetic checkpoints without touching CUDA:

```sh
CUDA_VISIBLE_DEVICES='' python tests/testdata/h3_group_warm_fixture.py \
  --root /absolute/owned/fixture --prepare-only
```

The GPU execution gate requires a root-controlled, explicitly owned GPU host,
with Runtime/TensorFS/Torch and the fixture's released dependencies installed.
Stage the package, harness and interpreter in directories traversable by UID65533
(for example a dedicated `/tmp/h3a093-...` tree and its venv). The runner verifies
actual UID access before constructing a worker. Do not chmod unrelated directories
or weaken the worker's private state permissions.

```sh
python tests/testdata/h3_group_warm_fixture.py \
  --root /absolute/owned/fixture-run --degree 4 --attention sdpa
```

The default allocator is the production `expandable_segments:True`. A separately
labelled `--allocator expandable_segments:False` run may compare that one policy
if authorized. Optional `--attention flash-attn3` or `sol-attn` resolves that exact
backend before warm; it never silently falls back. Sol requires a Runtime cohort
with h3a-087 integration. No NCCL debug flag or syscall exemption is injected.

Runtime h3a-092 owns pre-CUDA startup facts and structural exception chains. The
fixture additionally prints each rank's component entry, actual parameter/current
device and allocator segment summaries after CUDA initialization. These summaries
are not proof that any particular exchanged buffer uses an expandable segment.
Outcomes and workspace artifacts are retained under the supplied root. Capture
stdout/stderr continuously to the controlling machine before a rental can end.

CPU qualification: the official 24-row warm and both larger forwards passed with
these dimensions; the package interface builds; native TensorFS checkpoint creation
passes. GPU/UID/full-worker execution has not yet run. The independent overlay is
prepared but no foreign-module callback behavior is claimed.
