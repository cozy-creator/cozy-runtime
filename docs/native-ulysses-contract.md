# Native Ulysses choice and execution contract

The actual H3 sequence-parallel route now passes its selected backend/options to
`parallel.native_ulysses.exchange`; it does not pass an opaque compute callback.
The supported closed local providers are the existing Kitchen INT4, INT8 and
Kitchen Sol INT4/dense-prefix recipe, B=1, head width128, FP16/BF16, equal Q/K/V
shape and noncausal/unmasked inference. Unsupported recipes refuse before data
crossings. No different attention backend is substituted.

## Agreement and allocation

Each actual ProcessGroup has a host Store control instance and monotonic invocation
sequence. The prefix includes the actual group name; Runtime creates a fresh TCPStore
on each group formation. Stage keys include the sequence, immutable complete row/
shape/backend/transport identity and, after selection, the exact choice. Every rank
acknowledges reading each message before rank0 deletes it. Nothing in this transport
allocates a CUDA tensor or claims capacity.

1. Gather each rank's actual geometry and metadata refusal. Unequal and zero local
   row counts are preserved; every other semantic field must agree.
2. Query the actual selected provider for every model-legal head divisor. Sol uses
   its real `sol_attn_plan` byte/offset inventory. Existing INT4/INT8 explicit-buffer
   kernels use their exact documented code/scale/anchor/output shapes.
3. Each rank submits its complete tensor allocation vectors to native `split_select`.
   Returned eligible IDs are snapshots, not reservations. Python intersects IDs only
   for compatibility; it never compares capacity or memory-pressure numbers.
4. Rank0 submits the common menu in original model order to native `split_select`.
   Native phase-preservation/capacity ranking chooses the ID. Other ranks transport
   and accept that exact legal ID.
5. Each rank freshly selects/begins that singleton, forwarding its exact returned
   plan generation, then physically allocates and validates all its supplied buffers.
   All ranks report successful admission/allocation before the first tensor crossing.
   Full, common and singleton menus have distinct immutable native keys: native
   registration does not allow reusing one key with changed choices/preference.

The tensors held throughout one exchange are fixed output, three head-first send
packs, three receives, three contiguous landed Q/K/V tensors, one local attention
output, reverse send/receive and all local-provider scratch. Head padding is written
directly into the send packs; final rank/head assembly writes directly into the fixed
output. No permute/reshape allocation is hidden in these copies. Same scratch is
reused sequentially for model-legal head groups after same-stream dependencies.
NCCL uses `all_to_all_single` with supplied receivers; copy engines use the existing
crossing algorithm with `all_to_all_into`. Existing symmetric buffers, events,
communicators and private library holdings remain actual external owners. Their
unknown private pressure is not guessed or promised by tensor admission.

## Numerical and lifetime boundaries

The original selected dense/exact/sparse/prefix policy is retained. INT4 invokes the
existing `_attn4.attention` ABI. INT8 invokes the existing Kitchen quantize and
prequantized ABIs, with an explicit contiguous BHND output then the same layout copy.
Sol invokes `sol_attn` with queried workspace and the original tau/sinks, followed
by the selected INT8 protected-prefix recomputation. No kernel or approximation
threshold is changed. Original full-model quality/performance gates remain mandatory.

Preparation, allocation and numerical work use current owned S. Copy-engine peers
retain their actual S->copy-stream->S dependencies. Every local tensor/provider,
ProcessGroup and copy-engine owner survives checked S completion. If completion
cannot be proved, the actual owners are retained until executor teardown.

A preflight refusal is reported to every rank before data collectives. After the
all-rank ready acknowledgement, any failure uses the existing process-group abort
path, preserves the failure, and cannot replay or shrink the invocation. A native
selection/readmission may fail from changed phase or foreign pressure; no previous
snapshot is treated as a distributed reservation.

## Python fit/retry retirement

The old `ulysses._level` Room arithmetic, previous-pass maximum proposal and GPU
metadata gather are deleted. `execution.sizing` and its fit/descent ladder are
removed. Generic conv/channel norm/attention operations run ordinary eager once;
supported native RMSNorm/refiner/declared composites retain their native entrypoints.
An authored fixed window remains an authored fixed window. No memory estimate,
preflight byte percentage, same-level retry or OOM geometry descent is retained.
`catch.uncommitted_scope`, native eligibility and one-shot guards remain.

## Qualification

Actual Torch2.15.0 git d0f2 CPU Gloo+Store proof ran2 and4 ranks,24 packing/crossing/
return cells, including unequal/zero rows and head padding, bit-exact against actual
input tensor arithmetic. Actual CPU entry refuses GPU admission on every rank.
This is not CUDA, native attention or full-H3 qualification. GPU acceptance remains
full15s/362frames/1344x768/30steps/audio at ranks2 and4 with current provider/artifact
receipts, independent GPU repetition and output quality/performance checks.
