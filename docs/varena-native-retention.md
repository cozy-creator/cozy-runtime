# Native demand retention cutover

This branch removes the Runtime cache-window and retained-prefix byte decisions.
It is independent of the preserved ownership-guard experiment (#1992/#2374).
Clearing a streaming window only removes early eviction: native process/device
limits, live/escaped pins, loading state, completion fences, capture protection and
graph pools remain authoritative. Existing graph/cache-owner retirement remains.

The intended change removes all three window writes and the prefix/replanning
arithmetic together. Runtime still declares component phases, source/binding
identity and explicit graph/hold ownership. Native demand chooses reclaim when
actual capacity requires it. Host cache eligibility must follow actual pages and
read history, not a Python retained prefix. Generic split advice and P3 room
compatibility remain named separate migrations; they must not reconstruct a
second retained-weight allowance.

The paired native API is `Arena.clear_streaming_limit()` (Creator #2372). It
changes only native intent metadata, does not reclaim or call the driver, and
preserves other capacity limits. No Python maximum integer is supplied.

Source and GPU validation remain pending at this initial draft checkpoint. The
c05 prefetch-only full H3 arm failed performance (+4.56% denoise/+5.29% recorded
weight-source traffic); no fallback or threshold relaxation is proposed. Actual
cold/warm model reads, two capacities, aliases, failed fills and full H3/SDXL must
qualify the combined retention change. Nothing is merged or released.

## Implemented source boundary

`Memory` requires `clear_streaming_limit` when its native CUDA arena opens and
clears the legacy window once. Model registration, dtype/layout apply and phase
entry never set it again. `_native_region` reserves stable address space only;
normal demand calls native `Region.prepare` directly. If actual native allocation
still fails, the existing Python graph/cache-owner retirement adapter remains,
followed by an allocation-only retry only when it returned capacity. Capture,
idle preparation, source-fill publication and escaped Pin ownership are unchanged.

A component/request phase transition cancels the prior owned speculative read,
then calls metadata-only `reset_preferences`. Prior idle pages become native LRU
standby; pages touched by the active looping phase use native MRU order. There is
no Python retained prefix, missing-byte estimate, window, eager phase trim or
replanning budget. Binding-layout rollback retains identity/generation/site state
without the deleted prefix field.

Host-cache behavior deliberately changes: a raw page becomes eligible for idle
host pinning after its second completed read; baked/final bytes become eligible
after their second materialization. The first records history. This uses actual
reuse rather than predicting residency from `block.preferred`. Host capacity,
source identity, overlay validity and ordinary host demand are unchanged.

Generic cuts still observe actual driver-free, mapped working cache and idle
weights through `Request.room`, with their existing conservative reserve and
split/retry behavior. They no longer receive an independently planned phase
activation budget or train `_floor/_notes` for `_replan`. The unchanged legacy
TensorD pressure-floor multiplier is explicitly named `_ROOM_BLOCKS`; P3 removes
that separate compatibility path. This patch does not claim all P2/P3 policy is
gone, nor does it retire graph owners natively.

## Validation at source freeze

The matching before-retention control is `78cfbbe8d` (fresh master `ac870d96f`,
Runtime0.36.20, plus frozen c05). Do not attribute new-master interface/row changes
to retention by comparing only against the older0.36.18 c05 artifact.

- All three window setters and prefix/replan/note-split references are absent.
- AST/diff checks pass. Paired mypy has17 baseline/16 candidate findings, zero new;
  Ruff has12 inherited findings in each. No unrelated findings were repaired.
- An actual TensorFS checkpoint→Runtime `Declared/Weights/Memory.load` CPU smoke
  reads8,192 bytes, returns1,024 correct elements and closes owners. CUDA remains
  uninitialized; this is non-CUDA compatibility evidence, not GPU qualification.
- Independent native/Runtime source review finds no new lifetime blocker. The
  retained native allocation protections and Python cache-owner cleanup remain.

Receipts: `outputs/xs138-p2-native-retention/`; source check details are preserved
under this task worktree's `.task`. The real GPU fixture is separately owned by
the native reviewer. GPU results, whole-model quality/denoise/transfer acceptance
and release remain pending.
