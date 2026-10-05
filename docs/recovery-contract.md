# Recovery after an operation has started

Tracker #327; Runtime PR #1202. Base 50af76dd. Owner: Codex Runtime memory lane.

A successful call and a lease refusal before a forward starts keep their existing
behavior. An arbitrary Python forward does not run again after it starts. Tensor
inputs being unchanged says nothing about its counters, buffers or RNG.

An OOM or library allocation refusal escaping such a forward reports
`recovery_contract_unavailable`, without a capacity shortfall. The residency is
poisoned. The existing Executor consumer reports poison and refuses a subsequent
load as `poisoned_generation`; an explicit new attempt uses a fresh generation.
No started request is automatically submitted again.

Functional operator recovery uses a private list of native `aten` families, with
mutable overloads, seeded RNG and tensor subclasses excluded. A custom namespace
receives no authority from a read-only schema. The list covers allocation,
arithmetic, reduction, linear/convolution, normalization and inference attention;
an unlisted operator succeeds normally but cannot replay after an allocation
failure. Fused attention carrying a seeded-RNG tag remains excluded even when a
particular call might disable dropout. This is deliberately conservative.

The only whole-body token is private to SDK image decoder passes. It requires the
existing mirrored Diffusers source/branch check, the explicit native leaf set,
native forwards behind the residency's own currently installed weight hooks, no
custom/global hooks or compiled call replacement, and standard stateless attention
processors, including the post-quant convolution before the decoder. Custom
decoder calls remain valid; they receive no whole-pass token.
The outer failed-step retry uses the same check, including steps without a block
region. No generic pure annotation, state snapshot or input-mutation inference is
used. Module/Python state is never rolled back.

**Confirmed CPU:** the actual production wrapper's former three red cases now
stop after one failing counter/buffer/RNG forward. Observed state matches one
control execution; inputs remain exact, leases balance and no room action occurs.
Mutable, RNG and custom operator failures also execute once. A real native linear
operator retries within one Python forward and returns the control tensor.
Before-forward acquisition recovery still makes room and executes the body once.
The real tiny Diffusers SDK pass retries after its output writes and returns the
uninterrupted decode tensor exactly. Custom leaf/forward/hook/processor calls
succeed with that tensor, then fail once when their callback advances state. The
post-quant forward outside a decoder block has the same once-only failure proof.
The real in-process Executor exposes the distinct error and poison, refuses reuse,
and a fresh real Executor serves successfully.

Evidence: `outputs/codex-block-replay-contract-20261005/proof-08.log` (42 passed),
with the original red arm retained in `proof-01.log`. Leases and allocator/room I/O
are controlled CPU boundaries over a real CPU plane and real tensors. This proves
control flow and tensor equality for these cases, not physical CUDA OOM, complete
model inference or image equality across memory regimes. Supervised OS replacement
remains unrun here: the cgroup mount is read-only. No GPU or rental was used.

Full static gates remain red on the untouched base and this candidate: the same
51 mypy errors in the same 10 files, and the same two architecture fences
(`author/_calls.py` imports internal code; `Any` count 995 exceeds the limit 830).
No type error or `Any` use was added. CI bug-class Ruff and changed-file Ruff pass.
`FINAL-STATIC-SUMMARY.json` retains the matched base/candidate comparison, using
fresh same-lock Python environments; this is a recorded baseline failure, not a
successful full static check. No unrelated source was changed to hide these gates.
