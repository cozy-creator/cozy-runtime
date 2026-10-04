# Adaptive VAE capacity and replay safety

Work: tracker #320. CPU source6e reproduces an inherited full-decode retry through
an actual Diffusers AutoencoderKL subclass: decode increments Python state then
raises raw CUDA OOM before any hooked leaf. The adaptive wrapper changes tile mode
and invokes decode again, so the counter/input/RNG effects can happen twice while
no executor poison is recorded. A stock class with an authored pre-forward hook
can fail after the same sort of side effect; a class name or MRO does not establish
an exactly-once or rollback contract.

The minimal fix invokes the selected decode once. Its tile mode is chosen before
execution using existing measured peaks, owner shape seeds and available room.
Successful normal decodes retain their output and ordinary error behavior. A raw
OOM or an indivisible-operation refusal from decode becomes a typed
`recovery_contract_unavailable`, latches poison on all retained residencies, and
requires the executor/custody retirement fence. The original failure is included
in the detail; this does not assert that the whole model cannot physically fit.
Reviewed pure native operations may still release idle memory and retry that one
operation. The entire decoder is never automatically repeated.

No generic VAE replay promise is inferred from subclass membership, stock names,
processor names, successful prior runs or library version. Any future native
replay capability must prove input/RNG/buffer/Python/module-state semantics,
including authored overrides and hooks, before it is enabled. This patch adds
no source equality admission gate: harmless evolution and successful baseline
execution remain allowed.

Limits: an unmeasured first decode still uses existing initial placement and may
visibly refuse if a native operation cannot fit; it does not silently retry the
whole model or shrink the request. Learned pre-call tiles preserve output geometry
and step count but their numerical quality still needs actual same-engine GPU
reference comparison. The52 unconstrained image controls are reference/reuse proof;
none qualifies new low-pool behavior without the candidate SDK and physical receipts.

CPU tests use real native decoder/subclass calls and replace only device allocator
probes. They cover exact stock output, one successful subclass call, raw OOM and
indivisible refusal with input/RNG/Python effects exactly once, retained poison,
stock hook OOM, noncapacity errors and preselected tile geometry. GPU whole-actor
retirement, sparse-plane use and pressure quality remain separate gates.
