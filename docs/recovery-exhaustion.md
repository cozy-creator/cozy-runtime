# Exhausted operator recovery inside an entered forward

Tracker #327. Owner: Codex Runtime recovery follow-up. Base e651f89d.

The current CPU red probe executes a real native linear operation, then reports
OOM at its return boundary. When the controlled allocator/owner gives no room,
Weights.progressed raises its original operation_exceeds_device refusal. The
surrounding module's counter has advanced, yet its residency was still reusable.
The next forward ran with that changed state. This is a confirmed control-flow
defect; the room and lease I/O are CPU boundaries, not physical CUDA evidence.

The narrow change preserves that capacity error and its shortfall unchanged.
Only an entered generic forward is independently marked poisoned when its
operator recovery cannot complete. The existing Executor residency consumer
retires the generation even though capacity codes otherwise permit reuse.
Nothing replays a Python forward or invents a smaller device floor.

CPU proof: counter/buffer/RNG state advances once, the original capacity code is
retained, inputs remain exact, and a subsequent forward refuses poisoned state.
The actual selected Executor construction reports the capacity outcome plus
poison, refuses a next load, and a fresh real Executor serves successfully.
Known functional operator calls outside a Python body and private SDK decoder
passes retain their authority; the interrupted real decoder pass resumes to its
uninterrupted tensor exactly. All 48 nearby CPU tests currently pass.

A separate bind diagnostic is retained in the evidence directory: generic
Recovery currently covers SDK LoRA bind math and falsely poisons a seeded random
or mutable-copy failure before forward entry, despite dirty/base-refill state.
That diagnostic uses the actual binder and stochastic BF16 bake with real CPU
bytes; it is not yet a production binding repair or GPU refill qualification.
No wheels are frozen until the reviewed safety increment lands.

Evidence: /home/fidika/cozy_v2/outputs/codex-recovery-exhaustion-20261005;
red no-room receipt also lives in codex-current-artifact-cohort-20261005.
Scope remains CPU control flow/tiny tensor equality; OS/GPU/full-model gates open.
