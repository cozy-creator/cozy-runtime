# Two-GPU Ulysses correctness probe

This research-only package preserves full [1,109225,56,128] BF16 activations and uses
exactly two Runtime-granted RTX5090 ranks. The default rung and sequence_parallel
both require2. The frozen four-GPU version remains at source0dfd6c29 and the separate
h3-sage3-collective-correctness-20261010 output directory.

Primary control: packed Q/K/V with a2,348,774,400element span, token splits54613/54612,
28heads per actual exchanged callback, full-reference intermediate/output comparisons,
repeat/inverse/input-immutability controls. A research-local copy of the reviewed int64
fixed-means module preserves upstream BF16 matmul correction. Public Runtime0.23.5 and
TensorFS0.6.1 are unchanged. Three warm timing samples compare only this ordinary
single-pass two-GPU path; these are regression checks, not video generation times.

Sensitivity control: prior run5228's contiguous three-QKV generation, seed581204, showed
an upstream difference at14heads per group but not at28. On these same two GPUs, two
real exchanges of28input-head batches produce14local heads per callback and reconstruct
all56heads. This is explicitly a two-pass numerical sensitivity control, not four-rank
transport, and reports the observed mismatch rather than requiring it. Full geometry,
precision, and seed are fixed. No seed search, videos, sampling, or quality-mode study.

The133KB diagnostic checkpoint's ten tensors provide normal model admission; they are
not used as activation operands. Required override:

    model.model=paul/attention-kernel-benchmark-20260916#sha256:61dc0e651cf6ab7744213611b6ee1cb4a4ba0b8dac5da8f165d950c2c91a04ca

The report validates distinct GPUUUIDs/PIDs and complete rank coverage, retains transport
mismatch details before skipping attention, measures actual peak allocation, and reports
primary and sensitivity-control outcomes separately. A normal CLI --timeout=8m --await
invocation bounds work. Root owns rentals and must authorize execution after finalCPU,
capture/describe, checkpoint-fit, and independent review proof. No defaults change.

Report transport correction: run5245 reached the component output hook and failed because
Diffusers rejects dictionary outputs even for an empty CP plan. No numerical pass is
claimed from that run. Version0.0.2 returns an owned CPU uint8 JSON tensor through the
actual hook, then decodes it in the author model method. The real ModelMixin/CP hook
CPU regression reproduces the old refusal and verifies the new report round-trip.
