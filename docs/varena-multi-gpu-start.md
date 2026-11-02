# Per-rank native cooperation

Owner: `/root`, branch `codex/xs138-multi-gpu-start`. Creator companion has the
same branch name. This is source work; two/four-GPU execution is not qualified.

`Start.cooperation` is an ordered array of GPU owner descriptors, including for a
single GPU. Each descriptor contains the actual UUID, shared directory, sharing
mode and priority. Empty is the CPU/import-only form. The old single-object form
is removed; both sides of this private protocol must be installed together.

Runtime validates the complete inventory before starting followers or touching
CUDA. Missing ranks, repeated UUIDs and invalid directories fail at that boundary.
Rank r selects descriptor r and opens its native owner on local CUDA ordinal r,
retaining Working custody before Torch initialization. Native code checks that
the configured UUID is the actual device. Each rank has its own Memory and
execution stream. Rank zero requires every follower to acknowledge cooperation
before forming the collective group. A repeated Start cannot change any rank's
descriptor while the executor is active.

TensorD supplies categorical settings and actual device identities in sealed
device order. Byte admission, measured working peaks, reclaim and split selection
remain in Varena. Existing cross-rank split agreement remains in Runtime; this
change does not qualify that algorithm or change collective ordering.

Required GPU proof is separate: simultaneous independent jobs on two GPUs,
pressure/retirement on one while the other progresses, then full H3 Ulysses on
two and four GPUs with asymmetric pressure, output quality and matched timing.
Current leg17 keeps its frozen single-GPU protocol and artifacts. No master merge,
release or new rental is authorized by this source checkpoint.
