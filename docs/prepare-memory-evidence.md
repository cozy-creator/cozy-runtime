# Failed warmup memory evidence

Anima1898/1899 refused VAE decode during construction at1GiB. The original failure
records retained the17MB shortfall but discarded allocator readings, residency,
operation counters and planner decisions. The retired process could no longer
explain the remaining floor.

The warm refusal now snapshots the existing allocator reader and Planner.document
before backend poison/retirement. It does not request room, change the cap, retry,
or probe followers. Each reader can fail independently; absent data stays unknown.
These are **after-failure, before-retirement** observations, not a reconstruction
of the allocator at the exact failing allocation.

The load refusal carries the optional preparation_memory document into the existing
Loading model weights phase event. Ordinary full run history retains it under the
event payload; the phase's scalar fields and the existing triage ring stay unchanged.
Oversized diagnostic documents are shed at the control seam with the existing
observation_caps counter, preserving the original terminal refusal. Older replies
without the optional document continue through the same path.

CPU tests cover a real Model.warm exception, control-socket serialization, durable
phase history, unknown measurements and oversized evidence. They do not establish
GPU recovery or qualify the1GiB floor.
