# Demanded weight upload experiment

SDXL's constrained trace has many individually blocking pageable weight copies. This
experiment uses the reviewed HostTransfer mechanism on the current stream to copy one
already-demanded unit through the fill plane's existing pinned ring. It allocates no
additional model unit or host pool and does not predict or invoke a future model call.

Planner owns eligibility; fill lends existing capacity; tiers copies and joins before
publishing weights. Every transfer retires before authored code resumes. A failed drain
poisons the executor and keeps its transfer backings alive. Allocation failures follow
ordinary finite demand recovery. Insufficient or unsupported staging uses the prior copy
path. Group, compiled, mutable, pending-conversion and grad-enabled paths are excluded.

`residency.demand_upload` records eligible/uploaded/fallback bytes, unit counts, and
prepare/publication/retirement times. Preparation includes the added CPU staging copy;
retirement conservatively drains the device and may wait for unrelated queued work.
Neither phase is assumed free. This is not a claim of inference speedup.

Qualification must compare the same complete SDXL request and constrained pool against
R19, with identical pixels, residency, OOM/retry accounting and no additional host pool.
Report staging coverage and total denoise/request time. Counterbalance a promising result
before promotion. This branch intentionally excludes optional lookahead and scalar-stats
changes so a measured effect has one enabled behavioral difference from R19.
