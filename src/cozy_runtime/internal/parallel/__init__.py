"""Sequence parallelism inside ONE executor (cr-068, cr-018's door).

A group lane's executor is K processes under one cgroup, one seal, one supervision and one
epoch: rank 0 is the process the worker spawned and addresses; it spawns ranks 1..K-1
itself (`group.py`), each sealed to the same K devices and pinned to its own card. Rank 0
decides the plan and broadcasts it (`plan.py`); a follower that derives a different one
refuses `rank_divergence` and never adapts. Collectives happen only inside the gated call
(`cp.py`); the arguments a mirrored model call carries cross the rank seam in a CLOSED
vocabulary (`wire.py`). Ported from v1's `gen_worker.parallel` with its settings authority
removed: every number here is measured or handed down by the worker.
"""
