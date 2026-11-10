"""Sequence parallelism inside ONE executor (cr-068, cr-018's door).

A group lane's executor is K processes under one cgroup, one seal, one supervision and one
epoch: rank 0 is the process the worker spawned and addresses; it spawns ranks 1..K-1
itself (`group.py`), each sealed to the same K devices and pinned to its own card. Rank 0
decides the plan and broadcasts it (`plan.py`); a follower that derives a different one
refuses `rank_divergence` and never adapts. Every GPU runs each attempt's handler (`spmd.py`);
a `_cp_plan` component splits and gathers inside its own call (`cp.py`), and what one GPU
alone holds crosses to the others in a closed vocabulary (`wire.py`).
"""
