The Worker's GPUs are run by a stage scheduler; `gpu_scheduler.py` and the per-GPU device locks are gone.
- Each GPU call is placed once, width included, on the GPU set with the earliest predicted finish, from a cost book the Worker learns from every call and keeps across restarts (`stage-costs.json`). A call whose finish is unknown while every set is busy waits unplaced.
- An executor that speaks `stage/1` (one GPU, on the weight plane) asks for a turn at every component-use scope (`StageEnter`/`StageExit`). Older executors, Ulysses groups and device jobs hold one turn from their device entry to their exit, as before.
- The next queued call on a GPU is dispatched while the current one runs: its executor starts and its construction loads into the host tier, and its first stage waits for its turn. A call whose executor still holds device weights is dispatched at its turn instead, so the turn holder can cut it.
- A call past its last observed stage claims nothing. When the call wanting its GPUs next lacks room, it unmaps its weights (they stay pinned) and keeps the turn until it has; with room, the weights stay mapped.
- A wait for a turn is not counted as execution time, nor by the author's stage and step timers.
- A restarted machine prewarms the two installations each GPU set served last.
- Calls that can never run are refused at intake with numbers (`no_capacity`): some stage's measured floor exceeds every GPU at every width it may run at.
- New journal rows: `stage.prepare`, `stage.turn`, `stage.exit`. `gpu.grant`/`gpu.wait`/`gpu.release`/`gpu.lease` keep their bodies; `gpu.wait` also names the GPUs a call is placed on.
