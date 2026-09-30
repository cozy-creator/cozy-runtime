The Runtime has a stage-scheduler core, which is not wired in yet.
- `internal/stages.py` holds the typed stage graph, learned from the `@uses_components` scope calls a package already makes. It also holds a persisted cost book of measured wall, gap, passes, growth and cheapest-rung growth per shape cell and width, and the slot-ring streaming cost model.
- `internal/worker/stage_policy.py` is the pure placement, turn, bubble, prefetch and admission-floor decision core.
