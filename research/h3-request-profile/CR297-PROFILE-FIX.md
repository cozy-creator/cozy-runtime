# cr-297 diagnostic interval correction

The measurement-only build `b41d433fc122fb6e7c730d3c4172cfd0f4592743`
(`claude/cr-297-graph-measure`) adds CUDA event durations and calls the result
`busy_ms`. Concurrent streams and duplicate/nested events make that sum exceed
elapsed time. The H3 session 43442514074 dense-step receipt reports 49,596.04 ms
busy against 49,445.33 ms wall. That receipt has no interval timestamps, so its
true occupied time cannot be recovered; retain it as original evidence.

The deterministic regression calls that exact helper with two 70 ms CUDA events
on separate streams, `[1, 71]` and `[41, 111]` ms. The old result fails with
`AssertionError: 140.0 != 110.0`. The replacement reports 110 ms occupied and
140 ms aggregate device work. Root cause is established by this source-level
reproduction; speculative hypothesis testing is unnecessary.

## Owner integration

This change belongs to the cr-297 diagnostic subtask. No production graph hooks
are added, and the existing measurement branch remains owned by its graph agent.
After selecting an owned diagnostic worktree at the exact measurement commit,
copy these replacements from this research directory:

```sh
cp profile_intervals.py "$MEASUREMENT_WORKTREE/src/cozy_runtime/graphs/profile_intervals.py"
cp cr297_profiles.py "$MEASUREMENT_WORKTREE/src/cozy_runtime/graphs/profiles.py"
```

Do not install these files in the active in-call graph implementation. That branch
has no call sites for this measurement helper. This is a future measurement-build
correction, not a retrofit of historical aggregate-only receipts.

The original `describe()` still repeats the global profiles map beneath every
component. Treat matching labels there as the same measurement, never independent
samples or additive component totals. That separate presentation behavior is
unchanged by these replacements.

## Measurement contract

- `measurement_window`: the CPU `record_function("cozy.cr297.profiled-call")`
  start/end, in the same trace-relative microsecond clock as CUDA events. The
  annotation encloses the call and completion synchronization of the current GPU
  by default. A caller can explicitly pass `devices=(0, 1)` to measure that set.
  `synchronized_devices` records this scope; other visible GPUs are never touched.
  Initialization synchronization precedes the profiler.
- `profiler_context_wall_ms`: independent host-clock duration including profiler
  entry/exit. It is not used as a device denominator. The annotated call is still
  instrumented and is not a clean inference benchmark.
- `device_activity`: one row per selected device, with all of its streams unioned.
  Observed events from other devices are excluded and named in
  `excluded_cuda_device_indices`; their completion is outside this measurement. No total combines different GPUs into a single wall-time utilization.
- `busy_union_ms`: occupied traced device intervals inside the explicit window.
- `aggregate_device_work_ms`: sum of the same event/window intersections. It may
  exceed the window. CUDA activity includes kernels, copies and memory operations;
  the helper does not mislabel every activity as a kernel.
- `observed_span_ms`: first-to-last intersecting activity, zero when none exists.
- `unclassified_inactivity_ms`: window minus occupied intervals, including tails.
  It does not establish CPU launch overhead, paging, or any other cause.
- `top_aggregate_device_work_ms`: full event names ranked by aggregate work on
  that device, not wall-time attribution.

The old `wall_ms`, `busy_ms`, `kernels` and `top` fields are intentionally replaced.
Missing/ambiguous annotations or invalid event intervals report an `error` without
returning partial activity metrics or discarding the model output. Model failures
still propagate. Only the selected warm call is profiled.

PyTorch source inspected locally: version `2.15.0+cu132`, source commit
`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`.
`torch/autograd/profiler.py` constructs every FunctionEvent's `start_us` and
`end_us` from `(kineto_event.start_ns/end_ns - trace_start_ns) / 1000`.
`torch/autograd/profiler_util.py` documents `time_range` in microseconds and
`device_index` as the GPU index. No wall-clock/perf-counter origin is mixed into
interval arithmetic.

## Validation boundary

The local suite exercises the actual `profiled(torch, label, run)` adapter with
profiler-shaped events and deterministic context managers: overlapping streams,
containment, duplicate/disjoint events, explicit device separation, idle devices, bounded
clipping, tails, warm-call selection, missing/ambiguous annotations, invalid
intervals and model errors. A regression ensures visible but unselected GPUs are
untouched. The trace tool exercises the same accounting module and works both as a
CLI and under path-based dynamic import without changing `sys.path`.

This is source qualification. It is not a CUDA execution, H3 rerun, performance
claim, release or deployment. The graph owner must apply the replacement to a
measurement build and collect new timestamps to qualify real profiling behavior.
No GPU rental is needed to run the regressions.
