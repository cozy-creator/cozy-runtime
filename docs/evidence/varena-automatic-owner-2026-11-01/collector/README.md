# Read-only automatic-owner qualification collector

This helper performs only ordinary `cozy machine show`, `cozy machine logs ...
memory --lines 0 --json` and `cozy run show --once --json`. It never installs,
runs inference, rents or ends a machine. All raw stdout/stderr, command arguments,
return codes, timestamps and SHA256 values are retained. Each output directory
must be new; a failed collection is not overwritten.

Use it after installing all three paired P3 artifacts and before ending the
machine. Substitute the exact accepted rental ID; the collector refuses a
machine identity/version mismatch. Paths below are rooted at:

```bash
p3=/home/fidika/cozy_v2/outputs/xs138-p3-owner-candidate-afa6f9229
```

1. **Immediately after the paired SDK/worker update and before `/gpu`:**

```bash
python3 "$p3/qualification/collect.py" before \
  --machine columbina --rental-id EXACT_ACCEPTED_RENTAL_ID \
  --cohort "$p3/cohort.json" --out /absolute/results/owner-before
```

Require `valid=true` before proceeding. Then root runs the frozen ordinary
`$p3/fixture/owner-runtime/gpu` with `$p3/owner-payload-columbina.json`.

2. **After that run completes**, derive the same-PID/incarnation follow-up payload:

```bash
python3 "$p3/qualification/collect.py" observe-input \
  --run GPU_RUN_NUMBER --cohort "$p3/cohort.json" \
  --payload "$p3/owner-payload-columbina.json" \
  --out /absolute/results/owner-observe-input
```

This saves the unmodified run record, validates actual software/fixture identity,
and writes `observe-input.json` containing its `expected_pid` and `owner_row`.
Root runs the same frozen `$p3/fixture/owner-runtime/observe` with that payload.

3. **After the follow-up, while the machine is still ready:**

```bash
python3 "$p3/qualification/collect.py" after \
  --machine columbina --rental-id EXACT_ACCEPTED_RENTAL_ID \
  --cohort "$p3/cohort.json" --payload "$p3/owner-payload-columbina.json" \
  --before /absolute/results/owner-before \
  --run GPU_RUN_NUMBER --observe-run OBSERVE_RUN_NUMBER \
  --out /absolute/results/owner-after
```

`qualification.json` requires both actual fixture verdicts, matching Runtime /
TensorFS / TensorD source versions, the same reachable worker boot, same executor
PID/participant incarnation, and one exact `gpu_memory_policy=Cooperative` note
for that PID/device. It checks the returned interval from that policy note through
the follow-up's actual reply timestamp for legacy Room, automatic offload,
ended-for-room or GPU-pressure actions. This first cohort requires one
worker-visible GPU, so those events cannot be hidden by missing per-event PID.

The log query has **no default 200-line tail and no `--since` prefilter**. It reads
all retained current/prior memory files. An exact suffix of the pre-run snapshot
must remain in the post-run snapshot to anchor continuity. For an empty initial
log, the returned text must remain below the frozen 4 MiB rotation threshold.
A missing/ambiguous anchor, malformed note, failed read, changed worker boot or
missing positive policy binding makes the result inconclusive/failed. It never
asserts global absence from a truncated tail or claims that unlogged paths were
instrumented. Raw Start frames are not exposed by `run show`; the actual active
Memory owner, software versions and source-level acknowledgement check form that
boundary, stated explicitly in the report.

The fixture's temporary Shared participant opt-in and restoration remain
explicit. This is not machine shared-mode propagation, game coexistence,
automatic shortage/foreign-pressure policy or completed P3 qualification.

Source facts reviewed for this collector: frozen `afa6f9229` MemoryLog rotates
above 4 MiB to one prior file; `platform::tail::Log::read(Start::Last(0))` reads all
retained lines; CLI emits one JSON wrapper per complete raw line without the
Telemetry.log shortening path. Collector source/static checks do not constitute
remote test evidence.
