# Machine agent and resident Runtime

The same machine agent supervises the Runtime on an owned computer and a rental. The
agent owns endpoint identity, authorized callers, process start/reap/update, and provider
release. Runtime is the resident, torch-free coordinator: it owns acceptance, the durable
queue and journal, executor lifetimes, and model preparation. Its memory manager chooses
device residency; TensorFS acquires and retains bytes.

The private launch contract is detected with `cozy-runtime-worker capabilities --json`.
`machine-supervisor/1` accepts `COZY_RUNTIME_CONTROL_MODE=supervisor` and a fresh
`COZY_RUNTIME_PROCESS_INCARNATION`. Readiness echoes that incarnation so an old readiness
file cannot prove a new process ready. The inherited stop pipe and the process-lifetime
root lock remain mandatory. Existing clients still use the supported legacy launch mode.

Supervisor mode never reads or writes `ownership.json`. The authenticated local control
stream remains fenced within a process; it does not elect a durable owner. The existing
`cozy-local-client` journal namespace is established before recovery. Adopting or restarting
a machine does not reset its workspace ID, run numbers, attempts, checkpoint custody, or
finished outcomes. A new process incarnation is not a new attempt or workspace.

Runtime retains warm executor processes between requests. After the existing 900-second
machine idle grace, Runtime can park compute. It acquires the preparation and control locks
and checks the durable queue, active attempts, supervised work and preparation again. Only
then does its existing executor manager retire idle scopes. It never shuts down the
coordinator for idleness. A queued request starts its retained preparation through the same
execution path, and serialized admission prevents an idle decision from retiring work
accepted at the same time. Retained outcomes and observer attachment are not compute demand.

## Hub access without machine registration

An owned machine needs no Hub worker row, worker token, readiness receipt, or rental
lifecycle. Its controller may delegate limited catalog/download/publication access. The
agent atomically writes `run/cozy/bootstrap/machine-hubs.json` beneath the machine root:

```json
{"version":1,"hubs":[{"origin":"https://hub.example","access_token":"SCOPED_TOKEN","expires_at":1893456000,"object_storage_hosts":["objects.example"],"ca":"hub-ca.crt"}]}
```

`expires_at` is Unix seconds in this private projection. `ca` is an optional certificate
file name beside the document. Runtime re-reads this projection when it uses authority,
including publication-token renewal; rotation does not restart the coordinator. Unknown
fields and unreadable entries affect only the relevant Hub. Credentials never enter
captured arguments or executor environments.

Catalog reads use `Authorization: Bearer`; TensorFS receives `bearer <token>`. A publication
grant is renewed with `X-Cozy-Execution-Access`; model mutations carry only its short-lived
publication Bearer. Existing rentals may continue using `worker_id`/`worker_token` rows.
One row cannot combine rental credentials and delegated access. An expired delegated grant
fails the operation visibly; locally retained checkpoints still work without Hub access.

`runtime-model-preparation/1` moves the explicit models-only and package/model preparation
lanes into Runtime. Exact base and adapter checkpoints go through TensorFS `ensure/1`, with
the full preparation closure protected from eviction. Package installation can overlap
model transfer. Cached exact custody needs no network. The dependency floor is TensorFS
0.3.78, the first release with `ensure/1`.

## Read-only history while Runtime is unavailable

Runtime remains the sole writer of `<TensorFS root>/.cozy-workspace/journal.sqlite3`.
`machine-public-reads/1` exposes additive, explicitly selected SQLite views. The agent reads
with `mode=ro`; an unprivileged pod agent uses its root guardian's whitelisted read operation.
The private directory and WAL permissions are unchanged.

| View | Public columns |
| --- | --- |
| `machine_workspace_v1` | `workspace_id` |
| `machine_runs_v1` | `workspace_id`, `run_number`, `request_id`, `submission_id`, `attempt_ordinal`, `generation`, `state`, `collected`, `sequence`, `compacted_through`, `accepted_at_ms`, `finished_at_ms`, `worker_id`, `worker_boot_id` |
| `machine_events_v1` | `workspace_id`, `run_number`, `request_id`, `sequence`, `attempt_ordinal`, `at_ms`, `kind`, `body_json`, `invocation_spec_digest`, `outcome_id`, `outcome_digest`, `outcome_json` |
| `machine_outputs_v1` | `workspace_id`, `run_number`, `request_id`, `attempt_ordinal`, `slot`, `manifest_json`, `content_bytes`, `state` |

JSON columns are UTF-8 canonical JSON stored as BLOBs. Digests are raw bytes. Outcome
columns are populated only on outcome events and reconstruct `AttemptOutcome` directly;
product event bodies are canonical `RunProduct` documents. Readers preserve cursor and
compaction semantics and end an event page at its first outcome, as the live API does.
Runs views contain numbered root runs. Accepted boot identity is historical, not proof of
the currently running process. No capture, offer, preparation, or invocation document is
exposed by these views.
