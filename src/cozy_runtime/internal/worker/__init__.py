"""The no-CUDA worker: protocol session, records, grants, arbitration, supervision.

Five modules, one process:

| module | owns |
|---|---|
| `plan.py` | the worker-scoped host/pinned ledger, the PlanChooser, the ExecutorPreparer |
| `grants.py` | DeliveryGrant reads/writes and the CLOSED hub-call authorization table |
| `child.py` | the disposable executor: spawn, seal, epochs, cooperative -> forceful kill |
| `attempts.py` | the attempt state machine, terminals, metric attestation, the warm pass |
| `session.py` | the `cozy.worker.v1` client and the worker entrypoint |

Nothing here imports torch, and `scripts/worker-live.py nocuda` proves it against the
RUNNING process rather than against the import graph.
"""
