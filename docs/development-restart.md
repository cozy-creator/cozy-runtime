# Fixed-pod Runtime process replacement

The fixed `cozy-runtime-worker` entrypoint accepts SIGHUP from its local root
operator as a request to stop for process replacement. The handler marks the
restart and requests stop; the existing supervisor-pipe watcher closes transport
outside the handler. Ordinary executor/reader shutdown and the worker-root lease
still close. Source preparation drains under its existing lock, late preparation
refuses, and native source operations remain held for the replacement.

A fixed pod owns its TensorFS Store. It reads `Store.model_source_operations()`
from the native catalog on startup and again before ordinary shutdown. Thus a later
SIGTERM/EOF releases retained prepared and open operations even when the replacement
never replayed their preparation. A local worker may share a Store and releases
only the source operations it remembers owning. Runtime scans no native files and
stores no duplicate source registry.

This requires TensorFS0.3.11's native inventory and idempotent release. Release
metadata must adopt that floor before publication. Process replacement retains the
pod's existing boot/TLS/readiness identity and persisted control-stream fence;
missing or corrupt ownership history refuses startup.

The operator must quiesce execution before requesting replacement. This mechanism
does not preserve an in-flight producer's uncommitted computation, install wheels,
provide SSH, or reconcile an immutable image inventory with replacement software.
The development manager must survive Host exit, verify all old children stopped,
install the exact reviewed software, and reobserve control before resuming work.
The production source/derived cleanup protocol remains separate.
