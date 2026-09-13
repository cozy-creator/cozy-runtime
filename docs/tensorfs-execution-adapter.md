# TensorFS execution adapter (cr-174)

Owner: Codex `/root/tensorfs_scoped_finish`.
Worktree: `~/cozy/.worktrees/cozy-runtime/cr174-tensorfs-execution-adapter`.
Branch: `feat/cr174-tensorfs-execution-adapter`.
Base: `76dcf0bc44fd1c159c04e7af674c3cbb2d398a35`, fetched `origin/master`.

TensorFS PR167 supplies the native typed derivation/source surface and scoped
writer FD. Runtime binds accepted inputs, declared output slots/ceilings, work
identity and writer epochs, admits storage and records exact native receipts.
The existing workspace remains the sole execution/custody journal.

The implementation replaces Runtime's `WeightsSink`/`WeightsTransaction`, duplicate
declaration/source classes, `StoreProxy`/`WriterProxy` and complete-role spool
transport. Keep only interface output declarations and execution receipt/admission
bindings. The author entry points are `ctx.tensorfs_source(model)`,
`ctx.output(slot)` and `ctx.adopt_model(native_receipt)`; algorithms use TensorFS
declarations, source inspection and writer handles directly.

The existing executor control lane remains bounded at 64 KiB. File descriptors
cross through that accepted executor's existing local channel. Tensor/config bytes
and larger native metadata belong to scoped TensorFS handles, not Runtime files.
Socket shutdown must interrupt blocked reads on cancellation/replacement. Observer
disconnect or reconnect generation does not revoke an accepted local attempt.

Implementation order:

1. Bind the scoped native writer to current intent/receipt callbacks and the
   executor FD lifetime. Preserve native preflight before begin, per-write
   admission, pending commit custody and cancellation-vs-commit reconciliation.
2. Inject TensorFS source/output capabilities through Context. Migrate the shared
   quantizer and a tiny producer; prove actual ordinary Creator CLI calls using
   the source-identified TensorFS development wheel.
3. Migrate all Runtime/model-tool consumers and examples, then delete the old
   wrappers and spool protocol. Keep no compatibility alias without a consumer.
4. Prove caller-edited reuse, changed callee invalidation, mid-role interruption,
   native pending-commit recovery, two-consumer GC custody and zero Runtime tensor
   spool bytes through ordinary local/private CLI execution (cl-265).

The machine-execution owner separately owns worker session/control/workspace,
`workspace_executions.py`, machine capture/RPC and detached request lifetime. Any
change at that boundary must be coordinated. This ownership document records
work in progress, not completed integration or a public release.
