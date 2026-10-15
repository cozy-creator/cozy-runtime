# TensorFS execution adapter (cr-174, historical implementation record)

This record describes the earlier Python-worker migration. Current managed execution is
owned by TensorD; Runtime consumes its scoped data/buffer handles and performs executor
GPU work using TensorFS. See [Machine and Runtime roles](machine-agent-runtime.md) for the
current boundary. References below to the former worker/workspace journal are historical.

Owner: Codex `/root/tensorfs_native_resume`.
Worktree: `~/cozy/.worktrees/cozy-runtime/cr174-tensorfs-execution-adapter`.
Branch: `feat/cr174-tensorfs-execution-adapter`.
Base: `76dcf0bc44fd1c159c04e7af674c3cbb2d398a35`, fetched `origin/master`.

Published TensorFS0.3.41 (PR167/169) supplies the native typed derivation/source surface and scoped
writer FD. Runtime binds accepted inputs, declared output slots/ceilings, work
identity and writer epochs, admits storage and records exact native receipts.
The existing workspace remains the sole execution/custody journal.

The implementation replaces Runtime's `WeightsSink`/`WeightsTransaction`, duplicate
declaration/source classes, `StoreProxy`/`WriterProxy` and complete-role spool
transport. Keep only interface output declarations and execution receipt/admission
bindings. Model evaluation uses the same native source metadata/transport, with only float32 reconstruction in Runtime. The author entry points are `ctx.tensorfs_source(model)`,
`ctx.output(slot)` and `ctx.adopt_model(native_receipt)`; algorithms use TensorFS
declarations, source inspection and writer handles directly.

File descriptors cross through the accepted executor's existing local channel.
Tensor/config bytes and larger native metadata belong to scoped TensorFS handles, not
Runtime files.
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


Native component delivery: TensorFS0.3.41 is published from ec4eb902, wheel SHA256
c29a9f5370e84d79c4ec82a6592d055142d6814e1ab2d1250c4233756933029a. Runtime dfb74d99
removed the quantization compatibility path; the follow-up removes the full author writer
wrapper and executor Store path. Native source/writer/replay, managed quantizer recovery,
capacity and Workspace controls run against real scoped descriptors. The ordinary Creator
CLI producer/inheritance/shared-FP8 proof with nonlexical components is recorded under
`outputs/cr174-tensorfs-execution-adapter/order-cli-*` in the Cozy v2 workspace. The new
private/disconnected/default-home CLI qualification remains a joint integration gate.


The reader follow-up removes `ModelReaderProxy`/`ModelReaderBroker`, per-view metadata/part
spool files and `WeightsSource*` geometry classes. TensorFS PR170 supplies source checks,
bounded native part reads and opaque encoded tensor identities. Metadata-only inspection
leaves no persistent read lease; checking/reading pins the source until close. Resident
part reads require no disk-write grant. Runtime keeps only the logical float32 decoder,
invocation binding and resource cleanup. Eval's actual native measurement and instrument
normalization drivers qualify the new types and transport without claiming model quality.

Native `FileAsset` and `Tree` results can be passed directly to another managed
call. The worker selects only typed input fields, checks the parent's actual
received or input capability, and retains an independent native root for each
child input before dispatch. Existing grant loaders materialize those inputs for
both job and serving executors. Failed admission releases the child's holds;
parent cleanup and GC cannot release a live child's copy of the capability.
