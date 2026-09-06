# Draft: source operation lifetime across Runtime replacement

A source operation belongs to the owner's model transfer, not to one Python worker process.
Current `Worker.shutdown()` calls `release_model_source` for every prepared or partial source
operation. TensorFS release removes the operation's holds, source-selection record, conversion
journal and exported-head cursor. A graceful software update therefore loses more recovery
state than a process crash.

## Reproduced behavior

The public Runtime 0.2.23 / TensorFS 0.3.9 pair was exercised using the real builtin H3
native-dual profile. A complete 535-key header roster per DiT was split into a landed 384-byte
`final_layer.video_out.bias` carrier and an absent remaining carrier. The landed tensor values
were synthetic; the converter and journal were real, with no forged SQL or stored progress.

After SIGKILL, a new actual Worker over the same Store, with all original carriers removed and
no owner checkpoint supplied, returned the exact previous head and 791 checkpointed bytes.
After ordinary Worker.shutdown, the journal/cursor files were absent and that same restart
returned no head. Source bytes still present in CAS do not replace the lost op-to-part journal.

Evidence: `outputs/h3-resume-20260905/source-shutdown/process-proof.json`, `make-roster.py`,
`worker-process.py` and `reproduce.py`. The actual Host-to-public-Runtime source composite also
now proves metadata declaration, partial conversion, checkpoint paging and no-raw replay using
a checked-in 3.6 KiB H3 geometry fixture in Tensorhub's workerhost tests.

## Missing owner release consumer

The only production caller of Runtime's release helper is Worker.shutdown. Creator and Host
have no ModelSourceRelease/Abort/Finalize request. The native release method takes an operation
ID only, deletes its state completely and is not an authorization or replay boundary. A later
prepare can recreate that operation; a repeated release sees an absent root.

Do not remove shutdown cleanup and ship permanent source holds without connecting explicit
owner release. Do not infer release from a temporary disconnect, worker death, elapsed time,
individual attempt outcome, output receipt or a model slot name.

## Proposed minimal coordinated change

1. Creator keeps source operation ownership in its existing model-transfer record. It retains
   the holds while final source checkpoints need acknowledgement or the logical producer
   request can still retry. Successful final producer completion or explicit cancellation
   records release intent for that same operation. An attempt failure alone is insufficient.
2. A bounded owner request carries the existing operation ID and source-selection digest in
   the existing claimed-owner envelope. It has no new cleanup identity, object inventory,
   capability URL or time-based retention setting. Use the normal authenticated PodHost Call
   wrapper and loopback RuntimePreparation forwarding.
3. Host verifies the claim and the exact declared operation/selection, then journals the
   terminal release decision in its existing preparation lifecycle before dropping any holds.
   It closes new file grants, prepares and checkpoint transfers for that operation, drains an
   in-flight preparation and forwards native release. The existing terminal record makes lost
   replies and repeated release requests idempotent and rejects late work. No new database or
   second mutable source identity is needed, but the native delete itself cannot be that fence.
4. Runtime serializes explicit release with source preparation and calls TensorFS's existing
   release method. Ordinary Worker shutdown releases process-local readers/executors and
   leaves durable source operations intact. A replacement process needs no in-memory operation
   roster to recover native progress or honor an owner release forwarded by Host.
5. Explicit owner teardown removes source progress only after a successful native release
   verdict. Model output roots and their inherited objects retain their existing independent
   adoption/publication lifetime; source release does not dispose produced checkpoints.

## Required proof before enabling the behavior

- Same-Store SIGKILL and graceful replacement both preserve the exact head with no raw source
  and no owner ACK; the real converter reads no completed tensor bytes again.
- Explicit owner release removes operation holds/journal, replays after lost reply, and rejects
  a changed selection, foreign owner and late prepare/transfer. Race a queued prepare with
  release and prove no work recreates the released operation.
- Source release waits for logical completion/cancellation rather than a transient failed
  attempt; a retained producer retry still opens its exact source manifests.
- Confirm native GC no longer retains the released source journal, while adopted output
  manifests remain readable. Use existing filesystem roots and GC; no time-based cleanup.

This is a future coordinated owner/Host/Runtime change. It changes neither the currently
running H3 wire nor the published Runtime/TensorFS pair and proposes no release tag.
