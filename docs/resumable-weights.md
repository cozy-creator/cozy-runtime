# Resumable weights

Runtime 0.2.26 uses TensorFS 0.3.12 and the coordinated wire 38 checkpoint contract.
Clients probe the actual Runtime compatibility range before Claim or preparation; wire 37
cannot enter this changed layout. A local native commit or checkpoint ACK is not remote custody.

The Runtime parent authors the work fingerprint once from the selected job/environment,
payload and exact input/output identities plus observed protected framework/interpreter facts.
CUDA architecture and driver deliberately participate because AdaLN arithmetic is not presumed
bit-identical on another architecture. Deadlines, publication grants, paths, machine UUIDs,
memory measurements, boot IDs and attempt/owner counters are excluded. The existing transaction
ID remains scoped to the exact original owner/request/InvocationSpec/output; a matching work
fingerprint does not authorize cross-request reuse. Host and TensorFS carry this fingerprint
unchanged in the native-authored canonical declaration.

The parent owns native writer handles. The executor uses the existing bounded attempt spool
for parts and completed-role lists; control messages carry bounded metadata. A checkpoint
observation comes directly from the parent writer. Host records it before ACK and Creator
independently banks the Link closure before advancing its remote prefix. On replacement,
Creator restores source custody through existing acquisition and restores derived work under
the current intent. Native read-only validation precedes Ready and the new writer ACK; begin
imports completed work without importing the former writer's authority.

Quantization checks numerical acceptance before adding any role. A complete encoding group
can be skipped; an incomplete group is recomputed and identical accepted roles re-add
idempotently. Reused groups report their count and an unknown aggregate quality (`None`),
not a newly measured zero. Source reads and new bytes count work performed in this run.
Completed pending/adopted receipts replay only after native retained-closure validation.
Released receipt metadata is retained for audit and cannot establish reusable payload custody.
Explicit owner abandonment/disposal retains its terminal meaning and may release committed
outputs; checkpoint recovery does not reverse that decision.
