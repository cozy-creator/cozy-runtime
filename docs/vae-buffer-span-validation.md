# VAE edge buffer overlap validation

Distinct Torch Storage objects can describe overlapping bytes. The real CPU client
constructs two frombuffer tensors over the same bytearray: `_is_alias_of` is false,
data pointers match, and writes through one change the other. Actual CPU DLPack
exports also reproduce the distinct-Storage/same-address case.

Post-quant and output-tail binding now reject either Storage aliases or overlapping
actual byte intervals, using each accepted contiguous tensor's pointer and payload
size. The local checks require contiguous same-device operands and do not construct
bounds for noncontiguous tensors. Existing metadata/layout/policy/source checks and
native Plan validation remain unchanged. No allocator, admission or native policy
was changed. Prefix already performs the same interval check after its contiguous
same-device binding validation; its frozen source is unchanged by this follow-up.

Three concrete failure sequences are covered by the prepared real CUDA fixture:

- Post-quant: a duplicate DLPack view of the FP16 input is supplied as the NCDHW
  result. Its separate Storage identity could hide an overwrite of the caller input.
- Output tail: a duplicate DLPack export of the full projected rows supplies the
  pixel result's contiguous prefix. Pixel assembly could overwrite projection rows
  before they have all been consumed.
- Prefix: duplicate CUDA exports supply the same span for cosine and sine. Sine
  construction could overwrite the returned cosine table.

The new ordinary-child fixture queries the real prepared provider, admits the actual
inventory, creates each real CUDA duplicate export, and requires bind to reject it
before the consumer's execute method is called. It verifies distinct Storage identity,
matching pointers, unchanged bytes after the attempted bind and final cleanup. Its
sentinel fills are fixture initialization, not VAE operation writes. The three GPU
cells and same-child Request cleanup are **unrun**.

The committed CPU proof covers all three span predicates, same/partial overlap,
adjacent independent spans, ordinary Storage aliases and actual CPU DLPack. It is
not GPU binding, asynchronous lifetime, raw Source or full-model qualification.
The frozen post-quant, tail, prefix and root worktrees remain unchanged; this fix is
in a separate durable branch for root to compose before paired qualification.
