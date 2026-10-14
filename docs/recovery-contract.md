# Recovery after running out of device memory

Owner, 2026-10-05: "I want processes to succeed and be resistant to changes, not fail."

- **An op that runs out of memory** makes room (idle tenants and weights leave, maps page to
  the host, the grain gets finer) and runs again. It starts again from what it read: the
  generator state of a seeded op and the content of the tensors it writes into are put back
  first (a pointwise op allocates nothing, so it fails before it writes and is not copied).
- **Last, the process's own tensors move to the host** (`internal/spill.py`): a package's
  caches, an earlier stage's outputs, a decoder's feature maps, largest first and the op's
  operands last. Each storage is copied to page-locked host memory and repointed there in
  place: every tensor over it stays a CUDA tensor, read over the bus, and nothing brings it
  back. At most half of what the host has free; the pinned tier's weights give way first.
  A stage's idle cache goes back to the driver when the stage closes.
- **An out-of-memory that escapes a block's ops** (a library's own allocation) makes room and
  runs the whole block again from the state it started with:
  - The first run of each block is observed. It records whether the forward writes into
    tensors that existed before it (its inputs, its buffers, anything held elsewhere) and
    which generators its seeded ops draw from.
  - A block that writes nothing outside itself then runs unobserved. Only its generators
    are kept, for a second run.
  - A block that writes runs observed. The first content of every tensor it writes into is
    kept, and put back before it runs again.
  - The lists and dicts it was handed (a decoder's feature cache and its index) are put back
    as they were too, and from then on its ops recover one by one, never the whole block.
- **Not seen:** a plain Python attribute a forward changes, such as a counter on a module.
  A replay advances it once more. A module where that matters is handled by name, never
  by refusing replays.
- **A call fails only when nothing can make progress:** no room can be made even at the
  finest grain with every tensor it does not read on the host. It reports the measured
  shortfall, with the state its blocks found put back, so the executor serves the next call.
  A block shared by a GPU group cannot run again alone; its ops still recover one by one.

Tests: `tests/test_block_replay_contract.py`, `tests/test_recovery_exhaustion.py` (CPU) and
the weight plane's GPU tests.
