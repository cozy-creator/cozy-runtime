# Four-GPU Ulysses correctness probe — preparation checkpoint

This research package is **not GPU-qualified or ready to rent**. CPU integration tests,
normal CLI lock/capture/describe and independent review remain pending.

It declares exactly four RTX 5090 GPUs through one Model rung and sequence_parallel(4),
using the existing 133KB diagnostic checkpoint. Runtime mirrors the component forward and
installs its explicit process group. The component generates fixed BF16 [1,109225,56,128]
Q/K/V, takes unequal token shards [27307,27306,27306,27306], and calls the actual public
Runtime attention_ulysses.exchange. An identity callback checks QKV head ownership and O
inverse redistribution before attention. Full-head computation is a reference control;
partitioned attention receives only each rank's14heads. Repeated exchanges, intermediate
checks, candidate-versus-public differences, and three warm regression timing samples are
recorded separately. No videos, H3 weights, attention-default changes, or promotion.

The research-local fixed_means.py is copied unchanged from Runtime source checkpoint
2f0a2ca7. Its input addressing uses int64 before stride multiplication. The full packed
input spans2,348,774,400elements, exceeding signed int32. The public Runtime remains0.23.5;
this package neither installs a private Runtime nor monkeypatches upstream modules.

Before execution:

- Verify ModelMixin/mirroring/empty split-plan integration through the ordinary CLI.
- Exercise actual unequal Diffusers exchange over CPU Gloo with deterministic elementwise
  data, including inverse and rank coverage. This proves permutation only, not CUDA attention.
- Complete bounded memory/source/callback peer review; preserve all four actual GPU UUIDs.
- Capture the package and pin the exact diagnostic checkpoint through normal CLI.
- Root grants any rental and the exclusive GPU invocation. No smaller replacement shape.

The earlier video/layout package is preserved separately and is not part of this request.
