# Four-GPU Ulysses correctness probe — preparation checkpoint

This research package is **not GPU-qualified**. Four-process CPU Gloo and ordinary CLI
lock/capture/describe checks have passed. The exact same publicRuntime0.23.5/TensorFS0.6.1
capture environment also passed the four-process CPU exchange. Paid execution requires
the root owner’s explicit rental and run authorization.

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

Execution prerequisites and retained limits:

- ModelMixin/empty split-plan installation, explicit group binding and ten-tensor census
  passed the CPU gate; the CLI interface declares only the dit component and exact4GPU rung.
- CPU permutation gate passed on four processes at129/132tokens, using Runtime CpComms
  and the actual unequal Diffusers QKV/O helpers. This does not qualify CUDA/full-shape attention.
- Independent source reviews cover bounded memory, callbacks and failure propagation;
  the actual report validates four distinct UUIDs/PIDs and all rank identities.
- The invocation must pin `model.model=paul/attention-kernel-benchmark-20260916#sha256:61dc0e651cf6ab7744213611b6ee1cb4a4ba0b8dac5da8f165d950c2c91a04ca`.
  Exact full geometry and seed remain unchanged; an8-minute caller deadline bounds the run.
- Root grants any rental and the exclusive GPU invocation. No smaller replacement shape.

The earlier video/layout package is preserved separately and is not part of this request.
