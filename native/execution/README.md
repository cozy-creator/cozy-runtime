# Native execution

The Rust ABI3 extension `_kernels/_execution.abi3.so` runs recorded CUDA graphs with Python absent
from loop continuation. Its Python owner is `cozy_runtime.internal.prepared`; `author.denoise_loop`
(`cozy_runtime.internal.loop`) is the only thing that records and submits programs. Packages never
import either.

- **ResidentRepeat** clones one recorded graph into a conditional WHILE body. A device kernel
  advances the iteration index and decides whether the GPU repeats; the host launches once for any
  number of steps.
- **NativeSequence** advances ordered stages from one native worker. A half-open stage range can
  repeat (`repeat=(begin, end, passes)`) without Python between passes. A stage may own a TensorFS
  input: before its graph runs, the worker loads that block's raw (or stored-INT8) bytes at the
  block's own Varena addresses, and bounded lookahead prefetches the next reading stage.
- **Ownership.** Each definition retains its graph, pool, external tensors and storages, and its
  weight bindings. Every submission revalidates layouts, fill identities and the kernel epoch.
  Close cancels and joins before release; an unproven completion keeps every owner until process
  exit. Residency retires idle definitions before releasing their weights.
- **Cancellation** is a GPU atomic written from a control stream and observed after a complete body
  or stage; it is not preemption. Progress reads device counters through an ordered snapshot.
- **Graph contents.** Only kernel, empty, memcpy, memset and child-graph nodes are accepted;
  memcpy/memset become kernel nodes inside the conditional body. Host callbacks, allocations,
  events and nested conditionals are refused as capability refusals; real CUDA failures keep their
  driver codes.

Build with `scripts/build-execution-native.sh INSTALL_PREFIX` and `COZY_CUDA_TOOLKIT` pointing at a
CUDA 13 toolkit. The wheel carries eight native SASS targets, with no PTX and no serving JIT.
