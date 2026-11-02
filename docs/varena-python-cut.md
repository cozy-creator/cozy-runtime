# One public Varena Python surface

The public `tensorfs.varena` root exports the canonical `api.py` types. Old root
Arena/Cache/cooperation/source/graph aliases are removed. Runtime explicitly imports
`tensorfs.varena._runtime` for its source bindings, graph preparation and weak owner
contracts; `_cooperative` remains private. The old raw Cache wrapper is deleted;
native Cache-binding fixtures address `_ext._VarenaCache` directly and remain
internal proofs. The thin Torch adapter stays `tensorfs.varena.torch`.

Public `Gpu.use`, `Gpu.alloc` and `CacheView.release` require the actual stream.
Python examples retain nonblocking owned streams and prove their own completion.
The mixed C/Rust helper argument changes are coordinated with the native owner:
`mixed_c_alloc(gpu, bytes, stream, out)`, `mixed_rust_gpu(stream)`, and
`mixed_rust_retry_prepare(stream)`. Native helpers/ABI are a separate checkpoint.

Current Runtime no longer silently takes older-release Varena signatures. Linux
cooperation and no-store builds remain explicit platform/build capabilities. The
existing generic Room/sizing paths remain live in their unmigrated consumers; this
namespace change is not their retirement.

Canonical `Exhausted` is terminal (`sticky=False`, `varena_no_retry=True`) and is
classified as `Fault.EXHAUSTED`. The real CPU Request/cut proof invoked the body
once and propagated the identical exception. Namespace identity and required
stream signatures also passed. Actual executor Hello/Shutdown passed without
importing Torch or loading CUDA/NVML. Those host checks used the preserved P4 6ab
shared library under a task-owned current-Python overlay, not the pending composed
native SDK. Hello's distribution metadata came from its tooling venv and is not
an installed-cohort claim. No CUDA/GPU execution was run.

The private Arena retirement facade is copied exactly from native-owner commit
`e8da50e9aba9203192b2e82ce3933b5a7965b85d` (AST comparison passes), so the composed
SDK must include that native/stub successor. The portable
`release-gate/suites/varena/public-api/python_cut.py` additionally requires this
current native interface and awaits the composed artifact.

Fresh masters advanced during this work: Creator base `0888d1026` plus P4
`956f2b6fc`; Runtime base `6ac83520a` (0.37.1/shot contract) plus frozen `a09601d3`.
Both merges were clean. Runtime `kernels` and `_kernels` trees are byte-identical
to a096; all numerical implementation changes come from its preserved parent.
Creator native Rust and shim inputs are unchanged by this Python lane.

Source parsing and targeted public Python Ruff pass. Runtime Mypy reports the same
24 findings as its clean composition baseline, with no added errors; Ruff likewise
adds no findings. Exact comparisons and source receipts are in
`research/varena-python-cut/`. Original failed fixture import receipt is preserved
in `outputs/xs138-p4-python-cut/classifier-001-import-error.log`; it was an unused
checker import, not a product change or GPU failure.

Remaining gates: compose the final native Source/retirement/shared-service changes,
migrate their newly introduced private fixture imports, run the host gate against
that exact installed artifact, and run the real public mixed/retirement and Runtime
consumer GPU cohorts. Historical frozen sources and artifacts remain unchanged.
