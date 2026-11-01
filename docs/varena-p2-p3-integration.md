# Varena P2/P3 Runtime integration

This new source branch composes the frozen committed execution, sparse inference,
normalization/output ownership, Cache owner, native CUDA recovery and v3 cooperation
lanes. It does not promote the independent artifacts or replace their original evidence.

Base origin/master: `cb6c66bf343d15637eaeb4af35731a37fc56d61c`.

Planned frozen parents:
- `fcdebb8b4ece25d51041d1f40c0a0920431492f6`
- `167ffc828e3f84062d43641fcf53dbda54892631`
- `3043db98bb44472ddea03ed7c341b12d232d1569`
- `2626e3f48bc147f1f3d46b19bd385356e628680c`
- `8384bfd3d74f10d2c57800bb1a8ccb55aba46984`
- `6c8122412bf502f2913c35eaba6cf9d263d314b5`


One writer: /root/core_restart. Root owns Creator integration6c4f0e53a, packaging,
installation and all GPU execution. Source composition and native/kernel identity checks
precede any build. Actual combined qualification remains pending.

## Composition result

All six parents are present. The merge checkpoints were30b079f5a (fcde), c0ac699e2
(167),89519284f (3043),f8b560153 (2626),ac2a7b034 (8384) and99fb80ed5 (6c812).
The only conflict was `_prepare_native`: keep the corrected8384 direct native
`prepare(reclaim=reclaim)` call, removing the superseded Python victim/retry path.
The committed producer's exact source-hook ownership remains in Model/_paired/_hooks.

Thirty-six byte/AST/tree checks pass. Complete execution consumers and author APIs match
their frozen owners; Cache lifecycle/Loop and v3 request/cooperation methods retain their
owner versions; the CUDA recovery methods match8384. C++ sources, CMake/modules/toolchain
inputs, native interface stubs and Triton source are identical to both3043 and the ebc
native-build base. This supports native-file reuse but is not a newly packaged wheel claim.

The source contract audit against Creator6c4f0e53a confirms the owned Execution binding,
CacheOwner protocol1, cooperation owner protocol1/pressure protocol1, current Plan-generation
forwarding, clear_streaming_limit and native prepare(reclaim=...). Mandatory CacheOwner
refusal remains before native limits/Arena creation; an old a96 SDK demonstrably refuses
that CUDA path with CUDA uninitialized. Cooperative request activity includes preparation.
Native generation and reference/pin lifetimes remain distinct from Python source identity.

Source syntax and targeted mypy for six affected owner/execution modules pass. A real CPU
TensorFS checkpoint → two-layer forward → reclaim → refill retains the exact output and
closes its Model. No CUDA initialization, native build, GPU action or primary-checkout edit
was performed. The first mypy invocation used the old primary SDK source path and failed
on missing capabilities; its log is preserved, and the corrected check uses the actual
6c4 integration source. The requested workspace-git audit script/policy file is absent
from the current tracker; no workspace cleanup was attempted.

Receipts are under `research/varena-p2-p3-integration/`; detailed merge logs and rerunnable
checks are in `outputs/xs138-runtime-p2-p3-integration/`. All original branches, controls,
wheel/source cohorts and failed receipts remain intact. The next concrete consumer work
is described in [the remaining-policy inventory](varena-remaining-policy.md). Combined
wheel/ABI and actual CUDA consumer/owner/capture/replay/pressure qualification are pending.
