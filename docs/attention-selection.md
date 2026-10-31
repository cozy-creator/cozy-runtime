# Attention selection

Runtime picks each attention site's kernel from the GPU's family, never from a card list
(`internal/arch.py` rows, precision-tiers §2.1 and §2.4):

1. **Level.** 16 when the request asks for 16 bits (`bits` on `PrepareRequest`/`Invoke`,
   served as the `exact` pin); otherwise the model's `attention_bits` for the site; 16 when
   the model declares none. A sparse site's exact paths (`AttentionLayout.exact_paths`, and every
   layer whose key RMSNorm gain peaks at 3x its RMS or more) take Kitchen INT8 where the route
   declares it, else 16.
2. **Kernel.** The first kernel of the family's cell at that level (`attention.DENSE`) that
   serves the site's dtype, head dimension and degree, is compiled, imports and launches;
   then each level above; then SDPA. A sparse site (one a package producer validates: its
   model supplies an `AttentionLayout`) tries the `attention.SPARSE` cells at its level and
   each above before any dense one, its dense rows on the level's dense cell. A producer is
   tried just before its base. A cell lists only kernels of its own precision
   (`Candidate.level`), checked where the tables are built.

```python
class H3(Model[object], attention_bits={"fl2va_dit": 4, "ref2va_dit": 4}): ...
class Klein(Model[object], attention_bits={"transformer.single_transformer_blocks.": 8}): ...
```

INT4 Q·Kᵀ on the int4 rows (Ampere, Ada) is in no cell until the owner rules on its quality
(10-31): `kitchen-int4qk` (Kitchen INT8 attention with Q and K in int4) and its Sol form
`kitchen-sol-int4qk` (dense steps int4-QK, sparse steps and exact paths Kitchen INT8) serve only
a pin, `--attention-kernel=[component=]kitchen-int4qk`.

`attention_bits` is 16, 8 or 4, or a mapping from component-qualified module prefixes to them
(the longest prefix wins, unlisted sites 16). It is the model's tolerance, independent of its
weight lane. A kernel skipped because it is absent, compiling or failed is one run warning
(`fallbacks`) and is selected again once ready; one the row or the site's shape cannot run is
not in the walk.

| level | mma.sync | wgmma | tcgen05 |
|---|---|---|---|
| 16 | SDPA | FA3 | cuDNN |
| 8 | kitchen-int8, Sage2 (fp8 rows) | Sage2 | FA4 fp8 |
| 4 | Sage3 fp4 (fp4 rows), else 8 | 8 | 8 |
| sparse 16 / 8 / 4 | Sol / Kitchen Sol / Kitchen fp4 Sol | Sol / Sol + fixed-Kmean Sage2 / 8 | Sol (its exact tiles at its dense rows' precision) / 16 / 16 |

A card no row names takes the newest family whose 8-bit op builds for its own target and says
so on every run (`arch.inherited`).

The existing developer attention override string supports:

- `sageattention`: force every attention component, retaining the existing global
  pin meaning and refusing any unsupported site.
- `fl2va_dit=sageattention`: force only that component, retaining other prepared
  model choices (including auxiliary VAEs).
- `base_model/fl2va_dit=sageattention`: scope to a model parameter in a multi-model
  callable. An unqualified component scope matches that component in each model.

Unknown scopes refuse. A pin bypasses the level rule, `bits` included. Per-request backend changes still refuse compiled or context-parallel
frozen constructions. A group may repeat its prepared pin: every matching live
processor and its saved preparation must already select that backend. This is a
verified no-op, with no kernel search or mutation. Empty, unknown, mixed or changed
selections do not qualify. Remote request pins do not themselves choose a group's
preparation backend. No request silently substitutes a kernel or recompiles the model.
Prepared selections are saved and restored across all model executors before and
after request execution, including handler failures. An ordinary request therefore
retains its prepared selection, without repeating resolution or a previous
request's experimental choice.

`kitchen-int8` and `flashinfer-bf16-fp8` are newly registered explicit experimental
choices. See [upstream-attention.md](upstream-attention.md) for their actual limits.
The shared Diffusers registration bridge changes no global active backend and
refuses collisions with another implementation.

Validation includes actual Diffusers CPU processors and dispatch, model
hook/scoping/rollback tests and architecture fences. No GPU performance, model quality, worker deployment or CLI end-to-end
qualification is established by these tests.

Request observations record selected and restored backend names and per-component
site counts. The reply's `kernel_symbol` carries the selected backend
names, not a claim of a GPU profiler trace. Sol additionally records request-local
sparse, dense-step, dense-path and dense-prefix call counts, tagged with rank and
world size. The leader records its resolved dense reference. Follower counts travel
on existing component-call replies and include only successful acknowledgements;
an interrupted call is not fabricated as complete. A Turbo request whose policy stays dense is therefore not evidence
of sparse acceleration. Request-owned Sol hooks are removed even if applying or
executing an override fails; prepared Sol hooks remain owned by the generation.

Diffusers' classic `AttnProcessor2_0` (SDXL's UNet, AutoencoderKL) calls SDPA
directly. Selection first replaces it with `ClassicSdpa`, the same math through
backend dispatch: auto keeps SDPA there (bit-identical), and pins may choose another
kernel. Other legacy processors have no per-site backend selection contract. Runtime
leaves them untouched and does not report an installed backend for them. Explicit
matching pins refuse, including components mixing managed and legacy processors;
Runtime never creates an unused processor attribute to claim partial coverage.

## Kernels compiled on the machine, and Ulysses

Nothing is compiled on a builder. The image carries every kernel as pinned source
(`python -m cozy_runtime.internal.kernel_sources fetch /opt/cozy/kernel-sources`: sources
by commit or version and sha256, plus `index.json`) and the CUDA toolkit. At machine start
and after each install, while weights download, the machine runs the kernel boot for each
installed generation under that generation's seal (`internal/machine_kernels.py`): for each
card, the first kernel of each level of its row's cells and the 8-bit op, then Sol's card
objects, through the compile manager (`internal/kernel_compile.py`). One niced
build per exact key (sources, arch, driver, torch, CUDA, Python, compiler) per machine, into
the store its executors read; a key already built is never built again, so a restart
compiles nothing. An extension build runs as many compiler processes as the machine's CPUs
(affinity, cgroup quota) and memory (4 GiB each) allow, one such build at a time. Upstream's
own `setup.py` detects the card; no source is patched. A build is ended only when its whole
process tree stops burning CPU and writing output; a failure is recorded for that machine
run and retried at the next.

Selection serves a kernel only once it is compiled, imports and launches. A site that will
run a kernel still compiling waits for it, judged by the build's measured progress, and the
request shows `compiling <kernel>`; it never runs another kernel in its place. One whose
compile failed is skipped with why, and the chain falls through; SDPA is always ready. A
kernel no site runs is not waited for. Sol compiles one object per card, except on
SM90, where upstream specializes on the length's 64-token block layout; the first sight of
a length (intake or a dense step) submits that object, and its first sparse step waits for
it.

Every kernel is a Runtime-local forward (`attention_upstream`, `attention_sol`) that
`attention_ulysses` registers with Diffusers. Without a parallel config it calls the
kernel; under Runtime's Ulysses config it runs Diffusers' unequal-length all-to-all,
the kernel over the full sequence of this GPU's heads, and the exchange back. A kernel
whose reductions stay within a head therefore matches one GPU; FlashInfer's global V
amax does not, so it is registered without context parallelism. Ring is refused.

Each GPU's execution record lists every kernel of its chains: `ready` (served or not),
`compiling` with progress, `failed`, `absent` or `unsupported`, each with why and its
compile time.

Select with the existing `kernel.attention=dit=flash-attn3` or
`--attention-kernel=dit=flash-attn3` Creator argument. Eager requests restore the
prepared selection after success or failure, so an unpinned A / pinned B / unpinned
A sequence shares its prepared executor. A compiled or context-parallel construction
keeps the kernel it was prepared with; the request refuses a frozen change and does
not silently rebuild it.

Preparation facts and request `attention.artifact` observations identify the compiled
artifact by its store key. Each GPU's execution record carries `attention.impl`: the
implementation that actually served, such as the SDPA backend torch dispatched (`cudnn`
on H100) or Sol's `cute_sm90` route.
