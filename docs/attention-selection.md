# Attention selection

Runtime owns backend integration and preparation-time selection. Its current
curated default keeps floating-point attention: FA3 BF16/FP16 when this machine has
compiled it for the card and the site supports it, otherwise PyTorch SDPA. The table in
`internal/attention.py` links retained H3 timing evidence. It is not a universal
hardware optimum, and no accuracy check gates a compiled kernel. Approximate backends
remain explicit choices.

A model may choose a different eligible implementation before warm-up:

```python
from cozy_runtime.author import AttentionContext, Model


class VideoModel(Model[object]):
    def choose_attention(self, context: AttentionContext) -> str | None:
        if context.component == "dit" and "kitchen-int8" in context.eligible:
            return "kitchen-int8"  # This model's separately qualified quality policy.
        return None  # Keep Runtime's recommendation for other sites.
```

The immutable context contains component and module path, device kind/name/SM,
attention activation dtype, head dimension, parallel degree, eligible names and
the recommendation's evidence/reason. Activation dtype comes from active projection
output contracts (or ordinary floating Linear inference without external autocast),
never unrelated normalization parameters or encoded storage dtype. Unknown or
disagreeing projection contracts admit only a dtype-unbounded backend.
It has no tensors or device methods. FP8
checkpoint weights do not imply FP8 attention. Eligibility checks site/hardware and
context-parallel support; selecting an optional backend then resolves it: compiled,
imported and launched once. Nothing benchmarks every
request or promotes approximate attention automatically.

Returning an unknown or ineligible backend refuses preparation; it does not fall
back. Returning `None` keeps the recommendation. The entire site's selection set
is validated before processor fields change. The hook does not replace model
`load`/`warm` lifecycle or introduce an invocable model method.

The existing developer attention override string supports:

- `sageattention`: force every attention component, retaining the existing global
  pin meaning and refusing any unsupported site.
- `fl2va_dit=sageattention`: force only that component, retaining other prepared
  model choices (including auxiliary VAEs).
- `base_model/fl2va_dit=sageattention`: scope to a model parameter in a multi-model
  callable. An unqualified component scope matches that component in each model.

Unknown scopes refuse. A package policy cannot override a developer's matching
explicit pin. Per-request backend changes still refuse compiled or context-parallel
frozen constructions. A group may repeat its prepared pin: every matching live
processor and its saved preparation must already select that backend. This is a
verified no-op, with no kernel search or mutation. Empty, unknown, mixed or changed
selections do not qualify. Remote request pins do not themselves choose a group's
preparation backend; use the model's preparation policy for that selection. No
request silently substitutes a kernel or recompiles the model.
Prepared selections are saved and restored across all model executors before and
after request execution, including handler failures. An ordinary request therefore
retains its prepared model policy, without repeating resolution or a previous
request's experimental choice.

`kitchen-int8` and `flashinfer-bf16-fp8` are newly registered explicit experimental
choices. See [upstream-attention.md](upstream-attention.md) for their actual limits.
The shared Diffusers registration bridge changes no global active backend and
refuses collisions with another implementation.

Validation includes actual Diffusers CPU processors and dispatch, model
hook/scoping/rollback tests and architecture fences. No GPU performance, model quality, worker deployment or CLI end-to-end
qualification is established by these tests.

Request observations record selected and restored backend names and per-component
site counts. `ExecutionObservation.kernel_symbol` carries the selected backend
names, not a claim of a GPU profiler trace. Sol additionally records request-local
sparse, dense-step, dense-path and dense-prefix call counts, tagged with rank and
world size. The leader records its resolved dense reference. Follower counts travel
on existing component-call replies and include only successful acknowledgements;
an interrupted call is not fabricated as complete. A Turbo request whose policy stays dense is therefore not evidence
of sparse acceleration. Request-owned Sol hooks are removed even if applying or
executing an override fails; prepared Sol hooks remain owned by the generation.

Diffusers' classic `AttnProcessor2_0` (SDXL's UNet, AutoencoderKL) calls SDPA
directly. Selection first replaces it with `ClassicSdpa`, the same math through
backend dispatch: auto keeps SDPA there (bit-identical), and pins and model policies
may choose another kernel. Other legacy processors have no per-site backend selection
contract. Runtime leaves them untouched and does not report an installed backend for
them. Model policies receive an empty `eligible`
and `recommended` context explaining this limitation. Explicit matching pins or
policy choices refuse, including components mixing managed and legacy processors;
Runtime never creates an unused processor attribute to claim partial coverage.

## Kernels compiled on the machine, and Ulysses

Nothing is compiled on a builder. The worker image carries every kernel as pinned source
(`python -m cozy_runtime.internal.kernel_sources fetch /opt/cozy/kernel-sources`: sources
by commit or version and sha256, plus `index.json`) and the CUDA toolkit. At worker boot,
before any executor starts, the worker compiles the installed models' ladder for each card
(`internal/machine_kernels.py`: Sol's Python sources, SageAttention2, FA3 on sm90), best
first, through the compile manager (`internal/kernel_compile.py`): one niced build per exact
key (sources, arch, torch, CUDA, Python, compiler) per machine, into the worker's store
namespace that every executor reads. An extension build runs as many compiler processes as
the machine's CPUs (affinity, cgroup quota) and memory (4 GiB each) allow, one such build at
a time. An executor's warm submits what its model prefers too; a key the worker holds is
compiling, never built twice. Upstream's own `setup.py` detects the card; no source is
patched. A build is ended only when its whole process tree stops burning CPU and writing
output; a failure is recorded for that worker boot and retried at the next.

An executor is the Runtime its package locked. One before 0.18.70 compiles nothing and reads
only the image kernel site, `/opt/cozy/kernels` (`internal/kernel_site.py`): the worker
writes its manifest at boot and links each ready tree into it, and such an executor takes a
kernel at its next construction after it lands, through its own probe. It never gets Sol off
SM90 or the fused glue; relock the package for those.

Selection serves a kernel only once it is compiled, imports and launches. One still
compiling is skipped with its progress (`sageattention: compiling (42%)`), and the chain
falls through to the next ready kernel; SDPA is always ready. A request never compiles
and never waits: at its start, a construction that skipped a kernel now ready selects
again. Sol compiles one object per card, except on SM90, where upstream specializes on
the length's 64-token block layout; the first sight of a length (intake or a dense step)
submits that object, and a sparse step whose object is not ready runs dense.

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
A sequence shares its prepared executor. Compiled or context-parallel changes
require preparing a new model generation whose `choose_attention` policy selects
that backend before warm-up; the request refuses a frozen change and does not
silently rebuild it.

Preparation facts and request `attention.artifact` observations identify the compiled
artifact by its store key. Each GPU's execution record carries `attention.impl`: the
implementation that actually served, such as the SDPA backend torch dispatched (`cudnn`
on H100) or Sol's `cute_sm90` route.
