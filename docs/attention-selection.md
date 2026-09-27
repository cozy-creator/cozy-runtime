# Attention selection

Runtime owns backend integration and preparation-time selection. Its current
curated default keeps floating-point attention: FA3 BF16/FP16 when the installed
artifact and site support it, otherwise PyTorch SDPA. The table in
`internal/attention.py` links retained H3 timing evidence. It is not a universal
hardware optimum: a numerical probe proves the backend computes attention at a
production length, not that its videos retain quality. Approximate backends remain explicit choices.

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
checkpoint weights do not imply FP8 attention. Eligibility checks site/hardware,
installed dependency metadata and context-parallel support; selecting an optional
backend then resolves and checks that implementation. Nothing benchmarks every
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

Validation includes actual Diffusers CPU processors and dispatch, metadata-only
FA3 installation controls, model hook/scoping/rollback tests and architecture
fences. No GPU performance, model quality, worker deployment or CLI end-to-end
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

Legacy Diffusers processors such as `AttnProcessor2_0` call SDPA directly and have
no per-site backend selection contract. Runtime leaves them untouched and does not
report an installed backend for them. Model policies receive an empty `eligible`
and `recommended` context explaining this limitation. Explicit matching pins or
policy choices refuse, including components mixing managed and legacy processors;
Runtime never creates an unused processor attribute to claim partial coverage.

## Kernel site, Ulysses and probes

Kernels come from the package environment or the worker image's Runtime-owned site,
never from a package lock written for them. The image bakes `/opt/cozy/kernels/`:
the importable `site/`, the FA3 `kernels` snapshot `flash-attn3/` and the `manifest.json`
its kernel gate writes (`site`, `torch`, `cuda`, `python_abi`, per-kernel facts). At process entry the
executor appends `site/` after the package's own packages (`site.addsitedir`; the package
wins on conflict) only when its torch release, CUDA build and Python ABI equal the
manifest's. Otherwise every site kernel is typed-absent with the reason.

Every kernel is a Runtime-local forward (`attention_upstream`, `attention_sol`) that
`attention_ulysses` registers with Diffusers. Without a parallel config it calls the
kernel; under Runtime's Ulysses config it runs Diffusers' unequal-length all-to-all,
the kernel over the full sequence of this rank's heads, and the exchange back. A kernel
whose reductions stay within a head therefore matches one GPU; FlashInfer's global V
amax does not, so it is registered without context parallelism. Ring is refused.

Each rank probes the function it serves at `(1, 73439, 2, 128)` — one H3 rank's 10 s
document, where length-dependent error shows — against exact float64 attention over
64 sampled rows. Non-finite output fails first; then rel L2 against the kernel's
declared budget: past 1 it is `attention_kernel_wrong`, otherwise
`attention_kernel_out_of_budget` (FA3 FP8 at H3 length). `attention.capabilities`
is the per-rank document of every kernel: status, version/artifact and probe.

`cozy-runtime attention-qualify --degree 4 --out report.json` spawns one process per
GPU and, per kernel, runs the registered backend at degree 4 on identical global
`[1, 73439, 56, 128]` inputs against the same backend at degree 1 (byte equality and
rel L2 ≤ 2e-3). `--device cpu` runs the same path over gloo.

Select with the existing `kernel.attention=dit=flash-attn3` or
`--attention-kernel=dit=flash-attn3` Creator argument. Eager requests restore the
prepared selection after success or failure, so an unpinned A / pinned B / unpinned
A sequence shares its prepared executor. Compiled or context-parallel changes
require preparing a new model generation whose `choose_attention` policy selects
that backend before warm-up; the request refuses a frozen change and does not
silently rebuild it. Changing kernel bytes is a new worker image.

Preparation facts and request `attention.artifact` observations identify the installed
distribution/version and SHA256 of its RECORD, or the snapshot variant and metadata
digest. Each rank's execution record carries `attention.impl`: the implementation that
actually served, such as the SDPA backend torch dispatched (`cudnn` on H100) or Sol's
`cute_sm90` route.
