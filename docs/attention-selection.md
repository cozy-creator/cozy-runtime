# Attention selection

Runtime owns backend integration and preparation-time selection. Its current
curated default keeps floating-point attention: FA3 BF16/FP16 when the installed
artifact and site support it, otherwise PyTorch SDPA. The table in
`internal/attention.py` links retained H3 timing evidence. It is not a universal
hardware optimum: a small numerical smoke test proves the backend executes, not
that its videos retain quality. Approximate backends remain explicit choices.

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

## Optional wheels on an existing worker

Optional kernels are ordinary dependencies of the Python package's complete,
locked environment. Add a prebuilt wheel to that package and prepare its new
immutable generation on the same worker. Existing uv cache/hardlink reuse applies;
changing this closure replaces the affected executor generation, not the worker
image or Runtime service. Never install into a running generation or the worker's
system Python. No request runs pip, a shell, or a downloader.

SageAttention, Sol, Kitchen and FlashInfer use their existing distributions.
FA3/FA4 kernel-builder snapshots can be packaged without compiling or downloading:

```sh
uv run --no-project --with packaging python scripts/build-attention-wheel.py ./snapshot \
  --kernel flash-attn3 --version 0.1.0 \
  --revision FULL_UPSTREAM_COMMIT --source-url https://huggingface.co/kernels-community/flash-attn3 \
  --license BSD-3-Clause --requirement 'torch==EXACT_BUILD_VERSION' \
  --requirement 'kernels==EXACT_LOADER_VERSION' --out dist
```

Use the real immutable upstream revision, license and exact build dependency
versions; include every additional upstream dependency with `--requirement`.
The input must already contain `build/<variant>/metadata.json`, native/source
payload and upstream licenses. The wrapper conservatively tags CPython 3.12 on
Linux x86-64; a Torch extension is not advertised as abi3. Version each candidate
and lock the resulting wheel digest through the ordinary package dependency path.

Runtime discovers `cozy-kernel-flash-attn3` or `cozy-kernel-flash-attn4` from the
active environment's installed distribution metadata. The installed wheel takes
precedence over `/opt/cozy/kernels`; a broken installed candidate refuses rather
than loading different baked bytes. `kernels.get_local_kernel` still selects the
Torch/CUDA variant. Native architecture metadata and numerical admission still
apply. The wheel packaging step does not establish ABI compatibility, numerical
correctness or model quality. A JIT source kernel can still compile its device
code during normal upstream preparation.

Select with the existing `kernel.attention=dit=flash-attn3` or
`--attention-kernel=dit=flash-attn3` Creator argument. Eager requests restore the
prepared selection after success or failure, so an unpinned A / pinned B / unpinned
A sequence shares its prepared executor. Compiled or context-parallel changes
require preparing a new model generation whose `choose_attention` policy selects
that backend before warm-up; the request refuses a frozen change and does not
silently rebuild it. Changing kernel wheel bytes likewise requires a new ordinary
dependency generation, allowing rollback by selecting the prior package revision.

Preparation facts and request `attention.artifact` observations identify installed
distribution/version, SHA256 of its installed RECORD, immutable source URL when
provided, and the original wheel SHA256 when the installer retains it in standard
`direct_url.json`. The request's captured dependency closure remains authoritative
for exact bytes when that optional origin field is absent. An installed RECORD
digest is an inventory identity, not a signature or an additional integrity scan.
