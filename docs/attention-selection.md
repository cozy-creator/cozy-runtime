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
explicit pin. Per-request pins still refuse compiled or context-parallel frozen
constructions. They do not silently substitute a kernel or recompile the model.
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
names, not a claim of a GPU profiler trace. Sol additionally records its resolved
dense reference and request-local sparse, dense-step, dense-path and dense-prefix
call counts. A Turbo request whose policy stays dense is therefore not evidence
of sparse acceleration. Request-owned Sol hooks are removed even if applying or
executing an override fails; prepared Sol hooks remain owned by the generation.

Legacy Diffusers processors such as `AttnProcessor2_0` call SDPA directly and have
no per-site backend selection contract. Runtime leaves them untouched and does not
report an installed backend for them. Model policies receive an empty `eligible`
and `recommended` context explaining this limitation. Explicit matching pins or
policy choices refuse, including components mixing managed and legacy processors;
Runtime never creates an unused processor attribute to claim partial coverage.
