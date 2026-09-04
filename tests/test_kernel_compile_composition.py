"""cr-120: the fused CUDA kernel and a compiled graph COMPOSE, through the custom operator.

The kernel's nanobind entry point is opaque to Dynamo twice over — the module-level
`import_module` is on Dynamo's skip list, and the `nanobind` callable itself has no meta
implementation — so a compiled region that reaches it graph-breaks, and under
`fullgraph=True` refuses outright. Measured on the shipping Anima plan at two blocks:
9 breaks / 10 graphs through the raw entry point, 0 breaks / 1 graph through the operator.

`cozy_runtime._kernels.ops` is the fix and this file is its proof. Nothing here argues that
compilation should SHIP: the artifacts are never reused across pods (that was tried and it
never worked). What it argues is narrower and load-bearing — a compile-time discovery
oracle cannot study the plan we actually run if the plan's own kernel is where tracing
stops.
"""

from __future__ import annotations

from typing import Any

import pytest


def _why_not() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment probe
        return f"torch must be importable: {exc}"
    if not torch.cuda.is_available():
        return "the kernel is CUDA-only"
    try:
        import cozy_runtime._kernels._C  # noqa: F401
    except ImportError:
        return "this wheel carries no compiled kernels"
    return ""


SKIP = _why_not()
pytestmark = pytest.mark.skipif(bool(SKIP), reason=SKIP or "")

HEADS, HEAD_DIM, TOKENS = 16, 128, 512
EPSILON = 1e-5


def _inputs() -> tuple[Any, ...]:
    """The call site's real layout: a `(1, S, H*D)` projection, unflattened and transposed,
    so Q and K arrive NON-contiguous and `empty_like` carries that stride to the output."""
    import torch

    generator = torch.Generator(device="cuda").manual_seed(0)

    def projection() -> Any:
        flat: Any = torch.randn(
            1,
            TOKENS,
            HEADS * HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        return flat.unflatten(2, (HEADS, -1)).transpose(1, 2)

    cos = torch.randn(TOKENS, HEAD_DIM // 2, device="cuda", generator=generator)
    sin = torch.randn(TOKENS, HEAD_DIM // 2, device="cuda", generator=generator)
    return (
        projection(),
        projection(),
        torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, TOKENS, HEAD_DIM // 2, 2, 2),
        torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=generator),
        torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=generator),
    )


def test_the_operator_answers_exactly_what_the_kernel_computes() -> None:
    """Same bytes as the raw nanobind call — the operator adds a schema, not arithmetic."""
    import torch

    from cozy_runtime._kernels import _C
    from cozy_runtime._kernels.ops import operator

    rms_rope_split_half = operator()

    query, key, freqs, q_scale, k_scale = _inputs()
    with torch.no_grad():
        op_q, op_k = rms_rope_split_half(query, key, freqs, q_scale, k_scale, EPSILON, 0)
        raw_q, raw_k = torch.empty_like(query), torch.empty_like(key)
        _C.rms_rope_split_half(
            query,
            key,
            freqs,
            q_scale,
            k_scale,
            raw_q,
            raw_k,
            EPSILON,
            torch.cuda.current_stream(query.device).cuda_stream,
            0,
        )

    assert torch.equal(op_q, raw_q)
    assert torch.equal(op_k, raw_k)


def test_the_fake_answers_the_strides_the_kernel_actually_writes() -> None:
    """Not decoration: Q and K are the transpose of a projection, so a fake that claimed
    contiguous outputs would make Inductor plan a layout the kernel never produces, and
    the copy it inserted to repair that would eat the fusion the kernel exists for."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from cozy_runtime._kernels.ops import operator

    rms_rope_split_half = operator()

    args = _inputs()
    with torch.no_grad():
        real_q, real_k = rms_rope_split_half(*args, EPSILON, 0)
    with FakeTensorMode() as mode:
        fake_q, fake_k = rms_rope_split_half(*(mode.from_tensor(t) for t in args), EPSILON, 0)

    for fake, real in ((fake_q, real_q), (fake_k, real_k)):
        assert fake.shape == real.shape
        assert fake.stride() == real.stride()
        assert fake.dtype == real.dtype


def test_opcheck_accepts_the_operator() -> None:
    """Schema, fake, autograd registration and the AOT-dispatch path, from torch itself."""
    import torch

    from cozy_runtime._kernels.ops import operator

    operator()

    torch.library.opcheck(
        torch.ops.cozy_runtime.rms_rope_split_half,
        (*_inputs(), EPSILON, 0),
        test_utils=(
            "test_schema",
            "test_faketensor",
            "test_autograd_registration",
            "test_aot_dispatch_dynamic",
        ),
    )


def test_a_fullgraph_region_containing_the_kernel_compiles_and_matches_eager() -> None:
    """The claim in one arm: real work on both sides of the kernel, `fullgraph=True`, and
    the operator SURVIVES into the compiled graph as a node rather than being traced
    through or broken out."""
    import torch
    from torch import nn

    from cozy_runtime._kernels.ops import operator

    rms_rope_split_half = operator()

    class Region(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(
                HEADS * HEAD_DIM, HEADS * HEAD_DIM, bias=False, device="cuda", dtype=torch.bfloat16
            )
            self.out = nn.Linear(
                HEADS * HEAD_DIM, HEADS * HEAD_DIM, bias=False, device="cuda", dtype=torch.bfloat16
            )

        def forward(self, hidden: Any, key: Any, freqs: Any, q_scale: Any, k_scale: Any) -> Any:
            query = self.proj(hidden).unflatten(2, (HEADS, -1)).transpose(1, 2)
            query, key = rms_rope_split_half(query, key, freqs, q_scale, k_scale, EPSILON, 0)
            attention = torch.nn.functional.scaled_dot_product_attention(query, key, key)
            return self.out(attention.transpose(1, 2).flatten(2, 3))

    _, key, freqs, q_scale, k_scale = _inputs()
    hidden = torch.randn(1, TOKENS, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    call = (hidden, key, freqs, q_scale, k_scale)
    region = Region().eval()

    with torch.no_grad():
        explanation = torch._dynamo.explain(region)(*call)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    assert explanation.graph_count == 1

    captured: list[Any] = []

    def backend(graph: Any, _inputs: Any) -> Any:
        captured.append(graph)
        return graph.forward

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(region, backend=backend, fullgraph=True)(*call)
    assert [
        node
        for node in captured[0].graph.nodes
        if node.op == "call_function" and "rms_rope_split_half" in str(node.target)
    ]

    torch._dynamo.reset()
    with torch.no_grad():
        eager = region(*call)
        compiled = torch.compile(region, fullgraph=True)(*call)
    assert torch.equal(eager, compiled)


def test_the_raw_entry_point_is_what_breaks_the_graph() -> None:
    """The RED half: the same kernel reached the way the shipping plan reaches it today."""
    import torch

    from cozy_runtime._kernels import rms_rope_split_half as raw

    args = _inputs()

    def region(q: Any, k: Any) -> Any:
        q, k = raw(q, k, args[2], args[3], args[4], EPSILON)
        return q + k

    torch._dynamo.reset()
    with torch.no_grad():
        explanation = torch._dynamo.explain(region)(args[0], args[1])
    assert explanation.graph_break_count >= 1
    assert any("import_module" in reason.reason for reason in explanation.break_reasons)

    torch._dynamo.reset()
    with pytest.raises(Exception, match="raph"), torch.no_grad():
        torch.compile(region, fullgraph=True)(args[0], args[1])
