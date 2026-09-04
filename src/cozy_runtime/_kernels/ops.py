"""The kernels as PyTorch custom operators, so a compiled graph can contain them.

An opaque `nanobind` call is invisible to Dynamo, twice over: `rms_rope_split_half` reaches
the extension through `importlib.import_module`, which is on Dynamo's skip list, and the
native callable itself has no meta implementation. A compiled region that reaches it
graph-breaks, and under `fullgraph=True` refuses. Measured on the shipping Anima plan at two
blocks: 9 breaks / 10 graphs raw, 0 breaks / 1 graph through the operator.

`torch.library.custom_op` is the supported repair. The op becomes an ATen-level node whose
FAKE implementation answers shape, dtype AND STRIDE during tracing, while the real CUDA
launch happens only at runtime; Inductor then schedules around the op instead of stopping
at it. The stride half is load-bearing, not decoration: Q and K arrive as the
`(B, H, S, D)` transpose of a `(B, S, H, D)` projection and the kernel writes through
`torch.empty_like`, so a fake that claimed contiguous outputs would make Inductor plan a
layout the kernel never produces and insert a repair copy over the fusion the kernel exists
for.

None of this argues that compilation should SHIP. Compiled artifacts are not reused across
pods. What it buys is that `torch.compile` can be pointed at the plan we ACTUALLY run, as a
discovery oracle, instead of stopping at the plan's own kernel.

Registration is LAZY (`operator()`), because §1.0 keeps `import cozy_runtime` torch-free and
a module-scope `torch.library` decorator would import torch for everyone. Grab the operator
once where the plan is installed, exactly as the plan already grabs the eager entry point,
and call it in the hot path.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

OP_NAME = "cozy_runtime::rms_rope_split_half"

#: Stated, never inferred. `from __future__ import annotations` turns every annotation into
#: a string and torch is imported inside `_register`, so `infer_schema` cannot resolve
#: `torch.Tensor` — and a schema this load-bearing is better read than derived anyway.
OP_SCHEMA = (
    "(Tensor q, Tensor k, Tensor freqs, Tensor q_scale, Tensor k_scale, "
    "float epsilon=1e-6, int rot_dim=0) -> (Tensor, Tensor)"
)

_OPERATOR: Any = None


def operator() -> Any:
    """The registered `cozy_runtime::rms_rope_split_half` operator, registering it once."""
    global _OPERATOR
    if _OPERATOR is None:
        _OPERATOR = _register()
    return _OPERATOR


def _register() -> Any:
    import torch

    @torch.library.custom_op(OP_NAME, mutates_args=(), device_types="cuda", schema=OP_SCHEMA)
    def rms_rope_split_half(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        epsilon: float = 1e-6,
        rot_dim: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paired out-of-place RMS-normalized split-half RoPE, computed by the CUDA kernel."""
        from importlib import import_module

        tensors = (q, k, freqs, q_scale, k_scale)
        if not all(value.is_cuda for value in tensors):
            raise ValueError("RMSNorm+RoPE inputs must all be CUDA tensors")
        if any(value.device != q.device for value in tensors[1:]):
            raise ValueError("RMSNorm+RoPE inputs must occupy one CUDA device")
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        stream = torch.cuda.current_stream(q.device).cuda_stream
        native = import_module("cozy_runtime._kernels._C")
        native.rms_rope_split_half(
            q, k, freqs, q_scale, k_scale, q_out, k_out, epsilon, stream, rot_dim
        )
        return q_out, k_out

    @rms_rope_split_half.register_fake
    def _(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        epsilon: float = 1e-6,
        rot_dim: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shapes, dtypes and STRIDES of the two outputs, launching nothing.

        `empty_like` mirrors what the kernel does, which is what makes the strides right.
        """
        if q.shape != k.shape:
            raise ValueError("Q and K must share a shape")
        if q.ndim != 4:
            raise ValueError("Q/K must be 4D")
        return torch.empty_like(q), torch.empty_like(k)

    return rms_rope_split_half


__all__ = ["OP_NAME", "OP_SCHEMA", "operator"]
