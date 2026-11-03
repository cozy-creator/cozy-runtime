"""Prepared existing Kitchen attention providers for one Ulysses head group.

The exact selected dense/sparse/prefix policy survives the exchange. This closes
physical scratch and output holdings; no kernel, capacity policy or eager fallback
is implemented here.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.internal import attention, attention_sol, kernel_ready
from cozy_runtime.internal.execution.native_heads import _attention, _binary_identity, _route


@dataclass(frozen=True)
class Recipe:
    backend: str
    kind: str
    reason: str
    prefix: int
    tau: float
    scale: float
    identity: tuple[Any, ...]
    extension: Any
    exact_site: Any = None


def recipe(backend: str, options: dict[str, Any], tokens: int, device: int) -> Recipe:
    if (
        options.get("attn_mask") is not None
        or options.get("is_causal")
        or options.get("dropout_p", 0) != 0
        or options.get("enable_gqa")
    ):
        raise ValueError("native Ulysses requires unmasked noncausal equal-head inference")
    scale = options.get("scale")
    scale = 128**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("native Ulysses requires a finite attention scale")
    route = _route(backend, tokens)
    if route is not None:
        kernel = _attention(route, device)
        if kernel is None:
            raise ValueError("native Ulysses selected dense attention artifact is not ready")
        return Recipe(
            backend,
            route.implementation,
            route.reason,
            0,
            0.0,
            scale,
            (kernel.identity, route.layout),
            kernel.extension,
            route.exact_site,
        )
    candidate = next((c for c in attention.known().values() if c.backend == backend), None)
    original = attention.BY_NAME["kitchen-sol-int4qk"]
    base = (
        None
        if candidate is None
        else (candidate.producer.base if candidate.producer is not None else candidate.name)
    )
    if (
        candidate is None
        or base != original.name
        or candidate.dense != original.dense
        or candidate.local.forward is not original.local.forward
        or candidate.local.impl is not original.local.impl
    ):
        raise ValueError(
            "native Ulysses has no complete contract for the selected attention recipe"
        )
    layout, site = _ACTIVE_LAYOUT.get(), attention_sol._SITE.get()
    if layout is None or site is None or layout.live_tokens != tokens or not layout.sparse:
        raise ValueError("native Ulysses sparse recipe requires the actual complete model layout")
    dense_path, exact_path, prefix_heavy = attention_sol.resolved_policy(
        layout, site, quantized_sparse=True
    )
    if dense_path or exact_path or prefix_heavy:
        raise ValueError("native Ulysses sparse route disagrees with the selected model policy")
    ready = next(
        (
            r
            for r in kernel_ready.served(f"attention cuda:{device} ")
            if isinstance(r, attention.Ready) and r.candidate is candidate
        ),
        None,
    )
    extension = getattr(sys.modules.get("comfy_kitchen.backends.cuda"), "_C", None)
    filename = getattr(extension, "__file__", None)
    if (
        ready is None
        or not ready.artifact
        or not filename
        or not all(
            callable(getattr(extension, name, None))
            for name in (
                "sol_attn_plan",
                "sol_attn",
                "sage_sdpa_quantize",
                "sage_sdpa_prequantized",
            )
        )
    ):
        raise ValueError(
            "native Ulysses requires the actual served Kitchen artifact and out providers"
        )
    path = Path(filename)
    stat = path.stat()
    identity = (
        "kitchen-sol-direct/1",
        candidate.name,
        ready.artifact.get("artifact"),
        ready.artifact.get("key"),
        _binary_identity(str(path), stat.st_size, stat.st_mtime_ns),
        layout.live_tokens,
        layout.protected_prefix,
        layout.tau,
        site.path,
    )
    return Recipe(
        backend,
        "kitchen-sol",
        "sparse",
        layout.protected_prefix,
        layout.tau,
        scale,
        identity,
        extension,
    )


def _quant_specs(
    prefix: str, qrows: int, krows: int, heads: int, *, int4: bool, dtype: torch.dtype
) -> tuple[tuple[str, tuple[int, ...], Any], ...]:
    cta = 128 if krows > 1024 else 64
    blocks = (krows + cta - 1) // cta
    values = [
        ("q", (1, heads, qrows, 64 if int4 else 128), torch.int8),
        ("k", (1, heads, krows, 64 if int4 else 128), torch.int8),
        ("v", (heads * 128, blocks * cta), torch.int8),
        ("qs", (1, heads, ((qrows + 127) // 128) * 32), torch.float32),
        ("ks", (1, heads, blocks * 4), torch.float32),
        ("vs", (heads * 128,), torch.float32),
    ]
    values.append(
        ("sum", (1, heads, 128), torch.float32) if int4 else ("anchor", (heads,), torch.int32)
    )
    if not int4:
        values.append(("output_bhnd", (1, heads, qrows, 128), dtype))
    return tuple((f"{prefix}.{name}", shape, kind) for name, shape, kind in values)


@dataclass(frozen=True)
class Prepared:
    recipe: Recipe
    tokens: int
    heads: int
    dtype: torch.dtype
    device: torch.device
    stream: int
    pid: int
    sol_plan: tuple[tuple[str, int], ...]

    @property
    def identity(self) -> tuple[Any, ...]:
        r = self.recipe
        return (
            "native-ulysses-local/1",
            torch.version.git_version,
            r.backend,
            r.kind,
            r.reason,
            r.prefix,
            r.tau,
            r.scale,
            r.identity,
            self.tokens,
            self.heads,
            str(self.dtype),
            self.sol_plan,
        )

    def specs(self) -> tuple[tuple[str, tuple[int, ...], Any], ...]:
        r = self.recipe
        if r.kind == "kitchen-sol":
            values = (("sol_workspace", (dict(self.sol_plan)["total"],), torch.uint8),)
            if r.prefix:
                values += _quant_specs(
                    "prefix", r.prefix, self.tokens, self.heads, int4=False, dtype=self.dtype
                )
            return values
        return _quant_specs(
            "dense",
            self.tokens,
            self.tokens,
            self.heads,
            int4=r.kind == "kitchen-int4qk",
            dtype=self.dtype,
        )

    def validate(
        self, q: Tensor, k: Tensor, v: Tensor, output: Tensor, buffers: dict[str, Tensor]
    ) -> None:
        if (
            self.pid != os.getpid()
            or torch.cuda.current_stream(self.device).cuda_stream != self.stream
        ):
            raise RuntimeError("native Ulysses provider changed process or stream")
        shape = (1, self.tokens, self.heads, 128)
        for tensor in (q, k, v, output):
            if (
                type(tensor) is not Tensor
                or tensor.device != self.device
                or tensor.dtype != self.dtype
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or tensor.is_conj()
                or tensor.is_neg()
                or tensor.data_ptr() % 16
            ):
                raise ValueError("native Ulysses local operands differ from their declared layout")
        specs = self.specs()
        if set(buffers) != {name for name, _, _ in specs}:
            raise ValueError("native Ulysses scratch differs from its exact inventory")
        writes = [output]
        for name, shape, dtype in specs:
            tensor = buffers[name]
            if (
                type(tensor) is not Tensor
                or tensor.device != self.device
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or tensor.is_conj()
                or tensor.is_neg()
                or tensor.data_ptr() % 256
            ):
                raise ValueError("native Ulysses scratch owner differs from its queried contract")
            writes.append(tensor)
        for i, tensor in enumerate(writes):
            for other in (q, k, v, *writes[:i]):
                if torch._C._is_alias_of(tensor, other) or max(
                    tensor.data_ptr(), other.data_ptr()
                ) < min(
                    tensor.data_ptr() + tensor.numel() * tensor.element_size(),
                    other.data_ptr() + other.numel() * other.element_size(),
                ):
                    raise ValueError("native Ulysses writes overlap actual live operands")

    def execute(
        self, q: Tensor, k: Tensor, v: Tensor, output: Tensor, buffers: dict[str, Tensor]
    ) -> None:
        self.validate(q, k, v, output, buffers)
        r = self.recipe
        if r.kind == "kitchen-sol":
            r.extension.sol_attn(
                *(_capsule(t) for t in (q, k, v, output, buffers["sol_workspace"])),
                1,
                self.tokens,
                self.heads,
                128,
                r.tau,
                r.scale,
                0,
                (r.prefix + 63) // 64,
                0,
                0,
                self.stream,
                key_bias=None,
                threshold=None,
                block_len=None,
                tail=True,
                token_aug=0,
            )
            attention_sol._count("sparse")
            if r.prefix:
                self._int8(q[:, : r.prefix], k, v, output[:, : r.prefix], buffers, "prefix")
                attention_sol._count("dense_prefix")
        elif r.kind == "kitchen-int4qk":
            self._int4(q, k, v, output, buffers)
        else:
            self._int8(q, k, v, output, buffers, "dense")
        from cozy_runtime.internal import attention_ulysses

        attention_ulysses.record(r.backend, r.kind)
        if r.exact_site is not None:
            attention_sol.note_exact(r.exact_site)
        if r.reason and r.kind != "kitchen-sol":
            attention_sol._count(r.reason)

    def _int8(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        buffers: dict[str, Tensor],
        prefix: str,
    ) -> None:
        b = {
            name.removeprefix(prefix + "."): tensor
            for name, tensor in buffers.items()
            if name.startswith(prefix + ".")
        }
        cta = 128 if self.tokens > 1024 else 64
        code = 2 if self.dtype == torch.bfloat16 else 1
        self.recipe.extension.sage_sdpa_quantize(
            *(
                _capsule(t)
                for t in (
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    b["q"],
                    b["qs"],
                    b["k"],
                    b["ks"],
                    b["v"],
                    b["vs"],
                )
            ),
            cta,
            code,
            self.stream,
            b["anchor"].data_ptr(),
        )
        self.recipe.extension.sage_sdpa_prequantized(
            *(
                _capsule(t)
                for t in (b["q"], b["k"], b["v"], b["output_bhnd"], b["qs"], b["ks"], b["vs"])
            ),
            cta,
            self.recipe.scale,
            code,
            self.stream,
            None,
        )
        output.copy_(b["output_bhnd"].transpose(1, 2))

    def _int4(
        self, q: Tensor, k: Tensor, v: Tensor, output: Tensor, buffers: dict[str, Tensor]
    ) -> None:
        b = {name.removeprefix("dense."): tensor for name, tensor in buffers.items()}

        def strides(tensor: Tensor) -> tuple[int, int, int]:
            return tensor.stride(0), tensor.stride(2), tensor.stride(1)

        code = self.recipe.extension.attention(
            self.device.index or 0,
            *(
                t.data_ptr()
                for t in (
                    q,
                    k,
                    v,
                    output,
                    b["q"],
                    b["qs"],
                    b["k"],
                    b["ks"],
                    b["sum"],
                    b["v"],
                    b["vs"],
                )
            ),
            1,
            self.heads,
            self.heads,
            self.tokens,
            self.tokens,
            128 if self.tokens > 1024 else 64,
            self.dtype == torch.bfloat16,
            self.recipe.scale,
            strides(q),
            strides(k),
            strides(v),
            strides(output),
            self.stream,
        )
        if code:
            raise RuntimeError(
                f"native Ulysses INT4 attention: {self.recipe.extension.error_name(code)}"
            )


def _capsule(value: Tensor) -> Any:
    return value.__dlpack__(stream=-1)


def prepare(
    selected: Recipe, tokens: int, heads: int, dtype: torch.dtype, device: torch.device, stream: int
) -> Prepared:
    if (
        tokens < 1
        or tokens * heads * 128 > 2**31 - 1
        or heads < 1
        or dtype not in (torch.bfloat16, torch.float16)
        or device.type != "cuda"
        or stream <= 2
        or not 0 <= selected.prefix <= tokens
    ):
        raise ValueError("native Ulysses provider requires its restricted CUDA head geometry")
    plan = ()
    if selected.kind == "kitchen-sol":
        actual = selected.extension.sol_attn_plan(1, tokens, heads, token_aug=0)
        if type(actual) is not dict or type(actual.get("total")) is not int or actual["total"] <= 0:
            raise ValueError("native Ulysses Sol provider did not return its actual scratch layout")
        plan = tuple(sorted(actual.items()))
    return Prepared(selected, tokens, heads, dtype, device, stream, os.getpid(), plan)
