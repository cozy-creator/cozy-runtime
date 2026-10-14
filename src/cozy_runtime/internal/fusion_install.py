"""The substitution map for the fused glue (h3a-015): supported Diffusers H3 modules -> fused ones.

Applied after fill: parameters, buffers, state-dict keys and
storage identity are invariants (re-censused after the swap). Compatibility is the callable
and module topology contract consumed by the plan, not upstream source formatting or
annotations. Numerical agreement is exercised against supported Diffusers releases.
The swap itself is a class substitution: each module's `__class__` becomes the fused
subclass that overrides `forward` and nothing else. The map records `(path, was, now)` per site.

fp8 epilogues are chosen per site from what the fill actually installed: a site whose every
consumer is a `fp8-rowwise/1` encoded leaf gets the quantizing epilogue and hands the leaf
its operand directly (`leaves.PreQuantized`); any other consumer gets bf16.

The package consents on its class (`fusion="accept"`), and the executor applies this plan
only under that consent. CONSENT IS NOT A REQUIREMENT: `accept` installs the fused kernels
when this device's are compiled and otherwise serves eager, recording the typed code
(`NOT_INSTALLABLE`: compiling, failed, absent, no Triton, an unmatched diffusers).
`fusion="require"` waits for the build at construction, never in a request, and refuses
typed when it failed or cannot exist. BREAKAGE, a compiled build that does not load
(`fusion_kernels_corrupt`), refuses under either consent, as does a failure found mid-walk:
the model is part-substituted and there is no eager left.

Executor-only, like `fusion.py`: diffusers is imported at the top, guarded, and its absence
is the typed `fusion_dependency_absent`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from cozy_runtime.internal import fusion

try:
    import torch
    from diffusers.models.activations import SwiGLU
    from diffusers.models.attention import FeedForward
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AttnProcessor,
        MiniMaxH3Transformer3DModel,
        MiniMaxH3TransformerBlock,
        _apply_rotary_emb,
    )
except ImportError as _absent:  # pragma: no cover - exercised by the typed refusal
    DIFFUSERS_ABSENT = f"diffusers does not import ({_absent})"
else:
    DIFFUSERS_ABSENT = ""

CONTRACT = "h3-glue-call-and-topology/2"
_LOG = logging.getLogger(__name__)
ROWWISE_LEAF = "cozy.fp8-rowwise.native-leaf/1"
#: The codes a consenting package serves eager past; anything else is breakage and refuses.
#: Closed on purpose: a new code is a decision, not a default.
NOT_INSTALLABLE = frozenset(
    {
        "fusion_kernels_absent",
        "fusion_kernels_failed",
        "fusion_triton_absent",
        "fusion_dependency_absent",
        "fusion_contract_unsupported",
        "fusion_block_shape",
    }
)


class FusionRefusal(RuntimeError):
    """The package consented to fusion and the exact plan could not be installed."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class AppliedFusion:
    identity: str
    applied: bool
    reason: str
    substitutions: tuple[tuple[str, str, str], ...]
    """`(module path, was, now)` per site, in walk order."""
    fp8_sites: int
    warm_ms: float
    code: str = ""
    """The typed non-application code when `applied` is false: the fact an operator greps
    for. Empty on the applied path."""

    def identity_document(self) -> dict[str, object]:
        """The deterministic part: what the constructed generation's digest binds."""
        census: dict[str, int] = {}
        for _, was, now in self.substitutions:
            key = f"{was.rsplit(':', 1)[-1]}->{now.rsplit(':', 1)[-1]}"
            census[key] = census.get(key, 0) + 1
        return {
            "identity": self.identity,
            "applied": self.applied,
            "substitutions": census,
            "sites": len(self.substitutions),
            "fp8_sites": self.fp8_sites,
        }

    def document(self) -> dict[str, object]:
        return {
            **self.identity_document(),
            "code": self.code,
            "reason": self.reason,
            "warm_ms": round(self.warm_ms, 1),
        }


def _type_name(value: object) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def identity() -> str:
    document = {
        "kernels": fusion.source_digest() if not fusion.TRITON_ABSENT else "",
        "plan": CONTRACT,
    }
    return (
        "sha256:"
        + hashlib.sha256(
            b"cozy.runtime.glue-fusion\0"
            + json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _require_call(function: Callable[..., object], parameters: tuple[str, ...]) -> None:
    """Validate the inputs this plan substitutes; annotations are not an ABI.

    Positional inputs must keep their names and order. Additional optional inputs are
    harmless; additional required inputs or reordered inputs cannot use this forward.
    """
    try:
        signature = inspect.signature(function)
        bound = signature.bind(*(object() for _ in parameters))
        if tuple(bound.arguments) != parameters:
            raise TypeError(f"expected positional inputs {parameters}")
    except (TypeError, ValueError) as exc:
        raise FusionRefusal(
            "fusion_contract_unsupported", f"{function.__qualname__}: {exc}"
        ) from exc


def _require_capabilities() -> None:
    if DIFFUSERS_ABSENT:
        raise FusionRefusal("fusion_dependency_absent", DIFFUSERS_ABSENT)
    try:
        metadata.version("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise FusionRefusal(
            "fusion_dependency_absent", "diffusers is required by the fused glue plan"
        ) from exc
    _require_call(
        MiniMaxH3TransformerBlock.forward,
        ("self", "hidden_states", "temb", "adaln_indices", "rotary_emb", "attention_mask"),
    )
    _require_call(
        MiniMaxH3AttnProcessor.__call__,
        ("self", "attn", "hidden_states", "rotary_emb", "attention_mask"),
    )
    _require_call(_apply_rotary_emb, ("hidden_states", "cos", "sin"))
    _require_call(SwiGLU.forward, ("self", "hidden_states"))
    _require_call(FeedForward.forward, ("self", "hidden_states"))


def _record(model: object, result: AppliedFusion) -> AppliedFusion:
    document = result.document()
    if hasattr(model, "__dict__"):
        vars(model)["_cozy_fusion_report"] = document
    log = _LOG.info if result.applied else _LOG.warning
    log("H3 glue fusion: %s", json.dumps(document, sort_keys=True))
    return result


# ------------------------------------------------------------------ fused modules

if not DIFFUSERS_ABSENT:

    class FusedSwiGLU(SwiGLU):  # type: ignore[misc]
        _fusion_fp8: bool = False

        def forward(self, hidden_states: Any) -> Any:
            return fusion.swiglu(self.proj(hidden_states), fp8=self._fusion_fp8)

    class FusedMiniMaxH3AttnProcessor(MiniMaxH3AttnProcessor):  # type: ignore[misc]
        def __call__(
            self,
            attn: Any,
            hidden_states: Any,
            rotary_emb: tuple[Any, Any] | None = None,
            attention_mask: Any | None = None,
        ) -> Any:
            query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
            key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
            value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
            if rotary_emb is None:
                query, key = attn.norm_q(query), attn.norm_k(key)
            else:
                cos, sin = rotary_emb
                query = fusion.rmsnorm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
                key = fusion.rmsnorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3).type_as(query)
            return attn.to_out[1](attn.to_out[0](hidden_states))

    class FusedMiniMaxH3TransformerBlock(MiniMaxH3TransformerBlock):  # type: ignore[misc]
        _fusion_attn_fp8: bool = False
        _fusion_ff_fp8: bool = False

        def forward(
            self,
            hidden_states: Any,
            temb: Any,
            adaln_indices: Any,
            rotary_emb: tuple[Any, Any],
            attention_mask: Any | None = None,
        ) -> Any:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
            normed = fusion.rmsnorm_modulate(
                hidden_states,
                self.norm1.weight,
                scale_msa,
                shift_msa,
                adaln_indices,
                self.norm1.eps,
                fp8=self._fusion_attn_fp8,
            )
            attn_output = self.attn(normed, rotary_emb, attention_mask)
            hidden_states, normed = fusion.gate_add_rmsnorm_modulate(
                hidden_states,
                gate_msa,
                attn_output,
                adaln_indices,
                self.norm2.weight,
                scale_mlp,
                shift_mlp,
                self.norm2.eps,
                fp8=self._fusion_ff_fp8,
            )
            # `ff.net` is [SwiGLU, Dropout(0.0), Linear]; the identity dropout is skipped so
            # the fused SwiGLU can hand the out projection its fp8 operand directly.
            ff_output = self.ff.net[2](self.ff.net[0](normed))
            return fusion.gate_add(hidden_states, gate_mlp, ff_output, adaln_indices)


# ------------------------------------------------------------------------ install


def _rowwise_leaf(module: Any) -> bool:
    # Semantic hooks must receive the original Tensor, not a quantized carrier.
    return (
        getattr(module, "provider", None) == ROWWISE_LEAF
        and not module._forward_pre_hooks
        and not module._forward_hooks
    )


def _tensor_census(module: Any) -> tuple[object, ...]:
    return tuple(
        (name, tuple(value.shape), str(value.dtype), id(value), int(value.data_ptr()))
        for name, value in (
            *module.named_parameters(remove_duplicate=False),
            *module.named_buffers(remove_duplicate=False),
        )
    )


def _shape(condition: bool, detail: str) -> None:
    if not condition:
        raise FusionRefusal("fusion_block_shape", detail)


def _validate_transformer(transformer: Any, root: str) -> None:
    _shape(
        type(transformer).forward is MiniMaxH3Transformer3DModel.forward
        and "forward" not in vars(transformer),
        f"{root}: {_type_name(transformer)} overrides the Diffusers MiniMax-H3 forward",
    )
    for index, block in enumerate(transformer.transformer_blocks):
        path = f"{root}.transformer_blocks.{index}"
        attn, ff = block.attn, block.ff
        _shape(
            type(block) is MiniMaxH3TransformerBlock
            and "forward" not in vars(block)
            and type(attn.processor) is MiniMaxH3AttnProcessor
            and not attn.fused_projections
            and type(ff) is FeedForward
            and "forward" not in vars(ff)
            and len(ff.net) == 3
            and type(ff.net[0]) is SwiGLU
            and "forward" not in vars(ff.net[0])
            and type(ff.net[0].activation) is torch.nn.SiLU
            and "forward" not in vars(ff.net[0].activation)
            and type(ff.net[1]) is torch.nn.Dropout
            and "forward" not in vars(ff.net[1])
            and ff.net[1].p == 0.0
            and not any(
                m._forward_hooks or m._forward_pre_hooks
                for m in (
                    attn,
                    ff,
                    ff.net,
                    ff.net[0],
                    ff.net[1],
                    ff.net[0].activation,
                    block.norm1,
                    block.norm2,
                    attn.norm_q,
                    attn.norm_k,
                )
            )
            and all(
                type(m) is torch.nn.RMSNorm and "forward" not in vars(m)
                for m in (block.norm1, block.norm2, attn.norm_q, attn.norm_k)
            )
            and block.norm1.weight is not None
            and block.norm2.weight is not None
            and attn.norm_q.weight is not None
            and attn.norm_k.weight is not None,
            f"{path} is not the MiniMax-H3 block shape the fused plan was matched to ({CONTRACT})",
        )


def _install_transformer(transformer: Any, root: str) -> tuple[list[tuple[str, str, str]], int]:
    substitutions: list[tuple[str, str, str]] = []
    fp8_sites = 0
    for index, block in enumerate(transformer.transformer_blocks):
        path = f"{root}.transformer_blocks.{index}"
        attn, ff = block.attn, block.ff
        attn_fp8 = all(_rowwise_leaf(m) for m in (attn.to_q, attn.to_k, attn.to_v))
        ff_fp8 = _rowwise_leaf(ff.net[0].proj)
        out_fp8 = _rowwise_leaf(ff.net[2])
        fp8_sites += attn_fp8 + ff_fp8 + out_fp8

        processor = FusedMiniMaxH3AttnProcessor()
        processor._attention_backend = attn.processor._attention_backend
        processor._parallel_config = attn.processor._parallel_config
        substitutions.append(
            (f"{path}.attn.processor", _type_name(attn.processor), _type_name(processor))
        )
        attn.set_processor(processor)

        substitutions.append((f"{path}.ff.net.0", _type_name(ff.net[0]), _type_name(FusedSwiGLU)))
        ff.net[0].__class__ = FusedSwiGLU
        ff.net[0]._fusion_fp8 = out_fp8

        substitutions.append((path, _type_name(block), _type_name(FusedMiniMaxH3TransformerBlock)))
        block.__class__ = FusedMiniMaxH3TransformerBlock
        block._fusion_attn_fp8 = attn_fp8
        block._fusion_ff_fp8 = ff_fp8
    return substitutions, fp8_sites


def _transformers(model: object) -> Iterator[tuple[str, Any]]:
    """Every MiniMax-H3 transformer root the model holds, by component name.

    The loader's census reads component roots from a constructed object's `components`
    mapping (diffusers pipelines and the H3/Anima pipelines alike); this walks the same
    shape one level down from the model's own attributes.
    """
    seen: set[int] = set()
    for attribute, value in vars(model).items():
        table = getattr(value, "components", None)
        roots = table.items() if isinstance(table, Mapping) else ((attribute, value),)
        for name, member in roots:
            if isinstance(member, MiniMaxH3Transformer3DModel) and id(member) not in seen:
                seen.add(id(member))
                yield str(name), member


def capability_refusal(torch: Any, device: Any) -> str:
    """Why the fused plan does not apply on this device: "" when it does."""
    if device.type != "cuda":
        return f"fused glue serves CUDA only; this executor holds {device}"
    major, _ = torch.cuda.get_device_capability(device)
    if major < 8:
        return f"fused glue needs bf16 tensor cores (sm80+); this device is sm{major}x"
    return ""


def apply_fusion_plan(
    model: object, *, torch: Any, device: Any, required: bool = False
) -> AppliedFusion:
    """Install the fused glue on every H3 transformer the model holds.

    Called under the class's `fusion=` consent. A live build is waited for. Without `required`
    (`fusion="require"`) a `NOT_INSTALLABLE` code is a recorded non-application and eager
    serves; with it the same code refuses typed. Any other code refuses under either.
    """
    reason = capability_refusal(torch, device)
    if reason:
        return _record(
            model, AppliedFusion(identity(), False, reason, (), 0, 0.0, "fusion_device_incapable")
        )
    try:
        _require_capabilities()
        transformers = tuple(_transformers(model))
        for name, transformer in transformers:
            _validate_transformer(transformer, name)
        fusion.load(device)
    except (fusion.FusionUnavailable, FusionRefusal) as exc:
        if required or exc.code not in NOT_INSTALLABLE:
            raise FusionRefusal(exc.code, str(exc)) from exc
        # NOTHING HAS BEEN SUBSTITUTED: the eager model serves, and the code rides the
        # prepare's `execution_fusion` fact and the session's boot line.
        return _record(model, AppliedFusion(identity(), False, str(exc), (), 0, 0.0, exc.code))
    substitutions: list[tuple[str, str, str]] = []
    fp8_sites = 0
    warm_ms = 0.0
    for name, transformer in transformers:
        before = _tensor_census(transformer)
        installed, sites = _install_transformer(transformer, name)
        if _tensor_census(transformer) != before:
            raise FusionRefusal(
                "fusion_tensor_schema_changed",
                f"{name}: the substitution moved a parameter or buffer",
            )
        substitutions += installed
        fp8_sites += sites
        config = transformer.config
        try:
            warm_ms += fusion.warm(
                device,
                hidden=int(config.hidden_size),
                ffn=int(config.ffn_dim),
                heads=int(config.num_attention_heads),
                head_dim=int(config.attention_head_dim),
                rotary=6 * int(config.rope_freq_dim),
                fp8=sites > 0,
            )
        except fusion.FusionUnavailable as exc:
            # PAST THE POINT OF NO RETURN: the forwards are already substituted, so a
            # width no profile compiled refuses whatever the consent said.
            raise FusionRefusal(exc.code, str(exc)) from exc
    if not substitutions:
        return _record(
            model,
            AppliedFusion(
                identity(),
                False,
                "the model holds no MiniMax-H3 transformer to fuse",
                (),
                0,
                0.0,
                "fusion_no_transformer",
            ),
        )
    return _record(
        model, AppliedFusion(identity(), True, "", tuple(substitutions), fp8_sites, warm_ms)
    )
