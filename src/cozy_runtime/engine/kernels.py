"""The engine's Triton kernels (`_kernels`) as a declared set (xs-213): a model's step launches each
with constexprs from its dimensions and operands of its dtypes, so its set is generated from them
(`shapes_of`, `kernels_for`) and built where the Runtime builds its own sets: in the machine's
kernel store when the executor prepares the model, or copied from the wheel (`triton_prebuild`,
`PREBUILT`). A request never compiles one (`triton_kernels`): CozyEngine serves a model once its
set has landed (`CozyEngine.install`), Transformers' engine until then.

A step launches them through `launch.Run`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

from cozy_runtime.accel.kernels import TritonEntry, TritonKernels
from cozy_runtime.accel.moe import _kernels as grouped
from cozy_runtime.engine import _kernels, _moe, moe
from cozy_runtime.engine.launch import TRITON
from cozy_runtime.engine.qwen35 import CONV_BLOCK, MLP_BLOCK, NORM_WIDEST, PARTS, VALUE_BLOCK, layout_of, text_of

if TYPE_CHECKING:
    from cozy_runtime.accel.kernels import JitKernel
#: A drafted step's depth (`decoder.Decoder`'s layout).
DEPTH = 2
#: The block of a kept state's copy (`Qwen35.keep`).
KEEP_BLOCK = 1024


@dataclass(frozen=True)
class Shapes:
    """What a model's step launches depend on: its dimensions, its activations' dtype, the dtype of
    each linear's product as the fused ops read it (f32 from a ggml leaf, else the activations'),
    and its norms' and gated-delta parameters' dtypes."""

    act: str
    #: the K/V's dtype, and the gated-delta layers' conv and recurrent states' (the K/V layout's)
    kv: str
    states: tuple[str, ...]
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary: int
    attn_layers: int
    block_tokens: int
    k_heads: int
    v_heads: int
    dk: int
    dv: int
    conv: int
    #: each product role's dtype: q, k, v, o, gate, up, down, qkv, z, a, b, out (a routed MLP's gate, up and
    #: down are its shared expert's, and shared_gate that expert's gate)
    products: tuple[tuple[str, str], ...]
    #: each parameter role's dtype: norm, qk_norm, gated_norm, conv, a_log, dt_bias
    weights: tuple[tuple[str, str], ...]
    #: a mixture of experts' routing and experts' kind (`engine.moe`); None for a dense model
    moe: moe.Routed | None = None

    @property
    def digest(self) -> str:
        fields = asdict(self)
        if fields["moe"] is None:  # a dense model's sets keep the names they had before `moe`
            del fields["moe"]
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


class _Linear(Protocol):
    weight: torch.Tensor


def _dtype(tensor: torch.Tensor) -> str:
    return TRITON[tensor.dtype]


def _product(linear: object, act: str) -> str:
    """What `qwen35._product` returns for `linear`: a ggml leaf's f32 product, else the activations'."""
    return "fp32" if getattr(linear, "product", None) is not None else act


def _one(role: str, found: set[str]) -> str:
    if len(found) != 1:
        raise ValueError(f"its {role} take {len(found)} dtypes ({', '.join(sorted(found))})")
    return found.pop()


def shapes_of(model: torch.nn.Module) -> Shapes:
    """`model`'s `Shapes`, read from its modules without running them. Raises `ValueError` where a
    role's layers disagree (one entry a kernel cannot serve both)."""
    text = text_of(model)
    layout = layout_of(text, depth=DEPTH, drafting=getattr(model, "mtp", None) is not None)
    config = cast("_Configured", model).config.get_text_config(decoder=True)
    named = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or torch.bfloat16
    act = TRITON[getattr(torch, named) if isinstance(named, str) else named]
    root = cast("_Model", model).get_decoder()
    mtp = getattr(model, "mtp", None)
    layers = [*root.layers, *(mtp.layers if mtp is not None else [])]
    products: dict[str, set[str]] = {}
    weights: dict[str, set[str]] = {}
    routed: set[moe.Routed] = set()
    renorm = bool(getattr(config, "norm_topk_prob", True))

    def product(role: str, linear: object) -> None:
        products.setdefault(role, set()).add(_product(linear, act))

    def weight(role: str, tensor: torch.Tensor) -> None:
        weights.setdefault(role, set()).add(_dtype(tensor))

    for layer in layers:
        weight("norm", layer.input_layernorm.weight)
        weight("norm", layer.post_attention_layernorm.weight)
        mlp = layer.mlp
        if moe.is_routed(mlp):
            routed.add(moe.routed_of(mlp, renorm))
            product("shared_gate", moe.shared_gate(mlp))
            mlp = moe.shared(mlp)
        for role in ("gate", "up", "down"):
            product(role, getattr(mlp, f"{role}_proj"))
        attention, delta = getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None)
        if attention is not None:
            for role in ("q", "k", "v", "o"):
                product(role, getattr(attention, f"{role}_proj"))
            weight("qk_norm", attention.q_norm.weight)
            weight("qk_norm", attention.k_norm.weight)
        if delta is not None:
            for role, name in (("qkv", "in_proj_qkv"), ("z", "in_proj_z"), ("a", "in_proj_a"), ("b", "in_proj_b"),
                               ("out", "out_proj")):  # fmt: skip
                product(role, getattr(delta, name))
            weight("gated_norm", delta.norm.weight)
            weight("conv", delta.conv1d.weight)
            weight("a_log", delta.A_log)
            weight("dt_bias", delta.dt_bias)
    weight("norm", root.norm.weight)
    if mtp is not None:
        for norm in (mtp.pre_fc_norm_embedding, mtp.pre_fc_norm_hidden, mtp.norm):
            weight("norm", norm.weight)
    # The residual's addend is attention's, the gated delta's or the MLP's product: one entry reads all three.
    products["residual"] = products["o"] | products["out"] | products["down"]
    return Shapes(
        act=act, kv=TRITON[layout.dtype], states=tuple(TRITON[spec.dtype] for spec in layout.states),
        hidden=text.hidden, heads=text.heads,
        kv_heads=text.kv_heads, head_dim=text.head_dim, rotary=text.rotary, attn_layers=layout.attn_layers,
        block_tokens=layout.block_tokens, k_heads=text.k_heads, v_heads=text.v_heads, dk=text.dk, dv=text.dv,
        conv=text.conv,
        products=tuple((role, _one(f"{role} products", found)) for role, found in sorted(products.items())),
        weights=tuple((role, _one(f"{role} weights", found)) for role, found in sorted(weights.items())),
        moe=_routing(routed, len(layers)),
    )  # fmt: skip


def _routing(routed: set[moe.Routed], layers: int) -> moe.Routed | None:
    """The one routing every layer's MLP takes; None for a dense model. Raises `ValueError` for a mix."""
    if not routed:
        return None
    if len(routed) != 1:
        raise ValueError(f"its routed MLPs take {len(routed)} routings")
    return next(iter(routed))


class _Configured(Protocol):
    config: _Config


class _Config(Protocol):
    def get_text_config(self, decoder: bool = False) -> object: ...


class _Model(Protocol):
    def get_decoder(self) -> _Decoder: ...


class _Decoder(Protocol):
    layers: list[_Layer]
    norm: _Normed


class _Normed(Protocol):
    weight: torch.Tensor


class _Layer(Protocol):
    input_layernorm: _Normed
    post_attention_layernorm: _Normed
    mlp: _Mlp


class _Mlp(Protocol):
    gate_proj: _Linear


def _pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def _entry(kernel: JitKernel, types: dict[str, str], constexprs: dict[str, int], aligned: tuple[str, ...],
           warps: int) -> TritonEntry:  # fmt: skip
    """`kernel`'s entry: each argument's type from `types`, an upper-case name a constexpr; every
    other integer i32. Raises `ValueError` for a name the kernel doesn't take or a pointer left untyped."""
    names = set(kernel.arg_names)
    unknown = sorted((set(types) | set(aligned) | set(constexprs)) - names)
    untyped = [n for n in kernel.arg_names if n.endswith("_ptr") and n not in types]
    unset = sorted({n for n in names if n.isupper()} - set(constexprs))
    if unknown or untyped or unset:
        raise ValueError(f"{kernel.__name__}: no argument {unknown}; untyped {untyped}; constexprs unset {unset}")
    signature = {
        name: "constexpr" if name.isupper() else types.get(name, "i32") for name in kernel.arg_names
    }
    return TritonEntry(kernel, signature, constexprs, aligned, num_warps=warps)


def entries(shapes: Shapes) -> list[TritonEntry]:
    """Every entry a step of a model of `shapes` launches through the fused ops (`qwen35`, `attention`,
    `step`): a plain step's span of 1 and a drafted one's of `DEPTH + 1`."""
    if _kernels.TRITON_ABSENT:
        return []
    k = _kernels
    s, act = shapes, f"*{shapes.act}"
    p = {role: f"*{dtype}" for role, dtype in shapes.products}
    w = {role: f"*{dtype}" for role, dtype in shapes.weights}
    kv, i32, i64, f32 = f"*{shapes.kv}", "*i32", "*i64", "*fp32"
    window, recurrent = (f"*{dtype}" for dtype in shapes.states)
    group, spans = s.heads // s.kv_heads, (1, DEPTH + 1)
    width = _pow2(s.hidden)
    wide = ("width",) if s.hidden % 16 == 0 else ()
    found = [
        _entry(k.rms_norm_kernel, {"x_ptr": act, "w_ptr": w["norm"], "out_ptr": act, "eps": "fp32"},
               {"BLOCK": width}, ("x_ptr", "out_ptr", *wide), 8 if width >= 2048 else 4),
        _entry(k.add_norm_kernel, {"x_ptr": act, "h_ptr": p["residual"], "w_ptr": w["norm"], "sum_ptr": act,
                                   "out_ptr": act, "eps": "fp32"},
               {"BLOCK": width}, ("x_ptr", "h_ptr", "sum_ptr", "out_ptr", *wide), 8 if width >= 2048 else 4),
        _entry(k.silu_mul_kernel, {"gate_ptr": p["gate"], "up_ptr": p["up"], "out_ptr": act},
               {"BLOCK": MLP_BLOCK}, ("gate_ptr", "up_ptr", "out_ptr"), 4),
        _entry(k.qk_norm_rope_kernel, {"q_ptr": p["q"], "k_ptr": p["k"], "q_w_ptr": w["qk_norm"],
                                       "k_w_ptr": w["qk_norm"], "positions_ptr": i32, "inv_ptr": f32,
                                       "q_out_ptr": act, "k_out_ptr": act, "eps": "fp32"},
               {"D": s.head_dim, "R": s.rotary}, ("q_ptr", "k_ptr", "q_out_ptr", "k_out_ptr"), 2),
        _entry(k.keep_kernel, {"scratch_ptr": "*u8", "slots_ptr": i32, "kept_ptr": i32, "state_ptr": "*u8"},
               {"BLOCK": KEEP_BLOCK}, (), 4),
    ]  # fmt: skip
    if s.dv & (s.dv - 1) == 0 and s.dv <= NORM_WIDEST:
        found.append(_entry(
            k.gated_norm_kernel, {"x_ptr": act, "gate_ptr": p["z"], "w_ptr": w["gated_norm"], "out_ptr": act,
                                  "eps": "fp32"},
            {"D": s.dv, "WEIGHT_ROUNDS": int(w["gated_norm"] == act)}, ("x_ptr", "gate_ptr", "out_ptr"),
            1 if s.dv <= 256 else 4,
        ))  # fmt: skip
    keys = s.k_heads * s.dk * int(s.act[-2:]) // 8  # q, k and v are views of one row at these byte offsets
    lined = ("k_ptr", "v_ptr") if keys % 16 == 0 else ()
    for span in spans:
        found += [
            _entry(k.write_step_kernel, {"k_ptr": act, "v_ptr": p["v"], "card_ptr": kv, "slots_ptr": i64,
                                         "lengths_ptr": i32},
                   {"L": s.attn_layers, "S": span, "T": s.block_tokens, "KVH": s.kv_heads, "D": s.head_dim},
                   ("k_ptr", "v_ptr"), 4),
            _entry(k.paged_step_kernel, {"q_ptr": act, "card_ptr": kv, "host_ptr": kv, "table_ptr": i32,
                                         "lengths_ptr": i32, "part_m_ptr": f32, "part_l_ptr": f32,
                                         "part_acc_ptr": f32, "scale": "fp32"},
                   {"L": s.attn_layers, "S": span, "T": s.block_tokens, "KVH": s.kv_heads, "G": group,
                    "Q": _pow2(span * group), "D": s.head_dim, "P": PARTS},
                   ("q_ptr", "part_m_ptr", "part_l_ptr", "part_acc_ptr"), 4),
            _entry(k.paged_join_kernel, {"part_m_ptr": f32, "part_l_ptr": f32, "part_acc_ptr": f32,
                                         "gate_ptr": p["q"], "out_ptr": act},
                   {"S": span, "KVH": s.kv_heads, "G": group, "D": s.head_dim, "P": PARTS, "GATED": 1},
                   ("part_m_ptr", "part_l_ptr", "part_acc_ptr", "out_ptr"), 4),
            _entry(k.conv_span_kernel, {"x_ptr": p["qkv"], "weight_ptr": w["conv"], "slots_ptr": i32,
                                        "state_ptr": window, "scratch_ptr": window, "out_ptr": act},
                   {"S": span, "W": s.conv - 1, "BLOCK": CONV_BLOCK}, ("x_ptr", "out_ptr"), 4),
            _entry(k.delta_span_kernel, {"q_ptr": act, "k_ptr": act, "v_ptr": act, "a_ptr": p["a"], "b_ptr": p["b"],
                                         "a_log_ptr": w["a_log"], "dt_bias_ptr": w["dt_bias"], "slots_ptr": i32,
                                         "state_ptr": recurrent, "scratch_ptr": recurrent, "out_ptr": act,
                                         "scale": "fp32"},
                   {"S": span, "DK": s.dk, "DV": s.dv, "BV": VALUE_BLOCK, "GIVEN": 0},
                   ("q_ptr", *lined, "a_ptr", "b_ptr", "out_ptr"), 4),
            _entry(k.slots_kernel, {"start_ptr": i32, "live_ptr": "*i1", "table_ptr": i32, "out_ptr": i64},
                   {"T": s.block_tokens, "COUNT": span, "WIDTH": _pow2(span)}, (), 1),
        ]  # fmt: skip
    if (routed := s.moe) is not None:
        found += [
            _entry(_moe.route_kernel, {"logits_ptr": act, "index_ptr": f"*{TRITON[routed.index]}", "weights_ptr": act},
                   {"E_BLOCK": _pow2(routed.experts), "K": routed.top_k, "RENORM": routed.renorm}, (), 4),
            _entry(_moe.combine_kernel, {"pairs_ptr": f"*{routed.pairs}", "weights_ptr": act, "shared_ptr": p["down"],
                                        "gate_ptr": p["shared_gate"], "out_ptr": p["residual"]},
                   {"K": routed.top_k, "BLOCK": moe.COMBINE_BLOCK}, (), 4),
            *(grouped.entries(routed.kind) if routed.kind != "ggml" else ()),
        ]  # fmt: skip
    return found


def kernels_for(shapes: Shapes) -> TritonKernels:
    """The set a model of `shapes` launches, named by its shapes: one per model kind on a machine."""
    return TritonKernels(f"cozy-engine-{shapes.digest}", entries(shapes), origin=__name__)


_ROLES = ("a", "b", "down", "gate", "k", "o", "out", "q", "qkv", "residual", "up", "v", "z")
#: The canonical layout's parameters, every Qwen3.5 lane alike (44599741328's sets).
_WEIGHTS = (("a_log", "fp32"), ("conv", "bf16"), ("dt_bias", "bf16"), ("gated_norm", "fp32"), ("norm", "bf16"),
            ("qk_norm", "bf16"))  # fmt: skip


def _checkpoint(hidden: int, heads: int, full: int, v_heads: int, product: str) -> Shapes:
    """A Qwen3.5 checkpoint's shapes: its head dim 256 at a quarter rotary, 4 K/V heads, `full` attention layers and
    the drafting head's, 16 key heads of 128 and value heads of 128, a conv of 4; every product `product` (f32 on the
    ggml lanes, bf16 on bf16 and int8)."""
    return Shapes(
        act="bf16", kv="bf16", states=("bf16", "fp32"), hidden=hidden, heads=heads, kv_heads=4, head_dim=256,
        rotary=64, attn_layers=full + 1, block_tokens=16, k_heads=16, v_heads=v_heads, dk=128, dv=128, conv=4,
        products=tuple((role, product) for role in _ROLES), weights=_WEIGHTS,
    )  # fmt: skip


#: The checkpoints whose sets the wheel ships (`triton_prebuild`), so a fresh machine copies them as the model loads:
#: Qwen3.5-9B (q8_0 `cozy-engine-f26a776376f92032`; int8 and bf16 `cozy-engine-32a91780a70ae938`, as 44599741328
#: built them) and qwen3's default 27B (llmfan46's Qwen3.8-27B), each on the ggml lanes and on bf16 and int8. A
#: checkpoint whose shapes differ builds its own set on the machine, as before.
PREBUILT: tuple[Shapes, ...] = tuple(
    _checkpoint(*dims, product)
    for dims in ((4096, 16, 8, 32), (5120, 24, 16, 48))
    for product in ("fp32", "bf16")
)


def prebuilt() -> list[TritonKernels]:
    """The sets of `PREBUILT`."""
    return [kernels_for(shapes) for shapes in PREBUILT]
