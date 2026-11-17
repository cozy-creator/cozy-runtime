"""llama.cpp's lanes as cozytensors lanes: ggml's block types (`ggml.*`, cozytensors spec §6),
written by TensorFS's port of ggml's reference quantizers (`ggml-ref/1`), and llama-quantize's
per-tensor mixes.

A plan names a block type (`ggml.q8_0`: every selected weight in it) or a mix (`ggml.q4_k_m`).
In a mix each selected weight takes the type llama-quantize gives the GGUF tensor it becomes
(`llama_tensor_get_type`, `src/llama-quant.cpp`, llama.cpp b12466, MIT): the same categories,
counters, tensor order and fallbacks over the same tensors. A weight llama-quantize keeps as it
is (a router, a one-row gate) stays at source precision. Where it becomes is llama.cpp's
converter's map for the checkpoint's architecture (`ARCHITECTURES`); a checkpoint of another
architecture, or a weight that map does not place, is refused by name. Without an importance
matrix, as llama-quantize runs by default, the lane's bytes are its bytes.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cozy_runtime.author._errors import UnsupportedInput
from cozy_runtime.internal.encoding import ggml_types

if TYPE_CHECKING:
    from tensorfs import Checkpoint

    from .quantization import ArtifactQuantizationPlan

#: TensorFS's producer of every block type (ggml's `quantize_row_<type>_ref`).
STRATEGY = "ggml-ref/1"

#: The block types a plan may name.
TYPES = frozenset(ggml_types.WRITTEN)

#: llama-quantize's file types (`LLAMA_FTYPE_MOSTLY_*`) a plan may name: each mix's default type.
MIXES = {"ggml.q4_k_m": "ggml.q4_k"}

#: The type a block type falls back to when a row is not whole blocks (`tensor_type_fallback`).
_FALLBACK = {"ggml.q4_k": "ggml.q5_0", "ggml.q5_k": "ggml.q5_1", "ggml.q6_k": "ggml.q8_0"}

#: What llama-quantize writes for a row no block type fits; no lane writes it.
_F16 = "F16"


def block(encoding: str) -> int:
    """The columns a weight in `encoding` must be a multiple of: in a mix, its fallbacks'."""
    if encoding in MIXES:
        return min(ggml_types.block(t) for t in _FALLBACK.values())
    return ggml_types.block(encoding)


# ------------------------------------------------------------- llama.cpp's converter's names


@dataclass(frozen=True, slots=True)
class Architecture:
    """llama.cpp's converter for one Hugging Face architecture (`ModelBase.register`): each
    layer tensor's GGUF name after `blk.<i>.`, two of its weights sharing one name where it
    merges them. llama.cpp's loaders for these give no model `LLM_TYPE_70B`, so llama-quantize's
    70B rule never applies to them."""

    layer: Mapping[str, str]


#: A layer's tensors in Qwen3.5's converters (gguf-py `tensor_mapping.py`).
_QWEN35 = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
}
#: Qwen3.5's dense converter merges two pairs of linear-attention inputs (`_SHARED_INPUT_MERGES`):
#: `attn_qkv_gate` is `in_proj_qkv` then `in_proj_z`, `ssm_beta_alpha` is `in_proj_b` then
#: `in_proj_a`. Its MoE converter keeps them apart.
_QWEN35_DENSE = {
    **_QWEN35,
    "linear_attn.in_proj_qkv.weight": "attn_qkv_gate.weight",
    "linear_attn.in_proj_z.weight": "attn_qkv_gate.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta_alpha.weight",
    "linear_attn.in_proj_a.weight": "ssm_beta_alpha.weight",
}
ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration": Architecture(_QWEN35_DENSE),
    "Qwen3_5ForCausalLM": Architecture(_QWEN35_DENSE),
    "Qwen3_5MoeForConditionalGeneration": Architecture(_QWEN35),
    "Qwen3_5MoeForCausalLM": Architecture(_QWEN35),
}

#: A routed expert's weight, one per expert here, one stack per layer there. A fused gate and up
#: becomes two stacks: its first half of rows `ffn_gate_exps`, its second `ffn_up_exps`.
_EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down|gate_up)_proj\.weight")
#: The multi-token-prediction layer's own tensors (`_QwenMtpMixin`'s remapper), at block `n_layer`.
_NEXTN = {
    "fc.weight": "nextn.eh_proj.weight",
    "pre_fc_norm_embedding.weight": "nextn.enorm.weight",
    "pre_fc_norm_hidden.weight": "nextn.hnorm.weight",
    "norm.weight": "nextn.shared_head_norm.weight",
}
_TEXT = r"(?:model\.language_model|model)\."
_LAYER = re.compile(rf"^{_TEXT}layers\.(\d+)\.(.+)$")
_MTP_LAYER = re.compile(r"^(?:model\.)?mtp\.layers\.(\d+)\.(.+)$")
_MTP = re.compile(r"^(?:model\.)?mtp\.([a-z_]+\.weight)$")
#: The vision tower: llama.cpp's mmproj, another GGUF.
_MMPROJ = re.compile(r"^(?:model\.)?visual\.")


@dataclass(frozen=True, slots=True)
class Model:
    """A component's GGUF as llama.cpp's converter writes it: where each weight lands, and each
    tensor's dimensions (`ggml_n_dims`) and row length."""

    #: each name -> the GGUF tensors it lands in (none for mmproj's)
    tensors: Mapping[str, tuple[str, ...]]
    #: each GGUF tensor -> (ggml_n_dims, columns)
    gguf: Mapping[str, tuple[int, int]]
    #: `n_layer_all` (the main layers and the MTP layers after them) and `n_expert`
    n_layer: int
    n_expert: int


def model(config: Mapping[str, object], shapes: Mapping[str, tuple[int, ...]]) -> Model:
    """The GGUF llama.cpp's converter writes of a component with this `config` and these
    tensors (`shapes`, by name). Refuses an architecture it has no map for, and a weight of
    two or more axes the map does not place."""
    named = config.get("architectures")
    kind = named[0] if isinstance(named, list) and named else None
    arch = ARCHITECTURES.get(str(kind))
    if arch is None:
        raise UnsupportedInput(
            f"llama.cpp's tensor names for {kind!r} are not ported (only {sorted(ARCHITECTURES)})",
            code="quantization_plan",
        )
    text = config.get("text_config", config)
    assert isinstance(text, Mapping)
    main = int(text["num_hidden_layers"])
    n_layer = main + int(text.get("mtp_num_hidden_layers", 0) or 0)
    n_expert = int(text.get("num_experts", 0) or 0)
    tensors: dict[str, tuple[str, ...]] = {}
    rows: dict[str, int] = {}
    stacked: dict[str, int] = {}
    columns: dict[str, int] = {}
    flat: set[str] = set()
    for name, shape in shapes.items():
        found = _place(arch, name, main)
        if found is None:
            if len(shape) >= 2:
                raise UnsupportedInput(
                    f"llama.cpp's converter for {kind} places no {name} {list(shape)}",
                    code="quantization_plan",
                )
            continue
        tensors[name] = tuple(t for t, _ in found)
        for tensor, share in found:
            if len(shape) < 2:
                flat.add(tensor)
                continue
            rows[tensor] = rows.get(tensor, 0) + shape[0] // share
            columns[tensor] = shape[-1]
            if _EXPERT.search(name):
                stacked[tensor] = stacked.get(tensor, 0) + 1
    gguf = {t: (1, 0) for t in flat}
    for tensor, n in rows.items():
        experts = stacked.get(tensor, 1)
        n_dims = 3 if experts > 1 else (2 if n > 1 else 1)
        gguf[tensor] = (n_dims, columns[tensor])
    return Model(tensors, gguf, n_layer, n_expert)


def _place(arch: Architecture, name: str, main: int) -> list[tuple[str, int]] | None:
    """`name`'s GGUF tensors, each with the share of its rows it holds: none for mmproj's, None
    where the map does not place it."""
    if _MMPROJ.match(name):
        return []
    if re.fullmatch(rf"{_TEXT}embed_tokens\.weight", name):
        return [("token_embd.weight", 1)]
    if name == "lm_head.weight":
        return [("output.weight", 1)]
    if re.fullmatch(rf"{_TEXT}norm\.weight", name):
        return [("output_norm.weight", 1)]
    if m := _MTP.match(name):
        tail = _NEXTN.get(m.group(1))
        return None if tail is None else [(f"blk.{main}.{tail}", 1)]
    if m := _MTP_LAYER.match(name):
        return _layer(arch, main + int(m.group(1)), m.group(2))
    if m := _LAYER.match(name):
        return _layer(arch, int(m.group(1)), m.group(2))
    return None


def _layer(arch: Architecture, bid: int, rest: str) -> list[tuple[str, int]] | None:
    if e := _EXPERT.fullmatch(rest):
        if e.group(2) == "gate_up":
            return [(f"blk.{bid}.ffn_gate_exps.weight", 2), (f"blk.{bid}.ffn_up_exps.weight", 2)]
        return [(f"blk.{bid}.ffn_{e.group(2)}_exps.weight", 1)]
    tail = arch.layer.get(rest)
    return None if tail is None else [(f"blk.{bid}.{tail}", 1)]


# ------------------------------------------------------------------- llama_tensor_get_type


def _category(name: str) -> str:
    """`tensor_get_category`."""
    if name == "output.weight":
        return "output"
    if name in ("token_embd.weight", "per_layer_token_embd.weight"):
        return "token_embd"
    if "attn_qkv.weight" in name or "attn_qkv_gate.weight" in name:
        return "attn_qkv"
    for found, category in (
        ("attn_kv_b.weight", "attn_kv_b"),
        ("attn_v.weight", "attn_v"),
        ("attn_k.weight", "attn_k"),
        ("attn_q.weight", "attn_q"),
        ("attn_output.weight", "attn_output"),
        ("ffn_up", "ffn_up"),
        ("ffn_gate", "ffn_gate"),
        ("ffn_latent", "ffn_latent"),
        ("attn_gate.weight", "attn_gate"),
        ("ffn_down", "ffn_down"),
    ):
        if found in name:
            return category
    return "other"


_ATTN_V = frozenset({"attn_v", "attn_qkv", "attn_kv_b"})

#: `tensor_allows_quantization`'s name rules for every architecture (the ones for named other
#: architectures never meet these).
_KEPT = (
    "_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_tid2eid.weight",
    "engram_gate.weight",
    "altup",
    "laurel",
    "per_layer_model_proj",
    "ssm_conv1d",
    "shortconv.conv.weight",
    "conv2d_spatial.weight",
    "conv_dw",
    "indexer.k_proj.weight",
    "indexer.q_proj.weight",
    "time_mix_first.weight",
    "time_mix_w0.weight",
    "time_mix_w1.weight",
    "time_mix_w2.weight",
    "time_mix_v0.weight",
    "time_mix_v1.weight",
    "time_mix_v2.weight",
    "time_mix_a0.weight",
    "time_mix_a1.weight",
    "time_mix_a2.weight",
    "time_mix_g1.weight",
    "time_mix_g2.weight",
    "time_mix_decay_w1.weight",
    "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
    ".position_embd",
    "sam.pos_embd",
    "sam.neck.",
    "sam.net_",
    ".rel_pos",
    ".patch_embd",
    ".patch_merger",
    "a.rvq.codebook",
    "mm.a.code_embd",
)


def _allowed(name: str, n_dims: int) -> bool:
    """`tensor_allows_quantization`: two or more dimensions, a weight, no kept name."""
    return (
        n_dims >= 2
        and name.endswith("weight")
        and name not in ("position_embd.weight", "token_types.weight")
        and not name.startswith(("a.conv1d.", "a.gen.wav.quant."))
        and not any(kept in name for kept in _KEPT)
    )


def _order(name: str) -> tuple[int, str]:
    """`weight_name_comparer`: by `blk.<i>.` (none first), then by name."""
    m = re.match(r"blk\.(\d+)\.", name)
    return (int(m.group(1)) if m else -1, name)


def _more_bits(i_layer: int, n_layers: int) -> bool:
    """`use_more_bits`, in C's integer division."""
    return (
        i_layer < n_layers // 8
        or i_layer >= 7 * n_layers // 8
        or _c_mod(i_layer - n_layers // 8, 3) == 2
    )


def _c_mod(a: int, b: int) -> int:
    """C's `%`: the sign of the dividend."""
    return a - b * int(a / b)


def mix(encoding: str, model: Model, selected: Collection[str]) -> dict[str, str | None]:
    """Each selected weight's block type in mix `encoding`, as llama-quantize assigns the GGUF
    tensors it lands in; None where it keeps the tensor as it is. Weights merged into one tensor
    take its one type; a weight split over two takes theirs, which must agree."""
    default = MIXES[encoding]
    tensors = sorted(model.gguf, key=_order)
    categories = {t: _category(t) for t in tensors}
    n_attention_wv = sum(1 for t in tensors if categories[t] in _ATTN_V)
    tied = "output.weight" not in model.gguf
    counters = {"attention_wv": 0, "ffn_down": 0, "ffn_gate": 0, "ffn_up": 0}
    types: dict[str, str | None] = {}
    for tensor in tensors:
        n_dims, columns = model.gguf[tensor]
        if not _allowed(tensor, n_dims):
            types[tensor] = None
            continue
        category = categories[tensor]
        chosen = _get_type(
            default, tensor, category, counters, n_attention_wv, model, tied, columns
        )
        types[tensor] = _fallback(chosen, columns)
    out: dict[str, str | None] = {}
    for name in selected:
        landed = model.tensors.get(name)
        if not landed:
            raise ValueError(f"{name} is not in llama.cpp's model (its mmproj holds it)")
        found = {types[t] for t in landed}
        if len(found) != 1:
            raise ValueError(
                f"{name} becomes {', '.join(landed)}, of types {sorted(map(str, found))}"
            )
        (out[name],) = found
        if out[name] == _F16:
            raise ValueError(
                f"{name}'s rows are whole blocks of no ggml type (llama-quantize: F16)"
            )
    return out


def _get_type(
    new: str,
    name: str,
    category: str,
    counters: dict[str, int],
    n_attention_wv: int,
    model: Model,
    tied: bool,
    nx: int,
) -> str:
    """`llama_tensor_get_type_impl` for LLAMA_FTYPE_MOSTLY_Q4_K_M, every counter kept as it keeps
    them. Its Falcon, Muse Spark and GLM5-next rules are those architectures', and its 70B rule
    never meets these (`Architecture`)."""
    n_expert = max(1, model.n_expert)
    if category == "output" or (tied and category == "token_embd"):
        if nx % ggml_types.block(new) != 0:
            return "ggml.q8_0"
        return "ggml.q6_k" if new != "ggml.q8_0" else new
    if category == "token_embd":
        return new
    if category in _ATTN_V:
        if _more_bits(counters["attention_wv"], n_attention_wv):
            new = "ggml.q6_k"
        if model.n_expert == 8:
            new = "ggml.q8_0"
        counters["attention_wv"] += 1
    elif category == "attn_k":
        if model.n_expert == 8:
            new = "ggml.q8_0"
    elif category == "ffn_down":
        i_layer, n_layer = _layer_info(counters["ffn_down"], model.n_layer, name, n_expert)
        if _more_bits(i_layer, n_layer):
            new = "ggml.q6_k"
        counters["ffn_down"] += 1
    elif category == "attn_output":
        if model.n_expert == 8:
            new = "ggml.q5_k"
    elif category in ("ffn_gate", "ffn_up"):
        _layer_info(counters[category], model.n_layer, name, n_expert)
        counters[category] += 1
    return new


def _layer_info(i_layer: int, n_layer: int, name: str, n_expert: int) -> tuple[int, int]:
    """`layer_info`: a MoE's layer is its name's block, the counter is everyone else's."""
    if n_expert > 1:
        m = re.match(r"blk\.(\d+)\.", name)
        if m is None or not 0 <= int(m.group(1)) < n_layer:
            raise ValueError(f"no layer for {name}")
        i_layer = int(m.group(1))
    return i_layer, n_layer


def _fallback(chosen: str, cols: int) -> str:
    """`tensor_type_fallback`: a row of partial blocks takes the next type down, else F16."""
    if cols % ggml_types.block(chosen) == 0:
        return chosen
    chosen = _FALLBACK.get(chosen, chosen)
    return chosen if cols % ggml_types.block(chosen) == 0 else _F16


# ------------------------------------------------------------------------------- resolving


def resolve(source: Checkpoint, selection: ArtifactQuantizationPlan, encoding: str) -> None:
    """Each selected weight no rule names takes its type in mix `encoding`, over each of the
    selection's components; one llama-quantize keeps leaves the selection, at source precision."""
    free = [t for t in selection.tensors if t.encoding is None]
    kept: set[tuple[str, str]] = set()
    for component in dict.fromkeys(t.component for t in free):
        held = source.components[component]
        shapes = {key: tuple(info.shape) for key, info in held.tensors.items()}
        names = [t.key for t in free if t.component == component]
        try:
            types = mix(encoding, model(held.config, shapes), names)
        except (KeyError, ValueError) as error:
            raise UnsupportedInput(
                f"{encoding}: {component}: {error}", code="quantization_plan"
            ) from error
        for tensor in free:
            if tensor.component == component:
                chosen = types[tensor.key]
                if chosen is None:
                    kept.add((tensor.component, tensor.key))
                tensor.encoding = chosen
    selection.tensors[:] = [t for t in selection.tensors if (t.component, t.key) not in kept]
