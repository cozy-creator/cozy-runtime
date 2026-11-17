"""llama.cpp's lanes as cozytensors lanes: ggml's block types (`ggml.*`, cozytensors spec §6),
written in memory by TensorFS's port of ggml's reference quantizers (`ggml-ref/1`), and
llama-quantize's per-tensor mixes.

A plan names a block type (`ggml.q8_0`: every selected weight in it) or a mix (`ggml.q4_k_m`):
each selected weight takes the type llama-quantize gives the tensor it becomes in a GGUF
(`llama_tensor_get_type`, `src/llama-quant.cpp`, llama.cpp b12466, MIT), ported for the mix's
file type: the same categories, counters, tensor order and fallbacks. Without an importance
matrix, as llama-quantize runs by default, the lane's bytes are its bytes.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

#: TensorFS's producer of every block type below (ggml's `quantize_row_<type>_ref`).
STRATEGY = "ggml-ref/1"

#: The block types TensorFS writes: elements per block.
BLOCKS = {
    "ggml.q8_0": 32,
    "ggml.q5_0": 32,
    "ggml.q5_1": 32,
    "ggml.q4_k": 256,
    "ggml.q5_k": 256,
    "ggml.q6_k": 256,
}

#: Each block type's bytes per block (`ggml-common.h`).
BYTES = {
    "ggml.q8_0": 34,
    "ggml.q5_0": 22,
    "ggml.q5_1": 24,
    "ggml.q4_k": 144,
    "ggml.q5_k": 176,
    "ggml.q6_k": 210,
}

#: llama-quantize's file types (`LLAMA_FTYPE_MOSTLY_*`) a plan may name: each mix's default type.
MIXES = {"ggml.q4_k_m": "ggml.q4_k"}

#: The weights a block type falls back to when a row is not whole blocks (`tensor_type_fallback`).
_FALLBACK = {"ggml.q4_k": "ggml.q5_0", "ggml.q5_k": "ggml.q5_1", "ggml.q6_k": "ggml.q8_0"}


def block(encoding: str) -> int:
    """The columns a weight in `encoding` must be a multiple of: a mix's are its fallbacks'."""
    if encoding in MIXES:
        return 32
    return BLOCKS[encoding]


# ----------------------------------------------------------------------------- GGUF names

#: A layer's tensor, after `layers.<i>.`, as llama.cpp's converters name it (gguf-py
#: `tensor_mapping.py`, and Qwen3.5's `_SHARED_INPUT_MERGES`: `in_proj_qkv` and `in_proj_z` become
#: one `attn_qkv_gate`, `in_proj_b` and `in_proj_a` one `ssm_beta_alpha`).
_LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.qkv_proj.weight": "attn_qkv.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv_gate.weight",
    "linear_attn.in_proj_z.weight": "attn_qkv_gate.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta_alpha.weight",
    "linear_attn.in_proj_a.weight": "ssm_beta_alpha.weight",
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
#: A routed expert's weight, one per expert here, one stack per layer there.
_EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")
#: The multi-token-prediction layer's own tensors (`_QwenMtpMixin`'s remapper), at block `n_layer`.
_NEXTN = {
    "fc.weight": "nextn.eh_proj.weight",
    "pre_fc_norm_embedding.weight": "nextn.enorm.weight",
    "pre_fc_norm_hidden.weight": "nextn.hnorm.weight",
    "norm.weight": "nextn.shared_head_norm.weight",
}
_MAIN = re.compile(r"^(?:.*\.)?(?<!mtp\.)layers\.(\d+)\.(.+)$")
_MTP = re.compile(r"^(?:.*\.)?mtp\.(?:layers\.(\d+)\.(.+)|([a-z_]+\.weight))$")


@dataclass(frozen=True, slots=True)
class Model:
    """The tensors of a model's GGUF: each name's ggml name, its blocks and experts."""

    #: name -> its GGUF tensor's name, for every name that is one (the vision tower is mmproj's)
    ggml: Mapping[str, str]
    n_layer: int
    n_expert: int


def gguf_names(names: Iterable[str]) -> Model:
    """Where each of a checkpoint's tensors lands in the GGUF llama.cpp's converter writes."""
    names = list(names)
    main = [int(m.group(1)) for n in names if (m := _MAIN.match(n)) and ".mtp." not in f".{n}"]
    layers = max(main, default=-1) + 1
    drafts = {int(m.group(1)) for n in names if (m := _MTP.match(n)) and m.group(1) is not None}
    if not drafts and any((m := _MTP.match(n)) and m.group(3) for n in names):
        drafts = {0}
    experts = {int(m.group(1)) for n in names if (m := _EXPERT.search(n))}
    out: dict[str, str] = {}
    for name in names:
        ggml = _ggml_name(name, layers)
        if ggml is not None:
            out[name] = ggml
    return Model(out, layers + len(drafts), len(experts) if len(experts) > 1 else 0)


def _ggml_name(name: str, layers: int) -> str | None:
    if name.endswith("embed_tokens.weight") and "visual" not in name and ".mtp." not in f".{name}":
        return "token_embd.weight"
    if name.endswith("lm_head.weight"):
        return "output.weight"
    if "visual" not in name and re.search(r"(?:^|\.)(?:language_model|model)\.norm\.weight$", name):
        return "output_norm.weight"
    if m := _MTP.match(name):
        if m.group(3) is not None:
            tail = _NEXTN.get(m.group(3))
            return None if tail is None else f"blk.{layers}.{tail}"
        return _layer_name(layers + int(m.group(1)), m.group(2))
    if (m := _MAIN.match(name)) and "visual" not in name:
        return _layer_name(int(m.group(1)), m.group(2))
    return None


def _layer_name(bid: int, rest: str) -> str:
    if e := _EXPERT.fullmatch(rest):
        return f"blk.{bid}.ffn_{e.group(2)}_exps.weight"
    return f"blk.{bid}.{_LAYER.get(rest, rest)}"


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


def mix(
    encoding: str,
    model: Model,
    selected: Collection[str],
    columns: Mapping[str, int],
) -> dict[str, str]:
    """Each selected weight's block type in mix `encoding`, as llama-quantize assigns the GGUF
    tensor it becomes; `columns` is each selected weight's row length. Weights merged into one GGUF
    tensor take its one type."""
    default = MIXES[encoding]
    ggml = model.ggml
    tensors = sorted(set(ggml.values()), key=_order)
    categories = {t: _category(t) for t in tensors}
    n_attention_wv = sum(1 for t in tensors if categories[t] in _ATTN_V)
    tied = "output.weight" not in categories
    counters = {"attention_wv": 0, "ffn_down": 0, "ffn_gate": 0, "ffn_up": 0}
    wanted = {ggml[name] for name in selected}
    width = {ggml[name]: columns[name] for name in selected}
    types: dict[str, str] = {}
    for tensor in tensors:
        if not _quantized(tensor):
            if tensor in wanted:
                raise ValueError(f"llama-quantize keeps {tensor} as it is")
            continue
        category = categories[tensor]
        chosen = _get_type(
            default, tensor, category, counters, n_attention_wv, model, tied, width.get(tensor, 0)
        )
        if tensor in wanted:
            types[tensor] = _fallback(chosen, width[tensor])
    return {name: types[ggml[name]] for name in selected}


def _quantized(name: str) -> bool:
    """`tensor_allows_quantization`'s name rules that reach these models; a 1-D tensor (norms,
    `ssm_a`, biases) never gets there, and none of them has a counter."""
    return (
        name.endswith("weight")
        and "_norm.weight" not in name
        and "ffn_gate_inp.weight" not in name
        and "ssm_conv1d" not in name
    )


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
    them. Falcon, Muse Spark, GLM5-next and the 70B rule are other architectures'."""
    n_expert = max(1, model.n_expert)
    if category == "output" or (tied and category == "token_embd"):
        if nx % BLOCKS[new] != 0:
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
    """`tensor_type_fallback`: a row of partial blocks takes the next type down."""
    if cols % BLOCKS[chosen] == 0:
        return chosen
    chosen = _FALLBACK.get(chosen, chosen)
    if cols % BLOCKS[chosen] != 0:
        raise ValueError(f"a row of {cols} is whole blocks of no ggml type (llama-quantize: F16)")
    return chosen
