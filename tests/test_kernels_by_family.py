"""Runtime picks GEMM and attention by the GPU's row (precision-tiers §2.1, §2.3, §2.4, §2.6):
the real selection code, with only the card's facts (its capability, its compiler's verdict)
stated, since this box has no card to read them from."""

from __future__ import annotations

import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.author import AttentionProducer, ConformanceError, Model
from cozy_runtime.author import _attention as author_attention
from cozy_runtime.internal import (
    accel,
    arch,
    attention,
    int8_linear,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    machine_kernels,
    static_interface,
)
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor import _attention_pin
from cozy_runtime.internal.executor_commands import Invoke, PrepareRequest
from test_authored_model_defaults import FIXTURE, imported, project, slots

_8, _88, _884 = frozenset({"int8"}), frozenset({"int8", "fp8"}), frozenset({"int8", "fp8", "fp4"})
ROWS = {
    80: ("mma.sync", _8),
    86: ("mma.sync", _8),
    89: ("mma.sync", _88),
    90: ("wgmma", _88),
    100: ("tcgen05", _884),
    103: ("tcgen05", frozenset({"fp8", "fp4"})),
    120: ("mma.sync", _884),
    121: ("mma.sync", _884),
}


@pytest.fixture(autouse=True)
def _unlearned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arch, "_LEARNED", {})


def _cuda(sm: int) -> DeviceFacts:
    return DeviceFacts(kind="cuda", name=f"sm{sm}", sm=sm, driver="", configuration="", index=0)


def test_each_row_from_its_capability_and_a_set_is_their_intersection() -> None:
    for sm, (family, formats) in ROWS.items():
        assert (arch.row(sm).family, arch.row(sm).formats()) == (family, formats), sm
        assert not arch.row(sm).inherited and arch.families(sm) == (family,)
    for sm in (0, 75):
        assert arch.row(sm) == arch.NONE and arch.families(sm) == ()
    thor = arch.row(110)
    assert thor.inherited and arch.families(110) == ("tcgen05", "wgmma", "mma.sync")
    assert (thor.family, thor.formats()) == ("tcgen05", _884)
    assert arch.families(95) == ("wgmma", "mma.sync")
    mixed = arch.common([arch.row(89), arch.row(86)])
    assert (mixed.family, mixed.formats()) == ("mma.sync", _8)
    hopper_b300 = arch.common([arch.row(90), arch.row(103)])
    assert (hopper_b300.family, hopper_b300.formats()) == ("wgmma", frozenset({"fp8"}))
    assert arch.common([arch.row(90), arch.NONE]).family == ""


# ------------------------------------------------------------------------------ GEMM


_STUBS = {
    "setuptools/__init__.py": (
        "import json\ndef setup(**kw):\n    print(json.dumps(kw['ext_modules'][0]))\n"
    ),
    "torch/__init__.py": "",
    "torch/utils/__init__.py": "",
    "torch/utils/cpp_extension.py": (
        "BuildExtension = object\n"
        "def CUDAExtension(name, sources, **kw):\n"
        "    return {'sources': sources, 'nvcc': kw['extra_compile_args']['nvcc']}\n"
    ),
}


def _builder(tmp_path: Path, sm: int, row: list[str]) -> tuple[list[str], str, set[str]]:
    """The real `int8_op/setup.py` for one card, setuptools and torch's extension stated."""
    stubs = tmp_path / "stubs"
    for relative, text in _STUBS.items():
        (stubs / relative).parent.mkdir(parents=True, exist_ok=True)
        (stubs / relative).write_text(text)
    env = {"PYTHONPATH": str(stubs), "COZY_KERNEL_SM": str(sm), "COZY_KERNEL_ROW": " ".join(row)}
    found = subprocess.run(
        [sys.executable, "setup.py"],
        cwd=kernel_sources.OWN["cozy-int8-src"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    ext = json.loads(found.stdout.splitlines()[-1])
    [gencode] = [flag for flag in ext["nvcc"] if flag.startswith("-gencode")]
    defines = {f.removeprefix("-DCOZY_") for f in ext["nvcc"] if f.startswith("-DCOZY_")}
    return ext["sources"], gencode.rsplit("=", 1)[-1], defines


@pytest.fixture
def toolchain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry carrying the op's CUTLASS, and the torch/compiler it would build against."""
    root = tmp_path / "sources"
    root.mkdir()
    cutlass = {"version": "0", "file": "cutlass.tar.gz", "sha256": "0" * 64}
    (root / "index.json").write_text(json.dumps({"schema": 1, "sources": {"cutlass": cutlass}}))
    kernel_sources.registry.cache_clear()
    monkeypatch.setattr(kernel_sources, "job", partial(kernel_sources.job, root=root))
    monkeypatch.setattr(kernel_sources, "torch_build", lambda: ("2.14.0", "13.0"))
    monkeypatch.setattr(kernel_sources, "compiler", lambda: "toolkit")


_MMA, _TMA, _FP8 = "gemm_sm80.cu", "gemm_tma.cu", "gemm_fp8.cu"
#: card -> (its target, its GEMM sources): the TMA int8 GEMM and the CUTLASS fp8 one, or not
GEMM = {
    80: ("sm_80", [_MMA]),
    86: ("sm_86", [_MMA]),
    89: ("sm_89", [_MMA]),
    90: ("sm_90a", [_TMA, _FP8]),
    100: ("sm_100a", [_TMA, _FP8]),
    103: ("sm_103a", [_MMA, _FP8]),
    120: ("sm_120a", [_MMA, _FP8]),
    121: ("sm_121a", [_MMA, _FP8]),
    110: ("sm_110a", [_TMA, _FP8]),  # Thor inherits tcgen05
}


@pytest.mark.parametrize("sm", sorted(GEMM))
def test_the_8bit_op_builds_by_row_whatever_the_lane(
    sm: int, tmp_path: Path, toolchain: None
) -> None:
    """One op library per card serves every lane (int8, fp8), built for the row's family:
    12.1 and an unnamed 11.0 take their family's GEMMs."""
    job = kernel_sources.job(machine_kernels.INT8, sm)
    assert json.loads(job.key.inputs)["row"] == job.spec["row"]
    built, gencode, defines = _builder(tmp_path, sm, job.spec["row"])
    assert (gencode, built[2:]) == GEMM[sm]
    expected = {arch.row(sm).family.replace(".", "_").upper()}
    expected |= {"TMA"} if _TMA in built else set()
    expected |= {"FP8_GEMM"} if _FP8 in built else set()
    assert defines == expected


@pytest.mark.parametrize(("builds", "code"), [({"mma.sync"}, 0), (set(), 1)])
def test_an_unnamed_card_whose_target_takes_only_mma_sync_inherits_it(
    builds: set[str],
    code: int,
    tmp_path: Path,
    toolchain: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fake 13.0 whose compiler refuses tcgen05 and wgmma: the boot tries the newest family
    first and lands on the first that builds. The rejected families are probes: the boot exits
    0 with one warning; only the card's own family failing (here, every one) counts. The
    executor's op, row, attention cells and run warning follow the same verdicts."""
    tried: list[str] = []

    def verdict(_store: kernel_cache.Store, job: kernel_compile.Job, *_: str) -> Any:
        family = job.spec["row"][0]
        tried.append(family)
        state = "ready" if family in builds else "failed"
        return kernel_compile.State(job.key.kernel, state, path=str(tmp_path / family))

    monkeypatch.setattr(kernel_compile, "submit", verdict)
    monkeypatch.setattr(kernel_compile, "wait", verdict)
    monkeypatch.setattr(kernel_compile, "waiting", None)
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [None])
    monkeypatch.setattr(accel, "device_capability", lambda entry: 130)
    monkeypatch.setenv(kernel_cache.OWN_ENV, str(tmp_path / "u0"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-x")
    assert arch.row(130).family == "tcgen05"  # before any build: the newest candidate
    assert machine_kernels.main() == code
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert list(dict.fromkeys(tried)) == ["tcgen05", "wgmma", "mma.sync"]
    [warning] = [row["warning"] for row in lines if "warning" in row]
    assert warning.startswith("sm_130 (GPU-x) has no row; served as mma.sync with fp8 int8")
    assert [row["probe"] for row in lines if "probe" in row] == ["tcgen05", "wgmma"]
    assert (lines[-1]["kernels"], lines[-1]["failed"]) == (1, code)

    monkeypatch.setattr(arch, "_LEARNED", {})
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [kernel_cache.Store(tmp_path / "u1")])
    if builds:
        assert int8_linear._compiled(int8_linear.OP, 130) == tmp_path / "mma.sync" / "site"
    else:
        with pytest.raises(kernel_sources.Absent):
            int8_linear._compiled(int8_linear.OP, 130)
        return
    row = arch.row(130)
    assert row.inherited and (row.family, row.formats()) == ("mma.sync", _88)
    assert attention.walk(row, 8) == ["kitchen-int8", "sageattention", "sdpa"]
    assert arch.inherited(90, "H100") == ""
    built, gencode, defines = _builder(
        tmp_path, 130, kernel_sources.job("cozy-int8", 130).spec["row"]
    )
    assert built[2:] == ["gemm_sm80.cu"] and gencode == "sm_130" and defines == {"MMA_SYNC"}


# ------------------------------------------------------------------------- attention cells


_ADA_8 = (
    ["kitchen-int8", "sageattention", "sdpa"],
    ["sol-attn", "kitchen-sol", "kitchen-int8", "sageattention", "sdpa"],
)
_HOPPER_8 = (
    ["sageattention", "flash-attn3", "sdpa"],
    ["sol-attn-sage2-fixed-kmean", "sol-attn", "sageattention", "flash-attn3", "sdpa"],
)
_BLACKWELL_8 = (
    ["flash-attn4-fp8", "cudnn", "sdpa"],
    ["sol-attn", "flash-attn4-fp8", "cudnn", "sdpa"],
)
_AMPERE_8 = (["kitchen-int8", "sdpa"], ["kitchen-sol", "kitchen-int8", "sdpa"])
_RTX_4 = (
    ["sageattention3-fp4", *_ADA_8[0]],
    ["kitchen-sol-fp4-fixed-means", "kitchen-sol-fp4-lowmem", "sageattention3-fp4", *_ADA_8[1]],
)
#: card -> level -> (a site's walk, a sparse site's walk); no level-4 cell is level 8's
CELLS: dict[int, dict[int, tuple[list[str], list[str]]]] = {
    86: {16: (["sdpa"], ["sdpa"]), 8: _AMPERE_8, 4: _AMPERE_8},
    89: {16: (["sdpa"], ["sol-attn", "sdpa"]), 8: _ADA_8, 4: _ADA_8},
    90: {
        16: (["flash-attn3", "sdpa"], ["sol-attn", "flash-attn3", "sdpa"]),
        8: _HOPPER_8,
        4: _HOPPER_8,
    },
    100: {16: (["cudnn", "sdpa"], ["sol-attn", "cudnn", "sdpa"]), 8: _BLACKWELL_8, 4: _BLACKWELL_8},
    120: {16: (["sdpa"], ["sol-attn", "sdpa"]), 8: _ADA_8, 4: _RTX_4},
}
CELLS[80], CELLS[103], CELLS[121] = CELLS[86], CELLS[100], CELLS[120]


@pytest.mark.parametrize("sm", sorted(CELLS))
def test_every_row_has_one_cell_per_level(sm: int) -> None:
    for level, (dense, sparse) in sorted(CELLS[sm].items()):
        assert attention.walk(arch.row(sm), level) == dense, (sm, level)
        assert attention.walk(arch.row(sm), level, layout=True) == sparse, (sm, level)
    for level in attention.LEVELS:
        assert attention.walk(arch.NONE, level, layout=True) == ["sdpa"]


def test_a_package_producer_stands_in_before_its_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """H3's recipes on 12.x: each producer is tried just before the Runtime kernel it is
    served through; under Ulysses one without context parallelism steps to its base."""
    for name, base in (
        ("p-fixed", "kitchen-sol-fp4-fixed-means"),
        ("p-low", "kitchen-sol-fp4-lowmem"),
    ):
        monkeypatch.setitem(
            author_attention._PRODUCERS,
            name,
            AttentionProducer(name, base, lambda *a, **k: None, lambda module: None),
        )
    sparse = attention.walk(arch.row(120), 4, layout=True)
    assert sparse[:5] == [
        "p-fixed",
        "kitchen-sol-fp4-fixed-means",
        "p-low",
        "kitchen-sol-fp4-lowmem",
        "sageattention3-fp4",
    ]
    assert "p-fixed" not in attention.walk(arch.row(90), 4, layout=True)
    assert attention.sparse_site(object())


_PY = kernel_sources.PYTHON
#: card -> what its boot builds for attention: each level's first kernel the row runs
BOOT = {
    86: {"comfy-kitchen"},
    89: {_PY, "comfy-kitchen"},
    90: {_PY, "sageattention", "flash-attn3"},
    100: {_PY},
    103: {_PY},
    120: {_PY, "comfy-kitchen", "sageattn3"},
}


def test_the_boot_builds_each_levels_first_kernel_of_the_rows_cells() -> None:
    for sm, artifacts in BOOT.items():
        assert {c.artifact for c in machine_kernels.boot(arch.row(sm)) if c.artifact} == artifacts
    assert machine_kernels.boot(arch.NONE) == []


# ----------------------------------------------------------------------- the level rule


def test_attention_bits_prefixes_name_sites_and_the_longest_wins() -> None:
    bits = {
        "fl2va_dit": 4,
        "transformer.single_transformer_blocks.": 8,
        "transformer.single_transformer_blocks.3.": 16,
    }
    assert attention.level(bits, "fl2va_dit", "transformer_blocks.7.attn") == 4
    assert attention.level(bits, "base/fl2va_dit", "token_refiner.0.attn") == 4
    assert attention.level(bits, "fl2va_dit_extra", "blocks.0") == 16
    assert attention.level(bits, "transformer", "single_transformer_blocks.12.attn") == 8
    assert attention.level(bits, "transformer", "single_transformer_blocks.3.attn") == 16
    assert attention.level(bits, "transformer", "transformer_blocks.0.attn") == 16
    assert attention.level(bits, "video_vae", "mid.attn") == 16
    assert attention.level(8, "anything", "x") == 8


def test_attention_bits_is_a_class_keyword_with_two_forms_inherited() -> None:
    class Whole(Model[object], attention_bits=8):
        pass

    class Child(Whole):
        pass

    class Prefixes(Model[object], attention_bits={"dit": 4, "dit.blocks.0.": 16}):
        pass

    assert (Model.__attention_bits__, Whole.__attention_bits__, Child.__attention_bits__) == (
        16,
        8,
        8,
    )
    assert Prefixes.__attention_bits__ == {"dit": 4, "dit.blocks.0.": 16}
    for value in (6, 8.0, True, "8", {}, {"dit": 6}, {"": 8}, {1: 8}):
        with pytest.raises(ConformanceError) as refused:
            type("Bad", (Model,), {}, attention_bits=value)
        assert refused.value.code == "attention_bits_value", value


@pytest.mark.parametrize(
    "keyword", ['attention_bits={"dit": 4, "dit.blocks.0.": 16}', "attention_bits=8"]
)
def test_describe_reads_attention_bits_by_import_and_from_source(
    tmp_path: Path, keyword: str
) -> None:
    """Runtime reads `attention_bits` at selection; describe accepts it and carries nothing."""
    (tmp_path / "plain").mkdir()
    plain = slots(static_interface.build(project(tmp_path / "plain")))
    source = FIXTURE.replace("class Diffusion(Model):", f"class Diffusion(Model, {keyword}):")
    root = project(tmp_path, source)
    static = static_interface.build(root)
    assert imported(root) == static and slots(static) == plain


@pytest.mark.parametrize("value", ["6", '{"dit": 6}', "{}", "{1: 8}"])
def test_describe_refuses_out_of_vocabulary_attention_bits(tmp_path: Path, value: str) -> None:
    root = project(
        tmp_path,
        FIXTURE.replace(
            "class Diffusion(Model):", f"class Diffusion(Model, attention_bits={value}):"
        ),
    )
    with pytest.raises(ConformanceError, match="attention_bits"):
        imported(root)
    with pytest.raises(ConformanceError, match="attention_bits"):
        static_interface.build(root)


def test_a_request_at_16_bits_is_the_exact_override_and_a_pin_wins() -> None:
    prepared = msgspec.json.decode(
        b'{"cmd":"prepare_request","request_id":"r","entrypoint":"e","payload":null,"bits":16}',
        type=PrepareRequest,
    )
    assert (
        prepared.bits == 16 and _attention_pin(prepared.attention_kernel, prepared.bits) == "exact"
    )
    assert b"bits" not in msgspec.json.encode(
        Invoke(request_id="r", entrypoint="e", spool="s", deadline_s=None)
    )
    assert _attention_pin("", 8) == _attention_pin("", 0) == ""
    assert _attention_pin("dit=sdpa", 16) == "dit=sdpa"


# ------------------------------------------------- the level on real Diffusers sites


def _sites(*names: str) -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    processor: Any = WanAttnProcessor
    return {
        name: torch.nn.ModuleList(
            [WanAttention(dim=256, heads=2, dim_head=128, processor=processor())]
        ).to(torch.bfloat16)
        for name in names
    }


def _int8(site: Any) -> None:
    """Q/K/V as int8-convrot leaves: an int8 lane's projections."""
    import torch

    from cozy_runtime.author.kernels import INT8_CONVROT
    from cozy_runtime.internal.encoding.leaves import _native_leaf_class

    for name in ("to_q", "to_k", "to_v"):
        linear = getattr(site, name)
        roles = {
            "data": torch.zeros(linear.out_features, linear.in_features, dtype=torch.int8),
            "scale": torch.ones(linear.out_features),
        }
        leaf = _native_leaf_class(torch)(
            None, roles, None, torch.bfloat16, INT8_CONVROT, "int8-convrot256/1",
            linear.in_features, linear.out_features,
        )  # fmt: skip
        setattr(site, name, leaf)


def _ready(*names: str) -> list[attention.Ready]:
    """Kernels as resolution hands them over, with Diffusers' real members; no card to launch."""
    return [
        attention.Ready(attention.BY_NAME[n], attention._member(attention.BY_NAME[n]))
        for n in names
    ]


#: (projections, attention_bits) -> the kernel an Ada card's cell serves
LEVELS = {
    ("bf16", 16): "sdpa",
    ("int8", 16): "sdpa",
    ("bf16", 8): "kitchen-int8",
    ("int8", 8): "kitchen-int8",
    ("bf16", 4): "kitchen-int8",  # Ada has no level-4 cell: level 8
    ("int8", 4): "kitchen-int8",
}


@pytest.mark.parametrize(("projections", "bits"), sorted(LEVELS))
def test_the_level_is_the_models_tolerance_not_its_weight_lane(projections: str, bits: int) -> None:
    roots = _sites("transformer", "vae")
    if projections == "int8":
        _int8(roots["transformer"][0])
    ready = _ready("kitchen-int8", "sageattention", "sdpa")
    applied = attention.install(ready, roots, device=_cuda(89), bits={"transformer": bits})
    assert applied.hosts == {"transformer": {LEVELS[projections, bits]: 1}, "vae": {"sdpa": 1}}
    assert not applied.fallbacks


def test_bits_16_puts_every_site_at_level_16_for_the_request() -> None:
    roots = _sites("dit")
    ready = _ready("sageattention", "flash-attn3", "sdpa")
    attention.install(ready, roots, device=_cuda(90), bits=8)
    prepared = attention.snapshot(roots)
    assert attention.observed(roots).totals() == {"sageattention": 1}
    attention._READY.update({(entry.name, _cuda(90)): entry for entry in ready})
    try:
        swap = attention.plan(_cuda(90), roots, attention.EXACT)
        with attention.override(prepared, roots, swap, attention.EXACT) as served:
            assert served.totals() == {"flash-attn3": 1} and served.pin == "exact"
    finally:
        for entry in ready:
            attention._READY.pop((entry.name, _cuda(90)), None)
    assert attention.observed(roots).totals() == {"sageattention": 1}
