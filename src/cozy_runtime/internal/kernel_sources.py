"""Kernels as pinned SOURCE, compiled on the machine that runs them.

`PINS` is every kernel source this Runtime builds from, by version, release tag or commit and
sha256. Normal jobs use those pins without an image index. A missing compile acquires only
the sources it opens beneath sealed COZY_KERNEL_CACHE, through the Runtime egress/storage
boundary. Matching read-only preseeded archives may be reused. Explicit registry roots keep
the component/custom source layout:

    <root>/index.json
        {"schema": 1, "sources": {"<name>": {"version": str, "file": str, "sha256": str,
                                             "origin": str, "package"?: str}}}
    <root>/<file>   a .tar.gz source tree with one top directory, or a wheel

A `Recipe` names the sources an artifact builds from and how. `setup` runs the upstream
project's own `setup.py build` beside the GPU it will serve, so its arch detection picks this
card and no upstream patch is needed; a one-line incompatibility with this torch is a text
replacement applied at build time, in the open, in the recipe. `python` unpacks every Python
source (wheels, and a tree's pure package by its `package` path) into one importable tree: the
CuTe DSL, TVM-FFI and the JIT kernels built on them (Sol, FlashAttention-4, FlashInfer). Every
source's digest is a key input, with the card's architecture, the driver, torch, CUDA, Python
and the compiler, so another image, card or driver builds its own entry and never loads this
one's.
"""

from __future__ import annotations

import base64
import email.parser
import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata, util
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

from cozy_runtime.internal import accel, config, egress, kernel_cache, kernel_compile

ROOT = Path("/opt/cozy/kernel-sources")
INDEX = "index.json"
#: The artifact every Python source unpacks into.
PYTHON = "kernel-python"
KITCHEN_WHEEL = "comfy-kitchen-wheel"
#: Recipe fields outside a key: how many compiler processes a build runs, and other cards'
#: text fixes.
_UNKEYED = ("extensions", "unit_bytes", "replace_on", "build_deps")
#: What `setup.py` itself imports, from the registry when the environment lacks it; build-only.
_BUILD_DEPS = ("setuptools", "wheel", "packaging")


class Absent(Exception):
    """This machine cannot build the artifact at all: a missing source or toolkit."""


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    version: str
    file: str
    sha256: str
    origin: str = ""
    #: a tree source's pure Python package directory, unpacked by the `python` recipe
    package: str = ""
    #: Runtime-pinned sources are acquired only when their builder opens them.
    acquire: bool = False
    preseed: str = ""


@dataclass(frozen=True, slots=True)
class Pin:
    """A source by its version, release tag or commit, and the digest its download must have."""

    name: str
    version: str
    url: str
    sha256: str
    package: str = ""
    #: the archive paths recipes read; a worker image carries only these (`fetch`)
    keep: tuple[str, ...] = ()


_HEADERS = ("include", "tools/util/include", "LICENSE.txt")
_GITHUB = "https://github.com/{}/archive/{}.tar.gz"
_PYPI = "https://files.pythonhosted.org/packages/"
PINS: tuple[Pin, ...] = (
    Pin(
        "sageattention",
        "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5",
        _GITHUB.format("thu-ml/SageAttention", "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"),
        "74b6667164f9e368e3799bc2ab59b9b08c4591630f1c6029560208b6fcf354c4",
        keep=(
            "setup.py",
            "csrc",
            "sageattention",
            "sageattention3_blackwell",
            "README.md",
            "LICENSE",
        ),
    ),
    Pin(
        "cutlass",
        "098de2a652cf8f00fd70b2df54051c7eccbb855a",
        _GITHUB.format("NVIDIA/cutlass", "098de2a652cf8f00fd70b2df54051c7eccbb855a"),
        "a8a1b952c7f6179bc8076f20d45859f18bd81f21f3d22b6766d7f3f0cceb5d54",
        keep=_HEADERS,  # SageAttention3's (Blackwell)
    ),
    Pin(
        "flash-attention",
        "a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2",  # v2.8.3.post1
        _GITHUB.format("Dao-AILab/flash-attention", "a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2"),
        "e06c229e24ad61bcb31eea9446a5e6484cd8b3b5d10d90b0f8f48cc2b0d4c431",
    ),
    Pin(
        "flash-attention-cutlass",
        "dc4817921edda44a549197ff3a9dcf5df0636e7b",  # flash-attention's csrc/cutlass submodule
        _GITHUB.format("NVIDIA/cutlass", "dc4817921edda44a549197ff3a9dcf5df0636e7b"),
        "f2a3a9df5e6f010c8b02716aa2644a6f071827fafa606fac5f5241cab6a1ab56",
        keep=_HEADERS,
    ),
    Pin(
        KITCHEN_WHEEL,
        "0.2.37",
        _PYPI + "19/04/d24939d30b077679bfc8bf4a53c7392f9d8650df7ce69164238a673753a1/"
        "comfy_kitchen-0.2.37-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "03cca0d10c314aaa13bff8f9422d03974221a344162b23d4f10d27eb2a35d0e8",
    ),
    Pin(
        "nanobind",
        "3.1.0",
        _PYPI + "36/c1/6fd56f7a02a478d54a1a3e6000eccd34b446b1c45cbf0d002fe5d2937126/"
        "nanobind-3.1.0-py3-none-any.whl",
        "96984dbe8154ffb86e653b1b60aaeed6985c710c5e75ad0279eb87943c66292d",
    ),
    Pin(
        "comfy-kitchen",
        "be003b7c23c5b01328657955b8bc5d3f073d868e",  # v0.2.37: reviewed H3 matrix
        _GITHUB.format("Comfy-Org/comfy-kitchen", "be003b7c23c5b01328657955b8bc5d3f073d868e"),
        "364150b13466f51276c08fb77fa54328c9917ea90230151f2abed75a5c362e69",
    ),
    Pin(
        "sol-attn",
        "8e0db4fa562d727ea28b8d63015c196db7d97cae",
        _GITHUB.format("NVlabs/Sana", "8e0db4fa562d727ea28b8d63015c196db7d97cae"),
        "6e8c00037a2de9e48522c79c8984576e8e227fa7ab770be9502214c263dc1f01",
        package="techniques/sparse_backends/sol_attn",
        keep=("techniques/sparse_backends/sol_attn",),
    ),
    Pin(
        "apache-tvm-ffi",
        "0.1.14.post1",
        _PYPI + "e8/db/68786fdc0bfb3a9782df7304c1e577b9c5a7d14c75e98e06571cc1869b3f/"
        "apache_tvm_ffi-0.1.14.post1-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "eb759c039866b541ec2cb07d498082f8ba6ded74d58d9196180a5932f365c077",
    ),
    Pin(
        "einops",
        "0.8.2",
        _PYPI + "2a/09/f8d8f8f31e4483c10a906437b4ce31bdf3d6d417b73fe33f1a8b59e34228/"
        "einops-0.8.2-py3-none-any.whl",
        "54058201ac7087911181bfec4af6091bb59380360f069276601256a76af08193",
    ),
    Pin(
        "flash-attn-4",
        "4.0.0b32",
        _PYPI + "40/d0/b1e9ab8eefc9e67179a372c3e1a8e9f83d963c3a8e5bd4c738e55dd134c0/"
        "flash_attn_4-4.0.0b32-py3-none-any.whl",
        "9299da3524d579d84eb92954d3243aa095d55ebdafcf80d6a9766fd96fe8a562",
    ),
    Pin(
        "flashinfer-python",
        "0.7.0",
        _PYPI + "55/72/0ccab3f65c91b24e42fbb9abb9453639b738cd4af90718e95ee0431ac53f/"
        "flashinfer_python-0.7.0-py3-none-any.whl",
        "11e564820cde80ec13d351b68fc44cb54a537268bca135204abafc5fadd50240",
    ),
    Pin(
        "nvidia-cutlass-dsl",
        "4.8.0",
        _PYPI + "e4/f4/e96f452f7975dd2fd4d2a4af88c63205431dad7efbe63aff6e3a4b1ada17/"
        "nvidia_cutlass_dsl-4.8.0-py3-none-any.whl",
        "24f996e0e5fa88c8b417f4893e615af135bdfdfdbdbed7e239faf36a88df711b",
    ),
    Pin(
        "nvidia-cutlass-dsl-libs-base",
        "4.8.0",
        _PYPI + "12/1c/2f7a13626af1edfefb5ad69234d47e3b212064a42ad2e6db59351b5ae002/"
        "nvidia_cutlass_dsl_libs_base-4.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "5bdc80989b0fbbac3d58960057e74e2ce8676fafa76b852d9422b9c4b9b74bdd",
    ),
    Pin(
        "nvidia-cutlass-dsl-libs-core",
        "4.8.0",
        _PYPI + "85/66/d05f0ae372b80fc757e59d5211d93483c6ff721f574cb68cad84ceb87157/"
        "nvidia_cutlass_dsl_libs_core-4.8.0-py3-none-any.whl",
        "b32716d2ed10c185cdbbb6b5c6fca26a3113a60e6e2fe4e2f0cb15966422b621",
    ),
    Pin(
        "nvidia-cutlass-dsl-libs-cu13",
        "4.8.0",
        _PYPI + "df/75/6c52c58fc0fdab223feb9162886736390a6f902183d24df1508ac620a69d/"
        "nvidia_cutlass_dsl_libs_cu13-4.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "463cd7f0c452ffeeecb530a15aeb1b10cbbd2c8da6c5aa2fecf65f836258a732",
    ),
    Pin(
        "quack-kernels",
        "0.6.5",
        _PYPI + "73/d8/f5e56c38b286308d115a171e3c5c44f24e9d03980c3cfd22588df40261fa/"
        "quack_kernels-0.6.5-py3-none-any.whl",
        "df1ddd31366c82eb9692fe087c671209189dcbde53c909c81a865d4ee66b158d",
    ),
    Pin(
        "torch-c-dlpack-ext",
        "0.1.5",
        _PYPI + "e2/79/a914539b4785f3e44f891aa012a886edb8bc10fe081c440981c57543ce21/"
        "torch_c_dlpack_ext-0.1.5-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "e6f9da4bb9af70e27facc777458be62e10dbbbddda7672d16138db0553c5a524",
    ),
    Pin(
        "packaging",
        "26.3",
        _PYPI + "63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/"
        "packaging-26.3-py3-none-any.whl",
        "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c",
    ),
    Pin(
        "setuptools",
        "84.0.0",
        _PYPI + "95/9c/c510029fc6ef33a6275cd2c5d3cecd6613dfd6aa401d57c54f1c18852ccf/"
        "setuptools-84.0.0-py3-none-any.whl",
        "51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670",
    ),
    Pin(
        "wheel",
        "0.48.0",
        _PYPI + "2e/29/69cfbb602cd91690c55d38ba9fe53e6a7e76a6fa647bf38f19c138d25449/"
        "wheel-0.48.0-py3-none-any.whl",
        "3217dcc807155e45db462d7ef2431f5ddda0d7273b700d05a67b271ceb1287ab",
    ),
)


@dataclass(frozen=True, slots=True)
class Recipe:
    name: str
    kind: str  # setup | python
    sources: tuple[str, ...] = ()
    #: the directory holding `setup.py`, inside the first source's tree
    subdir: str = ""
    #: another source's tree -> where it goes inside the first (a submodule the archive lacks)
    place: Mapping[str, str] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    #: (file, old, new): a text fix applied to the extracted tree before building
    replace: tuple[tuple[str, str, str], ...] = ()
    #: more text fixes for one card architecture only (`sm`)
    replace_on: Mapping[int, tuple[tuple[str, str, str], ...]] = field(default_factory=dict)
    #: the extension modules upstream builds, all at once
    extensions: int = 1
    #: the peak memory of one of its compiler jobs, which the machine's compile ledger holds
    #: for it: SageAttention2's largest measured 3.2 GB (cicc on its sm89 kernels)
    unit_bytes: int = 4 << 30
    #: Recipe-specific, pinned build tools; staged ahead of ambient packages and keyed by SHA.
    build_deps: tuple[str, ...] = ()
    #: recipes whose kernels this build also holds under the same module: imported first, it
    #: serves them too, so one process never needs both
    serves: tuple[str, ...] = ()


#: The cards upstream SageAttention builds its SM89 and SM80 kernel families for.
_SAGE_SM89 = "HAS_SM89 or HAS_SM90 or HAS_SM100 or HAS_SM120 or HAS_SM121"
_SAGE_SM80 = "HAS_SM80 or HAS_SM86 or " + _SAGE_SM89
_SAGE = {"CXX_APPEND_FLAGS": "-std=c++20", "NVCC_APPEND_FLAGS": "-std=c++20 --threads=1"}
#: FA3 inference forward only, one KV split and no packed GQA: exactly what
#: `attention_upstream.fa3` calls (`num_splits=1`), a third of upstream's forward units.
_FA3_OFF = ("BACKWARD", "SM80", "PAGEDKV", "APPENDKV", "SOFTCAP", "LOCAL", "SPLIT", "PACKGQA")
#: FA3's head dimensions and dtypes the packages' attention sites run: H3's and Qwen Image's
#: transformers at 128 in BF16, H3's video VAE at 64 in FP16 (`attention.CANDIDATES`).
_FA3_SHAPES = (("128", "bf16"), ("64", "fp16"))
#: Upstream updates the submodule with git, and off CUDA 12.8 downloads NVIDIA's 12.6/12.8
#: compilers over this machine's toolkit; build with the toolkit present. It hard-codes C++17
#: and reads no append variable; only `flash_api.cpp`, a host unit, includes Torch.
_FA3_TOOLKIT = (
    ("hopper/setup.py", '"cxx": ["-O3", "-std=c++17"', '"cxx": ["-O3", "-std=c++20"'),
    (
        "hopper/setup.py",
        'subprocess.run(["git", "submodule", "update", "--init", "../csrc/cutlass"])',
        "pass",
    ),
    ("hopper/setup.py", 'if bare_metal_version != Version("12.8"):', "if False:"),
)
_FA3_UNIT = "if not (packgqa and (paged or split))]"
_HALF_64_DV = (
    "return run_mha_fwd_<Arch, cutlass::half_t, 64, {}, Split, PagedKVNonTMA, Has_softcap, "
    "PackGQA>(params, stream);"
)
_HALF_128 = "if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t"
_LOCAL_CHECK = (
    'TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");'
)
#: Torch 2.14's headers need C++20; every recipe below says so in upstream's own terms.
RECIPES: dict[str, Recipe] = {
    recipe.name: recipe
    for recipe in (
        Recipe(
            "sageattention",
            "setup",
            ("sageattention",),
            env=_SAGE,  # upstream appends these after its C++17 defaults
            # _qattn_sm80, _qattn_sm89, _qattn_sm90 (SM90 only) and _fused.
            extensions=4,
            # Upstream builds every kernel family the card can run; build the one
            # `sageattention` calls on it: SM90's own, SM89's on SM89/SM120. A family not
            # built is one upstream's `core` tolerates (`sageattention-full` builds them all).
            replace_on={
                sm: tuple(("setup.py", f"if {old}:", f"if {new}:") for old, new in fixes)
                for sm, fixes in (
                    (90, ((_SAGE_SM80, "False"), (_SAGE_SM89, "False"))),
                    (89, ((_SAGE_SM80, "False"),)),
                    (120, ((_SAGE_SM80, "False"),)),
                )
            },
        ),
        Recipe(
            "sageattention-full",
            "setup",
            ("sageattention",),
            env=_SAGE,
            extensions=4,
            serves=("sageattention",),
        ),
        Recipe(
            "sageattn3",
            "setup",
            ("sageattention", "cutlass"),
            subdir="sageattention3_blackwell",
            # Upstream clones CUTLASS HEAD when the directory is missing; place the pinned one.
            place={"cutlass": "sageattention3_blackwell/csrc/cutlass"},
            # Upstream hard-codes C++17 for both compilers and reads no append variable.
            replace=(("sageattention3_blackwell/setup.py", '"-std=c++17"', '"-std=c++20"'),),
        ),
        Recipe(
            "flash-attn3",
            "setup",
            ("flash-attention", "flash-attention-cutlass"),
            subdir="hopper",
            place={"flash-attention-cutlass": "csrc/cutlass"},
            env={
                f"FLASH_ATTENTION_DISABLE_{part}": "TRUE"
                for part in (*_FA3_OFF, "FP8", "HDIM96", "HDIM192", "HDIM256", "VARLEN")
            },
            # One unit per shape in `_FA3_SHAPES` (4.6-4.7 GB max RSS each, cicc), not
            # upstream's units of every head dimension (14.2 GB, ~190 s apiece on H100 pods);
            # dispatch to a shape not built is compiled out, and so are causal and variable
            # length attention, which no package's site runs (bidirectional DiTs and VAEs).
            unit_bytes=5 << 30,
            replace=(
                (
                    "hopper/static_switch.h",
                    "CAUSAL_CONST_NAME = true;" + " " * 42 + "\\\n        return __VA_ARGS__();",
                    "CAUSAL_CONST_NAME = false;" + " " * 41 + "\\\n        return __VA_ARGS__();",
                ),
                (
                    "hopper/flash_api.cpp",
                    _LOCAL_CHECK,
                    _LOCAL_CHECK
                    + '\n    TORCH_CHECK(!params.is_causal, "This build is non-causal.");',
                ),
                (
                    "hopper/setup.py",
                    'HEAD_DIMENSIONS_FWD = ["all", "diff"]',
                    f"HEAD_DIMENSIONS_FWD = {sorted({h for h, _ in _FA3_SHAPES})!r}",
                ),
                (
                    "hopper/setup.py",
                    _FA3_UNIT,
                    _FA3_UNIT[:-1] + f" and (hdim, dtype) in {_FA3_SHAPES!r}]",
                ),
                (
                    "hopper/flash_api.cpp",
                    "if (params.is_bf16) {\n            #ifndef FLASHATTENTION_DISABLE_HDIM64",
                    "if (params.is_bf16) {\n            #if 0",
                ),
                (
                    "hopper/flash_api.cpp",
                    f"#ifndef FLASHATTENTION_DISABLE_HDIM128\n            {_HALF_128}",
                    f"#if 0\n            {_HALF_128}",
                ),
                *(
                    ("hopper/flash_api.cpp", _HALF_64_DV.format(dv), "TORCH_CHECK(false);")
                    for dv in (512, 256)
                ),
                *_FA3_TOOLKIT,
            ),
        ),
        # Every forward head dimension and dtype, FP8 included, one unit per dimension: what
        # the `flash-attn3-fp8` pin runs, built when a pin asks for it.
        Recipe(
            "flash-attn3-full",
            "setup",
            ("flash-attention", "flash-attention-cutlass"),
            subdir="hopper",
            place={"flash-attention-cutlass": "csrc/cutlass"},
            env={f"FLASH_ATTENTION_DISABLE_{part}": "TRUE" for part in _FA3_OFF},
            unit_bytes=5 << 30,
            serves=("flash-attn3",),
            replace=(
                (
                    "hopper/setup.py",
                    'HEAD_DIMENSIONS_FWD = ["all", "diff"]',
                    'HEAD_DIMENSIONS_FWD = ["64", "96", "128", "192", "256", "64_256", "64_512",'
                    ' "192_128"]',
                ),
                (
                    "hopper/setup.py",
                    _FA3_UNIT,
                    _FA3_UNIT[:-1] + ' and os.path.exists(f"instantiations/flash_fwd_hdim'
                    '{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu")]',
                ),
                *_FA3_TOOLKIT,
            ),
        ),
        Recipe("comfy-kitchen", "setup", ("comfy-kitchen",), build_deps=("nanobind",)),
        Recipe(PYTHON, "python"),
    )
}


# ------------------------------------------------------------------------------ registry


@functools.cache
def registry(root: Path) -> tuple[dict[str, Source], str]:
    """The image's sources, or none and why. Unknown fields are ignored; a source without a
    file and a digest is not a source."""
    try:
        raw = json.loads((root / INDEX).read_text())
    except FileNotFoundError:
        return {}, f"this image carries no kernel sources ({root / INDEX})"
    except (OSError, ValueError) as exc:
        return {}, f"{root / INDEX} is unreadable: {exc}"
    rows = raw.get("sources") if isinstance(raw, dict) else None
    sources = {
        str(name): Source(
            str(name),
            str(row.get("version", "")),
            str(root / str(row["file"])),
            str(row["sha256"]),
            str(row.get("origin", "")),
            str(row.get("package", "")),
        )
        for name, row in (rows or {}).items()
        if isinstance(row, dict) and row.get("file") and row.get("sha256")
    }
    return sources, "" if sources else f"{root / INDEX} names no sources"


def torch_build() -> tuple[str, str]:
    """This environment's torch release and CUDA build, read from its files, never imported:
    the executor imports torch only after its environment seal is re-imposed."""
    try:
        installed = metadata.distribution("torch")
        text = Path(str(installed.locate_file("torch/version.py"))).read_text()
    except (metadata.PackageNotFoundError, OSError):
        return "", ""
    cuda = re.search(r"^cuda\b[^=\n]*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    return installed.version, cuda.group(1) if cuda else ""


def _nvcc() -> str:
    home = config.inherited_environment().get("CUDA_HOME") or "/usr/local/cuda"
    return shutil.which("nvcc") or str(Path(home) / "bin" / "nvcc")


@functools.cache
def compiler() -> str:
    """The CUDA toolkit and host compiler that will build, as they identify themselves."""
    environment = config.inherited_environment()
    nvcc = _nvcc()
    try:
        found = [
            subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
            for tool in (nvcc, environment.get("CXX") or "c++")
        ]
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Absent(f"no CUDA toolkit to build with ({nvcc}): {exc}") from exc
    return hashlib.sha256("\n".join(found).encode()).hexdigest()[:16]


def _compatible_kitchen_wheel(source: Source | None) -> bool:
    """Only the reviewed wheel's actual platform/Python/ABI tags authorize reuse."""
    if source is None or sys.platform != "linux":
        return False
    try:
        distribution, version, _, tags = parse_wheel_filename(Path(source.file).name)
    except ValueError:
        return False
    if distribution != "comfy-kitchen" or str(version) != source.version:
        return False
    native = {tag for tag in tags if tag.abi == "abi3" and tag.platform.startswith("manylinux_")}
    return bool(native) and not native.isdisjoint(sys_tags())


@functools.cache
def reviewed_kitchen_wheel_compatible() -> bool:
    """Hardware-independent eligibility for automatic use of the reviewed native wheel."""
    pin = next(pin for pin in PINS if pin.name == KITCHEN_WHEEL)
    return _compatible_kitchen_wheel(Source(pin.name, pin.version, _filename(pin), pin.sha256))


def job(name: str, sm: int, root: Path | None = None) -> kernel_compile.Job:
    """The exact artifact `name` is on a card of capability `sm` in this environment."""
    recipe = RECIPES[name]
    sources, why = registry(root) if root is not None else (_pinned_sources(), "")
    inputs: dict[str, Any]
    wheel = sources.get(KITCHEN_WHEEL) if name == "comfy-kitchen" else None
    if _compatible_kitchen_wheel(wheel):
        assert wheel is not None
        # This exact published binary is the reviewed matrix artifact. Its key is
        # the installed wheel's bytes and Python ABI, not a source compiler recipe.
        wheel = replace(wheel, name=name)
        inputs = {"python": _python(), "sources": {name: wheel.sha256}, "layout": 2}
        return kernel_compile.Job(
            "python",
            kernel_cache.Key.of(name, inputs),
            {"recipe": name, "sm": sm, "sources": {name: asdict(wheel)}, "build_deps": {}},
        )
    if recipe.kind == "python":
        build_only = set(_BUILD_DEPS).union(*(r.build_deps for r in RECIPES.values()))
        wanted = {
            n: s
            for n, s in sources.items()
            if n not in build_only | {KITCHEN_WHEEL} and (s.file.endswith(".whl") or s.package)
        }
        if not wanted:
            raise Absent(why or "the kernel sources carry no Python sources")
        inputs = {"python": _python()}
    else:
        missing = [n for n in (*recipe.sources, *recipe.build_deps) if n not in sources]
        if missing:
            raise Absent(why or f"the kernel sources carry no {', '.join(missing)}")
        wanted = {n: sources[n] for n in recipe.sources}
        release, cuda = torch_build()
        if not release:
            raise Absent("this environment has no torch to build against")
        inputs = {
            "arch": f"sm_{sm}",
            "driver": accel.driver_version(),
            "torch": release,
            "cuda": cuda,
            "python": _python(),
            "compiler": compiler(),
            "recipe": json.loads(
                json.dumps(
                    {k: v for k, v in asdict(recipe).items() if k not in _UNKEYED}
                    | ({"replace_on": recipe.replace_on[sm]} if sm in recipe.replace_on else {})
                )
            ),
        }
    if recipe.build_deps:
        inputs["build_deps"] = {n: sources[n].sha256 for n in recipe.build_deps}
    inputs["sources"] = {n: s.sha256 for n, s in sorted(wanted.items())}
    # 2: every tree carries an installed distribution's dist-info.
    inputs["layout"] = 2
    spec = {
        "recipe": name,
        "sm": sm,
        "sources": {n: asdict(s) for n, s in wanted.items()},
        "build_deps": {
            n: asdict(sources[n])
            for n in (*_BUILD_DEPS, *recipe.build_deps)
            if n in sources and recipe.kind == "setup"
        },
    }
    return kernel_compile.Job(recipe.kind, kernel_cache.Key.of(name, inputs), spec)


def _filename(pin: Pin) -> str:
    name = Path(urlparse(pin.url).path).name
    if name.endswith(".tar.gz") and not name.startswith(pin.name):
        name = f"{pin.name}-{name}"
    return name


def _pinned_sources() -> dict[str, Source]:
    """No I/O: cache replay uses the Runtime pins even when archives/index are absent."""
    store = kernel_cache.configured() or kernel_cache.from_sealed(config.inherited_environment())
    if store is None:
        raise Absent("kernel source acquisition requires the sealed kernel cache namespace")
    return {
        pin.name: Source(
            pin.name,
            pin.version,
            str(store.own / "sources" / pin.sha256 / _filename(pin)),
            pin.sha256,
            pin.url,
            pin.package,
            True,
            str(ROOT / _filename(pin)),
        )
        for pin in PINS
    }


def _python() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


# ------------------------------------------------------------------------------ builders


def _verified(source: Mapping[str, Any]) -> Path:
    path = Path(source["file"])
    if source.get("acquire"):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (path.parent / ".fetch.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.is_file() or _digest(path) != source["sha256"]:
                preseed = Path(source["preseed"])
                if preseed.is_file() and _digest(preseed) == source["sha256"]:
                    return preseed
                egress.fetch_public_wheel(
                    path, source["origin"], f"sha256:{source['sha256']}", 1 << 30
                )
    if _digest(path) != source["sha256"]:
        raise kernel_compile.CompileFailed(
            "compile_source_mismatch", f"{path} is not the source {source['sha256'][:12]}"
        )
    return path


def _fetched(sources: Iterable[Mapping[str, Any]]) -> None:
    """A build's sources, downloaded side by side (the image may carry them already)."""
    with ThreadPoolExecutor() as pool:
        list(pool.map(_verified, [row for row in sources if not _tree(row).is_dir()]))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(source: Mapping[str, Any]) -> Path:
    """The part of a source a worker image carries, unpacked (`fetch`)."""
    named = str(source.get("file") or "")
    return Path(named if named.endswith(".tree") else f"{source.get('preseed') or '-'}.tree")


def _extract(source: Mapping[str, Any], into: Path) -> Path:
    """A verified source tree, its one top directory stripped; the image's copy when it has."""
    if _tree(source).is_dir():
        shutil.copytree(_tree(source), into)
        return into
    into.mkdir(parents=True)
    with tarfile.open(_verified(source)) as archive:
        archive.extractall(into, filter="data")
    tops = list(into.iterdir())
    return tops[0] if len(tops) == 1 and tops[0].is_dir() else into


def _unpack(source: Mapping[str, Any], site: Path) -> None:
    """A wheel into an importable tree: its purelib/platlib data folded into the root."""
    with zipfile.ZipFile(_verified(source)) as wheel:
        for member in wheel.infolist():
            parts = Path(member.filename).parts
            if len(parts) > 2 and parts[0].endswith(".data"):
                if parts[1] not in ("purelib", "platlib"):
                    continue
                parts = parts[2:]
            target = site.joinpath(*parts)
            if member.is_dir() or not target.resolve().is_relative_to(site.resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wheel.read(member))
            mode = member.external_attr >> 16
            if mode & 0o111:
                target.chmod(0o755)


def build_python(
    spec: Mapping[str, Any], staging: Path, report: Callable[[float], None]
) -> dict[str, object]:
    site = staging / kernel_cache.SITE
    site.mkdir()
    rows = sorted(spec["sources"].values(), key=lambda row: row["name"])
    _fetched(rows)
    with tempfile.TemporaryDirectory(prefix="cozy-python-") as scratch:
        for done, row in enumerate(rows, 1):
            if row.get("package"):
                tree = _extract(row, Path(scratch) / row["name"])
                package = site / Path(row["package"]).name
                shutil.copytree(tree / row["package"], package)
                _dist_info(site, row["name"], f"0+{row['version'][:12]}", [package])
            else:
                _unpack(row, site)
            report(done / len(rows))
    return {"sources": [f"{row['name']}=={row['version']}" for row in rows]}


#: `PYTORCH_NVCC` for a build: its compiler jobs, each admitted by the machine's compile ledger.
_COMPILER = """#!{python} -I
import sys
sys.path[:0] = {path!r}
from cozy_runtime.internal.kernel_compile import compiler_job
compiler_job(*{job!r})
"""


def build_setup(
    spec: Mapping[str, Any], staging: Path, report: Callable[[float], None]
) -> dict[str, object]:
    """Upstream's `setup.py build` into the staging `site/`, against this interpreter's torch
    and the card this process sees."""
    recipe = RECIPES[spec["recipe"]]
    sources = spec["sources"]
    _fetched(sources.values())
    with tempfile.TemporaryDirectory(prefix=f"cozy-build-{recipe.name}-") as scratch:
        work = Path(scratch)
        tree = _extract(sources[recipe.sources[0]], work / "src")
        for name, where in recipe.place.items():
            placed = _extract(sources[name], work / f"place-{name}")
            shutil.rmtree(tree / where, ignore_errors=True)
            (tree / where).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(placed, tree / where)
        for relative, old, new in (*recipe.replace, *recipe.replace_on.get(spec.get("sm", 0), ())):
            path = tree / relative
            text = path.read_text()
            if old not in text:
                raise kernel_compile.CompileFailed(
                    "compile_recipe_stale", f"{relative} no longer contains {old!r}"
                )
            path.write_text(text.replace(old, new))
        deps = work / "deps"
        deps.mkdir()
        for name, row in spec.get("build_deps", {}).items():
            if name in recipe.build_deps or util.find_spec(name) is None:
                _unpack(row, deps)
        environment = {
            key: value
            for key, value in config.inherited_environment().items()
            if key != "TORCH_CUDA_ARCH_LIST"
        }
        # Every compiler job goes through the machine's compile ledger, which admits it.
        nvcc = work / "nvcc"
        nvcc.write_text(
            _COMPILER.format(
                python=sys.executable,
                path=sys.path,
                job=(spec["ledger"], spec["key"], recipe.unit_bytes, _nvcc()),
            )
        )
        nvcc.chmod(0o755)
        jobs = kernel_compile.cpus()
        environment.update(
            {
                # Upstream's own knobs: extensions at once, and ninja jobs within each.
                "EXT_PARALLEL": str(recipe.extensions),
                "MAX_JOBS": str(jobs),
                "PYTORCH_NVCC": str(nvcc),
                "PYTHONPATH": os.pathsep.join([str(deps), *sys.path]),
                "PYTHONNOUSERSITE": "1",
                **recipe.env,
            }
        )
        site = staging / kernel_cache.SITE
        output = kernel_compile.run(
            [
                sys.executable,
                "setup.py",
                "build",
                "--build-base",
                str(work / "build"),
                "--build-lib",
                str(site),
                "egg_info",
                "--egg-base",
                str(work),
            ],
            cwd=tree / recipe.subdir,
            env=environment,
            report=report,
            alongside=functools.partial(kernel_compile.compilers, Path(spec["ledger"])),
        )
        [info] = work.glob("*.egg-info/PKG-INFO")
        found = email.parser.Parser().parsestr(info.read_text())
        _dist_info(site, found["Name"], found["Version"], list(site.iterdir()))
        units = _units(work / "build")
    return {
        "sources": {name: row["version"] for name, row in sources.items()},
        "max_jobs": jobs,
        "units_ms": dict(sorted(units.items(), key=lambda unit: -unit[1])[:12]),
        "log_tail": output[-400:],
    }


def _units(build: Path) -> dict[str, int]:
    """Each compiled unit's own wall time, from ninja's log: what bounds a wider build."""
    units: dict[str, int] = {}
    for log in build.rglob(".ninja_log"):
        for line in log.read_text().splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) == 5 and fields[0].isdigit() and fields[1].isdigit():
                units[Path(fields[3]).name] = int(fields[1]) - int(fields[0])
    return units


def _dist_info(site: Path, name: str, version: str, installed: list[Path]) -> None:
    """The installed distribution's record a tree needs to be one: an executor reads a
    kernel's version and RECORD from it."""
    info = site / f"{re.sub(r'[-_.]+', '_', name)}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (info / "INSTALLER").write_text("cozy-runtime\n")
    files = sorted(
        path
        for top in [*installed, info]
        for path in ([top] if top.is_file() else top.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    )
    rows = []
    for path in files:
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{path.relative_to(site).as_posix()},sha256={digest},{len(data)}")
    rows.append(f"{(info / 'RECORD').relative_to(site).as_posix()},,")
    (info / "RECORD").write_text("\n".join(rows) + "\n")


def fetch(root: Path) -> dict[str, Any]:
    """Download every pinned source into `root` side by side through the Runtime's one egress
    boundary, each held to its pinned sha256, and write the index beside them. A worker image
    carries them, so a pod's boot compile downloads none; of a source with `keep`, only those
    paths, unpacked beside where its archive would be (`_tree`)."""
    root.mkdir(parents=True, exist_ok=True)

    def download(pin: Pin) -> None:
        name = _filename(pin)
        if (root / f"{name}.tree").is_dir():
            return
        if not (root / name).exists() or _digest(root / name) != pin.sha256:
            egress.fetch_public_wheel(root / name, pin.url, f"sha256:{pin.sha256}", 1 << 30)
        if pin.keep:
            staging = root / f".{name}.tree"
            shutil.rmtree(staging, ignore_errors=True)
            with tarfile.open(root / name) as archive:
                kept = [
                    member.replace(name=rest)
                    for member in archive.getmembers()
                    if (rest := member.name.partition("/")[2])
                    and any(rest == k or rest.startswith(f"{k}/") for k in pin.keep)
                ]
                archive.extractall(staging, members=kept, filter="data")
            os.replace(staging, root / f"{name}.tree")
            (root / name).unlink()

    with ThreadPoolExecutor() as pool:
        list(pool.map(download, PINS))
    sources = {
        pin.name: {
            "version": pin.version,
            "file": _filename(pin) + (".tree" if pin.keep else ""),
            "sha256": pin.sha256,
            "origin": pin.url,
            **({"package": pin.package} if pin.package else {}),
        }
        for pin in PINS
    }
    index = {"schema": 1, "sources": sources}
    (root / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    return index


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "fetch":
        raise SystemExit("usage: python -m cozy_runtime.internal.kernel_sources fetch <dir>")
    print(f"{len(fetch(Path(sys.argv[2]))['sources'])} kernel sources in {sys.argv[2]}")
