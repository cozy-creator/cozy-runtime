"""A package's Triton kernels (`author.kernels.TritonKernels`): built when its executor starts,
for each sealed GPU's architecture, and loaded at construction.

Each entry compiles into Triton's own keyed cache (`TRITON_CACHE_DIR`, inside the machine's
kernel store), then Triton's launcher module, in the compile manager's niced background builder
(`BUILDERS["triton"]`), which imports each kernel by module and name; the store's entry marks the
cache filled. The key is Triton's own hash of every entry (sources and the functions they call,
signature, architecture, options, Triton) plus the driver, the launcher module's key and the
cache location. `load` waits for a live build by its measured progress, then serves it or
raises `KernelsUnavailable`: a request never compiles.

Executor-only: torch and Triton are imported at the top, guarded; their absence is
`kernels_absent`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

from cozy_runtime.author.kernels import (
    KernelsUnavailable,
    Launch,
    TritonEntry,
    TritonKernels,
    entry_key,
    readable,
)
from cozy_runtime.internal import kernel_cache, kernel_compile
from cozy_runtime.internal.accel import driver_version

try:
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import driver as _cuda_driver
    from triton.compiler import ASTSource
    from triton.compiler.compiler import (
        CompiledKernel,
        get_cache_invalidating_env_vars,
        make_backend,
    )
    from triton.runtime.build import _get_cache_manager
    from triton.runtime.cache import get_cache_key, get_cache_manager
except ImportError as _absent:  # pragma: no cover - exercised by the typed refusal
    TRITON_ABSENT = f"torch and triton do not import ({_absent})"
else:
    TRITON_ABSENT = ""

WARP_SIZE = 32


def _require_triton() -> None:
    if TRITON_ABSENT:
        raise KernelsUnavailable("kernels_absent", TRITON_ABSENT)


def _source(entry: TritonEntry) -> Any:
    """The exact `ASTSource` an entry compiles from: builder and loader share it."""
    fn = entry.kernel
    attrs = {(fn.arg_names.index(name),): [["tt.divisibility", 16]] for name in entry.aligned}
    return ASTSource(fn, dict(entry.signature), constexprs=dict(entry.constexprs), attrs=attrs)


def _backend(arch: int) -> Any:
    """One compiler backend per architecture: constructing one runs `ptxas --version`."""
    if arch not in _BACKENDS:
        _BACKENDS[arch] = make_backend(GPUTarget("cuda", arch, WARP_SIZE))
    return _BACKENDS[arch]


_BACKENDS: dict[int, Any] = {}


def _cache_key(entry: TritonEntry, arch: int) -> str:
    """Triton's own cache hash for (sources, signature, architecture, options, Triton)."""
    backend = _backend(arch)
    options = backend.parse_options({"num_warps": entry.num_warps})
    key = get_cache_key(_source(entry), backend, options, get_cache_invalidating_env_vars())
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _launcher_key() -> str:
    """Triton's cache key for its launcher module (`CudaUtils`) on this host, its own way."""
    source = (Path(_cuda_driver.__file__).parent / "driver.c").read_bytes()
    options = {
        "language": ["c"],
        "library_dirs": _cuda_driver.library_dirs(),
        "include_dirs": _cuda_driver.include_dirs,
        "libraries": _cuda_driver.libraries,
        "ccflags": None,
    }
    return str(_get_cache_manager(source, config=options).key)


def _document(entry: TritonEntry) -> dict[str, object]:
    return {
        "module": entry.kernel.fn.__module__,
        "name": entry.kernel.__name__,
        "signature": dict(entry.signature),
        "constexprs": dict(entry.constexprs),
        "aligned": list(entry.aligned),
        "num_warps": entry.num_warps,
    }


def _entry(document: Mapping[str, Any]) -> TritonEntry:
    return TritonEntry(
        getattr(import_module(document["module"]), document["name"]),
        document["signature"],
        document["constexprs"],
        tuple(document["aligned"]),
        int(document["num_warps"]),
    )


def job(kernels: TritonKernels, arch: int) -> kernel_compile.Job:
    """`kernels` for `arch`, compiled into this process's Triton cache."""
    _require_triton()
    entries = {entry.key: _cache_key(entry, arch) for entry in kernels.entries}
    inputs = {
        "arch": arch,
        "driver": driver_version(),
        "triton": triton.__version__,
        "entries": entries,
        "launcher": _launcher_key(),
        "cache": triton.knobs.cache.dir,
    }
    spec = {
        "arch": arch,
        "entries": entries,
        "kernels": [_document(entry) for entry in kernels.entries],
    }
    return kernel_compile.Job("triton", kernel_cache.Key.of(kernels.name, inputs), spec)


def build(
    spec: dict[str, Any], staging: Path, report: Callable[[float], None]
) -> dict[str, object]:
    """The compile manager's builder: every entry into Triton's own cache, then Triton's
    launcher module, so a launch compiles nothing. `staging` stays empty: the published
    entry marks the cache filled."""
    _require_triton()
    arch, entries = int(spec["arch"]), spec["entries"]
    todo = [_entry(document) for document in spec["kernels"]]
    for done, entry in enumerate(todo, 1):
        compiled = triton.compile(
            _source(entry),
            target=GPUTarget("cuda", arch, WARP_SIZE),
            options={"num_warps": entry.num_warps},
        )
        if compiled.hash != entries[entry.key]:
            raise kernel_compile.CompileFailed(
                "compile_key_mismatch",
                f"{entry.key} compiled under {compiled.hash[:12]}, keyed {entries[entry.key][:12]}",
            )
        report(done / (len(todo) + 1))
    _cuda_driver.CudaUtils()
    return {"entries": len(todo)}


def _store() -> kernel_cache.Store:
    store = kernel_cache.configured()
    if store is None:
        raise KernelsUnavailable(
            "kernels_absent", "no machine kernel store: this executor has no rooted worker"
        )
    return store


def expect(kernels: TritonKernels, arch: int) -> str:
    """Submit `kernels`' build for `arch` without waiting, before any device exists: the
    state's line. Keyed in the process that will load it (its Triton and cache directory)."""
    try:
        return kernel_compile.submit(_store(), job(kernels, arch)).line()
    except KernelsUnavailable as exc:
        return f"{exc.code}: {exc}"


class _Launcher:
    """A set's compiled entries on one GPU. Tensor operands are read by pointer, so each has
    real bytes behind it first (`readable`): a norm scale read from a sibling module may be a
    weight the plane has not mapped."""

    def __init__(self, arch: int, compiled: dict[str, Any], names: dict[str, tuple[str, ...]]):
        self.arch, self.compiled, self.names = arch, compiled, names

    def __call__(self, kernel: Any, grid: tuple[int, ...], *args: object) -> None:
        params = kernel.arg_names
        constexprs = {name: args[params.index(name)] for name in self.names[kernel.__name__]}
        found = self.compiled.get(entry_key(kernel.__name__, constexprs))
        if found is None:
            raise KernelsUnavailable(
                "kernels_absent",
                f"{entry_key(kernel.__name__, constexprs)} is not compiled for sm{self.arch}",
            )
        at = [i for i, value in enumerate(args) if isinstance(value, torch.Tensor)]
        with readable(*(args[i] for i in at)) as real:
            operands = list(args)
            for i, value in zip(at, real, strict=True):
                operands[i] = value
            found[grid](*operands)


#: Loaded sets by (name, device index).
_LOADED: dict[tuple[str, int], _Launcher] = {}


def load(kernels: TritonKernels, *, device: Any) -> Launch:
    """`kernels` compiled for `device`, loaded onto it from Triton's cache; never a compile
    here. Waits for the key's live build by its measured progress; a build that ended without
    publishing is typed. Idempotent per (process, set, device)."""
    _require_triton()
    device = torch.device(device)
    if device.type != "cuda":
        raise KernelsUnavailable("kernels_absent", f"{kernels.name} serves CUDA, not {device}")
    index = device.index or 0
    if (kernels.name, index) in _LOADED:
        return _LOADED[(kernels.name, index)]
    major, minor = torch.cuda.get_device_capability(index)
    arch, store = major * 10 + minor, _store()
    found = job(kernels, arch)
    state = kernel_compile.wait(store, found, kernels.name)
    if state.state != "ready":
        raise KernelsUnavailable("kernels_failed", f"{kernels.name} sm{arch}: {state.line()}")
    compiled: dict[str, Any] = {}
    names: dict[str, tuple[str, ...]] = {}
    for entry in kernels.entries:
        digest = found.spec["entries"][entry.key]
        name = f"{entry.kernel.__name__}.json"
        group = get_cache_manager(digest).get_group(name) or {}
        if name not in group:
            raise KernelsUnavailable(
                "kernels_corrupt",
                f"{entry.key}: the store marks sm{arch} built, Triton's cache lacks {digest[:12]}",
            )
        kernel = CompiledKernel(_source(entry), group, digest)
        try:
            kernel._init_handles()
        except Exception as exc:
            raise KernelsUnavailable(
                "kernels_corrupt", f"{entry.key} does not load on sm{arch}: {exc}"
            ) from exc
        compiled[entry.key] = kernel
        names[entry.kernel.__name__] = tuple(sorted(entry.constexprs))
    launcher = _LOADED[(kernels.name, index)] = _Launcher(arch, compiled, names)
    return launcher
