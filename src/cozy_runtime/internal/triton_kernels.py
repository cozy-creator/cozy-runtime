"""A package's Triton kernels (`author.kernels.TritonKernels`): built when its executor starts,
for each sealed GPU's architecture, and loaded at construction.

Each entry compiles into Triton's own keyed cache (`TRITON_CACHE_DIR`, inside the machine's
kernel store), then Triton's launcher module, in the compile manager's niced background builder
(`BUILDERS["triton"]`), which imports each kernel by module and name; the store's entry marks the
cache filled. The key is Triton's own hash of every entry (sources and the functions they call,
signature, architecture, options, Triton) plus the launcher module's key and the cache
location; a driver update rebuilds nothing. `load` waits for a live build by its measured
progress, then serves it or raises `KernelsUnavailable`: a request never compiles. The Runtime's
own sets open through `opened` (`kernel_ready.TRITON`): loaded, then verified on the card.

The native wheel carries the Runtime's own sets prebuilt (`PREBUILT`, `triton_prebuild`): every
entry's compiled kernel by its Triton key, and Triton's launcher module per Python tag. The builder
copies a set's kernels into the cache, as Triton writes them, when every key is shipped, and
compiles otherwise; the launcher goes in under the key this host gives it. So a machine with no C
compiler, which Triton needs for its launcher, still serves them.

Executor-only: torch and Triton are imported at the top, guarded; their absence is
`kernels_absent`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sysconfig
from collections.abc import Callable, Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

import msgspec
from msgspec import Struct

from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._model import LANDED
from cozy_runtime.author.kernels import (
    KernelsUnavailable,
    Launch,
    TritonEntry,
    TritonKernels,
    entry_key,
    readable,
)
from cozy_runtime.internal import kernel_cache, kernel_compile, kernel_ready, tools, verify

try:
    import torch
    import triton
    from triton.backends.compiler import BaseBackend, GPUTarget
    from triton.backends.nvidia import driver as _cuda_driver
    from triton.compiler import ASTSource
    from triton.compiler.compiler import (
        CompiledKernel,
        get_cache_invalidating_env_vars,
        make_backend,
    )
    from triton.knobs import CompileTimes
    from triton.runtime.build import _get_cache_manager
    from triton.runtime.cache import get_cache_key, get_cache_manager
    from triton.runtime.driver import driver as triton_driver
    from triton.runtime.jit import JITFunction
except ImportError as _absent:  # pragma: no cover - exercised by the typed refusal
    TRITON_ABSENT = f"torch and triton do not import ({_absent})"
else:
    TRITON_ABSENT = ""

WARP_SIZE = 32


def _require_triton() -> None:
    if TRITON_ABSENT:
        raise KernelsUnavailable("kernels_absent", TRITON_ABSENT)


def _source(entry: TritonEntry) -> ASTSource:
    """The exact `ASTSource` an entry compiles from: builder and loader share it."""
    fn = entry.kernel
    attrs = {(fn.arg_names.index(name),): [["tt.divisibility", 16]] for name in entry.aligned}
    return ASTSource(fn, dict(entry.signature), constexprs=dict(entry.constexprs), attrs=attrs)


def _backend(arch: int) -> BaseBackend:
    """One compiler backend per architecture: constructing one runs `ptxas --version`."""
    if arch not in _BACKENDS:
        _BACKENDS[arch] = make_backend(GPUTarget("cuda", arch, WARP_SIZE))
    return _BACKENDS[arch]


_BACKENDS: dict[int, BaseBackend] = {}


def _cache_key(entry: TritonEntry, arch: int) -> str:
    """Triton's own cache hash for (sources, signature, architecture, options, Triton)."""
    backend = _backend(arch)
    options = backend.parse_options(entry.options)
    key = get_cache_key(_source(entry), backend, options, get_cache_invalidating_env_vars())
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def launcher_driver() -> Path:
    """Triton's launcher module source (`CudaUtils`)."""
    return Path(_cuda_driver.__file__).parent / "driver.c"


def launcher_source() -> bytes:
    return launcher_driver().read_bytes()


def launcher_config() -> dict[str, list[str] | None]:
    """What Triton builds its launcher module (`CudaUtils`) with on this host, its own way."""
    return {
        "language": ["c"],
        "library_dirs": _cuda_driver.library_dirs(),
        "include_dirs": _cuda_driver.include_dirs,
        "libraries": _cuda_driver.libraries,
        "ccflags": None,
    }


def _launcher_key() -> str:
    """Triton's cache key for its launcher module on this host."""
    return str(_get_cache_manager(launcher_source(), config=launcher_config()).key)


#: The wheel's prebuilt Runtime sets (`triton_prebuild`): `manifest.json`, `kernels/<key>/` and
#: `launcher/`.
PREBUILT = Path(__file__).resolve().parents[1] / "_triton"
#: The launcher module's file name in Triton's cache, this Python's.
LAUNCHER = f"cuda_utils{sysconfig.get_config_var('EXT_SUFFIX')}"


@functools.cache
def _manifest() -> dict[str, Any]:
    try:
        found = json.loads((PREBUILT / "manifest.json").read_text())
    except (OSError, ValueError):
        return {}
    return found if found.get("triton") == triton.__version__ else {}


def _shipped(keys: Iterable[str]) -> bool:
    kernels = _manifest().get("kernels", {})
    return all(key in kernels for key in keys)


def _launcher_shipped() -> dict[str, str] | None:
    row = _manifest().get("launcher", {}).get(LAUNCHER)
    if row is None or row.get("driver_sha256") != hashlib.sha256(launcher_source()).hexdigest():
        return None
    return dict(row)


def _prebuilt(found: kernel_compile.Job) -> bool:
    """Whether the wheel serves this set's build: its every kernel and this Python's launcher."""
    return _shipped(found.spec["entries"].values()) and _launcher_shipped() is not None


def _seed(keys: Iterable[str]) -> None:
    """The wheel's compiled kernels into Triton's cache, each group as Triton writes one."""
    shipped = _manifest()["kernels"]
    for key in keys:
        row, cache = shipped[key], get_cache_manager(key)
        if cache.get_group(row["group"]) is not None:
            continue
        files = PREBUILT / "kernels" / key
        group = {name: cache.put((files / name).read_bytes(), name) for name in row["files"]}
        cache.put_group(row["group"], group)


@functools.cache
def _seed_launcher() -> None:
    """The wheel's launcher module into Triton's cache under this host's key, if it has none;
    once per process."""
    row = _launcher_shipped()
    if row is None:
        return
    cache = _get_cache_manager(launcher_source(), config=launcher_config())
    if cache.get_file(LAUNCHER) is None:
        cache.put((PREBUILT / "launcher" / row["file"]).read_bytes(), LAUNCHER)


class _Kernel(Struct, frozen=True):
    """One entry as the out-of-process builder reads it: the kernel by module and name."""

    module: str
    name: str
    signature: dict[str, str]
    constexprs: dict[str, int]
    aligned: tuple[str, ...]
    num_warps: int
    num_stages: int
    fp_fusion: bool

    @classmethod
    def of(cls, entry: TritonEntry) -> _Kernel:
        kernel = entry.kernel
        signature, constexprs = dict(entry.signature), dict(entry.constexprs)
        return cls(
            kernel.fn.__module__,
            kernel.__name__,
            signature,
            constexprs,
            entry.aligned,
            entry.num_warps,
            entry.num_stages,
            entry.fp_fusion,
        )

    def entry(self) -> TritonEntry:
        kernel = getattr(import_module(self.module), self.name)
        return TritonEntry(
            kernel,
            self.signature,
            self.constexprs,
            self.aligned,
            self.num_warps,
            self.num_stages,
            self.fp_fusion,
        )


def job(kernels: TritonKernels, arch: int) -> kernel_compile.Job:
    """`kernels` for `arch`, compiled into this process's Triton cache."""
    _require_triton()
    entries = {entry.key: _cache_key(entry, arch) for entry in kernels.entries}
    inputs = {
        "arch": arch,
        "triton": triton.__version__,
        "entries": entries,
        "launcher": _launcher_key(),
        "cache": triton.knobs.cache.dir,
    }
    spec = {
        "arch": arch,
        "entries": entries,
        "kernels": [msgspec.to_builtins(_Kernel.of(entry)) for entry in kernels.entries],
    }
    return kernel_compile.Job("triton", kernel_cache.Key.of(kernels.name, inputs), spec)


@request_path.forbid("kernel build: triton")
def build(
    spec: dict[str, Any], staging: Path, report: Callable[[float], None]
) -> dict[str, object]:
    """The compile manager's builder: every entry into Triton's own cache (the wheel's, when it
    ships them all), then Triton's launcher module, so a launch compiles nothing. `staging` stays
    empty: the published entry marks the cache filled."""
    _require_triton()
    prebuilt = _shipped(spec["entries"].values())
    if prebuilt:
        _seed(spec["entries"].values())
    else:
        compile_entries(spec, report)
    _seed_launcher()
    _cuda_driver.CudaUtils()
    return {"entries": len(spec["kernels"]), "prebuilt": prebuilt}


def compile_entries(spec: dict[str, Any], report: Callable[[float], None]) -> None:
    """Every entry of a `job`'s spec compiled into Triton's own cache, each under its key."""
    arch, entries = int(spec["arch"]), spec["entries"]
    todo = [msgspec.convert(row, _Kernel).entry() for row in spec["kernels"]]
    for done, entry in enumerate(todo, 1):
        compiled = triton.compile(
            _source(entry), target=GPUTarget("cuda", arch, WARP_SIZE), options=entry.options
        )
        if compiled.hash != entries[entry.key]:
            raise kernel_compile.CompileFailed(
                "compile_key_mismatch",
                f"{entry.key} compiled under {compiled.hash[:12]}, keyed {entries[entry.key][:12]}",
            )
        report(done / (len(todo) + 1))


class _Card:
    """The compile manager's stand-in for a card: Triton's active driver while a builder runs
    `.warmup` calls, so they compile for `arch` with no GPU (and no context) in the builder."""

    def __init__(self, arch: int) -> None:
        self.target = GPUTarget("cuda", arch, WARP_SIZE)

    def get_current_device(self) -> int:
        return 0

    def get_current_stream(self, device: int) -> int:
        return 0

    def get_current_target(self) -> GPUTarget:
        return self.target


def warm_job(
    kernel: str, function: str, args: list[object], arch: int, inputs: dict[str, object]
) -> kernel_compile.Job:
    """Upstream Triton kernels a Runtime `module:function(*args)` builds by `.warmup`, for
    `arch`, into this process's Triton cache, keyed by `inputs` (their sources) and the
    arguments: the out-of-process build of kernels Runtime does not declare (`build_warm`).
    Once it is ready, the same call in an executor loads what it built and compiles nothing."""
    _require_triton()
    key = {**inputs, "arch": arch, "args": args, "triton": triton.__version__,
           "cache": triton.knobs.cache.dir}  # fmt: skip
    spec = {"arch": arch, "function": function, "args": args}
    return kernel_compile.Job(
        "triton-warm", kernel_cache.Key.of(kernel, key), spec, {"CUDA_VISIBLE_DEVICES": ""}
    )


@request_path.forbid("kernel build: triton warm")
def build_warm(
    spec: dict[str, Any], staging: Path, report: Callable[[float], None]
) -> dict[str, object]:
    """The compile manager's builder for a `warm_job`: its function's `.warmup` calls compile
    for the job's card through `_Card`. `staging` stays empty, as `build`'s does."""
    _require_triton()
    triton_driver.set_active(_Card(int(spec["arch"])))
    module, name = str(spec["function"]).split(":")
    getattr(import_module(module), name)(*spec["args"])
    return {"function": spec["function"]}


def _store() -> kernel_cache.Store:
    store = kernel_cache.configured()
    if store is None:
        raise KernelsUnavailable(
            "kernels_absent", "no machine kernel store: this executor has no rooted worker"
        )
    return store


def _submit(store: kernel_cache.Store, found: kernel_compile.Job) -> kernel_compile.State:
    """`kernel_compile.submit`; a set neither built, nor prebuilt in the wheel, nor buildable
    (Triton has no C compiler here) is absent."""
    if kernel_compile.status(store, found).state == "ready" or _prebuilt(found) or tools.triton():
        return kernel_compile.submit(store, found)
    raise KernelsUnavailable("kernels_absent", f"{found.key.kernel}: cc not found")


def expect(kernels: TritonKernels, arch: int) -> str:
    """Submit `kernels`' build for `arch` without waiting, before any device exists: the
    state's line. Keyed in the process that will load it (its Triton and cache directory)."""
    try:
        return _submit(_store(), job(kernels, arch)).line()
    except KernelsUnavailable as exc:
        return f"{exc.code}: {exc}"


class _Launcher:
    """A set's compiled entries on one GPU. Tensor operands are read by pointer, so each has
    real bytes behind it first (`readable`): a norm scale read from a sibling module may be a
    weight the plane has not mapped."""

    def __init__(
        self,
        arch: int,
        compiled: dict[str, CompiledKernel],
        names: dict[str, tuple[str, ...]],
        key: str,
    ) -> None:
        self.arch, self.compiled, self.names, self.key = arch, compiled, names, key

    def __call__(self, kernel: JITFunction, grid: tuple[int, ...], *args: object) -> None:
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


def load(kernels: TritonKernels, *, device: object) -> Launch:
    """`kernels` compiled for `device`, loaded onto it from Triton's cache; never a compile
    here. Waits for the key's live build by its measured progress; a build that ended without
    publishing is typed. Idempotent per (process, set, device)."""
    _require_triton()
    target = torch.device(device)
    if target.type != "cuda":
        raise KernelsUnavailable("kernels_absent", f"{kernels.name} serves CUDA, not {target}")
    index = target.index or 0
    if (kernels.name, index) in _LOADED:
        return _LOADED[(kernels.name, index)]
    _seed_launcher()  # this environment's key: an image boot's builder seeded its own
    major, minor = torch.cuda.get_device_capability(index)
    arch, store = major * 10 + minor, _store()
    found = job(kernels, arch)
    _submit(store, found)
    state = kernel_compile.wait(store, found, kernels.name)
    if state.state != "ready":
        raise KernelsUnavailable("kernels_failed", f"{kernels.name} sm{arch}: {state.line()}")
    compiled: dict[str, CompiledKernel] = {}
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
    launcher = _LOADED[(kernels.name, index)] = _Launcher(arch, compiled, names, found.key.digest)
    return launcher


def ready(kernels: TritonKernels, wait: bool = False, *, device: object) -> Launch | None:
    """`load`, when this machine's build of `kernels` for `device` is done; None while it
    compiles (started here when nobody had). Waits for it only when `wait`."""
    if wait:
        return load(kernels, device=device)
    _require_triton()
    target = torch.device(device)
    if target.type != "cuda":
        raise KernelsUnavailable("kernels_absent", f"{kernels.name} serves CUDA, not {target}")
    if (kernels.name, target.index or 0) in _LOADED:
        return _LOADED[(kernels.name, target.index or 0)]
    if torch.cuda.is_current_stream_capturing():  # a load would end the graph being recorded
        return None
    major, minor = torch.cuda.get_device_capability(target.index or 0)
    if _submit(_store(), job(kernels, major * 10 + minor)).state == "compiling":
        return None
    launch = load(kernels, device=device)
    LANDED[0] += 1
    return launch


def opened(
    kernels: TritonKernels,
    probe: Callable[[Launch], verify.Verdict],
    wait: bool = False,
    *,
    device: object,
) -> Launch | None:
    """A Runtime set's opener: `ready`, then `probe` of its launcher against the exact law on
    this card (`kernel_ready.verified`, keyed by the build), else `kernel_ready.Refused`."""
    launch = ready(kernels, wait, device=device)
    if launch is None:
        return None
    loaded = _LOADED[(kernels.name, torch.device(device).index or 0)]
    return kernel_ready.verified(launch, kernels.name, loaded.key, loaded.arch, probe)


kernel_ready.TRITON[:] = [opened]


def _compiled(*, src: ASTSource, times: CompileTimes, cache_hit: bool, **_: object) -> None:
    """Triton compiling a kernel at its first launch: on a request thread, it blocked it."""
    if not cache_hit:
        request_path.note(f"triton compile: {src.name}", times.total / 1000)


def _tuned(*, fn: JITFunction, duration: float | None, cache_hit: bool, **_: object) -> None:
    """Triton timing an autotuned kernel's configurations to choose one."""
    if not cache_hit and duration:
        request_path.note(f"triton autotune: {fn.__name__}", duration * 1000)


if not TRITON_ABSENT:
    triton.knobs.compilation.listener = _compiled
    triton.knobs.autotuning.listener = _tuned
