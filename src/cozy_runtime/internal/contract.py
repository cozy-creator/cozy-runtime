"""The package contract's seal (CONTRACT §9, ENFORCEMENT §2): what a package may not reach raises
`ContractViolation` with the contract's API named. Applied once per executor before its package
imports (a zygote's forks inherit it); nothing is checked per op.

A call is the package's when the innermost frame that is not the standard library or a library
(a distribution other than the package's) is package code; the Runtime's own frames, and anything
under `owned()`, pass. So a library's call on the package's behalf is the package's, and a
library's call under a Runtime frame is the Runtime's. Code the Runtime builds and loads on the
package's behalf is the Runtime's by where it lives: a kernel tree in the kernel store
(SageAttention, comfy-kitchen), and what it calls. The room readers raise for every caller:
the Runtime reads varena (`vt.stats`, `vt.room`). An env write that leaves the variable as it
is passes for every caller, since it changes nothing: torch's `cache_dir()` writes TensorD's own
`TORCHINDUCTOR_CACHE_DIR` back at import.

Behaviour found after the fact warns instead (a swallowed OOM, a package's thread), since its
output is already made; the release gate fails on those warnings."""

from __future__ import annotations

import ctypes
import functools
import importlib.metadata
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import FrameType
from typing import Any, NoReturn

from cozy_runtime.author import _model
from cozy_runtime.author._errors import ContractViolation
from cozy_runtime.author._model import WARNINGS, WarningRow
from cozy_runtime.internal import kernel_cache, numerical_environment

#: The API each rule's construct is replaced by (static_contract's refusals say the same).
USE = {
    "oom_catch": "author.split(fn, x, Cut(...)): varena picks a level that fits",
    "room_query": "author.split or author.retained: a package never asks for room",
    "allocator": "nothing: varena owns the allocator and reclaims",
    "env_write": "Config or Settings: TensorD seals the environment",
    "global_flag": "the Runtime's numerics policy, hold_dtype, "
    "@uses_components(deterministic=True)",
    "own_stream": "denoise_loop, uncaptured, graph_keys: the Runtime owns streams",
    "compile": "accel.kernels.TritonKernels, prebuilt",
    "weights_io": "loader.construct or loader.load, loader.assets.read or .open",
    "own_decode": "lm.generate (allowed=lm.Constraint for a constrained decode)",
    "global_rng": "rng.noise, rng.generator(seed), view.generator",
    "process": "concurrently, spread or an invocable child call",
    "thread": "author.ThreadPoolExecutor",
    "internal_import": "the public names (CONTRACT.md); a package split is Final",
    "attention": "cozy_runtime.attention.Attention(heads, head_dim, dtype, causal=) or a "
    "Diffusers attention module",
}

#: The Runtime's own work done through a library on its behalf (`lm.generate`'s Transformers
#: loop): its calls pass.
OWNED: ContextVar[bool] = ContextVar("cozy_contract_owned", default=False)


@contextmanager
def owned() -> Iterator[None]:
    token = OWNED.set(True)
    try:
        yield
    finally:
        OWNED.reset(token)


# ------------------------------------------------------------------------------- whose call

_RUNTIME = frozenset({"cozy_runtime", "tensorfs", "__main__"})
_STDLIB = frozenset(sys.stdlib_module_names)
#: Import names of the installed distributions other than the package's, set by `seal`; before
#: it no frame is the package's.
_LIBRARIES: list[frozenset[str] | None] = [None]


def _site(frame: FrameType | None, *, through: bool = True) -> str | None:
    """`file:line` of the package frame a call comes from, else None. `through`: past library
    frames to the code that called the library; otherwise a library's own call is its own."""
    libraries = _LIBRARIES[0]
    if libraries is None or OWNED.get():
        return None
    while frame is not None:
        top = str(frame.f_globals.get("__name__") or "").partition(".")[0]
        if top in _RUNTIME or kernel_cache.holds(frame.f_code.co_filename):
            return None
        library = not top or top in libraries
        if not library and top not in _STDLIB:
            return f"{frame.f_code.co_filename}:{frame.f_lineno}"
        if library and not through:
            return None
        frame = frame.f_back
    return None


def _refuse(rule: str, what: str, site: str) -> NoReturn:
    raise ContractViolation(
        f"{what} is not in the package contract ({rule}, {site}): use {USE[rule]}",
        code="contract_violation",
        fields=[rule],
    )


def _guard(rule: str, what: str, original: Callable[..., Any], *, owned_only: bool = False) -> Any:
    """`original`, refusing the package's calls; `owned_only`: every call outside `owned()`."""

    @functools.wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if owned_only:
            if not OWNED.get() and _LIBRARIES[0] is not None:
                _refuse(rule, what, "outside cozy_runtime.lm")
        elif (site := _site(sys._getframe(1))) is not None:
            _refuse(rule, what, site)
        return original(*args, **kwargs)

    guarded.__contract_original__ = original  # type: ignore[attr-defined]
    return guarded


# ------------------------------------------------------------------------------- the seal

#: Room and allocator readers: memory is varena's, and nothing sizes work from these.
_ROOM = (
    "mem_get_info",
    "memory_stats",
    "memory_allocated",
    "memory_reserved",
    "max_memory_allocated",
    "max_memory_reserved",
    "reset_peak_memory_stats",
    "set_per_process_memory_fraction",
)
#: The sealed room readers' originals: for varena's own gate cells, which read the driver.
_ORIGINAL: dict[str, Callable[..., Any]] = {}


def unsealed(torch: Any, name: str) -> Callable[..., Any]:
    """`torch.cuda.<name>` as torch has it, sealed or not. Never for sizing work."""
    return _ORIGINAL.get(name) or getattr(torch.cuda, name)


def _raiser(name: str) -> Callable[..., NoReturn]:
    def refused(*_: Any, **__: Any) -> NoReturn:
        caller = sys._getframe(1)
        _refuse(
            "room_query", f"torch.cuda.{name}", f"{caller.f_code.co_filename}:{caller.f_lineno}"
        )

    refused.__name__ = refused.__qualname__ = name
    return refused


def _replace(rule: str, owners: tuple[Any, ...], name: str, spelled: str) -> None:
    """Guard `name` on each of `owners` that has it (one function, several import paths)."""
    for owner in owners:
        found = getattr(owner, name, None)
        if found is not None and not hasattr(found, "__contract_original__"):
            setattr(owner, name, _guard(rule, spelled, found))


def _construction(rule: str, cls: Any, spelled: str, method: str = "__new__") -> None:
    """Guard building `cls` (or calling `method`) in place, so `isinstance` and subclasses
    keep their class."""
    if (
        cls is None
        or method not in vars(cls)
        or hasattr(getattr(cls, method), "__contract_original__")
    ):
        return
    found = getattr(cls, method)
    guarded = _guard(rule, spelled, found)
    setattr(cls, method, staticmethod(guarded) if method == "__new__" else guarded)


def seal(torch: Any, application: str = "") -> None:
    """Seal `torch` for the package whose App is `application` (`module:attribute`)."""
    for name in _ROOM:
        if hasattr(torch.cuda, name) and name not in _ORIGINAL:
            _ORIGINAL[name] = getattr(torch.cuda, name)
    for module in (torch.cuda, torch.cuda.memory):
        for name in _ROOM:
            if hasattr(module, name):
                setattr(module, name, _raiser(name))
    cuda, graphs = torch.cuda, getattr(torch.cuda, "graphs", None)
    _replace("allocator", (cuda, cuda.memory), "empty_cache", "torch.cuda.empty_cache")
    for name in ("synchronize", "set_device"):
        _replace("own_stream", (cuda,), name, f"torch.cuda.{name}")
    _replace("own_stream", (cuda, graphs), "make_graphed_callables", "make_graphed_callables")
    for cls, spelled in (
        (getattr(cuda, "Stream", None), "torch.cuda.Stream"),
        (getattr(cuda, "ExternalStream", None), "torch.cuda.ExternalStream"),
        (getattr(cuda, "Event", None), "torch.cuda.Event"),
        (getattr(cuda, "CUDAGraph", None), "torch.cuda.CUDAGraph"),
    ):
        _construction("own_stream", cls, spelled)
    _construction("own_stream", getattr(cuda, "graph", None), "torch.cuda.graph", "__init__")
    for cls in (getattr(cuda, "Stream", None), getattr(cuda, "Event", None)):
        _construction(
            "own_stream", cls, f"{getattr(cls, '__name__', '')}.synchronize", "synchronize"
        )
    _replace("compile", (torch,), "compile", "torch.compile")
    _construction("compile", torch.nn.Module, "nn.Module.compile", "compile")
    random = getattr(cuda, "random", None)
    _replace("global_rng", (torch, torch.random), "manual_seed", "torch.manual_seed")
    _replace("global_rng", (torch.random,), "fork_rng", "torch.random.fork_rng")
    for name in ("manual_seed", "manual_seed_all"):
        _replace("global_rng", (cuda, random), name, f"torch.cuda.{name}")
    for name in ("set_float32_matmul_precision", "use_deterministic_algorithms"):
        _replace("global_flag", (torch,), name, f"torch.{name}")
    seal_libraries()
    if _EXPECTED[0] is None:
        pin(torch)
    if application and _LIBRARIES[0] is None:
        _LIBRARIES[0] = _libraries(application)
        _model.CONTRACT[0] = _Scopes(torch)


def seal_libraries() -> None:
    """Transformers' own decode (`GenerationMixin.generate`, a `DynamicCache`) passes only under
    `owned()` (`lm.generate`'s). Sealed once Transformers is imported; nothing is imported here."""
    utils = sys.modules.get("transformers.generation.utils")
    caches = sys.modules.get("transformers.cache_utils")
    if utils is not None:
        mixin = getattr(utils, "GenerationMixin", None)
        if mixin is not None and not hasattr(mixin.generate, "__contract_original__"):
            mixin.generate = _guard(
                "own_decode", "GenerationMixin.generate", mixin.generate, owned_only=True
            )
    cache = getattr(caches, "DynamicCache", None) if caches is not None else None
    if cache is not None and not hasattr(cache.__init__, "__contract_original__"):
        cache.__init__ = _guard("own_decode", "DynamicCache", cache.__init__, owned_only=True)


def _libraries(application: str) -> frozenset[str]:
    """Import names of every installed distribution but the one holding `application`."""
    found = importlib.metadata.packages_distributions()
    top = application.partition(":")[0].partition(".")[0]
    own = set(found.get(top, ()))
    return frozenset(t for t, owners in found.items() if not own & set(owners)) - _RUNTIME - {top}


# ------------------------------------------------------------------------------- the flags

#: The process-wide numerics a package could move, with how to put each back.
_FLAGS: tuple[tuple[str, Callable[[Any], object], Callable[[Any, Any], object]], ...] = (
    (
        "torch.backends.cuda.matmul.allow_tf32",
        lambda t: t.backends.cuda.matmul.allow_tf32,
        lambda t, v: setattr(t.backends.cuda.matmul, "allow_tf32", v),
    ),
    (
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
        lambda t: t.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        lambda t, v: setattr(t.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", v),
    ),
    (
        "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction",
        lambda t: t.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        lambda t, v: setattr(t.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", v),
    ),
    (
        "torch.backends.cudnn.allow_tf32",
        lambda t: t.backends.cudnn.allow_tf32,
        lambda t, v: setattr(t.backends.cudnn, "allow_tf32", v),
    ),
    (
        "torch.backends.cudnn.benchmark",
        lambda t: t.backends.cudnn.benchmark,
        lambda t, v: setattr(t.backends.cudnn, "benchmark", v),
    ),
    (
        "torch.backends.cudnn.deterministic",
        lambda t: t.backends.cudnn.deterministic,
        lambda t, v: setattr(t.backends.cudnn, "deterministic", v),
    ),
    (
        "torch.backends.cudnn.enabled",
        lambda t: t.backends.cudnn.enabled,
        lambda t, v: setattr(t.backends.cudnn, "enabled", v),
    ),
    *(
        (
            f"torch.backends.cuda.{kind}_sdp_enabled",
            lambda t, kind=kind: getattr(t.backends.cuda, f"{kind}_sdp_enabled")(),
            lambda t, v, kind=kind: getattr(t.backends.cuda, f"enable_{kind}_sdp")(v),
        )
        for kind in ("flash", "mem_efficient", "math", "cudnn")
    ),
    (
        "torch.get_float32_matmul_precision",
        lambda t: t.get_float32_matmul_precision(),
        lambda t, v: _unguarded(t.set_float32_matmul_precision)(v),
    ),
    (
        "torch.get_default_dtype",
        lambda t: t.get_default_dtype(),
        lambda t, v: t.set_default_dtype(v),
    ),
    (
        "torch.are_deterministic_algorithms_enabled",
        lambda t: t.are_deterministic_algorithms_enabled(),
        lambda t, v: _unguarded(t.use_deterministic_algorithms)(v),
    ),
)
#: cuDNN's mode while a `deterministic=True` scope is open (numerical_environment).
_DETERMINISTIC = {
    "torch.backends.cudnn.deterministic": True,
    "torch.backends.cudnn.benchmark": False,
}
_EXPECTED: list[dict[str, object] | None] = [None]


def _unguarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    return getattr(fn, "__contract_original__", fn)


def _flags(torch: Any) -> dict[str, object]:
    read: dict[str, object] = {}
    for name, get, _ in _FLAGS:
        try:
            read[name] = get(torch)
        except Exception as exc:  # torch 2.15's fp32_precision makes the allow_tf32 getters raise
            read[name] = type(exc).__name__
    return read


def pin(torch: Any) -> None:
    """The flags as the Runtime's numerics policy set them: what every scope leaves."""
    _EXPECTED[0] = _flags(torch)


def restore_flags(torch: Any) -> list[str]:
    """Put back every flag moved since `pin`; the names of those moved."""
    expected = _EXPECTED[0]
    if expected is None:
        return []
    want = dict(expected)
    if numerical_environment.deterministic_open():
        want.update(_DETERMINISTIC)
    moved = [name for name, value in _flags(torch).items() if value != want[name]]
    for name, _, put in _FLAGS:
        if name in moved:
            put(torch, want[name])
    return moved


def check_flags(torch: Any, during: str) -> None:
    """Refuse a package that moved a process-wide flag `during` an import or a scope, once the
    flag is put back."""
    if moved := restore_flags(torch):
        _refuse("global_flag", ", ".join(moved), f"changed during {during}")


# ------------------------------------------------------------------------------- the scopes


#: The executor's `() -> (varena's refused allocations on its GPU, the execution request)`, once
#: its memory opens; None from a TensorFS that counts none.
REFUSALS: list[Callable[[], tuple[int, Any] | None] | None] = [None]


class _Scopes:
    """Each local component scope (`author/_model.py`): at its exit the flags are put back and
    checked, and a refusal varena counted while it ran (`vt.ooms`) that the scope returned past,
    and that no Runtime Retry answered (`finer_splits`), is a swallowed OOM: a run warning
    (`oom_swallowed`) and the request's `swallowed_ooms`. Scopes overlapping on two threads are
    not judged for it: varena's count is the device's."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self._lock = threading.Lock()
        self._open: dict[int, int] = {}
        self._shared = 0
        self._accounted: ContextVar[int] = ContextVar("cozy_contract_ooms", default=0)

    def enter(self) -> tuple[int, int, int] | None:
        thread = threading.get_ident()
        with self._lock:
            if any(t != thread for t in self._open):
                self._shared += 1
            self._open[thread] = self._open.get(thread, 0) + 1
            shared = self._shared
        read = REFUSALS[0]() if REFUSALS[0] is not None else None
        if read is None:
            return None
        return read[0], getattr(read[1], "finer_splits", 0), shared

    def leave(self, entered: tuple[int, int, int] | None, method: str, ok: bool) -> None:
        thread = threading.get_ident()
        with self._lock:
            depth = self._open.pop(thread, 1) - 1
            if depth:
                self._open[thread] = depth
            alone = (
                entered is not None
                and self._shared == entered[2]
                and not self._open.keys() - {thread}
            )
        moved = restore_flags(self.torch)
        if ok and alone and REFUSALS[0] is not None and (read := REFUSALS[0]()) is not None:
            assert entered is not None
            ooms, request = read
            since = max(entered[0], self._accounted.get())
            swallowed = ooms - since - (getattr(request, "finer_splits", 0) - entered[1])
            self._accounted.set(ooms)
            if swallowed > 0:
                if request is not None and hasattr(request, "swallowed_ooms"):
                    request.swallowed_ooms += swallowed
                WARNINGS.append(
                    (
                        f"{method} returned past {swallowed} refused allocation(s) (an OOM it "
                        f"swallowed): use {USE['oom_catch']}",
                        {"code": "oom_swallowed", "method": method, "refused": swallowed},
                    )
                )
        if moved and ok:
            _refuse("global_flag", ", ".join(moved), f"changed during {method}")


# ------------------------------------------------------------------------------- the audit hook

_SPAWNS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.forkpty",
        "pty.spawn",
    }
)
_ENV_WRITES = frozenset({"os.putenv", "os.unsetenv"})
#: The events `audit` judges, folded into the executor's one audit hook (`sandbox.refuse_network`).
AUDITED = _SPAWNS | _ENV_WRITES | {"open", "_thread.start_new_thread"}
_WEIGHT_FILES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
_READ_ONLY = ("r", "rb", "rt", "br", "tr")


# `_COPY` is `os.environ` keyed as `putenv` and `unsetenv` are audited (bytes on POSIX); `_held`
# reads the process's own environment, which they write and `os.environ` only copies.
if sys.platform == "win32":
    _COPY: Mapping[Any, Any] = os.environ

    def _held(key: str) -> str | None:
        get = ctypes.WinDLL("kernel32").GetEnvironmentVariableW
        if not (size := get(key, None, 0)):
            return None
        value = ctypes.create_unicode_buffer(size)
        get(key, value, size)
        return value.value

else:
    _COPY: Mapping[Any, Any] = os.environb

    @functools.cache
    def _getenv() -> Callable[[bytes], bytes | None]:
        getenv = ctypes.CDLL(None).getenv
        getenv.argtypes, getenv.restype = (ctypes.c_char_p,), ctypes.c_char_p
        return getenv

    def _held(key: bytes) -> bytes | None:
        return _getenv()(key)


def _unchanged(event: str, args: tuple[Any, ...]) -> bool:
    """An env write (`putenv(key, value)`, `unsetenv(key)`) that leaves `key` as it is, both in
    `os.environ` and in the process's environment: it changes nothing."""
    key, value = args[0], args[1] if event == "os.putenv" else None
    return _COPY.get(key) == value and _held(key) == value


def audit(event: str, args: tuple[Any, ...]) -> None:
    """One of `AUDITED`, called by the audit hook itself (its frame and the hook's are skipped):
    refuse the package's env change, spawn or weight read; warn on its thread. The Runtime's own
    (zygote forks, compile children, media encodes) pass."""
    if _LIBRARIES[0] is None or OWNED.get():
        return
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else None
        if isinstance(path, bytes):
            path = path.decode(errors="replace")
        if not (isinstance(path, str) and path.endswith(_WEIGHT_FILES)):
            return
        if isinstance(mode, str) and mode not in _READ_ONLY:
            return
        if mode is None and len(args) > 2 and isinstance(args[2], int) and args[2] & 3:
            return  # os.open for writing
        if (site := _site(sys._getframe(2))) is not None:
            _refuse("weights_io", f"reading {path}", site)
    elif event == "_thread.start_new_thread":
        if (site := _site(sys._getframe(2), through=False)) is not None:
            row: WarningRow = (
                f"a package thread started at {site}: use {USE['thread']}",
                {"code": "contract_thread", "site": site},
            )
            if row not in WARNINGS:
                WARNINGS.append(row)
    elif (site := _site(sys._getframe(2))) is not None:
        if event not in _ENV_WRITES:
            _refuse("process", f"{event}({str(args[:1])[1:-2][:80]})", site)
        if not _unchanged(event, args):
            _refuse("env_write", f"{event}({str(args[:1])[1:-2][:80]}) changing it", site)
