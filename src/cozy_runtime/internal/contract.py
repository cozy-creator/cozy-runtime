"""The package contract's seal (CONTRACT §9, ENFORCEMENT §2): what harms the process a package
shares raises `ContractViolation` naming the rule and the harm. Applied once per executor before
its package imports (a zygote's forks inherit it); nothing is checked per op. A package's own
code shape is not judged: its loops, attention calls, host reads, waits (`synchronize`), threads
and library decode pass.

A call is the package's when the innermost frame that is not the standard library or a library
(a distribution other than the package's) is package code; the Runtime's own frames, and anything
under `owned()`, pass. So a library's call on the package's behalf is the package's (the refusal
names the library), and a library's call under a Runtime frame is the Runtime's. Never the
package's:
- what a library's module body does as it is imported (an env default, a helper it spawns);
- a library reading a file of its own installed tree;
- code the Runtime builds and loads, by where it lives: a kernel tree in the kernel store
  (SageAttention, comfy-kitchen), and what it calls;
- wrapping the stream that is already current (`torch.cuda.current_stream()`);
- an env write that leaves the variable as it is, since it changes nothing: torch's
  `cache_dir()` writes TensorD's own `TORCHINDUCTOR_CACHE_DIR` back at import.
The room readers raise for every caller: the Runtime reads varena (`vt.stats`, `vt.room`).

Behaviour found after the fact warns instead (a swallowed OOM, a package thread that ran a
component), since its output is already made; the release gate fails on those warnings."""

from __future__ import annotations

import ctypes
import functools
import importlib.metadata
import os
import sys
import sysconfig
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

from cozy_runtime.author import _model
from cozy_runtime.author._errors import ContractViolation
from cozy_runtime.author._model import WARNINGS, WarningRow
from cozy_runtime.internal import kernel_cache, numerical_environment

if TYPE_CHECKING:
    from cozy_runtime.internal.residency import Request

#: Each rule's harm, said in its refusal (static_contract's refusals say the same).
WHY = {
    "oom_catch": "memory is varena's, and a caught OOM hides its refusal while the call goes on",
    "room_query": "memory is varena's, and room read from the driver counts other work's bytes",
    "allocator": "the allocator is varena's, which reclaims on its own",
    "env_write": "TensorD seals the process environment that every call here shares",
    "global_flag": "the flag is process-wide and changes every other call's numerics",
    "own_stream": "the Runtime orders all device work on its own streams, graphs and device",
    "compile": "it spawns compiler processes and makes a first call take minutes; ship kernels "
    "prebuilt",
    "weights_io": "weights come only from the bound artifact through TensorFS, where their "
    "memory is counted",
    "global_rng": "the global RNG is process-wide; seed a generator of your own",
    "device_rng": "a CUDA generator's draws depend on the card; use rng.generator(seed)",
    "process": "a child process is outside the sandbox and holds memory varena cannot see",
    "internal_import": "Runtime internals bypass the contract and move without notice",
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
    """`file:line` of the package frame a call comes from (`, via <library>` when a library made
    it for the package), else None. `through`: past library frames to the code that called the
    library; otherwise a library's own call is its own. A library's module body is always its
    own: the package asked only for the import."""
    libraries = _LIBRARIES[0]
    if libraries is None or OWNED.get():
        return None
    via = ""
    while frame is not None:
        top = str(frame.f_globals.get("__name__") or "").partition(".")[0]
        if top in _RUNTIME or kernel_cache.holds(frame.f_code.co_filename):
            return None
        library = not top or top in libraries
        if not library and top not in _STDLIB:
            return f"{frame.f_code.co_filename}:{frame.f_lineno}{via}"
        if library:
            if not through or frame.f_code.co_name == "<module>":
                return None
            via = f", via {top}" if top else via
        frame = frame.f_back
    return None


def _refuse(rule: str, what: str, site: str) -> NoReturn:
    raise ContractViolation(
        f"{what} is not allowed in a package ({rule}, {site}): {WHY[rule]}",
        code="contract_violation",
        fields=[rule],
    )


def _guard[**P, R](
    rule: str,
    what: str,
    original: Callable[P, R],
    *,
    passes: Callable[[Mapping[str, object]], bool] | None = None,
) -> Callable[P, R]:
    """`original`, refusing the package's calls; `passes(kwargs)`: a call that does no harm."""

    @functools.wraps(original)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        harmless = passes is not None and passes(kwargs)
        if not harmless and (site := _site(sys._getframe(1))) is not None:
            _refuse(rule, what, site)
        return original(*args, **kwargs)

    guarded.__contract_original__ = original  # type: ignore[attr-defined]  # `_unguarded` reads it
    return guarded


class _Function(Protocol):
    """A torch function the seal replaced, as torch has it: any of several signatures."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


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
_ORIGINAL: dict[str, _Function] = {}


def unsealed(torch: ModuleType, name: str) -> _Function:
    """`torch.cuda.<name>` as torch has it, sealed or not. Never for sizing work."""
    found: _Function = _ORIGINAL.get(name) or getattr(torch.cuda, name)
    return found


def _raiser(name: str) -> _Function:
    def refused(*_: object, **__: object) -> NoReturn:
        caller = sys._getframe(1)
        _refuse(
            "room_query", f"torch.cuda.{name}", f"{caller.f_code.co_filename}:{caller.f_lineno}"
        )

    refused.__name__ = refused.__qualname__ = name
    return refused


def _replace(rule: str, owners: tuple[ModuleType | None, ...], name: str, spelled: str) -> None:
    """Guard `name` on each of `owners` that has it (one function, several import paths)."""
    for owner in owners:
        found = getattr(owner, name, None)
        if found is not None and not hasattr(found, "__contract_original__"):
            setattr(owner, name, _guard(rule, spelled, found))


def _construction(
    rule: str,
    cls: type | None,
    spelled: str,
    method: str = "__new__",
    passes: Callable[[Mapping[str, object]], bool] | None = None,
) -> None:
    """Guard building `cls` (or calling `method`) in place, so `isinstance` and subclasses
    keep their class."""
    if (
        cls is None
        or method not in vars(cls)
        or hasattr(getattr(cls, method), "__contract_original__")
    ):
        return
    found = getattr(cls, method)
    guarded = _guard(rule, spelled, found, passes=passes)
    setattr(cls, method, staticmethod(guarded) if method == "__new__" else guarded)


def seal(torch: ModuleType, application: str = "") -> None:
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
    _replace("own_stream", (cuda,), "set_device", "torch.cuda.set_device")
    _replace("own_stream", (cuda, graphs), "make_graphed_callables", "make_graphed_callables")
    # `current_stream()` and `default_stream()` wrap a stream that exists: they make none.
    _construction(
        "own_stream", getattr(cuda, "Stream", None), "torch.cuda.Stream", passes=_wraps_stream
    )
    for cls, spelled in (
        (getattr(cuda, "ExternalStream", None), "torch.cuda.ExternalStream"),
        (getattr(cuda, "Event", None), "torch.cuda.Event"),
        (getattr(cuda, "CUDAGraph", None), "torch.cuda.CUDAGraph"),
    ):
        _construction("own_stream", cls, spelled)
    _construction("own_stream", getattr(cuda, "graph", None), "torch.cuda.graph", "__init__")
    _replace("compile", (torch,), "compile", "torch.compile")
    _construction("compile", torch.nn.Module, "nn.Module.compile", "compile")
    random = getattr(cuda, "random", None)
    _replace("global_rng", (torch, torch.random), "manual_seed", "torch.manual_seed")
    _replace("global_rng", (torch.random,), "fork_rng", "torch.random.fork_rng")
    for name in ("manual_seed", "manual_seed_all"):
        _replace("global_rng", (cuda, random), name, f"torch.cuda.{name}")
    for name in ("set_float32_matmul_precision", "use_deterministic_algorithms"):
        _replace("global_flag", (torch,), name, f"torch.{name}")
    if _EXPECTED[0] is None:
        pin(torch)
    if application and _LIBRARIES[0] is None:
        _LIBRARIES[0] = _libraries(application)
        _model.CONTRACT[0] = _Scopes(torch)


def _wraps_stream(kwargs: Mapping[str, object]) -> bool:
    return "stream_id" in kwargs and "stream_ptr" not in kwargs


def _libraries(application: str) -> frozenset[str]:
    """Import names of every installed distribution but the one holding `application`."""
    found = importlib.metadata.packages_distributions()
    top = application.partition(":")[0].partition(".")[0]
    own = set(found.get(top, ()))
    return frozenset(t for t, owners in found.items() if not own & set(owners)) - _RUNTIME - {top}


# ------------------------------------------------------------------------------- the flags

#: The process-wide numerics a package could move, with how to put each back.
#: torch is read and set by attribute path here, so it is the module object itself
_FLAGS: tuple[
    tuple[str, Callable[[ModuleType], object], Callable[[ModuleType, object], object]], ...
] = (
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


def _unguarded[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    original: Callable[P, R] = getattr(fn, "__contract_original__", fn)
    return original


def _flags(torch: ModuleType) -> dict[str, object]:
    read: dict[str, object] = {}
    for name, get, _ in _FLAGS:
        try:
            read[name] = get(torch)
        except Exception as exc:  # torch 2.15's fp32_precision makes the allow_tf32 getters raise
            read[name] = type(exc).__name__
    return read


def pin(torch: ModuleType) -> None:
    """The flags as the Runtime's numerics policy set them: what every scope leaves."""
    _EXPECTED[0] = _flags(torch)


def restore_flags(torch: ModuleType) -> list[str]:
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


# ------------------------------------------------------------------------------- the scopes


#: The executor's `() -> (varena's refused allocations on its GPU, the execution request)`, once
#: its memory opens; None from a TensorFS that counts none.
REFUSALS: list[Callable[[], tuple[int, Request | None] | None] | None] = [None]


class _Scopes:
    """Each local component scope (`author/_model.py`): at its exit the flags are put back and
    checked, and a refusal varena counted while it ran (`vt.ooms`) that the scope returned past,
    and that no Runtime Retry answered (`retries`), is a swallowed OOM: a run warning
    (`oom_swallowed`) and the request's `swallowed_ooms`. Scopes overlapping on two threads are
    not judged for it: varena's count is the device's. A scope that ran on a thread the package
    started is a run warning (`contract_thread`): a package's thread launches no device work."""

    def __init__(self, torch: ModuleType) -> None:
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
        return read[0], getattr(read[1], "retries", 0), shared

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
        if (started := _THREADS.get(threading.current_thread())) is not None:
            row: WarningRow = (
                f"{method} ran on a thread the package started at {started}: a package thread "
                "launches no device work, which the Runtime orders per call",
                {"code": "contract_thread", "site": started, "method": method},
            )
            if row not in WARNINGS:
                WARNINGS.append(row)
        if ok and alone and REFUSALS[0] is not None and (read := REFUSALS[0]()) is not None:
            assert entered is not None
            ooms, request = read
            since = max(entered[0], self._accounted.get())
            swallowed = ooms - since - (getattr(request, "retries", 0) - entered[1])
            self._accounted.set(ooms)
            if swallowed > 0:
                if request is not None and hasattr(request, "swallowed_ooms"):
                    request.swallowed_ooms += swallowed
                WARNINGS.append(
                    (
                        f"{method} returned past {swallowed} refused allocation(s) (an OOM it "
                        f"swallowed): {WHY['oom_catch']}",
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
_THREAD_STARTS = frozenset({"_thread.start_new_thread", "_thread.start_joinable_thread"})
#: The events `audit` judges, folded into the executor's one audit hook (`sandbox.refuse_network`).
AUDITED = _SPAWNS | _ENV_WRITES | _THREAD_STARTS | {"open"}
#: Each live thread the package's own code started, with where.
_THREADS: weakref.WeakKeyDictionary[threading.Thread, str] = weakref.WeakKeyDictionary()
_WEIGHT_FILES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
_READ_ONLY = ("r", "rb", "rt", "br", "tr")


# `_COPY` is `os.environ` keyed as `putenv` and `unsetenv` are audited (bytes on POSIX); `_held`
# reads the process's own environment, which they write and `os.environ` only copies.
if sys.platform == "win32":
    _COPY: Mapping[str, str] = os.environ

    def _held(key: str) -> str | None:
        get = ctypes.WinDLL("kernel32").GetEnvironmentVariableW
        if not (size := get(key, None, 0)):
            return None
        value = ctypes.create_unicode_buffer(size)
        get(key, value, size)
        return value.value

else:
    _COPY: Mapping[bytes, bytes] = os.environb

    @functools.cache
    def _getenv() -> Callable[[bytes], bytes | None]:
        getenv = ctypes.CDLL(None).getenv
        getenv.argtypes, getenv.restype = (ctypes.c_char_p,), ctypes.c_char_p
        return getenv

    def _held(key: bytes) -> bytes | None:
        return _getenv()(key)


def _unchanged(event: str, args: tuple[Any, ...]) -> bool:  # type: ignore[explicit-any]  # an audit event's arguments
    """An env write (`putenv(key, value)`, `unsetenv(key)`) that leaves `key` as it is, both in
    `os.environ` and in the process's environment: it changes nothing."""
    key, value = args[0], args[1] if event == "os.putenv" else None
    return _COPY.get(key) == value and _held(key) == value


def _library_file(path: str) -> bool:
    """`path` is inside an installed library's own tree: its data, never a model's weights."""
    libraries = _LIBRARIES[0] or frozenset()
    return any(
        path.startswith(root) and path[len(root) :].partition(os.sep)[0] in libraries
        for root in _installed()
    )


@functools.cache
def _installed() -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    return tuple({os.path.join(paths[name], "") for name in ("purelib", "platlib")})


def audit(event: str, args: tuple[Any, ...]) -> None:  # type: ignore[explicit-any]  # an audit event's arguments
    """One of `AUDITED`, called by the audit hook itself (its frame and the hook's are skipped):
    refuse the package's env change, spawn or weight read; remember its thread (`_Scopes`). The
    Runtime's own (zygote forks, compile children, media encodes) pass."""
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
        if _library_file(path):
            return
        if (site := _site(sys._getframe(2))) is not None:
            _refuse("weights_io", f"reading {path}", site)
    elif event in _THREAD_STARTS:
        thread = getattr(args[0], "__self__", None) if args else None
        site = _site(sys._getframe(2), through=False)
        if site is not None and isinstance(thread, threading.Thread):
            _THREADS[thread] = site
    elif (site := _site(sys._getframe(2))) is not None:
        if event not in _ENV_WRITES:
            _refuse("process", f"{event}({str(args[:1])[1:-2][:80]})", site)
        if not _unchanged(event, args):
            _refuse("env_write", f"{event}({str(args[:1])[1:-2][:80]}) changing it", site)
