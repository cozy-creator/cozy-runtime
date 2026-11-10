"""The package contract's seal (CONTRACT §9, ENFORCEMENT §2), applied once per executor before
its package imports (a zygote's forks inherit it); nothing is checked per op.

Ordinary PyTorch runs. A call that means nothing under the Runtime does nothing, with one run
warning per kind: `torch.compile`, `empty_cache`, `cudnn.benchmark`, a stream or device switch.
A call that means something for one request is scoped to it, since one package's requests run
side by side here:
- a seed seeds that request's own generators (torch's, `random`'s and NumPy's);
- a numerics flag applies within that request's model calls and is put back after. One set
  outside every request (at import, in `load`) is the package's, for all its model calls.
Seven things harm the process a package shares, and raise `ContractViolation` in one line saying
what to do instead: catching OOM, reading free memory, replacing the allocator, raw CUDA graphs,
subprocesses, reading weight files and importing Runtime internals (`static_contract` refuses
the ones a seal cannot see at describe).

A call is the package's when the innermost frame that is not the standard library or a library
(a distribution other than the package's) is package code; the Runtime's own frames, and anything
under `owned()`, pass. So a library's call on the package's behalf is the package's (the refusal
names the library), and a library's call under a Runtime frame is the Runtime's. Never the
package's:
- what a library's module body does as it is imported (an env default, a helper it spawns);
- a library reading a file of its own installed tree;
- code the Runtime builds and loads, by where it lives: a kernel tree in the kernel store
  (SageAttention, comfy-kitchen), and what it calls;
- an env write that leaves the variable as it is, since it changes nothing: torch's
  `cache_dir()` writes TensorD's own `TORCHINDUCTOR_CACHE_DIR` back at import.
The room readers raise for every caller: the Runtime reads varena (`vt.stats`, `vt.room`).

Behaviour found after the fact warns instead (a swallowed OOM, a package thread that ran a
component), since its output is already made; the release gate fails on those warnings."""

from __future__ import annotations

import ctypes
import functools
import importlib
import importlib.metadata
import inspect
import os
import random
import sys
import sysconfig
import threading
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, SupportsInt, cast

from cozy_runtime.author import _model
from cozy_runtime.author._context import CURRENT
from cozy_runtime.author._errors import ContractViolation
from cozy_runtime.author._model import WARNINGS, WarningRow
from cozy_runtime.internal import kernel_cache, numerical_environment

if TYPE_CHECKING:
    from numpy.random import RandomState
    from torch import Generator, Tensor, nn
    from torch import device as TorchDevice

    from cozy_runtime.internal.residency import Request

#: What each refusal says: why, then what to do instead (static_contract's say the same).
WHY = {
    "oom_catch": "varena retries a call short of memory itself: delete the except, and cut a "
    "big op with cozy_runtime.author.split",
    "room_query": "memory is varena's, and the driver's numbers count other work: size work "
    "from the request, and cut a big op with cozy_runtime.author.split",
    "allocator": "memory is varena's: allocate with torch tensors",
    "cuda_graph": "the Runtime records each component's CUDA graphs itself: call the module",
    "env_write": "TensorD sets the environment every call here shares: read variables, don't "
    "change them",
    "weights_io": "weights come only from the bound artifact, where their memory is counted: "
    "build the model in load() with the loader",
    "process": "a child process is outside the sandbox and its memory: do the work in this "
    "process (DataLoader(num_workers=0); out.save_video encodes)",
    "internal_import": "Runtime internals bypass the contract and change without notice: use "
    "cozy_runtime's public modules",
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


def _package(frame: FrameType | None) -> bool:
    return _site(frame) is not None


def _refuse(rule: str, what: str, site: str) -> NoReturn:
    raise ContractViolation(
        f"{what} is not allowed in a package ({rule}, {site}): {WHY[rule]}",
        code="contract_violation",
        fields=[rule],
    )


def _warn(call: str, said: str) -> None:
    """One run warning per kind of call the Runtime ignored or narrowed."""
    row: WarningRow = (said, {"code": "ignored_call", "call": call})
    if row not in WARNINGS:
        WARNINGS.append(row)


class _Function(Protocol):
    """A torch function the seal replaced, as torch has it: any of several signatures."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


def _marked[F](wrapper: F, original: object) -> F:
    object.__setattr__(wrapper, "__contract_original__", original)  # `_unguarded` reads it
    return wrapper


def _guard(rule: str, what: str, original: _Function) -> _Function:
    """`original`, refusing the package's calls."""

    @functools.wraps(original)
    def guarded(*args: object, **kwargs: object) -> object:
        if (site := _site(sys._getframe(1))) is not None:
            _refuse(rule, what, site)
        return original(*args, **kwargs)

    return _marked(guarded, original)


def _instead(original: _Function, call: str, said: str, instead: _Function) -> _Function:
    """`original` for the Runtime; `instead` for the package, with one run warning."""

    @functools.wraps(original)
    def chosen(*args: object, **kwargs: object) -> object:
        if not _package(sys._getframe(1)):
            return original(*args, **kwargs)
        _warn(call, said)
        return instead(*args, **kwargs)

    return _marked(chosen, original)


def _swap(
    owners: Iterable[object | None], name: str, make: Callable[[_Function], _Function]
) -> None:
    """Replace `name` on each of `owners` that has it (one function, several import paths)."""
    for owner in owners:
        found = getattr(owner, name, None)
        if callable(found) and not hasattr(found, "__contract_original__"):
            setattr(owner, name, make(found))


def _ignoring(call: str, said: str, instead: _Function) -> Callable[[_Function], _Function]:
    return lambda original: _instead(original, call, said, instead)


def _nothing(*_: object, **__: object) -> None:
    return None


def _first(*args: object, **kwargs: object) -> object:
    return args[0] if args else next(iter(kwargs.values()), None)


def _words(names: str) -> tuple[str, ...]:
    return tuple(names.split())


def _unguarded[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    original: Callable[P, R] = getattr(fn, "__contract_original__", fn)
    return original


# ------------------------------------------------------------------------------- the seal

#: Room readers: memory is varena's, and nothing sizes work from these.
_ROOM = _words(
    "mem_get_info memory_stats memory_allocated memory_reserved max_memory_allocated "
    "max_memory_reserved"
)
#: Allocator calls that mean nothing under varena, each with why.
_MEMORY_NOOPS = {
    "empty_cache": "varena reclaims memory on its own",
    "reset_peak_memory_stats": "memory statistics are varena's",
    "reset_max_memory_allocated": "memory statistics are varena's",
    "reset_max_memory_cached": "memory statistics are varena's",
    "reset_accumulated_memory_stats": "memory statistics are varena's",
    "set_per_process_memory_fraction": "varena decides each run's memory",
}
#: The sealed originals: for varena's own gate cells, which read the driver.
_ORIGINAL: dict[str, _Function] = {}
_COMPILE = "the Runtime records CUDA graphs and picks kernels itself"
_ATTENTION = "the Runtime picks each attention kernel"
_STREAM = "the Runtime orders device work on its own stream"
_PLACES = "the Runtime places work on this executor's device"
_CUDNN = "the Runtime's numerics policy picks cuDNN's algorithms"


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


def _construction(rule: str, cls: type | None, spelled: str, method: str = "__new__") -> None:
    """Refuse building `cls` (or calling `method`) in place, so `isinstance` and subclasses
    keep their class."""
    if (
        cls is None
        or method not in vars(cls)
        or hasattr(getattr(cls, method), "__contract_original__")
    ):
        return
    guarded = _guard(rule, spelled, getattr(cls, method))
    setattr(cls, method, staticmethod(guarded) if method == "__new__" else guarded)


def seal(torch: ModuleType, application: str = "") -> None:
    """Seal `torch` for the package whose App is `application` (`module:attribute`)."""
    cuda, mps = torch.cuda, getattr(torch, "mps", None)
    graphs = getattr(cuda, "graphs", None)
    for name in (*_ROOM, *_MEMORY_NOOPS):
        if hasattr(cuda, name) and name not in _ORIGINAL:
            _ORIGINAL[name] = getattr(cuda, name)
    for module in (cuda, cuda.memory):
        for name in _ROOM:
            if hasattr(module, name):
                setattr(module, name, _raiser(name))
    for name, why in _MEMORY_NOOPS.items():
        for owners, spelled in (
            ((cuda, cuda.memory), f"torch.cuda.{name}"),
            ((mps,), f"torch.mps.{name}"),
        ):
            _swap(owners, name, _ignoring(spelled, f"{spelled} does nothing here: {why}", _nothing))
    _construction("cuda_graph", getattr(cuda, "CUDAGraph", None), "torch.cuda.CUDAGraph")
    _construction("cuda_graph", getattr(cuda, "graph", None), "torch.cuda.graph", "__init__")
    said = f"torch.cuda.make_graphed_callables returns its callables unchanged: {_COMPILE}"
    _swap(
        (cuda, graphs),
        "make_graphed_callables",
        _ignoring("torch.cuda.make_graphed_callables", said, _first),
    )
    said = f"torch.compile does nothing here: {_COMPILE}"
    _swap((torch,), "compile", _ignoring("torch.compile", said, _compiled))
    module_compile = vars(torch.nn.Module).get("compile")
    if module_compile is not None and not hasattr(module_compile, "__contract_original__"):
        said = f"nn.Module.compile does nothing here: {_COMPILE}"
        torch.nn.Module.compile = _instead(module_compile, "nn.Module.compile", said, _nothing)
    _attention(torch)
    _devices(torch)
    _RNG.install(torch)
    NUMERICS.install(torch)
    if _EXPECTED[0] is None:
        pin(torch)
    if application and _LIBRARIES[0] is None:
        _LIBRARIES[0] = _libraries(application)
        _model.CONTRACT[0] = _Scopes(torch)


def _compiled(*args: object, **kwargs: object) -> object:
    """`torch.compile(model)` is `model`; `torch.compile(**options)` decorates as nothing."""
    model = _first(*args, **{k: v for k, v in kwargs.items() if k == "model"})
    return model if model is not None else _first


def _attention(torch: ModuleType) -> None:
    """Attention backend choices do nothing: the Runtime picks each site's kernel."""
    backends = torch.backends.cuda
    for name in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
        "enable_cudnn_sdp",
        "allow_fp16_bf16_reduction_math_sdp",
    ):
        spelled = f"torch.backends.cuda.{name}"
        _swap(
            (backends,),
            name,
            _ignoring(spelled, f"{spelled} does nothing here: {_ATTENTION}", _nothing),
        )
    try:
        attention: ModuleType | None = importlib.import_module("torch.nn.attention")
    except ImportError:
        attention = None
    for owner, name in ((attention, "sdpa_kernel"), (backends, "sdp_kernel")):
        spelled = f"{getattr(owner, '__name__', '')}.{name}"
        _swap(
            (owner,),
            name,
            _ignoring(spelled, f"{spelled} does nothing here: {_ATTENTION}", _no_block),
        )


def _no_block(*_: object, **__: object) -> AbstractContextManager[None]:
    return nullcontext()


def _devices(torch: ModuleType) -> None:
    """Device and stream choices are the Runtime's: a new stream is the current one, and a
    switch to another device or stream does nothing (so the package's code stays correct and
    runs in order). Events, waits and `current_stream()` are the package's to use."""
    cuda = torch.cuda

    def own(device: object) -> bool:
        if device is None:
            return True
        if not cuda.is_available():
            return False
        try:
            index = int(cuda._get_device_index(device, optional=True))
        except (RuntimeError, ValueError, TypeError, AssertionError):
            return False
        return index < 0 or index == int(cuda.current_device())

    def device_choice(original: _Function, *, spelled: str) -> _Function:
        @functools.wraps(original)
        def chosen(*args: object, **kwargs: object) -> object:
            device = _first(*args, **kwargs)
            if own(device) or not _package(sys._getframe(1)):
                return original(*args, **kwargs)
            _warn(spelled, f"{spelled}({device!r}) does nothing here: {_PLACES}")
            return None

        return _marked(chosen, original)

    def stream_choice(original: _Function, *, spelled: str, current: _Function) -> _Function:
        @functools.wraps(original)
        def chosen(*args: object, **kwargs: object) -> object:
            stream = _first(*args, **kwargs)
            if (
                stream is None
                or stream == current(getattr(stream, "device", None))
                or not _package(sys._getframe(1))
            ):
                return original(*args, **kwargs)
            _warn(spelled, f"{spelled} does nothing here: {_STREAM}")
            return None

        return _marked(chosen, original)

    accelerator = getattr(torch, "accelerator", None)
    for owner, name in ((cuda, "set_device"), (accelerator, "set_device_index")):
        spelled = f"{getattr(owner, '__name__', '')}.{name}"
        _swap((owner,), name, functools.partial(device_choice, spelled=spelled))
    for owner in (cuda, accelerator):
        current = getattr(owner, "current_stream", None)
        spelled = f"{getattr(owner, '__name__', '')}.set_stream"
        if current is not None:
            _swap(
                (owner,),
                "set_stream",
                functools.partial(stream_choice, spelled=spelled, current=current),
            )
    scope: type[_DeviceScope] | None = getattr(cuda, "device", None)
    if scope is not None and not hasattr(scope.__enter__, "__contract_original__"):
        entering = scope.__enter__

        def enter(self: _DeviceScope) -> object:
            if self.idx >= 0 and not own(self.idx) and _package(sys._getframe(1)):
                _warn(
                    "torch.cuda.device",
                    f"torch.cuda.device({self.idx}) does nothing here: {_PLACES}",
                )
                self.idx = -1
            return entering(self)

        scope.__enter__ = _marked(enter, entering)  # type: ignore[method-assign]  # torch's own, wrapped
    for cls, spelled, at in (
        (getattr(cuda, "Stream", None), "torch.cuda.Stream()", 0),
        (getattr(cuda, "ExternalStream", None), "torch.cuda.ExternalStream()", 1),
    ):
        if isinstance(cls, type) and "__new__" in vars(cls):
            made: _Function = vars(cls)["__new__"].__func__
            if not hasattr(made, "__contract_original__"):
                new = _marked(_current_stream(cuda, made, spelled, at), made)
                cls.__new__ = staticmethod(new)  # type: ignore[assignment]  # torch's own, wrapped


class _DeviceScope(Protocol):
    """`torch.cuda.device`: the index it enters (-1: none)."""

    idx: int

    def __enter__(self) -> object: ...


def _current_stream(cuda: ModuleType, made: _Function, said: str, at: int) -> _Function:
    """A stream class's `__new__`: for the package, the current stream (one run warning).
    `current_stream()` and `default_stream()` wrap a stream that exists: they make none."""

    def new(*args: object, **kwargs: object) -> object:
        wraps = "stream_id" in kwargs and "stream_ptr" not in kwargs
        if wraps or not _package(sys._getframe(1)):
            return made(*args, **kwargs)
        _warn(said, f"{said} is the current stream here: {_STREAM}")
        device = kwargs.get("device", args[at + 1] if len(args) > at + 1 else None)
        return cuda.current_stream(device)

    return new


def _libraries(application: str) -> frozenset[str]:
    """Import names of every installed distribution but the one holding `application`."""
    found = importlib.metadata.packages_distributions()
    top = application.partition(":")[0].partition(".")[0]
    own = set(found.get(top, ()))
    return frozenset(t for t, owners in found.items() if not own & set(owners)) - _RUNTIME - {top}


# ------------------------------------------------------------------------------- per request


class _Request:
    """One request's own random streams and numerics flags (`Context._contract`), made at its
    first seed, flag or model call."""

    __slots__ = ("flags", "generators", "numpy", "random", "seeds")

    def __init__(self) -> None:
        self.flags: dict[str, object] = {}
        #: torch's seeds by device: "*" every device, "cuda" every GPU, "cpu", "cuda:N"
        self.seeds: dict[str, int] = {}
        self.generators: dict[str, Generator] = {}
        self.random: random.Random | None = None
        self.numpy: RandomState | None = None

    def generator(self, torch: ModuleType, where: TorchDevice) -> Generator | None:
        """This request's generator for `where`, once it seeded one; else None (the global)."""
        index: int | None = where.index
        if where.type == "meta":
            return None
        if where.type == "cuda" and index is None:
            where = torch.device("cuda", _cuda_index(torch))
        key = str(where)
        found = self.generators.get(key)
        if found is not None:
            return found
        seed = self.seeds.get(key, self.seeds.get(where.type, self.seeds.get("*")))
        if seed is None:
            return None
        made: Generator = torch.Generator(device=where)
        made.manual_seed(seed)
        self.generators[key] = made
        return made


def _request(create: bool = True) -> _Request | None:
    """The running request's own state, None outside every request (import, `load`, idle)."""
    context = CURRENT.get()
    if context is None:
        return None
    state = context._contract
    if isinstance(state, _Request):
        return state
    if not create:
        return None
    made = _Request()
    context._contract = made
    return made


def _cuda_index(torch: ModuleType) -> int:
    return int(torch.cuda.current_device()) if torch.cuda.is_available() else 0


def _fresh_seed() -> int:
    return int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)


# ------------------------------------------------------------------------------- random streams

#: torch's draws that take a `generator`: module functions, then tensor methods.
_DRAWS = _words(
    "rand rand_like randn randn_like randint randint_like randperm normal bernoulli multinomial "
    "poisson binomial"
)
_TENSOR_DRAWS = _words(
    "normal_ uniform_ random_ bernoulli_ bernoulli exponential_ geometric_ cauchy_ log_normal_ "
    "multinomial"
)


class _Rng:
    """A seed inside a request seeds that request's own streams, so its draws repeat whatever
    else runs: torch's (a draw with no `generator` takes the request's for its device),
    `random`'s and NumPy's legacy functions. A request that never seeds draws from the global
    streams, as before; a seed outside every request seeds them."""

    def install(self, torch: ModuleType) -> None:
        self.torch = torch
        for name in _DRAWS:
            _swap((torch,), name, lambda f: self._draw(f, method=False))
        for name in _TENSOR_DRAWS:
            original = getattr(torch.Tensor, name, None)
            if callable(original) and not hasattr(original, "__contract_original__"):
                setattr(torch.Tensor, name, self._draw(original, method=True))
        host, gpu = (torch, torch.random), (torch.cuda, getattr(torch.cuda, "random", None))
        on = functools.partial
        bodies: tuple[tuple[tuple[object, ...], str, _Body], ...] = (
            (host, "manual_seed", self._manual_seed),
            (host, "seed", self._manual_seed),
            (host, "initial_seed", on(self._initial, default="cpu")),
            (host, "get_rng_state", on(self._get, default="cpu")),
            (host, "set_rng_state", on(self._set, default="cpu")),
            ((torch.random,), "fork_rng", self._fork),
            (gpu, "manual_seed", self._cuda_seed),
            (gpu, "seed", self._cuda_seed),
            (gpu, "manual_seed_all", self._cuda_all),
            (gpu, "seed_all", self._cuda_all),
            (gpu, "initial_seed", on(self._initial, default="cuda")),
            (gpu, "get_rng_state", on(self._get, default="cuda")),
            (gpu, "set_rng_state", on(self._set, default="cuda")),
        )
        for owners, name, body in bodies:
            _swap(owners, name, functools.partial(_per_request, body=body, read=name in _READS_RNG))
        self._python()
        numpy = sys.modules.get("numpy")
        if numpy is not None:
            self._numpy(numpy)

    def _draw(self, original: _Function, *, method: bool) -> _Function:
        torch = self.torch

        @functools.wraps(original)
        def draw(*args: object, **kwargs: object) -> object:
            context = CURRENT.get()
            state = context._contract if context is not None else None
            if (
                isinstance(state, _Request)
                and (state.seeds or state.generators)
                and kwargs.get("generator") is None
                and not any(isinstance(a, torch.Generator) for a in args)
            ):
                where = args[0].device if method else _draw_device(torch, args, kwargs)  # type: ignore[attr-defined]  # a tensor method's self
                generator = state.generator(torch, where)
                if generator is not None:
                    kwargs["generator"] = generator
            return original(*args, **kwargs)

        return _marked(draw, original)

    def _where(self, device: object, default: str) -> TorchDevice:
        torch = self.torch
        named = default if device is None else device
        found: TorchDevice = (
            torch.device("cuda", named) if isinstance(named, int) else torch.device(named)
        )
        index: int | None = found.index
        if found.type == "cuda" and index is None:
            found = torch.device("cuda", _cuda_index(torch))
        return found

    def _manual_seed(self, state: _Request, *args: object, **kwargs: object) -> object:
        """`torch.manual_seed(seed)`, or `torch.seed()` (a fresh one, returned)."""
        seed = _seed(args, kwargs)
        state.seeds = {"*": _fresh_seed() if seed is None else seed}
        state.generators.clear()
        if seed is None:
            return state.seeds["*"]
        return state.generator(self.torch, self.torch.device("cpu"))

    def _cuda_seed(self, state: _Request, *args: object, **kwargs: object) -> None:
        """`torch.cuda.manual_seed(seed)`, or `torch.cuda.seed()`: the current GPU's."""
        key = str(self._where(None, "cuda"))
        seed = _seed(args, kwargs)
        state.seeds[key] = _fresh_seed() if seed is None else seed
        state.generators.pop(key, None)

    def _cuda_all(self, state: _Request, *args: object, **kwargs: object) -> None:
        for held in (state.seeds, state.generators):
            for key in [k for k in held if k.startswith("cuda")]:
                del held[key]
        seed = _seed(args, kwargs)
        state.seeds["cuda"] = _fresh_seed() if seed is None else seed

    def _initial(self, state: _Request, *_: object, default: str, **__: object) -> object:
        found = state.generator(self.torch, self._where(None, default))
        return _GLOBAL if found is None else found.initial_seed()

    def _get(self, state: _Request, *args: object, default: str, **kwargs: object) -> object:
        found = state.generator(self.torch, self._where(_first(*args, **kwargs), default))
        return _GLOBAL if found is None else found.get_state()

    def _set(self, state: _Request, *args: object, default: str, **kwargs: object) -> None:
        new = kwargs.get("new_state", args[0] if args else None)
        where = self._where(kwargs.get("device", args[1] if len(args) > 1 else None), default)
        made = state.generator(self.torch, where) or self.torch.Generator(device=where)
        made.set_state(cast("Tensor", new))
        state.generators[str(where)] = made

    def _fork(self, state: _Request, *args: object, **kwargs: object) -> object:
        enabled = kwargs.get("enabled", args[1] if len(args) > 1 else True)
        return _forked(state) if enabled else nullcontext()

    def _python(self) -> None:
        shared = random._inst
        for name in random.__all__:
            bound = getattr(random, name, None)
            if not callable(bound) or getattr(bound, "__self__", None) is not shared:
                continue
            seeding = name in ("seed", "setstate")
            setattr(random, name, _routed(name, bound, "random", random.Random, seeding))

    def _numpy(self, numpy: ModuleType) -> None:
        legacy = numpy.random.mtrand._rand
        for name in dir(numpy.random):
            bound = getattr(numpy.random, name)
            seeding = name in ("seed", "set_state")
            if (getattr(bound, "__self__", None) is not legacy and name != "seed") or hasattr(
                bound, "__contract_original__"
            ):
                continue
            setattr(
                numpy.random, name, _routed(name, bound, "numpy", numpy.random.RandomState, seeding)
            )


#: A per-request body's answer: the request has no stream of its own here, so torch's global.
_GLOBAL = object()
#: Seed functions that only read: they never make a request's state.
_READS_RNG = frozenset({"initial_seed", "get_rng_state"})


class _Body(Protocol):
    """A seed function's work inside a request, on its own streams (or `_GLOBAL`)."""

    def __call__(self, state: _Request, *args: object, **kwargs: object) -> object: ...


def _per_request(original: _Function, *, body: _Body, read: bool) -> _Function:
    """`original` outside every request; inside one, `body(state, ...)` on its own streams."""

    @functools.wraps(original)
    def scoped(*args: object, **kwargs: object) -> object:
        state = _request(create=not read)
        if state is None or (found := body(state, *args, **kwargs)) is _GLOBAL:
            return original(*args, **kwargs)
        return found

    return _marked(scoped, original)


def _seed(args: tuple[object, ...], kwargs: Mapping[str, object]) -> int | None:
    value = _first(*args, **kwargs)
    return None if value is None else int(cast("SupportsInt", value))


def _routed(
    name: str, shared: _Function, slot: str, make: Callable[[], object], seeding: bool
) -> _Function:
    """`random.<name>` (or NumPy's): the request's own stream once it seeded one."""

    @functools.wraps(shared)
    def routed(*args: object, **kwargs: object) -> object:
        if CURRENT.get() is None:
            return shared(*args, **kwargs)
        state = _request(create=seeding)
        own = getattr(state, slot) if state is not None else None
        if own is None and seeding and state is not None:
            own = make()
            setattr(state, slot, own)
        if own is None:
            return shared(*args, **kwargs)
        method: _Function = getattr(own, name)
        return method(*args, **kwargs)

    return _marked(routed, shared)


def _draw_device(
    torch: ModuleType, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> TorchDevice:
    """The device a torch draw lands on: its `device=`, its `out=`, its first tensor, else the
    default."""
    named = kwargs.get("device")
    if named is not None:
        found: TorchDevice = torch.device(named)
        return found
    for value in (kwargs.get("out"), *args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            where: TorchDevice = value.device
            return where
    default: TorchDevice = _unguarded(torch.get_default_device)()
    return default


@contextmanager
def _forked(state: _Request) -> Iterator[None]:
    """`torch.random.fork_rng` in a request: its own streams, put back after."""
    seeds = dict(state.seeds)
    held = {key: generator.get_state() for key, generator in state.generators.items()}
    try:
        yield
    finally:
        state.seeds = seeds
        for key in [k for k in state.generators if k not in held]:
            del state.generators[key]
        for key, saved in held.items():
            state.generators[key].set_state(saved)


_RNG = _Rng()


# ------------------------------------------------------------------------------- numerics flags


class _Prop(Protocol):
    """torch.backends' `ContextProp`: a module attribute read and set through torch."""

    getter: Callable[[], object]
    setter: Callable[[object], object]

    def __get__(self, obj: object, objtype: type | None = None) -> object: ...
    def __set__(self, obj: object, value: object) -> None: ...


_MISSING = object()


class _Numerics:
    """The numerics flags a package sets: inside a request that request's (`_Request.flags`),
    outside every request the package's own (`own`). Model calls take turns by flags: the first
    in applies its flags, calls wanting the same flags share the turn, and the last out puts
    torch's back, so a request's flags never reach another request's model call. A request that
    sets a flag inside its own model call, alone, has it applied at once. A thread outside every
    request never waits: it joins the turn running."""

    def __init__(self) -> None:
        self.own: dict[str, object] = {}
        #: each flag's (read, set) on torch itself
        self.access: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {}
        #: the flags the package ever set: torch's reads of the others skip the scoping
        self.named: set[str] = set()
        self._changed = threading.Condition()
        self._holders: dict[_Request | None, int] = {}
        self._applied: dict[str, object] | None = None
        self._saved: dict[str, object] = {}
        self.torch: ModuleType | None = None

    def _wanted(self, holder: _Request | None) -> dict[str, object]:
        return self.own | holder.flags if holder is not None else dict(self.own)

    def take(self, holder: _Request | None) -> None:
        with self._changed:
            while True:
                if holder in self._holders or (holder is None and self._holders):
                    self._holders[holder] = self._holders.get(holder, 0) + 1
                    return
                wanted = self._wanted(holder)
                if not self._holders:
                    try:
                        for name, value in wanted.items():
                            read, put = self.access[name]
                            self._saved[name] = read()
                            put(value)
                    except BaseException:
                        self._put_back()
                        raise
                    self._applied = wanted
                if wanted == self._applied:
                    self._holders[holder] = 1
                    return
                self._changed.wait()

    def give(self, holder: _Request | None) -> list[str]:
        """The flags torch had moved besides the package's, put back once the last call ends."""
        with self._changed:
            count = self._holders.get(holder, 0) - 1
            if count > 0:
                self._holders[holder] = count
                return []
            self._holders.pop(holder, None)
            if self._holders:
                return []
            self._put_back()
            moved = restore_flags(self.torch) if self.torch is not None else []
            self._changed.notify_all()
            return moved

    def _put_back(self) -> None:
        for name, value in self._saved.items():
            self.access[name][1](value)
        self._saved.clear()
        self._applied = None

    def write(self, name: str, value: object) -> None:
        holder = _request()
        (holder.flags if holder is not None else self.own)[name] = value
        self.named.add(name)
        self._now(holder, name)

    def forget(self, name: str) -> None:
        holder = _request()
        (holder.flags if holder is not None else self.own).pop(name, None)
        self._now(holder, name)

    def _now(self, holder: _Request | None, name: str) -> None:
        with self._changed:
            if list(self._holders) != [holder]:
                return  # applied at this caller's next model call, never into another's
            read, put = self.access[name]
            wanted = self._wanted(holder)
            if name in wanted:
                self._saved.setdefault(name, read())
                put(wanted[name])
            elif name in self._saved:
                put(self._saved.pop(name))
            self._applied = wanted

    def read(self, name: str) -> object:
        holder = _request(create=False)
        if holder is not None and name in holder.flags:
            return holder.flags[name]
        if name in self.own:
            return self.own[name]
        return self.access[name][0]()

    # -------------------------------------------------------------- torch's flags

    def install(self, torch: ModuleType) -> None:
        self.torch = torch
        backends = torch.backends
        modules = [backends, *(m for m in vars(backends).values() if isinstance(m, ModuleType))]
        for module in modules:
            for attribute, prop in list(vars(type(module)).items()):
                if type(prop).__name__ == "ContextProp":
                    setattr(type(module), attribute, self._prop(module, attribute, prop))
        self._by_name(type(backends.cuda.matmul), lambda _: "torch.backends.cuda.matmul")
        precision = getattr(backends, "_FP32Precision", None)
        if isinstance(precision, type):
            self._by_name(precision, lambda o: "torch.backends.{backend}.{op}".format_map(vars(o)))
        pairs = (
            ("set_float32_matmul_precision", "get_float32_matmul_precision", ""),
            ("set_deterministic_debug_mode", "get_deterministic_debug_mode", ""),
            ("set_default_dtype", "get_default_dtype", "dtype"),
            ("set_default_device", "get_default_device", "device"),
        )
        for setter, getter, made in pairs:
            if hasattr(torch, setter) and hasattr(torch, getter):
                self._pair(torch, setter, getter, made)
        self._deterministic(torch)
        self._choices(torch)
        self._blocks(torch)

    def _prop(self, module: ModuleType, attribute: str, prop: _Prop) -> object:
        name = f"{module.__name__}.{attribute}"
        self.access[name] = (prop.getter, lambda value: prop.__set__(module, value))
        numerics = self
        benchmark = name == "torch.backends.cudnn.benchmark"

        class Scoped:
            """`torch.backends.<x>.<flag>`: the package's reads and writes are scoped."""

            def __get__(self, obj: object, objtype: type | None = None) -> object:
                if obj is None:
                    return self
                if name not in numerics.named or not _package(sys._getframe(1)):
                    return prop.__get__(obj, objtype)
                return numerics.read(name)

            def __set__(self, obj: object, value: object) -> None:
                if not _package(sys._getframe(1)):
                    prop.__set__(obj, value)
                elif benchmark:
                    if value:
                        _warn(name, f"{name} = True does nothing here: {_CUDNN}")
                else:
                    numerics.write(name, value)

        return Scoped()

    def _by_name(self, cls: type, prefix: Callable[[object], str]) -> None:
        """A class whose instances read and set their flags by attribute name."""
        reading: Callable[[object, str], object] = vars(cls)["__getattr__"]
        setting: Callable[[object, str, object], None] = vars(cls)["__setattr__"]
        if hasattr(reading, "__contract_original__"):
            return
        numerics = self

        def get(obj: object, attribute: str) -> object:
            if attribute.startswith("_"):
                return reading(obj, attribute)
            name = f"{prefix(obj)}.{attribute}"
            if name not in numerics.named or not _package(sys._getframe(1)):
                return reading(obj, attribute)
            return numerics.read(name)

        def put(obj: object, attribute: str, value: object) -> None:
            if attribute in ("backend", "op") or not _package(sys._getframe(1)):
                setting(obj, attribute, value)
                return
            name = f"{prefix(obj)}.{attribute}"
            reading(obj, attribute)  # an unknown flag raises as torch has it
            numerics.access.setdefault(
                name, (lambda: reading(obj, attribute), lambda v: setting(obj, attribute, v))
            )
            numerics.write(name, value)

        for attribute, method in (
            ("__getattr__", _marked(get, reading)),
            ("__setattr__", _marked(put, setting)),
        ):
            setattr(cls, attribute, method)

    def _pair(self, torch: ModuleType, setter: str, getter: str, made: str) -> None:
        """`torch.set_x(value)` and `torch.get_x()`. `made`: a default for the tensors code makes.
        A library sets those around its own construction (Transformers' `torch_dtype`), and a
        package in `load` around its, so they are scoped only for the package's own call in a
        request: inside its model calls, with one run warning."""
        set_, get_ = getattr(torch, setter), getattr(torch, getter)
        if hasattr(set_, "__contract_original__"):
            return
        name = f"torch.{getter}"
        self.access[name] = (get_, set_)

        def scoped(frame: FrameType) -> bool:
            if not made:
                return _package(frame)
            return CURRENT.get() is not None and _site(frame, through=False) is not None

        @functools.wraps(set_)
        def set_flag(value: object) -> None:
            if not scoped(sys._getframe(1)):
                set_(value)
                return
            if made:
                said = f"torch.{setter} applies only inside model calls here: pass {made}= to "
                _warn(setter, said + "tensors made outside them")
            self.write(name, value)

        @functools.wraps(get_)
        def get_flag() -> object:
            if name not in self.named or not scoped(sys._getframe(1)):
                return get_()
            return self.read(name)

        for attribute, made_flag in (
            (setter, _marked(set_flag, set_)),
            (getter, _marked(get_flag, get_)),
        ):
            setattr(torch, attribute, made_flag)

    def _deterministic(self, torch: ModuleType) -> None:
        """`use_deterministic_algorithms(mode, warn_only=)`, read back by its two getters."""
        use, enabled = (
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled,
        )
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled
        if hasattr(use, "__contract_original__"):
            return
        name = "torch.use_deterministic_algorithms"
        self.access[name] = (
            lambda: (enabled(), warn_only()),
            lambda value: use(value[0], warn_only=value[1])
            if isinstance(value, tuple)
            else use(value),
        )

        @functools.wraps(use)
        def use_flag(mode: bool, *, warn_only: bool = False) -> None:
            if not _package(sys._getframe(1)):
                use(mode, warn_only=warn_only)
                return
            self.write(name, (bool(mode), bool(warn_only)))

        def part(index: int, original: _Function) -> Callable[[], object]:
            @functools.wraps(original)
            def get() -> object:
                if name not in self.named or not _package(sys._getframe(1)):
                    return original()
                value = self.read(name)
                return value[index] if isinstance(value, tuple) else value

            return _marked(get, original)

        for attribute, function in (
            ("use_deterministic_algorithms", _marked(use_flag, use)),
            ("are_deterministic_algorithms_enabled", part(0, enabled)),
            ("is_deterministic_algorithms_warn_only_enabled", part(1, warn_only)),
        ):
            setattr(torch, attribute, function)

    def _choices(self, torch: ModuleType) -> None:
        """`preferred_linalg_library(backend=None)` and its kin: read with no argument."""
        for attribute in ("preferred_linalg_library", "preferred_blas_library"):
            original = getattr(torch.backends.cuda, attribute, None)
            if original is not None and not hasattr(original, "__contract_original__"):
                name = f"torch.backends.cuda.{attribute}"
                self.access[name] = (original, original)
                setattr(
                    torch.backends.cuda, attribute, _marked(self._chooser(name, original), original)
                )

    def _chooser(self, name: str, original: _Function) -> _Function:
        def choose(*args: object, **kwargs: object) -> object:
            backend = _first(*args, **kwargs)
            if not _package(sys._getframe(1)):
                return original(*args, **kwargs)
            if backend is not None:
                self.write(name, backend)
            return self.read(name)

        return choose

    def _blocks(self, torch: ModuleType) -> None:
        """`with torch.backends.cudnn.flags(...)` and its kin: the package's flags for the block."""
        backends = torch.backends
        for module in (
            backends,
            getattr(backends, "cudnn", None),
            getattr(backends, "mkldnn", None),
        ):
            original = getattr(module, "flags", None)
            if (
                module is not None
                and original is not None
                and not hasattr(original, "__contract_original__")
            ):
                module.flags = _marked(self._flags_block(module.__name__, original), original)

    def _flags_block(self, prefix: str, original: _Function) -> _Function:
        signature = inspect.signature(original)

        def flags(*args: object, **kwargs: object) -> object:
            if not _package(sys._getframe(1)):
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            wanted = {
                f"{prefix}.{param}": value
                for param, value in bound.arguments.items()
                if value is not None and f"{prefix}.{param}" in self.access
            }
            return self._block(wanted)

        return flags

    @contextmanager
    def _block(self, wanted: Mapping[str, object]) -> Iterator[None]:
        holder = _request()
        held = holder.flags if holder is not None else self.own
        before = {name: held.get(name, _MISSING) for name in wanted}
        for name, value in wanted.items():
            if name == "torch.backends.cudnn.benchmark":
                if value:
                    _warn(name, f"{name} = True does nothing here: {_CUDNN}")
                continue
            self.write(name, value)
        try:
            yield
        finally:
            for name, value in before.items():
                if name == "torch.backends.cudnn.benchmark":
                    continue
                if value is _MISSING:
                    self.forget(name)
                else:
                    self.write(name, value)


NUMERICS = _Numerics()


def install(roots: Iterable[nn.Module]) -> None:
    """Each call of a component root, or of one of its children (`vae.decode` calls only those),
    takes the numerics turn (`_Numerics`) as a model method's scope does: so a plain loop
    calling a component gets its request's flags too."""
    held = threading.local()

    def taken(module: nn.Module, *_: object) -> None:
        holder = _request()
        NUMERICS.take(holder)
        stack: list[tuple[int, _Request | None]] = held.__dict__.setdefault("stack", [])
        stack.append((id(module), holder))

    def given(module: nn.Module, *_: object) -> None:
        stack: list[tuple[int, _Request | None]] = held.__dict__.get("stack", [])
        if stack and stack[-1][0] == id(module):
            holder = stack.pop()[1]
            if moved := NUMERICS.give(holder):
                _moved(moved, type(module).__name__)

    for root in roots:
        for module in (root, *root.children()):
            if getattr(module, "_cozy_numerics", False):
                continue
            module.register_forward_pre_hook(taken, prepend=True)
            module.register_forward_hook(given, always_call=True)
            object.__setattr__(module, "_cozy_numerics", True)


def _moved(moved: list[str], method: str) -> None:
    _warn(
        "flags",
        f"{', '.join(moved)} changed during {method} outside torch's flag API and was put back: "
        "set it through torch.backends, where it applies per request",
    )


# ------------------------------------------------------------------------------- pinned flags

#: The process-wide numerics, with how to put each back: checked when the last model call ends.
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
            lambda t, v, kind=kind: _unguarded(getattr(t.backends.cuda, f"enable_{kind}_sdp"))(v),
        )
        for kind in ("flash", "mem_efficient", "math", "cudnn")
    ),
    (
        "torch.get_float32_matmul_precision",
        lambda t: _unguarded(t.get_float32_matmul_precision)(),
        lambda t, v: _unguarded(t.set_float32_matmul_precision)(v),
    ),
    (
        "torch.get_default_dtype",
        lambda t: _unguarded(t.get_default_dtype)(),
        lambda t, v: _unguarded(t.set_default_dtype)(v),
    ),
    (
        "torch.are_deterministic_algorithms_enabled",
        lambda t: _unguarded(t.are_deterministic_algorithms_enabled)(),
        lambda t, v: _unguarded(t.use_deterministic_algorithms)(v),
    ),
)
#: cuDNN's mode while a `deterministic=True` scope is open (numerical_environment).
_DETERMINISTIC = {
    "torch.backends.cudnn.deterministic": True,
    "torch.backends.cudnn.benchmark": False,
}
_EXPECTED: list[dict[str, object] | None] = [None]


def _flags(torch: ModuleType) -> dict[str, object]:
    read: dict[str, object] = {}
    for name, get, _ in _FLAGS:
        try:
            read[name] = get(torch)
        except Exception as exc:  # torch 2.15's fp32_precision makes the allow_tf32 getters raise
            read[name] = type(exc).__name__
    return read


def pin(torch: ModuleType) -> None:
    """The flags as the Runtime's numerics policy set them: what every model call leaves."""
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

type _Entered = tuple[_Request | None, tuple[int, int, int] | None]


class _Scopes:
    """Each local component scope (`author/_model.py`) takes the numerics turn for its request
    (`_Numerics`). A refusal varena counted while it ran (`vt.ooms`) that the scope returned
    past, and that no Runtime Retry answered (`finer_splits`), is a swallowed OOM: a run warning
    (`oom_swallowed`) and the request's `swallowed_ooms`. Scopes overlapping on two threads are
    not judged for it: varena's count is the device's. A scope that ran on a thread the package
    started is a run warning (`contract_thread`): a package's thread launches no device work."""

    def __init__(self, torch: ModuleType) -> None:
        self.torch = torch
        self._lock = threading.Lock()
        self._open: dict[int, int] = {}
        self._shared = 0
        self._accounted: ContextVar[int] = ContextVar("cozy_contract_ooms", default=0)

    def enter(self) -> _Entered:
        holder = _request()
        NUMERICS.take(holder)
        thread = threading.get_ident()
        with self._lock:
            if any(t != thread for t in self._open):
                self._shared += 1
            self._open[thread] = self._open.get(thread, 0) + 1
            shared = self._shared
        read = REFUSALS[0]() if REFUSALS[0] is not None else None
        if read is None:
            return holder, None
        return holder, (read[0], getattr(read[1], "finer_splits", 0), shared)

    def leave(self, entered: _Entered, method: str, ok: bool) -> None:
        holder, counted = entered
        thread = threading.get_ident()
        with self._lock:
            depth = self._open.pop(thread, 1) - 1
            if depth:
                self._open[thread] = depth
            alone = (
                counted is not None
                and self._shared == counted[2]
                and not self._open.keys() - {thread}
            )
        if moved := NUMERICS.give(holder):
            _moved(moved, method)
        if (started := _THREADS.get(threading.current_thread())) is not None:
            row: WarningRow = (
                f"{method} ran on a thread the package started at {started}: a package thread "
                "launches no device work, which the Runtime orders per call",
                {"code": "contract_thread", "site": started, "method": method},
            )
            if row not in WARNINGS:
                WARNINGS.append(row)
        if ok and alone and REFUSALS[0] is not None and (read := REFUSALS[0]()) is not None:
            assert counted is not None
            ooms, request = read
            since = max(counted[0], self._accounted.get())
            swallowed = ooms - since - (getattr(request, "finer_splits", 0) - counted[1])
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
