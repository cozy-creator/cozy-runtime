"""Ordinary Diffusers and Transformers calls on the Runtime's models.

Inside `load()`, `from_pretrained` of a Diffusers or Transformers model on `loader.path` builds it
from its config on the construction substrate, each parameter in its stored dtype; everything a
`load()` built so is filled as one construction when it returns (`Loader._finish`), through
TensorFS like any other. A `torch_dtype=` is a hold, from placement on.

A component root answers its own `.to()`, `.cuda()`, `.cpu()`, `.half()`, `.float()`,
`.bfloat16()` and `.double()`, ahead of its class's: placement is the Runtime's, so a device is
ignored, and a dtype is served through `hold_dtype` (the stored bytes are cast at the next fill,
counted). Diffusers' offload helpers do nothing: varena already pages every component.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from cozy_runtime.author._errors import CapabilityError, ConformanceError
from cozy_runtime.author._loader import (
    CONSTRUCTION,
    LOADING,
    TORCH_DTYPES,
    Loader,
    canonical_dtype,
)
from cozy_runtime.author._model import PLACED, WARNINGS, WarningRow
from cozy_runtime.internal.accel import StaleBinding
from cozy_runtime.internal.encoding.formats import PLAIN

if TYPE_CHECKING:
    import tensorfs
    import torch
    from torch import nn

#: The module methods a component root answers itself.
MOVES = ("to", "cuda", "cpu", "half", "float", "bfloat16", "double")

type Hold = Callable[[torch.dtype], None]


def dtype_of(args: tuple[object, ...], kwargs: Mapping[str, object]) -> torch.dtype | None:
    """The dtype a `.to(...)` asks for: `to(dtype)`, `to(device, dtype)`, `to(dtype=)` or
    `to(tensor)`."""
    import torch

    for value in (*args[:2], kwargs.get("dtype"), kwargs.get("tensor")):
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, torch.Tensor):
            return value.dtype
    return None


def answer_moves(root: nn.Module, hold: Hold) -> None:
    """`root`'s own device and dtype methods: a device is ignored, a dtype goes to `hold`."""
    import torch

    def to(*args: object, **kwargs: object) -> nn.Module:
        dtype = dtype_of(args, kwargs)
        if dtype is not None:
            hold(dtype)
        return root

    def stay(*_: object, **__: object) -> nn.Module:
        return root

    def cast(dtype: torch.dtype) -> Callable[[], nn.Module]:
        def held() -> nn.Module:
            hold(dtype)
            return root

        return held

    vars(root).update(
        to=to,
        cuda=stay,
        cpu=stay,
        half=cast(torch.float16),
        float=cast(torch.float32),
        bfloat16=cast(torch.bfloat16),
        double=cast(torch.float64),
    )


def forget_moves(root: nn.Module) -> None:
    for name in MOVES:
        vars(root).pop(name, None)


def _warn(row: WarningRow) -> None:
    if row not in WARNINGS:
        WARNINGS.append(row)


def _held(root: nn.Module, dtype: torch.dtype, keep: tuple[str, ...] = ()) -> None:
    """A placed root's dtype change: its placeholders change now, its bytes at the next fill
    (`keep` modules in float32). A dtype its lane cannot serve (encoded weights; not f16, bf16 or
    f32) is served as stored."""
    placed = PLACED.get(root)
    if placed is None:
        return
    residency, component = placed
    try:
        change = residency.represent(component, dtype, keep=keep)
    except StaleBinding as exc:
        _warn((
            f"{component}.to({str(dtype).removeprefix('torch.')}) is served as stored: {exc}",
            {"code": "dtype_held", "component": component},
        ))  # fmt: skip
        return
    change()


def place(roots: Mapping[str, nn.Module]) -> None:
    """Placed component roots answer `.to()` and its kin from here on, and those
    `from_pretrained` built take the dtype asked of them."""
    for root in roots.values():
        answer_moves(root, functools.partial(_held, root))
        if (pending := _PENDING.pop(root, None)) is not None:
            _held(root, *pending)


def _nothing(*_: object, **__: object) -> None:
    return None


#: Diffusers' memory helpers: varena pages every component, so each is a no-op.
_OFFLOAD = {
    "diffusers.pipelines.pipeline_utils:DiffusionPipeline": (
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_offload",
    ),
    "diffusers.models.modeling_utils:ModelMixin": (
        "enable_group_offload",
        "enable_layerwise_casting",
    ),
}


def _installed(library: str) -> bool:
    return library in sys.modules or importlib.util.find_spec(library) is not None


def prepare() -> None:
    """Before a construction: Diffusers' offload helpers do nothing. Diffusers is imported here when
    the package's environment has it, so a helper called inside `load()` is already one."""
    if not _installed("diffusers"):
        return
    for where, names in _OFFLOAD.items():
        module, _, name = where.partition(":")
        owner: object = getattr(importlib.import_module(module), name)
        for method in names:
            if hasattr(owner, method):
                setattr(owner, method, _nothing)


# ------------------------------------------------------------------------- from_pretrained

#: Each root `from_pretrained` built and the dtype asked of it (`torch_dtype=`, a `.to()` in
#: `load()`), with the modules it keeps in float32: held from its placement on.
_PENDING: weakref.WeakKeyDictionary[nn.Module, tuple[torch.dtype, tuple[str, ...]]] = (
    weakref.WeakKeyDictionary()
)
#: Roots answering moves for `load()`; the fill gets their own methods back (`settle`).
_LOADING: list[weakref.ref[nn.Module]] = []

#: Where a library finds files and how it loads them: the binding chooses, the Runtime loads.
_LOADING_KWARGS = frozenset({
    "cache_dir", "force_download", "proxies", "local_files_only", "token", "revision",
    "use_safetensors", "weights_only", "low_cpu_mem_usage", "disable_mmap",
    "ignore_mismatched_sizes", "user_agent", "mirror", "resume_download", "from_tf", "from_flax",
    "use_flashpack", "flashpack_kwargs", "dduf_entries", "tqdm_class", "_from_auto",
    "_from_pipeline", "_commit_hash", "trust_remote_code", "allow_all_kernels", "use_kernels",
    "kernel_config", "key_mapping", "fusion_config", "adapter_kwargs", "adapter_name",
})  # fmt: skip
#: Of those, the ones asking for a placement, a source or an encoding: set, they warn.
_PLACING = frozenset({
    "device_map", "max_memory", "offload_folder", "offload_state_dict", "offload_buffers",
    "quantization_config", "variant", "gguf_file", "state_dict", "tp_plan", "tp_size",
    "device_mesh", "distributed_config", "parallel_config",
})  # fmt: skip


def _ignored(cls: type, kwargs: dict[str, object]) -> None:
    for key in [k for k in kwargs if k in _LOADING_KWARGS or k in _PLACING]:
        value = kwargs.pop(key)
        if key in _PLACING and value is not None and value is not False:
            _warn((
                f"{cls.__name__}.from_pretrained({key}=...) is ignored: the binding selects the "
                "weights and the Runtime places them",
                {"code": "from_pretrained_ignored", "argument": key},
            ))  # fmt: skip


def _requested(kwargs: dict[str, object]) -> torch.dtype | None:
    """`dtype=` or `torch_dtype=`; "auto" and None serve the stored dtype."""
    import torch

    found = [kwargs.pop(key, None) for key in ("dtype", "torch_dtype")]
    return next((value for value in found if isinstance(value, torch.dtype)), None)


def _where(cls: type, path: object, subfolder: object) -> tuple[Loader, Path, str] | None:
    """In a `load()`: the loader, the component's folder and its name. A path outside the bound
    model's folder refuses: weights come only from the model the binding selects."""
    from cozy_runtime.internal.fill import ROOT_COMPONENT

    loader = LOADING.get()
    if loader is None:
        return None
    folder = loader._folder
    directory = (Path(str(path)) / str(subfolder or "")).resolve()
    if folder is None or not directory.is_relative_to(folder):
        raise CapabilityError(
            f"{cls.__name__}.from_pretrained({str(path)!r}) in load(): weights come from the "
            "model the binding selects, `from_pretrained(loader.path, ...)`",
            code="weights_io",
        )
    parts = directory.relative_to(folder).parts
    if len(parts) > 1:
        raise CapabilityError(
            f"{cls.__name__}.from_pretrained: {'/'.join(parts)} is not a component of the model, "
            "each of which is one folder of loader.path",
            code="weights_io",
        )
    return loader, directory, parts[0] if parts else ROOT_COMPONENT


@contextmanager
def _constructing() -> Iterator[None]:
    """A factory call of its own: the guard reads this build's initializers as initialization."""
    token = CONSTRUCTION.set(object())
    try:
        yield
    finally:
        CONSTRUCTION.reset(token)


def _kept(name: str, keep: tuple[str, ...]) -> bool:
    return any(module in name.split(".") for module in keep)


def _as_stored(
    loader: Loader,
    component: str,
    root: nn.Module,
    requested: torch.dtype | None,
    keep: tuple[str, ...],
) -> bool:
    """Each parameter and stored buffer in the dtype the checkpoint stores, which the fill binds
    as it is. Returns whether `requested` (`keep` modules in float32) needs a hold; an encoded
    component serves its lane, with a warning."""
    import torch

    differs = encoded = False
    cast: set[int] = set()
    for path, module in root.named_modules(remove_duplicate=False):
        stored = [
            (name, tensor)
            for name, tensor in module._buffers.items()
            if name not in module._non_persistent_buffers_set
        ]
        for name, tensor in [*module._parameters.items(), *stored]:
            full = f"{path}.{name}" if path else name
            spec = loader._stored(f"{component}.{full}")
            if tensor is None or spec is None or id(tensor) in cast:
                continue
            cast.add(id(tensor))
            dtype = getattr(torch, TORCH_DTYPES.get(canonical_dtype(spec.dtype), "-"), None)
            if not isinstance(dtype, torch.dtype):
                continue
            if tensor.dtype != dtype:
                if isinstance(tensor, torch.nn.Parameter):
                    tensor.data = tensor.data.to(dtype)
                else:
                    module._buffers[name] = tensor.to(dtype)
            want = torch.float32 if _kept(full, keep) else requested
            if want is not None and dtype.is_floating_point and dtype != want:
                differs = True
                encoded = encoded or spec.encoding != PLAIN
    if differs and encoded:
        _warn((
            f"{component} serves the dtypes its lane stores: a dtype applies to plain weights only",
            {"code": "dtype_held", "component": component},
        ))  # fmt: skip
        return False
    return differs


def _built[M: nn.Module](
    loader: Loader,
    component: str,
    root: M,
    requested: torch.dtype | None,
    keep: tuple[str, ...],
) -> M:
    """A root `from_pretrained` built: as stored, pending its dtype, answering moves until the
    fill, one of the components this `load()` fills."""
    root.eval()
    if _as_stored(loader, component, root, requested, keep) and requested is not None:
        _PENDING[root] = (requested, keep)

    def pend(dtype: torch.dtype) -> None:
        _PENDING[root] = (dtype, ())

    answer_moves(root, pend)
    _LOADING.append(weakref.ref(root))
    loader._pretrained[component] = root
    return root


def _existing(loader: Loader, cls: type, component: str) -> object | None:
    held = loader._pretrained.get(component)
    if held is not None and type(held) is not cls:
        raise ConformanceError(
            f"component {component!r} is loaded as {type(held).__name__} and as {cls.__name__}: "
            "one folder of the model is one model",
            code="component_twice",
        )
    return held


def _answer(model: object, info: bool) -> object:
    """`output_loading_info=True` adds what was missing or unexpected: nothing (the fit judges)."""
    if not info:
        return model
    none: list[str] = []
    return model, {
        "missing_keys": none, "unexpected_keys": none, "mismatched_keys": none, "error_msgs": none,
    }  # fmt: skip


class _Original(Protocol):
    def __get__(self, owner: None, cls: type) -> Callable[..., object]: ...


class _DiffusersClass(Protocol):
    """A Diffusers model class as its `from_pretrained` builds one (`ConfigMixin`, `ModelMixin`)."""

    _keep_in_fp32_modules: Sequence[str] | str | None

    def load_config(
        self, path: Path, *, return_unused_kwargs: bool, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]: ...

    def from_config(self, config: dict[str, object], **kwargs: object) -> nn.Module: ...


class _Registers(Protocol):
    def register_to_config(self, **kwargs: object) -> None: ...


class _ConfigClass(Protocol):
    def from_pretrained(
        self, path: str | Path, *, return_unused_kwargs: bool, **kwargs: object
    ) -> tuple[_Named, dict[str, object]]: ...


class _Named(Protocol):
    name_or_path: str


class _TransformersClass(Protocol):
    """A Transformers model class as its `from_pretrained` builds one (`PreTrainedModel`)."""

    config_class: _ConfigClass

    def _from_config(self, config: _Named, **kwargs: object) -> nn.Module: ...


class _TransformersModel(Protocol):
    config: _Named
    generation_config: object
    _keep_in_fp32_modules: Iterable[str] | None
    _keep_in_fp32_modules_strict: Iterable[str] | None

    def tie_weights(self) -> None: ...

    def can_generate(self) -> bool: ...


def _diffusers(original: _Original, cls: type, path: object, /, **kwargs: object) -> object:
    """`ModelMixin.from_pretrained` on the bound model's folder: `from_config` of the component's
    config, weightless on the substrate."""
    where = _where(cls, path, kwargs.get("subfolder"))
    if where is None:
        return original.__get__(None, cls)(path, **kwargs)
    loader, directory, component = where
    if (held := _existing(loader, cls, component)) is not None:
        return held
    from diffusers.models.modeling_utils import no_init_weights

    kwargs.pop("subfolder", None)
    requested = _requested(kwargs)
    info = kwargs.pop("output_loading_info", False) is True
    _ignored(cls, kwargs)
    built = cast("_DiffusersClass", cls)
    config, unused = built.load_config(directory, return_unused_kwargs=True, **kwargs)
    with no_init_weights(), _constructing():
        model = built.from_config(config, **unused)
    cast("_Registers", model).register_to_config(_name_or_path=str(path))
    kept = built._keep_in_fp32_modules or ()
    keep = ((kept,) if isinstance(kept, str) else tuple(kept)) if requested is not None else ()
    return _answer(_built(loader, component, model, requested, keep), info)


def _transformers(
    original: _Original, cls: type, path: object, /, *args: object, **kwargs: object
) -> object:
    """`PreTrainedModel.from_pretrained` on the bound model's folder: `_from_config` of its
    config, weightless on the substrate, ties and generation config as Transformers sets them."""
    where = _where(cls, path, kwargs.get("subfolder"))
    if where is None:
        return original.__get__(None, cls)(path, *args, **kwargs)
    loader, directory, component = where
    if (held := _existing(loader, cls, component)) is not None:
        return held
    if args:
        raise CapabilityError(
            f"{cls.__name__}.from_pretrained: pass the model's own arguments by keyword",
            code="from_pretrained_arguments",
        )
    import copy

    import torch
    from transformers import GenerationConfig, PretrainedConfig
    from transformers.initialization import no_init_weights

    kwargs.pop("subfolder", None)
    requested = _requested(kwargs)
    info = kwargs.pop("output_loading_info", False) is True
    given = kwargs.pop("config", None)
    generation = kwargs.pop("generation_config", None)
    attention = kwargs.pop("attn_implementation", None)
    _ignored(cls, kwargs)
    built = cast("_TransformersClass", cls)
    if isinstance(given, PretrainedConfig):
        config, extra = cast("_Named", copy.deepcopy(given)), kwargs
    else:
        source = given if isinstance(given, str | Path) else directory
        config, extra = built.config_class.from_pretrained(
            source, return_unused_kwargs=True, **kwargs
        )
    if requested is not None:
        extra["dtype"] = requested
    if attention is not None:
        extra["attn_implementation"] = attention
    with no_init_weights(), _constructing():
        root = built._from_config(config, **extra)
    model = cast("_TransformersModel", root)
    model.tie_weights()
    model.config.name_or_path = str(path)
    if isinstance(generation, GenerationConfig):
        model.generation_config = generation
    elif model.can_generate() and (directory / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(directory)
    keep: set[str] = set()
    if requested == torch.float16:
        keep |= set(model._keep_in_fp32_modules or ())
    if requested in (torch.float16, torch.bfloat16):
        keep |= set(model._keep_in_fp32_modules_strict or ())
    return _answer(_built(loader, component, root, requested, tuple(sorted(keep))), info)


#: The model loaders a `load()` routes: library, owner, route.
_ROUTES = (
    ("diffusers", "diffusers.models.modeling_utils:ModelMixin", _diffusers),
    ("transformers", "transformers.modeling_utils:PreTrainedModel", _transformers),
)


@contextmanager
def routing() -> Iterator[None]:
    """For one construction: `from_pretrained` of a Diffusers or Transformers model on the bound
    model's folder builds it weightless, for the construction's one fill (`Loader._finish`)."""
    patched: list[tuple[object, object]] = []
    method = "from_pretrained"
    try:
        for library, where, route in _ROUTES:
            if not _installed(library):
                continue
            module, _, name = where.partition(":")
            owner: object = getattr(importlib.import_module(module), name)
            original = vars(owner)[method]
            patched.append((owner, original))
            setattr(owner, method, classmethod(functools.partial(route, original)))
        yield
    finally:
        for owner, original in reversed(patched):
            setattr(owner, method, original)


def settle() -> None:
    """The fill begins: roots `from_pretrained` built answer moves with their own methods again."""
    while _LOADING:
        root = _LOADING.pop()()
        if root is not None:
            forget_moves(root)


def folder(root: Path, main: tensorfs.Checkpoint, bound: Mapping[str, tensorfs.Checkpoint]) -> Path:
    """The model's folder as its Diffusers or Transformers release lays it out, without weights:
    `model_index.json`, each component's config and files, from the checkpoint binding it. Made
    once per checkpoint set under `root`; its files are read-only."""
    from cozy_runtime.internal.fill import ROOT_COMPONENT

    ids = "\n".join([main.id, *(f"{name}={ck.id}" for name, ck in sorted(bound.items()))])
    made = root / hashlib.sha256(ids.encode()).hexdigest()[:24]
    if made.is_dir():
        return made
    making = root / f".{made.name}.{os.getpid()}"
    shutil.rmtree(making, ignore_errors=True)
    for name, component in main.components.items():
        held = bound[name].components.get(name, component) if name in bound else component
        at = making if main.pipeline is None and name == ROOT_COMPONENT else making / name
        held.export_files(at)
    if main.pipeline is not None:
        (making / "model_index.json").write_text(json.dumps(main.pipeline.config, indent=2))
    for path in making.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    try:
        making.rename(made)
    except OSError:  # another construction made it first
        shutil.rmtree(making, ignore_errors=True)
    return made
