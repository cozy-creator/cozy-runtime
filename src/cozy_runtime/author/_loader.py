"""The load/unload-ONLY surface: `Loader`, its one construction primitive, and the
construction records `describe` and exact binding read.

`Loader` is a DISTINCT TYPE passed only to `Model.load`/`unload` (param name `loader`, so
`ctx` always means the request Context). Using the loader at request time, or a request
service at load time, is a type error rather than a footnote (§1.3).

`construct(T, factory=)` is the ONE construction primitive and does exactly four things
(§1.1): (1) hands the factory a typed READ-ONLY config capability — never a path; (2) runs
the SAME factory under derive and under serve; (3) censuses the returned object and asks
the backend for the ONE fit verdict (`tensorfs.fit` over the census and the checkpoint
header, model-code-fit §3) BEFORE allocating anything; (4) fills every destination through
the single backend path. `load(T)` is sugar over a REGISTERED exact adapter and never
duck-types `from_pretrained`. The runtime keeps no matcher of its own: a verdict that is
not `ok` is the typed refusal `model_fit_refused`, carrying the Fit document verbatim.

Deliberately absent, and not to be added here: any checkpoint path, source carrier, catalog
ref, device/dtype override, offload/pin/eviction knob, compile hook or selection surface.
`Loader.engine(EngineSpec)` is the engine-hosted tier and rides cr-018/se-007 — no stub.
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, NoReturn, Protocol, TypeVar, cast, final, runtime_checkable

import msgspec

from cozy_runtime.author._context import AdapterRef, Device
from cozy_runtime.author._errors import CapabilityError, ConformanceError, InvalidRequest

T = TypeVar("T")
S = TypeVar("S", bound=msgspec.Struct)
V = TypeVar("V")

Mode = Literal["derive", "serve"]
Custody = Literal["canonical", "local"]


# ---------------------------------------------------------------------- tensor schema


@runtime_checkable
class TensorLike(Protocol):
    """A logical tensor destination. `torch.Tensor` satisfies this structurally — which is
    why the construction plane needs no torch dependency (cr-005 brings the real fills)."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> object: ...


@runtime_checkable
class ModuleLike(Protocol):
    """A component root. The tensor-schema walk can only see what `state_dict()` reaches
    (§1.1 authoring rules) — `torch.nn.Module` satisfies this structurally."""

    def state_dict(self) -> Mapping[str, TensorLike]: ...


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One tensor-schema row: logical shape plus how the bytes represent it."""

    shape: tuple[int, ...]
    dtype: str = "bf16"
    encoding: str = "plain"
    nbytes: int = 0


@dataclass(frozen=True, slots=True)
class Destination:
    """One logical destination the package's construction demands."""

    key: str
    component: str
    spec: TensorSpec


#: TensorFS dtype name -> torch dtype name. This is the ONE thing the store cannot own: it
#: is a fact about torch, and TensorFS has no business knowing torch exists. It lives HERE
#: because the census is where a torch dtype first becomes a runtime fact, and the author
#: surface may import nothing deeper; the encoding planes re-export it. The store still owns
#: the set of names and their widths, so this map is CHECKED against `tensorfs.DTYPES` by
#: the fill plane rather than trusted — a name the store knows and this map does not is a
#: refusal, not a silent passthrough that would land the wrong bytes.
TORCH_DTYPES: dict[str, str] = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "f8_e4m3fn": "float8_e4m3fn",
    "f8_e5m2": "float8_e5m2",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "u8": "uint8",
    "bool": "bool",
}

_CANONICAL_DTYPES = {torch_name: name for name, torch_name in TORCH_DTYPES.items()}


def canonical_dtype(spelling: object) -> str:
    """Any spelling of a dtype — `torch.bfloat16`, `"bfloat16"`, `"bf16"` — into TensorFS's
    closed dtype namespace. THE one canonicalization: every census row and every dtype that
    crosses toward TensorFS is spelled through here, so the fit/census documents carry only
    closed-enum tokens. An unknown name passes through for the fill plane to refuse."""

    name = str(getattr(spelling, "name", spelling)).removeprefix("torch.")
    return name if name in TORCH_DTYPES else _CANONICAL_DTYPES.get(name, name)


def _dtype_of(tensor: TensorLike) -> str:
    return canonical_dtype(tensor.dtype)


# --------------------------------------------------------------------------- config

#: Config keys that would smuggle a source carrier past the one construction primitive.
#: The runtime refuses to BUILD a config carrying one, so a factory can never read a path.
PATH_KEYS = frozenset(
    {
        "path",
        "dir",
        "file",
        "filename",
        "folder",
        "repo",
        "repo_id",
        "checkpoint",
        "checkpoint_path",
        "weights",
        "weights_path",
        "pretrained_model_name_or_path",
        "cache_dir",
        "local_files_only",
        "url",
    }
)


#: Key SUFFIXES that name a carrier however the key is spelled (`unet_path`, `vae_dir`).
PATH_SUFFIXES = ("_path", "_dir", "_file", "_url", "_repo")


def _carrier(key: str, value: object) -> str | None:
    """Why this entry is a source carrier, if it is. Keys AND values are checked: a value
    that is a filesystem path or a URL is one whatever its key is called."""
    lowered = key.lower()
    if lowered in PATH_KEYS or lowered.endswith(PATH_SUFFIXES):
        return f"{key!r} names a source carrier"
    if isinstance(value, str) and (
        value.startswith(("/", "./", "../", "~/", "file://")) or "://" in value
    ):
        return f"the value {value!r} is a path or URL"
    return None


@final
class Config:
    """The artifact's IMMUTABLE configuration, presented typed and read-only.

    Accessors are explicit (`as_int`, `section`, `typed`) rather than attribute proxying:
    a dynamic `__getattr__` would be exactly the declared `Any` seam §1.7 deletes.
    """

    __slots__ = ("_data", "_tensor_dtypes", "_where")

    _data: Mapping[str, object]
    _where: str
    _tensor_dtypes: Mapping[str, str] | None

    def __init__(
        self,
        data: Mapping[str, object],
        where: str = "config",
        *,
        tensor_dtypes: Mapping[str, str] | None = None,
    ) -> None:
        """Freeze one in-memory construction config."""
        for key, value in data.items():
            carrier = _carrier(key, value)
            if carrier is not None:
                raise ConformanceError(
                    f"{where}.{key}: {carrier} — the construction config is a typed "
                    "capability, never a path: no source carrier, checkpoint path or "
                    "catalog ref reaches a factory (§1.1)",
                    code="config_path",
                    fields=[f"{where}.{key}"],
                )
            if isinstance(value, Mapping):
                Config(value, f"{where}.{key}")
        frozen = dict(data)
        object.__setattr__(self, "_data", frozen)
        object.__setattr__(self, "_where", where)
        object.__setattr__(
            self,
            "_tensor_dtypes",
            None if tensor_dtypes is None else MappingProxyType(dict(tensor_dtypes)),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise CapabilityError(
            f"{self._where} is read-only: construction configuration is an immutable "
            "artifact fact, not a knob",
            code="config_readonly",
        )

    def __repr__(self) -> str:
        return f"Config({self._where}: {', '.join(sorted(self._data))})"

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._data))

    def section(self, name: str) -> Config:
        value = self._value(name)
        if not isinstance(value, Mapping):
            raise InvalidRequest(
                f"{self._where}.{name} is {type(value).__name__}, not a config section",
                code="config_kind",
            )
        return Config(value, f"{self._where}.{name}", tensor_dtypes=self._tensor_dtypes)

    def tensor_dtype(self, key: str, *, default: str) -> str:
        """Logical dtype of a component-qualified checkpoint tensor, before construction.

        This is native header metadata, separate from ``mapping()`` and not a compute
        dtype override. ``default`` applies only to config-only construction; once a
        checkpoint is supplied, a missing key refuses rather than hiding a typo.
        """
        if self._tensor_dtypes is None:
            return default
        try:
            return canonical_dtype(self._tensor_dtypes[key])
        except KeyError:
            raise ConformanceError(
                f"checkpoint has no tensor {key!r}", code="config_tensor_missing", fields=[key]
            ) from None

    def as_int(self, name: str, default: int | None = None) -> int:
        return self._scalar(name, int, default)

    def as_float(self, name: str, default: float | None = None) -> float:
        value = self._data.get(name, default)
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return self._scalar(name, float, default)

    def as_str(self, name: str, default: str | None = None) -> str:
        return self._scalar(name, str, default)

    def as_bool(self, name: str, default: bool | None = None) -> bool:
        return self._scalar(name, bool, default)

    def mapping(self) -> dict[str, object]:
        """The whole section as a plain, canonically ordered COPY.

        The typed accessors are the door for the author's own code; this one is the door
        for the third-party constructor a package brings as a dependency, every one of
        which takes `**config` (`from_config`, `from_dict`, a dataclass). Handing it back
        field by field would mean restating a forty-field upstream schema in the package,
        which is the declaration surface §1.1 deletes. It carries no new authority: a
        source carrier was already refused at construction, and the result is a copy, so
        this is still read-only configuration and never a path.
        """
        return cast("dict[str, object]", _canonical(self._data))

    def typed(self, schema: type[S]) -> S:
        """The author's own strict schema over this section — the fully typed door."""
        try:
            return msgspec.convert(self._data, type=schema)
        except (msgspec.ValidationError, TypeError) as exc:
            raise InvalidRequest(
                f"{self._where} does not validate against {schema.__name__}: {exc}",
                code="config_invalid",
            ) from exc

    def _value(self, name: str) -> object:
        try:
            return self._data[name]
        except KeyError:
            raise InvalidRequest(
                f"{self._where} declares no {name!r} (has: {', '.join(self.keys()) or 'nothing'})",
                code="config_missing",
                fields=[name],
            ) from None

    def _scalar(self, name: str, kind: type[V], default: V | None) -> V:
        value = self._data.get(name, default) if default is not None else self._value(name)
        if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
            raise InvalidRequest(
                f"{self._where}.{name} is {type(value).__name__}, declared {kind.__name__}",
                code="config_kind",
                fields=[name],
            )
        return value


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {k: _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


# --------------------------------------------------------------------------- artifact


@dataclass(frozen=True, slots=True)
class Artifact:
    """The RESOLVED binding a slot loads: one exact snapshot, its tensor schema and immutable
    config.

    Selection produced this (package.toml default -> `--model` -> deploy binding); author
    code never names any part of it. Nothing here is a compatibility claim: whether the
    code fits these bytes is a function of the census and the header (model-code-fit §3),
    and the backend answers it from the header it holds.
    """

    snapshot: str
    tensor_schema: Mapping[str, TensorSpec]
    config: Config
    assets: Mapping[str, bytes] = field(default_factory=dict)
    variant: str = "plain"
    custody: Custody = "canonical"
    """`canonical` refuses a stored key inside a constructed component that the code does
    not consume (conversion drift); `local` lists it and ignores it (D5)."""
    unnormalized: str = ""
    """Set when the checkpoint was stored as-is from an unrecognized layout: why, and the
    fix. A construction that fails on it refuses `checkpoint_unnormalized` with this text."""
    _prepare: Callable[[object], None] | None = field(default=None, repr=False, compare=False)
    _adapters: tuple[AdapterRef, ...] = ()
    _selection_snapshot: str = ""
    """The originally admitted base when a prepared view becomes the serving snapshot."""


#: The `Fit` document `tensorfs.fit` returns, verbatim: `code` (one of the eight),
#: `detail` (the §3 text), `component`, `key`, `encoding`, `device`, `custody`, `ignored`,
#: `routes` and `ok`. The runtime never authors one and never rephrases one.
Fit = Mapping[str, object]


UNNORMALIZED = "checkpoint_unnormalized"


class CheckpointUnnormalized(ConformanceError):
    """A factory failed on a checkpoint stored as-is; the failure is its cause."""

    default_code = UNNORMALIZED


class ModelFitRefused(ConformanceError):
    """The construction does not fit its checkpoint. Raised before a byte is reserved; the
    Fit document rides whole so the prepare refusal can carry it verbatim (§3). On a
    checkpoint stored as-is it is `checkpoint_unnormalized`, prefixed with why and the fix."""

    default_code = "model_fit_refused"

    def __init__(self, fit: Fit, unnormalized: str = "") -> None:
        verdict = f"{fit.get('code')} — {fit.get('detail')}"
        super().__init__(
            f"{unnormalized}; {verdict}" if unnormalized else verdict,
            code=UNNORMALIZED if unnormalized else None,
        )
        self.fit = dict(fit)


@final
class ModelAssets:
    """Verified read-only model files embedded in one CozyTensors artifact.

    Names are header paths, never config values. Reads are bounded and return immutable bytes
    straight from the checkpoint; there is no filesystem view. A path-only constructor is fed
    the parsed bytes instead (e.g. a tokenizer's vocab dict and merges list): model loading
    writes nothing, and a derivation's sandbox refuses any write.
    """

    MAX_BYTES = 64 << 20

    def __init__(self, values: Mapping[str, bytes], guard: Callable[[str], None]) -> None:
        self._values = {str(name): bytes(value) for name, value in values.items()}
        self._guard = guard

    def names(self) -> tuple[str, ...]:
        self._guard("loader.assets")
        return tuple(sorted(self._values))

    def read(self, name: str) -> bytes:
        self._guard("loader.assets.read")
        raw = self._get(name)
        if len(raw) > self.MAX_BYTES:
            raise CapabilityError(
                f"model asset {name!r} is {len(raw)} B; the bounded Loader limit is "
                f"{self.MAX_BYTES} B",
                code="model_asset_too_large",
            )
        return raw

    def open(self, name: str) -> io.BytesIO:
        """An in-memory binary stream over the same immutable verified bytes."""

        return io.BytesIO(self.read(name))

    def _get(self, name: str) -> bytes:
        try:
            return self._values[name]
        except KeyError:
            self._missing(name)

    def _missing(self, name: str) -> NoReturn:
        raise InvalidRequest(
            f"the CozyTensors artifact declares no model asset {name!r} "
            f"(has: {', '.join(sorted(self._values)) or 'none'})",
            code="model_asset_missing",
            fields=[name],
        )


# --------------------------------------------------------------------------- backend


class Backend(Protocol):
    """The ONE fill path (§1.1). cr-005 supplies the TensorFS stream behind it; the
    package cannot observe which backend served it — that is the backend-blindness fence.

    Four calls, one transaction: `fit` judges the census against the checkpoint header
    (`tensorfs.fit`, model-code-fit §3) with nothing reserved, `materialize` reserves the
    destinations the census just agreed on, `fill` validates and claims each one, and
    `commit` moves the bytes and fences the copies. `fill` is deliberately NOT the thing
    that moves bytes — a plane that copies inside the walk cannot refuse a late
    destination without a partial generation to undo.

    `encoded_leaves` is the model class's own `ENCODED_LEAVES` declaration, carried in
    rather than reached for. It rides `fit` (it is the `accepts` of every code row) and
    `materialize` (the last moment before a destination is reserved): a package that did
    not consent must refuse with nothing to undo, not after its bytes are on the card.
    """

    def fit(self, walked: Census, *, encoded_leaves: str) -> Fit | None:
        """The one verdict, or None when this backend holds no checkpoint header to judge
        against (derive mode; a local lane tree). The runtime keeps no other matcher."""
        ...

    def materialize(
        self, obj: object, walked: Census, *, encoded_leaves: str
    ) -> Mapping[str, TensorLike]: ...

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None: ...

    def commit(self) -> None: ...

    def poison(self, keys: Sequence[str]) -> None:
        """Abort a construction whose fill failed. The backend marks the generation
        unusable; it does not promise to free anything — the executor process is the
        transaction and its replacement reclaims the card (cr-025, decisions #613)."""
        ...


@dataclass(slots=True)
class _NullBackend:
    """Derive mode's backend: the walk runs, nothing is judged, allocated or filled."""

    def fit(self, walked: Census, *, encoded_leaves: str) -> Fit | None:
        return None

    def materialize(
        self, obj: object, walked: Census, *, encoded_leaves: str
    ) -> Mapping[str, TensorLike]:
        return walked.live

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None:
        raise CapabilityError(
            "derive mode allocates nothing: construction walks and matches under fake "
            "tensors, and fills only under serve (§1.1)",
            code="derive_fill",
        )

    def commit(self) -> None:
        return None

    def poison(self, keys: Sequence[str]) -> None:
        return None


# --------------------------------------------------------------------------- cache


@final
class DerivedCache:
    """A model-generation-scoped handle for immutable, digest-keyed derived entries.

    MEMORY-ONLY and declared-bounded: entries die with the generation, `get` is legal only
    inside an active component-use scope, and there is no persistence surface to declare
    (durable derived tables live in pruned-DiT artifacts produced by jobs, §1.3). At its
    bound the least recently used entries leave to admit a new one; an entry larger than
    the whole bound refuses. cr-008b owns the ledger accounting and lease mechanics.
    """

    __slots__ = ("_bytes", "_entries", "_owner", "max_bytes", "max_entries", "name")

    def __init__(self, name: str, *, max_entries: int, max_bytes: int, owner: _ScopeHost) -> None:
        self.name = name
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[object, tuple[object, int]] = OrderedDict()
        self._bytes = 0
        self._owner = owner

    def __repr__(self) -> str:
        return (
            f"DerivedCache({self.name!r}, entries={len(self._entries)}/{self.max_entries}, "
            f"bytes={self._bytes}/{self.max_bytes})"
        )

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    @property
    def entries(self) -> int:
        return len(self._entries)

    def get(self, key: object, *, build: Callable[[], V]) -> V:
        """Return the cached entry, or build it single-flight under the ACTIVE scope."""
        self._owner._cozy_require_scope(f"{self.name}.get")
        try:
            hash(key)
        except TypeError:
            raise ConformanceError(
                f"cache {self.name!r}: key {type(key).__name__} is mutable — entries are "
                "keyed by immutable digests, never by a live object",
                code="cache_key",
            ) from None
        if key in self._entries:
            self._entries.move_to_end(key)
            return cast(V, self._entries[key][0])
        value = build()
        _require_immutable(self.name, value)
        size = _entry_bytes(value)
        if size > self.max_bytes or self.max_entries < 1:
            raise CapabilityError(
                f"cache {self.name!r}: one {size} B entry exceeds its declared bound "
                f"({self.max_entries} entries, {self.max_bytes} bytes)",
                code="cache_bound",
            )
        while len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes:
            _, (_, dropped) = self._entries.popitem(last=False)
            self._bytes -= dropped
        self._entries[key] = (value, size)
        self._bytes += size
        return value

    def drop(self) -> None:
        self._entries.clear()
        self._bytes = 0


def _require_immutable(name: str, value: object) -> None:
    if isinstance(value, (list, dict, set, bytearray)):
        raise ConformanceError(
            f"cache {name!r}: {type(value).__name__} entries are mutable — the cache accepts "
            "only immutable derived entries (§1.3)",
            code="cache_mutable",
        )
    if isinstance(value, tuple):
        for item in value:
            _require_immutable(name, item)


def _entry_bytes(value: object) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, tuple):
        return sum(_entry_bytes(v) for v in value)
    if isinstance(value, (bytes, bytearray, str)):
        return len(value)
    return 8


class _ScopeHost(Protocol):
    """What a cache needs from the model generation that declared it."""

    def _cozy_require_scope(self, what: str) -> None: ...


# --------------------------------------------------------------------------- adapters

_ADAPTERS: dict[type, Callable[[Config], object]] = {}


def register_adapter[T](cls: type[T], factory: Callable[[Config], T]) -> None:
    """Register the EXACT construction adapter for a pipeline class, making `load(T)` legal.

    A library ships this beside the class it constructs (the runtime ships the diffusers
    one outside the author fence). There is no reflective fallback: an unregistered class
    refuses, and so does a `from_pretrained` that performs I/O.
    """
    if cls in _ADAPTERS:
        raise ConformanceError(
            f"{cls.__name__} already has a registered construction adapter: two producers of "
            "one construction is exactly the drift this fence exists for",
            code="duplicate_adapter",
        )
    _ADAPTERS[cls] = cast("Callable[[Config], object]", factory)


def registered_adapters() -> tuple[str, ...]:
    return tuple(sorted(c.__name__ for c in _ADAPTERS))


# --------------------------------------------------------------------------- record


@dataclass(frozen=True, slots=True)
class ConstructionRecord:
    """What executing `load()` derived: the tensor requirements and the fill's bookkeeping."""

    destinations: tuple[Destination, ...]
    components: tuple[str, ...]
    schedulers: tuple[str, ...]
    filled: int
    filled_bytes: int
    ignored_extras: tuple[str, ...] = ()
    """LOCAL-custody extras the code does not consume: warned, ignored and CONFESSED."""
    fit: dict[str, object] | None = None
    """The `ok` Fit document this construction was judged by, verbatim; None where no
    header was there to judge against (derive mode)."""
    weightless: tuple[int, int, int] = (0, 0, 0)
    """DID THE WEIGHTLESS SUBSTRATE FIRE (cr-102 / proto-038 q1): parameters on `meta`,
    parameters on a real device, and the bytes those real ones hold, observed after the
    factory returned. `serving_substrate` promises "parameters weightless"; nothing
    observed whether the promise held, and a construction that silently allocates its
    weights on the host costs seconds and gigabytes that no reported number names."""
    stage_ms: tuple[tuple[str, float], ...] = ()
    """THE CONSTRUCTION'S OWN LEGS (cr-102): the package factory, the census, the fit
    verdict and the fill, in execution order. The backend's `fill_ms` measures only the
    object-stream wall, so without these the difference between a slow package factory
    and a slow materialization is not observable from a prepare reply."""


# --------------------------------------------------------------------------- loader


@final
class Loader:
    """The load/unload-ONLY capability. Its whole surface is `construct`, `load`, `cache`
    and `device` — there is no path, dtype, placement, offload, compile or catalog door."""

    __slots__ = (
        "_artifact",
        "_assets",
        "_backend",
        "_caches",
        "_device",
        "_mode",
        "_objects",
        "_open",
        "_owner",
        "_records",
    )

    def __init__(
        self,
        artifact: Artifact,
        *,
        owner: _ScopeHost,
        backend: Backend | None = None,
        mode: Mode = "serve",
        device: Device | None = None,
    ) -> None:
        self._artifact = artifact
        self._assets = ModelAssets(artifact.assets, self._require_open)
        self._backend: Backend = backend if backend is not None else _NullBackend()
        self._mode: Mode = mode
        self._device = device or Device()
        self._owner = owner
        self._open = True
        self._records: list[ConstructionRecord] = []
        self._objects: list[object] = []
        self._caches: list[DerivedCache] = []

    @property
    def device(self) -> Device:
        """Read-only device identity. Placement is the runtime's; naming it is not a door."""
        self._require_open("loader.device")
        return self._device

    @property
    def assets(self) -> ModelAssets:
        """The selected CozyTensors artifact's verified, read-only model assets."""

        self._require_open("loader.assets")
        return self._assets

    def construct(self, t: type[T], *, factory: Callable[[Config], T]) -> T:
        """THE construction primitive: typed config in, walk and match, then one fill path."""
        self._require_open("loader.construct")
        identity = factory_identity(factory)
        _legs: list[tuple[str, float]] = []
        _t0 = time.perf_counter()
        config = self._artifact.config
        if self._artifact.tensor_schema:
            config = Config(
                config.mapping(),
                config._where,
                tensor_dtypes=config._tensor_dtypes
                if config._tensor_dtypes is not None
                else {key: spec.dtype for key, spec in self._artifact.tensor_schema.items()},
            )
        try:
            obj = factory(config)
        except Exception as exc:
            if self._artifact.unnormalized:
                raise CheckpointUnnormalized(
                    f"{self._artifact.unnormalized}; {type(exc).__name__}: {exc}"
                ) from exc
            raise
        _legs.append(("factory", round((time.perf_counter() - _t0) * 1000, 1)))
        _t0 = time.perf_counter()
        if not isinstance(obj, t):
            raise ConformanceError(
                f"factory {identity} returned {type(obj).__name__}, declared {t.__name__} — "
                "a factory returns ONE whole object; no partial-binding tier exists (§1.1)",
                code="factory_result",
            )
        if self._artifact._prepare is not None:
            self._artifact._prepare(obj)
        _weightless = _weightless_census(obj)
        walked = census(obj)
        _legs.append(("census", round((time.perf_counter() - _t0) * 1000, 1)))
        _t0 = time.perf_counter()
        # THE ONE VERDICT, before a byte is reserved (model-code-fit §3). The backend holds
        # the checkpoint header and asks `tensorfs.fit`; this plane matches nothing itself.
        fit = self._backend.fit(walked, encoded_leaves=self._consent())
        if fit is not None and not fit.get("ok"):
            raise ModelFitRefused(fit, self._artifact.unnormalized)
        ignored = tuple(str(k) for k in cast("Sequence[object]", (fit or {}).get("ignored", ())))
        _legs.append(("fit", round((time.perf_counter() - _t0) * 1000, 1)))
        _t0 = time.perf_counter()
        filled, filled_bytes = self._fill(walked, obj)
        _legs.append(("fill", round((time.perf_counter() - _t0) * 1000, 1)))
        record = ConstructionRecord(
            destinations=walked.destinations,
            components=walked.components,
            schedulers=walked.schedulers,
            filled=filled,
            filled_bytes=filled_bytes,
            ignored_extras=ignored,
            fit=dict(fit) if fit is not None else None,
            stage_ms=tuple(_legs),
            weightless=_weightless,
        )
        self._records.append(record)
        self._objects.append(obj)
        return obj

    def load(self, t: type[T]) -> T:
        """Sugar over a REGISTERED exact adapter. Never a `from_pretrained` duck-type."""
        self._require_open("loader.load")
        adapter = _ADAPTERS.get(t)
        if adapter is None:
            hint = (
                f" — {t.__name__}.from_pretrained exists, but a reflective fallback is exactly "
                "what this refusal deletes: it would hand a filesystem path to third-party "
                "code and desynchronize derive from serve"
                if hasattr(t, "from_pretrained")
                else ""
            )
            raise ConformanceError(
                f"load({t.__name__}) has no registered construction adapter{hint}. Name the "
                "factory explicitly: loader.construct(T, factory=...)",
                code="unregistered_adapter",
            )
        return self.construct(t, factory=cast("Callable[[Config], T]", adapter))

    def cache(self, name: str, *, max_entries: int = 64, max_bytes: int = 1 << 30) -> DerivedCache:
        """Declare a model-generation-scoped derived cache. Memory-only, by construction."""
        self._require_open("loader.cache")
        if not name or not name.replace("_", "").isalnum():
            raise ConformanceError(
                f"cache name {name!r} must be a plain identifier", code="cache_name"
            )
        cache = DerivedCache(name, max_entries=max_entries, max_bytes=max_bytes, owner=self._owner)
        self._caches.append(cache)
        return cache

    def records(self) -> tuple[ConstructionRecord, ...]:
        return tuple(self._records)

    def constructed(self) -> tuple[object, ...]:
        """What this load built, exactly as the factory returned it — nothing wraps it."""
        return tuple(self._objects)

    def caches(self) -> tuple[DerivedCache, ...]:
        """Declared derived caches. They die with the generation (§1.3)."""
        return tuple(self._caches)

    def close(self) -> None:
        """load/unload returned: the loader is dead. Holding one is not holding a capability."""
        self._open = False

    def _consent(self) -> str:
        """The consent the MODEL CLASS declared, read off the generation that owns this
        loader. `__encoded_leaves__` is a class attribute with a conservative default on
        `Model` itself, so a host that never declared one says "refuse" rather than raising
        here — the refusal belongs to the plane that would replace a leaf, with the
        artifact's own encoding named in it."""
        return str(getattr(type(self._owner), "__encoded_leaves__", "refuse"))

    def _require_open(self, what: str) -> None:
        if not self._open:
            raise CapabilityError(
                f"{what} outside load/unload: the Loader is the LOAD-ONLY surface and request "
                "code cannot reach it — `ctx` always means the request Context (§1.3)",
                code="loader_closed",
            )

    def _fill(self, walked: Census, obj: object) -> tuple[int, int]:
        """The SINGLE fill path. Failure rolls the whole construction back (§1.1).

        The destinations filled are the ones `materialize` reserved, re-censused — never
        the meta placeholders the walk agreed on. That is the seam where placement enters:
        the construction names shapes and dtypes, and the runtime alone decides where they
        live, so this loop can be identical under every backend.
        """
        if self._mode == "derive":
            return 0, 0
        done: list[str] = []
        total = 0
        try:
            live = self._backend.materialize(obj, walked, encoded_leaves=self._consent())
            for d in walked.destinations:
                self._backend.fill(d.key, self._artifact.tensor_schema[d.key], live[d.key])
                done.append(d.key)
                total += self._artifact.tensor_schema[d.key].nbytes
            self._backend.commit()
        except Exception:
            self._backend.poison(done)
            raise
        return len(done), total


@dataclass(frozen=True, slots=True)
class Census:
    """One walk of a constructed object. `live` is dropped with the walk — a construction
    record never retains tensor references (a handle asserts nothing about residency)."""

    destinations: tuple[Destination, ...]
    components: tuple[str, ...]
    schedulers: tuple[str, ...]
    live: Mapping[str, TensorLike]


def _weightless_census(obj: object) -> tuple[int, int, int]:
    """Parameters on `meta`, parameters on a real device, and the real ones' bytes."""
    meta = real = real_bytes = 0
    members = getattr(obj, "components", None)
    values = members.values() if isinstance(members, Mapping) else ()
    for member in values:
        parameters = getattr(member, "parameters", None)
        if not callable(parameters):
            continue
        for tensor in parameters():
            if getattr(getattr(tensor, "device", None), "type", "") == "meta":
                meta += 1
            else:
                real += 1
                real_bytes += tensor.numel() * tensor.element_size()
    return meta, real, real_bytes


def census(obj: object) -> Census:
    """Walk the constructed object: logical destinations, component roots, schedulers.

    Components are read from the object's `components` mapping when it has one (diffusers
    pipelines and modular pipelines both expose it) and from its public module-like
    attributes otherwise. The walk can only see what `state_dict()` reaches (§1.1).
    """
    destinations: list[Destination] = []
    components: list[str] = []
    schedulers: list[str] = []
    live: dict[str, TensorLike] = {}
    for name, member in _members(obj):
        if isinstance(member, Scheduler):
            schedulers.append(name)
            continue
        if not isinstance(member, ModuleLike):
            continue
        components.append(name)
        for key, tensor in member.state_dict().items():
            full = f"{name}.{key}"
            live[full] = tensor
            nbytes = getattr(tensor, "nbytes", 0)
            destinations.append(
                Destination(
                    full,
                    name,
                    TensorSpec(
                        tuple(tensor.shape),
                        _dtype_of(tensor),
                        nbytes=nbytes if isinstance(nbytes, int) else 0,
                    ),
                )
            )
    return Census(tuple(destinations), tuple(components), tuple(schedulers), live)


def tensor_schema_of(obj: object) -> dict[str, TensorSpec]:
    """The tensor schema a construction DEMANDS — the same rows a checkpoint header
    supplies. cr-004's derivation harness produces this per candidate under fake tensors."""
    return {d.key: d.spec for d in census(obj).destinations}


def _members(obj: object) -> Iterator[tuple[str, object]]:
    table = getattr(obj, "components", None)
    if isinstance(table, Mapping):
        for name in table:
            yield str(name), table[name]
        return
    for name in dir(obj):
        if not name.startswith("_"):
            yield name, getattr(obj, name)


@runtime_checkable
class Scheduler(Protocol):
    """A request-lifetime scheduler family instance. `view.make_scheduler(name)` returns a
    FRESH one per request; a scheduler is never shared across attempts (§1.3)."""

    @property
    def config(self) -> Mapping[str, object]: ...

    @property
    def compatibles(self) -> tuple[str, ...]: ...

    def clone(self, sampler: str | None = None) -> Scheduler: ...


def factory_identity(factory: Callable[[Config], object]) -> str:
    """The derive/serve fence's first term. Closure values are part of it, so a ROLE-SCOPED
    lambda (`task="fl2va"`) is a different factory from its twin (§1.1)."""
    code = getattr(factory, "__code__", None)
    where = getattr(factory, "__qualname__", type(factory).__name__)
    module = getattr(factory, "__module__", "?")
    parts = [f"{module}:{where}"]
    if code is not None:
        closure = getattr(factory, "__closure__", None) or ()
        for name, cell in zip(code.co_freevars, closure, strict=True):
            try:
                parts.append(f"{name}={cell.cell_contents!r}")
            except ValueError:  # an unset cell: identity falls back to code position
                parts.append(f"{name}=<unset>")
        parts.append(f"@{code.co_firstlineno}")
    return "|".join(parts)


@contextmanager
def loading(loader: Loader) -> Iterator[Loader]:
    """A loader lives exactly as long as the load/unload call it was made for."""
    try:
        yield loader
    finally:
        loader.close()
