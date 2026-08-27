"""The load/unload-ONLY surface: `Loader`, its one construction primitive, and the
construction records `describe` and exact binding read.

`Loader` is a DISTINCT TYPE passed only to `Model.load`/`unload` (param name `loader`, so
`ctx` always means the request Context). Using the loader at request time, or a request
service at load time, is a type error rather than a footnote (§1.3).

`construct(T, factory=)` is the ONE construction primitive and does exactly four things
(§1.1): (1) hands the factory a typed READ-ONLY config capability — never a path; (2) runs
the SAME factory under derive and under serve; (3) walks and matches the returned object
against the checkpoint topology BEFORE allocating anything; (4) fills every destination
through the single backend path. `load(T)` is sugar over a REGISTERED exact adapter and
never duck-types `from_pretrained`.

Deliberately absent, and not to be added here: any checkpoint path, source carrier, catalog
ref, device/dtype override, offload/pin/eviction knob, compile hook or selection surface.
`Loader.engine(EngineSpec)` is the engine-hosted tier and rides cr-018/se-007 — no stub.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar, cast, final, runtime_checkable

import msgspec

from cozy_runtime.author._context import Device
from cozy_runtime.author._errors import CapabilityError, ConformanceError, InvalidRequest

T = TypeVar("T")
S = TypeVar("S", bound=msgspec.Struct)
V = TypeVar("V")

Mode = Literal["derive", "serve"]
Custody = Literal["canonical", "local"]


# --------------------------------------------------------------------------- topology


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
    """A component root. The topology walk can only see what `state_dict()` reaches
    (§1.1 authoring rules) — `torch.nn.Module` satisfies this structurally."""

    def state_dict(self) -> Mapping[str, TensorLike]: ...


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One row of a topology table: the logical shape plus how the bytes represent it."""

    shape: tuple[int, ...]
    dtype: str = "bf16"
    encoding: str = "plain"
    nbytes: int = 0


@dataclass(frozen=True, slots=True)
class Destination:
    """One logical destination the endpoint's construction demands."""

    key: str
    component: str
    spec: TensorSpec


def _dtype_of(tensor: TensorLike) -> str:
    dtype = tensor.dtype
    name = getattr(dtype, "name", None)
    return str(name if isinstance(name, str) else dtype).removeprefix("torch.")


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

    __slots__ = ("_data", "_document_digest", "_where")

    _data: Mapping[str, object]
    _document_digest: str
    _where: str

    def __init__(
        self,
        data: Mapping[str, object],
        where: str = "config",
        *,
        document_digest: str | None = None,
    ) -> None:
        """Freeze one canonical config document and its one SHA-256 identity.

        The proof seat supplies ``document_digest`` only after verifying the exact stored
        canonical bytes. In-memory fixtures omit it and thereby author their own exact
        canonical document from ``data``. Both paths use the same digest spelling; the old
        short BLAKE2 projection was not a document identity and produced MCC bytes the
        Runtime reader itself refused.
        """
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
        rendered = msgspec.json.encode(_canonical(frozen))
        measured = "sha256:" + hashlib.sha256(rendered).hexdigest()
        if document_digest is not None:
            if re.fullmatch(r"sha256:[0-9a-f]{64}", document_digest) is None:
                raise ConformanceError(
                    f"{where} document digest {document_digest!r} is not sha256:<64 lowercase hex>",
                    code="config_digest",
                )
            measured = document_digest
        object.__setattr__(self, "_data", frozen)
        object.__setattr__(self, "_document_digest", measured)
        object.__setattr__(self, "_where", where)

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
        return Config(value, f"{self._where}.{name}")

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
        for the third-party constructor an endpoint brings as a dependency, every one of
        which takes `**config` (`from_config`, `from_dict`, a dataclass). Handing it back
        field by field would mean restating a forty-field upstream schema in the endpoint,
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

    def digest(self) -> str:
        """The exact canonical config-document identity used by MCC and binding readers."""
        return self._document_digest

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


#: KNOWN MARKER RULES (§1.1). A checkpoint key whose PRESENCE states an inference-contract
#: fact rather than supplying a weight. The table is deliberately tiny and closed: a marker
#: earns a row by being a real dialect convention with a real consequence, and everything
#: not listed here is an ordinary unknown extra that warns and is ignored.
MARKER_RULES: Mapping[str, tuple[str, str]] = {
    "v_pred": ("objective", "v_prediction"),
    "v_pred_like": ("objective", "v_prediction"),
    "ztsnr": ("noise_schedule", "zero_terminal_snr"),
    "edm_vpred": ("objective", "edm_v_prediction"),
}


def marker_rule(key: str) -> tuple[str, str] | None:
    """The rule a checkpoint key matches, by its LEAF name. `model.v_pred` is `v_pred`."""
    return MARKER_RULES.get(key.rsplit(".", 1)[-1])


@dataclass(frozen=True, slots=True)
class Artifact:
    """The RESOLVED binding a slot loads: one exact snapshot, its topology, its immutable
    config, and the uploader-minted stamps a class's keywords must match.

    Selection produced this (endpoint.toml default -> `--model` -> deploy binding); author
    code never names any part of it.
    """

    snapshot: str
    topology: Mapping[str, TensorSpec]
    config: Config
    stamps: Mapping[str, str | Sequence[str]] = field(default_factory=dict)
    """The checkpoint's OPEN uploader-minted stamp map, as its catalog lane carries it: one
    key to a canonically ordered SET of values (tensorhub.md §lane). A class keyword joins
    it by MEMBERSHIP and fails closed on absence — which is what makes the N-ary packaging
    rule servable at all: the dual H3 artifact carries `task=fl2va+ref2va` because both
    task-role transformers are in it, and a scalar equality could never admit either class.
    Stamps are per CHECKPOINT, so every lane over one snapshot carries the same set.

    A bare string is the one-element set and is accepted verbatim: `__post_init__`
    normalizes, so `stamp_set` is the one reading and nothing downstream branches on which
    spelling a caller used."""
    variant: str = "plain"
    custody: Custody = "canonical"
    strict_keys: bool = False
    """`--strict-keys`: a LOCAL bind's unknown extra keys refuse instead of confessing.
    Meaningless on a canonical lane, where every extra already refuses."""

    def __post_init__(self) -> None:
        # ONE STAMP SPELLING INSIDE, ACCEPTED EITHER WAY AT THE DOOR. A narrow artifact's
        # stamp is a one-element set and every caller that writes `task="fl2va"` means
        # exactly that; normalizing here is what keeps `_admit` a single membership test
        # instead of two readings that can disagree.
        normalized: dict[str, str | Sequence[str]] = {
            key: (value,) if isinstance(value, str) else tuple(value)
            for key, value in self.stamps.items()
        }
        object.__setattr__(self, "stamps", normalized)

    def stamp_set(self, key: str) -> tuple[str, ...]:
        """This checkpoint's values for one stamp key. Absent is the EMPTY set, which is
        what "unstamped means nobody looked" spells — never a value."""
        value = self.stamps.get(key)
        if value is None:
            return ()
        return (value,) if isinstance(value, str) else tuple(value)


# --------------------------------------------------------------------------- backend


class Backend(Protocol):
    """The ONE fill path (§1.1). cr-005 supplies the TensorFS stream behind it; the
    endpoint cannot observe which backend served it — that is the backend-blindness fence.

    Three calls, one transaction: `materialize` reserves the destinations the census just
    agreed on, `fill` validates and claims each one, and `commit` moves the bytes and
    fences the copies. `fill` is deliberately NOT the thing that moves bytes — a plane
    that copies inside the walk cannot refuse a late destination without a partial
    generation to undo.

    `encoded_leaves` is the model class's own `ENCODED_LEAVES` declaration, carried in
    rather than reached for. It rides `materialize` because that is the first call of the
    transaction and the last moment before a destination is reserved: an endpoint that did
    not consent must refuse with nothing to undo, not after its bytes are on the card.
    """

    def materialize(
        self, obj: object, walked: Census, *, encoded_leaves: str
    ) -> Mapping[str, TensorLike]: ...

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None: ...

    def commit(self) -> None: ...

    def rollback(self, keys: Sequence[str]) -> None: ...


@dataclass(slots=True)
class _NullBackend:
    """Derive mode's backend: walk and match run, nothing is allocated or filled."""

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

    def rollback(self, keys: Sequence[str]) -> None:
        return None


# --------------------------------------------------------------------------- cache


@final
class DerivedCache:
    """A model-generation-scoped handle for immutable, digest-keyed derived entries.

    MEMORY-ONLY and declared-bounded: entries die with the generation, `get` is legal only
    inside an active component-use scope, and there is no persistence surface to declare
    (durable derived tables are baked artifacts produced by jobs, §1.3). cr-008b owns the
    ledger accounting and lease mechanics.
    """

    __slots__ = ("_bytes", "_entries", "_owner", "max_bytes", "max_entries", "name")

    def __init__(self, name: str, *, max_entries: int, max_bytes: int, owner: _ScopeHost) -> None:
        self.name = name
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: dict[object, object] = {}
        self._bytes = 0
        self._owner = owner

    def __repr__(self) -> str:
        return (
            f"DerivedCache({self.name!r}, entries={len(self._entries)}/{self.max_entries}, "
            f"bytes={self._bytes}/{self.max_bytes})"
        )

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
            return cast(V, self._entries[key])
        value = build()
        _require_immutable(self.name, value)
        size = _entry_bytes(value)
        if len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes:
            raise CapabilityError(
                f"cache {self.name!r} is over its declared bound "
                f"({len(self._entries) + 1} entries, {self._bytes + size} bytes vs "
                f"{self.max_entries}/{self.max_bytes}) — an unbounded entry poisons the "
                "generation rather than growing silently",
                code="cache_bound",
            )
        self._entries[key] = value
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


def register_adapter(cls: type[T], factory: Callable[[Config], T]) -> None:
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
    """What executing `load()` derived — the ModelConstructionContract's launch subset.

    cr-004 owns the derivation HARNESS (per-candidate execution under fake tensors, the
    published digest map); this is the record it produces and exact binding matches.
    """

    factory_identity: str
    config_digest: str
    destinations: tuple[Destination, ...]
    components: tuple[str, ...]
    schedulers: tuple[str, ...]
    filled: int
    filled_bytes: int
    ignored_extras: tuple[str, ...] = ()
    """LOCAL-custody extras the code does not consume: warned, ignored and CONFESSED."""
    transcribed_markers: tuple[str, ...] = ()
    """Known marker keys and the stamp each was transcribed to (`v_pred->objective=…`)."""

    @property
    def topology_digest(self) -> str:
        # The published ModelConstructionContract sorts logical destinations by key.  The
        # serve-side projection must therefore be independent of a module's state_dict walk
        # order too; otherwise the exact same derived contract can name a different logical
        # model merely because two equivalent module implementations enumerate differently.
        rows = sorted(f"{d.key}:{list(d.spec.shape)}:{d.spec.dtype}" for d in self.destinations)
        return "blake2b:" + hashlib.blake2b("\n".join(rows).encode(), digest_size=16).hexdigest()

    @property
    def identity(self) -> str:
        """The derive/serve fence: the SAME factory and config bytes construct both graphs."""
        return (
            "blake2b:"
            + hashlib.blake2b(
                f"{self.factory_identity}|{self.config_digest}|{self.topology_digest}".encode(),
                digest_size=16,
            ).hexdigest()
        )


# --------------------------------------------------------------------------- loader


@final
class Loader:
    """The load/unload-ONLY capability. Its whole surface is `construct`, `load`, `cache`
    and `device` — there is no path, dtype, placement, offload, compile or catalog door."""

    __slots__ = (
        "_artifact",
        "_backend",
        "_caches",
        "_device",
        "_markers",
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
        self._backend: Backend = backend if backend is not None else _NullBackend()
        self._mode: Mode = mode
        self._device = device or Device()
        self._owner = owner
        self._open = True
        self._markers: tuple[str, ...] = ()
        self._records: list[ConstructionRecord] = []
        self._objects: list[object] = []
        self._caches: list[DerivedCache] = []

    @property
    def device(self) -> Device:
        """Read-only device identity. Placement is the runtime's; naming it is not a door."""
        self._require_open("loader.device")
        return self._device

    def construct(self, t: type[T], *, factory: Callable[[Config], T]) -> T:
        """THE construction primitive: typed config in, walk and match, then one fill path."""
        self._require_open("loader.construct")
        identity = factory_identity(factory)
        obj = factory(self._artifact.config)
        if not isinstance(obj, t):
            raise ConformanceError(
                f"factory {identity} returned {type(obj).__name__}, declared {t.__name__} — "
                "a factory returns ONE whole object; no partial-binding tier exists (§1.1)",
                code="factory_result",
            )
        walked = census(obj)
        destinations = walked.destinations
        extras = self._match(destinations, identity)
        filled, filled_bytes = self._fill(walked, obj)
        record = ConstructionRecord(
            factory_identity=identity,
            config_digest=self._artifact.config.digest(),
            destinations=destinations,
            components=walked.components,
            schedulers=walked.schedulers,
            filled=filled,
            filled_bytes=filled_bytes,
            ignored_extras=extras,
            transcribed_markers=self._markers,
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
        """What this load built — the runtime installs guarded roots over it."""
        return tuple(self._objects)

    def caches(self) -> tuple[DerivedCache, ...]:
        """Declared derived caches. They die with the generation (§1.3)."""
        return tuple(self._caches)

    def close(self) -> None:
        """load/unload returned: the loader is dead. Holding one is not holding a capability."""
        self._open = False

    def _require_open(self, what: str) -> None:
        if not self._open:
            raise CapabilityError(
                f"{what} outside load/unload: the Loader is the LOAD-ONLY surface and request "
                "code cannot reach it — `ctx` always means the request Context (§1.3)",
                code="loader_closed",
            )

    def _match(self, destinations: Sequence[Destination], identity: str) -> tuple[str, ...]:
        """Match the code topology against the checkpoint topology BEFORE allocating."""
        table = self._artifact.topology
        if self._mode == "derive" and not table:
            # Derivation and matching are two steps on one clock, not one step. A release
            # can be derived before any binding exists (§1.0), and an open/BYOM slot binds
            # checkpoints that did not exist at build — so an artifact with no header table
            # yields the CODE topology alone, and the match runs when a candidate arrives.
            # Serve mode has no such case: an empty table there is a checkpoint supplying
            # nothing, and the missing-key refusal below is the correct answer.
            return ()
        missing = [d.key for d in destinations if d.key not in table]
        if missing:
            raise ConformanceError(
                f"{identity} vs {self._artifact.snapshot}: the code constructs "
                f"{len(missing)} destination(s) the checkpoint does not supply: "
                f"{', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}",
                code="topology_missing_keys",
                fields=missing[:6],
            )
        wrong = [
            f"{d.key} code{list(d.spec.shape)} vs bytes{list(table[d.key].shape)}"
            for d in destinations
            if tuple(table[d.key].shape) != tuple(d.spec.shape)
        ]
        if wrong:
            raise ConformanceError(
                f"{identity} vs {self._artifact.snapshot}: logical shapes differ: "
                f"{'; '.join(wrong[:4])}",
                code="topology_shape",
            )
        # A key belonging to a component this class does not construct is an UNBOUND
        # SIBLING — the dual artifact's other transformer — simply outside this
        # construction, never an "extra key" (§1.1). Surplus is judged per constructed
        # component: bytes the code DOES bind a component for but does not consume.
        built = {d.component for d in destinations}
        consumed = {d.key for d in destinations}
        extras = sorted(k for k in table if k not in consumed and k.split(".")[0] in built)
        if extras and self._artifact.custody == "canonical":
            raise ConformanceError(
                f"{self._artifact.snapshot}: {len(extras)} key(s) the code does not consume on a "
                f"CANONICAL lane — the border already classified every key, so a surplus is "
                f"conversion drift: {', '.join(extras[:6])}",
                code="topology_extra_keys",
                fields=extras[:6],
            )
        return self._local_extras(extras)

    def _local_extras(self, extras: Sequence[str]) -> tuple[str, ...]:
        """The LOCAL/dev custody tier (§1.1): markers are never ignored, unknowns confess.

        A local bind of an un-bordered checkpoint is the pull-and-run path, and its extras
        are not all the same thing. A key matching a KNOWN MARKER RULE carries meaning the
        inference contract depends on — the `v_pred` class — so it is transcribed to its
        stamp or REFUSED naming the marker; ignoring it would serve the wrong objective
        silently. Everything else warns, is ignored, and is CONFESSED: named in the bind
        result and the attempt record, which is what "loudly ignored" has to mean if it is
        to mean anything. `--strict-keys` opts the unknowns into refusal too.
        """
        if not extras:
            return ()
        marked: list[tuple[str, str, str]] = []
        unknown: list[str] = []
        for key in extras:
            rule = marker_rule(key)
            if rule is None:
                unknown.append(key)
            else:
                marked.append((key, *rule))
        for key, stamp, value in marked:
            # A CONTRADICTION IS NON-MEMBERSHIP IN A NON-EMPTY SET, not inequality with a
            # scalar: a checkpoint stamped `objective=v_prediction+epsilon` (a package
            # carrying both regimes) is not contradicted by a `v_pred` marker key. An
            # UNSTAMPED key is nobody having looked, and the marker is what transcribes it.
            declared = self._artifact.stamp_set(stamp)
            if declared and value not in declared:
                raise ConformanceError(
                    f"{self._artifact.snapshot} carries the marker key {key!r}, which means "
                    f"{stamp}={value}, but the artifact is stamped "
                    f"{stamp}={'+'.join(declared)} — a "
                    "known marker is never ignored, so a contradiction refuses NAMING it",
                    code="marker_conflict",
                    fields=[key],
                )
        self._markers = tuple(f"{key}->{stamp}={value}" for key, stamp, value in marked)
        if unknown and self._artifact.strict_keys:
            raise ConformanceError(
                f"{self._artifact.snapshot}: {len(unknown)} extra key(s) the code does not "
                f"consume, and --strict-keys refuses rather than ignoring them: "
                f"{', '.join(unknown[:6])}",
                code="strict_extra_keys",
                fields=unknown[:6],
            )
        return tuple(unknown)

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
            # The consent the MODEL CLASS declared, read off the generation that owns this
            # loader. `__encoded_leaves__` is a class attribute with a conservative default
            # on `Model` itself, so a host that never declared one says "refuse" rather
            # than raising here — the refusal belongs to the plane that would replace a
            # leaf, with the artifact's own encoding named in it.
            live = self._backend.materialize(
                obj,
                walked,
                encoded_leaves=str(getattr(type(self._owner), "__encoded_leaves__", "refuse")),
            )
            for d in walked.destinations:
                self._backend.fill(d.key, self._artifact.topology[d.key], live[d.key])
                done.append(d.key)
                total += self._artifact.topology[d.key].nbytes
            self._backend.commit()
        except Exception:
            self._backend.rollback(done)
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


def topology_of(obj: object) -> dict[str, TensorSpec]:
    """The topology table a construction DEMANDS — the same rows a checkpoint header
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
