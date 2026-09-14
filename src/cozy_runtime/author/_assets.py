"""Typed assets — the ONE asset type, both directions (§1.3/§3.4).

Input assets are ordinary typed request fields or an explicit ordered Assets input.
Their bytes arrive digest-verified; output assets are what `out.save_*` returns and
what the result struct declares. Both use the same handles and granted reader.

`Asset` is deliberately NOT a msgspec-native type. The only way one comes into existence
from the wire is `asset_dec_hook`, which accepts a REF STRING and nothing else — so
hydration state is unforgeable by request data rather than merely undocumented, and local
paths cannot enter a schema, a digest, or serialized output.
"""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Self, cast, get_args, overload

from PIL.Image import Image as Image

from cozy_runtime.author._errors import CapabilityError, PreflightViolation, RuntimeFailure
from cozy_runtime.author._markers import AssetBound, ImagePreparation
from cozy_runtime.author._walker import strip, unwrap_optional

_UNBOUND: Final = ""
Fidelity = Literal["auto", "low", "medium", "high"]


# Private process-handoff facts, never input identity or wire metadata.
FileState = tuple[int, int, int, int, int]


def file_state(path: Path) -> FileState:
    try:
        facts = path.lstat()
    except OSError as exc:
        raise RuntimeFailure(
            "verified input file is unavailable", code="input_unavailable"
        ) from exc
    if not stat.S_ISREG(facts.st_mode):
        raise RuntimeFailure("verified input is no longer a regular file", code="input_changed")
    return (facts.st_dev, facts.st_ino, facts.st_size, facts.st_mtime_ns, facts.st_ctime_ns)


class Asset:
    """A digest-identified immutable blob, referenced by an opaque handle."""

    __slots__ = (
        "_attempt",
        "_fidelity",
        "_file_state",
        "_grant_token",
        "_image_preparation",
        "_input_id",
        "_label",
        "_local",
        "_max_decoded_bytes",
        "_position",
        "_read_guard",
        "digest",
        "media_type",
        "ref",
        "size_bytes",
    )

    kind: ClassVar[str] = "file"

    def __init__(
        self,
        ref: str,
        *,
        media_type: str = "",
        size_bytes: int = 0,
        digest: str = "",
        local: Path | None = None,
        attempt: str = _UNBOUND,
    ) -> None:
        self.ref = ref
        self.media_type = media_type
        self.size_bytes = size_bytes
        self.digest = digest
        self._local = local
        self._attempt = attempt
        self._input_id = ""
        self._image_preparation: ImagePreparation | None = None
        self._fidelity: Fidelity = "auto"
        self._label = ""
        self._position = -1
        self._read_guard: Callable[[], None] | None = None
        self._max_decoded_bytes = 0
        self._file_state: FileState | None = None
        self._grant_token: object | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.ref!r}, media_type={self.media_type!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Asset) and type(other) is type(self) and other.ref == self.ref

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.ref))

    @property
    def id(self) -> str:
        """This input occurrence's request-local ID; empty before it is bound."""
        return self._input_id

    @property
    def position(self) -> int:
        """The input binding's order; -1 before an occurrence is assigned."""
        return self._position

    @property
    def label(self) -> str:
        """An optional caller-supplied name for this occurrence; empty when omitted."""
        return self._label

    @property
    def fidelity(self) -> Fidelity:
        """Per-occurrence model hint; the generic loader never resizes because of it."""
        return self._fidelity

    def with_fidelity(self, fidelity: Fidelity) -> Self:
        if fidelity not in get_args(Fidelity):
            raise ValueError("asset fidelity must be auto, low, medium or high")
        result = copy(self)
        result._fidelity = fidelity
        return result

    def with_label(self, label: str) -> Self:
        """Name a copy of this handle without changing its bytes or the original."""
        if not isinstance(label, str):
            raise TypeError("an asset label must be a string")
        result = copy(self)
        result._label = label
        return result

    @property
    def hydrated(self) -> bool:
        """True once the worker verified the granted file at its local path."""
        return self._local is not None

    def read_bytes(self) -> bytes:
        """The verified bytes. Refuses before hydration — preflight sees METADATA ONLY."""
        if self._local is None:
            raise PreflightViolation(
                f"asset {self.ref!r} has no bytes here: preflight and describe read counts, "
                "kinds and typed metadata only — bytes exist after hydration (§1.3)",
                code="asset_bytes_unavailable",
            )
        if self._read_guard is not None:
            self._read_guard()
        self._check_file()
        try:
            with self._local.open("rb") as source:
                data = source.read(self.size_bytes + 1) if self._file_state else source.read()
            if self._file_state and len(data) != self.size_bytes:
                raise RuntimeFailure("input file size changed during read", code="input_changed")
            return data
        finally:
            self._check_file()

    def _check_file(self) -> None:
        if self._file_state is not None and (
            self._local is None or file_state(self._local) != self._file_state
        ):
            raise RuntimeFailure(
                "input file changed after verification; keep sources unchanged until completion",
                code="input_changed",
                fields=[self._input_id] if self._input_id else (),
            )

    def row(self) -> dict[str, Any]:
        """The manifest row. Names the fields it emits, so no local path can leak. A
        registered frame nobody has encoded yet (cr-079) has no size or digest to claim,
        and the row says so by omission: the output manifest is where those facts live."""
        row: dict[str, Any] = {"ref": self.ref, "kind": self.kind, "media_type": self.media_type}
        if self.digest:
            row["size_bytes"] = self.size_bytes
            row["digest"] = self.digest
        return row


class ImageAsset(Asset):
    kind: ClassVar[str] = "image"


class AudioAsset(Asset):
    kind: ClassVar[str] = "audio"


class VideoAsset(Asset):
    kind: ClassVar[str] = "video"


class FileAsset(Asset):
    kind: ClassVar[str] = "file"


@dataclass(frozen=True, slots=True)
class AssetInfo:
    """Readonly declared input metadata; accessing it never reads or decodes bytes."""

    id: str
    position: int
    label: str
    fidelity: Fidelity
    kind: str
    media_type: str
    size_bytes: int
    digest: str


class Assets[T](Sequence[T]):
    """Ordered granted inputs, accessed by position or an optional unique label.

    ``Assets[Image]`` yields real PIL images; ``Assets[Mixed]`` yields decoded image,
    video or audio values. Decode is lazy and uses the attempt's shared decoder.
    ``info(key)`` exposes metadata without decoding, including during preflight.
    Explicit ``Assets[ImageAsset]`` and other handle types remain a raw input view.
    Construction holds references for forwarding; value access starts at injection.
    """

    __slots__ = ("_cache", "_decoded", "_guard", "_items", "_project")

    def __init__(self, values: Iterable[Asset] = ()) -> None:
        items: list[Asset] = []
        for position, value in enumerate(values):
            if not isinstance(value, Asset):
                raise TypeError("Assets construction accepts granted asset handles")
            item = copy(value)
            item._input_id = f"assets.{position}.asset"
            item._position = position
            items.append(item)
        self._items = tuple(items)
        self._project: Callable[[Asset], object] | None = None
        self._decoded = True
        self._cache: dict[int, object] = {}
        self._guard: Callable[[], None] | None = None
        self._check_labels()

    @classmethod
    def _from_bound(
        cls,
        values: Iterable[Asset],
        *,
        decoded: bool = False,
        project: Callable[[Asset], object] | None = None,
        guard: Callable[[], None] | None = None,
    ) -> Assets[T]:
        """Retain exact slot IDs/order after binding; no byte acquisition or decode."""
        result = cls()
        items: list[Asset] = []
        seen: set[str] = set()
        for position, value in enumerate(values):
            if not isinstance(value, Asset):
                raise TypeError("Assets accepts typed asset handles")
            if not value.id or value.id in seen or value.position != position:
                raise ValueError("bound Assets requires distinct input IDs in binding order")
            seen.add(value.id)
            items.append(copy(value))
        result._items = tuple(items)
        result._project = project
        result._decoded = decoded
        result._guard = guard
        result._check_labels()
        return result

    def _check_labels(self) -> None:
        labels = [item.label for item in self._items if item.label]
        if len(labels) != len(set(labels)):
            raise ValueError("nonempty asset labels must be unique within the collection")

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return (self[index] for index in range(len(self)))

    def _index(self, key: int | str) -> int:
        if isinstance(key, str):
            if key:
                for index, item in enumerate(self._items):
                    if item.label == key:
                        return index
            raise KeyError(key)
        index = key if key >= 0 else len(self) + key
        if not 0 <= index < len(self):
            raise IndexError(key)
        return index

    @overload
    def __getitem__(self, key: int | str) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> Assets[T]: ...

    def __getitem__(self, key: int | str | slice) -> T | Assets[T]:
        if isinstance(key, slice):
            return self.select(*range(len(self))[key])
        index = self._index(key)
        if not self._decoded:
            return cast(T, self._items[index])
        if self._guard is not None:
            self._guard()
        if self._project is None:
            raise PreflightViolation(
                "Assets values are available after hydration; use info(key) for metadata",
                code="asset_bytes_unavailable",
            )
        if index not in self._cache:
            self._cache[index] = self._project(self._items[index])
        return cast(T, self._cache[index])

    @overload
    def get(self, key: int | str, default: None = None) -> T | None: ...

    @overload
    def get[D](self, key: int | str, default: D) -> T | D: ...

    def get(self, key: int | str, default: object = None) -> object:
        """Read an optional occurrence, returning default only when the key is absent."""
        try:
            index = self._index(key)
        except (IndexError, KeyError):
            return default
        return self[index]

    def info(self, key: int | str) -> AssetInfo:
        item = self._items[self._index(key)]
        return AssetInfo(
            item.id,
            item.position,
            item.label,
            item.fidelity,
            item.kind,
            item.media_type,
            item.size_bytes,
            item.digest,
        )

    def select(self, *keys: int | str) -> Assets[T]:
        """Select/reorder occurrences without decoding or changing their granted bytes."""
        result: Assets[T] = Assets(self._items[self._index(key)] for key in keys)
        result._decoded = self._decoded
        result._project = self._project
        result._guard = self._guard
        return result

    def by_id(self, input_id: str) -> T:
        for index, item in enumerate(self._items):
            if item.id == input_id:
                return self[index]
        raise KeyError(input_id)

    def by_label(self, label: str) -> T:
        return self[label]


def asset_kinds(annotation: object) -> frozenset[str]:
    """Asset kinds reached through containers/unions, stopping at nested records.

    A bound on ``list[ImageAsset]`` belongs to that field. A bound on
    ``list[SomeReference]`` does not: the concrete reference record owns the bound on its
    own asset field. Stopping at records keeps tagged-union reference schemas from growing
    a second, lossy path-based authority over their leaf fields.
    """
    base, _ = strip(annotation)
    base, _ = unwrap_optional(base)
    if isinstance(base, type):
        if base is Tree:
            return frozenset({"tree"})
        if issubclass(base, Asset):
            return frozenset({base.kind})
        return frozenset()
    return frozenset(kind for argument in get_args(base) for kind in asset_kinds(argument))


class Tree:
    """A digest-verified MATERIALIZED input tree — the typed job input (jobs.md §1).

    A model, dataset or checkpoint input is an ordinary typed request FIELD, and the local
    path rides the field's VALUE (`payload.base_model.path`) rather than a context accessor.
    `resume_from: Tree | None` is therefore just another verified input, never a reserved
    payload name (§2).

    The path exists only after the runtime materialized the tree under THIS attempt's grant.
    A field naming a tree the grant does not cover never hydrates, so a job cannot read one
    it was not granted — the capability is the grant, and there is no ambient store handle.
    """

    __slots__ = (
        "_attempt",
        "_grant_token",
        "_member_token",
        "_read_guard",
        "_root",
        "digest",
        "ref",
        "size_bytes",
    )

    kind: ClassVar[str] = "tree"

    def __init__(
        self, ref: str, *, digest: str = "", root: Path | None = None, attempt: str = _UNBOUND
    ) -> None:
        self.ref = ref
        self.digest = digest
        self._root = root
        self._attempt = attempt
        self.size_bytes = 0
        self._read_guard: Callable[[], None] | None = None
        self._grant_token: object | None = None
        self._member_token: object | None = None

    def __repr__(self) -> str:
        return f"Tree({self.ref!r}, hydrated={self.hydrated})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tree) and other.ref == self.ref

    def __hash__(self) -> int:
        return hash(("Tree", self.ref))

    @property
    def hydrated(self) -> bool:
        return self._root is not None

    @property
    def path(self) -> Path:
        """The materialized tree's root. Refuses before hydration — never a silent miss."""
        if self._root is None:
            raise CapabilityError(
                f"input tree {self.ref!r} was never materialized for this attempt: a tree is "
                "granted, verified and materialized before the body runs, and a job cannot "
                "reach one it was not granted",
                code="tree_unhydrated",
            )
        if self._read_guard is not None:
            self._read_guard()
        return self._root

    def files(self) -> tuple[Path, ...]:
        """Every regular file in the tree, sorted. The whole read surface of an input."""
        return tuple(p for p in sorted(self.path.rglob("*")) if p.is_file())

    def member[A: Asset](self, path: str, asset_type: type[A], *, bound: AssetBound) -> A:
        """Project a verified native member as an attempt-scoped typed read capability.

        The explicit bound governs encoded bytes, admitted media and decoded bytes.
        Runtime verifies the path and native membership; use Outputs to copy a member
        into a declared output when returning it.
        """
        from cozy_runtime.author._tree_members import project

        return project(self, path, asset_type, bound=bound)

    def row(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }


def asset_dec_hook(target: type, obj: Any) -> Any:
    """msgspec `dec_hook` for asset and tree request fields: a bare ref STRING, never a
    record.

    Anything else — a mapping carrying `local`, a pre-filled digest — is refused, so the
    request cannot hand itself a hydrated handle or dictate identity-bearing fields.
    """
    if isinstance(target, type) and issubclass(target, (Asset, Tree)):
        if not isinstance(obj, str):
            raise TypeError(
                f"{target.__name__} takes a reference string; "
                f"got {type(obj).__name__} — hydration state is never wire data"
            )
        return target(obj)
    raise NotImplementedError(f"no decoder for {target!r}")


def digest_bytes(data: bytes) -> str:
    return "blake2b:" + hashlib.blake2b(data, digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class InputMetadata:
    """Identity and declared metadata from one input binding, before byte hydration."""

    input_id: str
    media_type: str
    digest: str
    length: int
    order: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class GrantedInput(InputMetadata):
    """One input the worker verified at its granted or materialized local path.

    Everything on it was established BEFORE acceptance: the bytes were read under the
    grant through the one bounded reader, checked against the grant's declared length and
    content digest, and sniffed. The executor never re-fetches and never re-hashes — it
    matches this against the FIELD it is about to fill, which is the half only it knows.
    """

    local: Path
    file_state: FileState | None = None


def bind(
    asset: Asset,
    *,
    local: Path,
    attempt: str,
    media_type: str,
    digest: str,
    length: int,
    max_decoded_bytes: int = 0,
    input_id: str = "",
    order: int = 0,
    read_guard: Callable[[], None] | None = None,
    source_state: FileState | None = None,
) -> None:
    """Hydrate an asset in place. Runtime-only: author code has no path into this.

    The facts come from the caller because the caller is the runtime and the runtime
    already established them under the grant. Re-reading the file here to recompute a
    digest — which is what this did — reads every input twice and produces a SECOND
    identity for one object, so a handler comparing `asset.digest` to the one the
    RecordOwner granted would find two different spellings of the same bytes.
    """
    if asset._attempt not in (_UNBOUND, attempt):
        raise CapabilityError(
            f"asset {asset.ref!r} belongs to attempt {asset._attempt!r}", code="foreign_asset"
        )
    asset._local = local
    asset._file_state = source_state
    asset._attempt = attempt
    asset._input_id = input_id
    asset._position = order
    asset._read_guard = read_guard
    asset.media_type = media_type
    asset.size_bytes = length
    asset.digest = digest
    asset._max_decoded_bytes = max_decoded_bytes


def bind_tree(tree: Tree, *, root: Path, digest: str, attempt: str) -> None:
    """Materialize an input tree in place. Runtime-only, like `bind`."""
    if tree._attempt not in (_UNBOUND, attempt):
        raise CapabilityError(
            f"tree {tree.ref!r} belongs to attempt {tree._attempt!r}", code="foreign_tree"
        )
    tree._root = root
    tree.digest = digest
    tree._attempt = attempt
