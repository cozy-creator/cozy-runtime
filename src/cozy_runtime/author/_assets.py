"""Typed assets — the ONE asset type, both directions (§1.3/§3.4).

Input assets are ordinary typed REQUEST FIELDS that arrive digest-verified; output assets
are what `out.save_*` returns and what the result struct declares. There is no separate
Inputs capability, no second asset type, and no author-managed fetch.

`Asset` is deliberately NOT a msgspec-native type. The only way one comes into existence
from the wire is `asset_dec_hook`, which accepts a REF STRING and nothing else — so
hydration state is unforgeable by request data rather than merely undocumented, and local
paths cannot enter a schema, a digest, or serialized output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from cozy_runtime.author._errors import CapabilityError, PreflightViolation

_UNBOUND: Final = ""


class Asset:
    """A digest-identified immutable blob, referenced by an opaque handle."""

    __slots__ = ("_attempt", "_local", "digest", "media_type", "ref", "size_bytes")

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.ref!r}, media_type={self.media_type!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Asset) and type(other) is type(self) and other.ref == self.ref

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.ref))

    @property
    def hydrated(self) -> bool:
        """True once the supervisor materialized the bytes into the attempt spool."""
        return self._local is not None

    def read_bytes(self) -> bytes:
        """The verified bytes. Refuses before hydration — preflight sees METADATA ONLY."""
        if self._local is None:
            raise PreflightViolation(
                f"asset {self.ref!r} has no bytes here: preflight and describe read counts, "
                "kinds and typed metadata only — bytes exist after hydration (§1.3)",
                code="asset_bytes_unavailable",
            )
        return self._local.read_bytes()

    def row(self) -> dict[str, Any]:
        """The manifest row. Names the fields it emits, so no local path can leak."""
        return {
            "ref": self.ref,
            "kind": self.kind,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
        }


class ImageAsset(Asset):
    kind: ClassVar[str] = "image"


class AudioAsset(Asset):
    kind: ClassVar[str] = "audio"


class VideoAsset(Asset):
    kind: ClassVar[str] = "video"


class FileAsset(Asset):
    kind: ClassVar[str] = "file"


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

    __slots__ = ("_attempt", "_root", "digest", "ref")

    kind: ClassVar[str] = "tree"

    def __init__(
        self, ref: str, *, digest: str = "", root: Path | None = None, attempt: str = _UNBOUND
    ) -> None:
        self.ref = ref
        self.digest = digest
        self._root = root
        self._attempt = attempt

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
        return self._root

    def files(self) -> tuple[Path, ...]:
        """Every regular file in the tree, sorted. The whole read surface of an input."""
        return tuple(p for p in sorted(self.path.rglob("*")) if p.is_file())

    def row(self) -> dict[str, Any]:
        return {"ref": self.ref, "kind": self.kind, "digest": self.digest}


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
class GrantedInput:
    """ONE input the supervisor already fetched, verified and spooled (cr-012).

    Everything on it was established BEFORE acceptance: the bytes were read under the
    grant through the one bounded reader, checked against the grant's declared length and
    content digest, and sniffed. The executor never re-fetches and never re-hashes — it
    matches this against the FIELD it is about to fill, which is the half only it knows.
    """

    input_id: str
    """The payload field PATH, list index included — worker-protocol/01's stable identity.
    Inputs are IDENTIFIED, ORDERED subjects, never a digest-sorted anonymous list."""
    local: Path
    media_type: str
    """What the BYTES are, from the sniff — not what the caller declared them to be."""
    digest: str
    length: int
    order: int = 0


def bind(
    asset: Asset, *, local: Path, attempt: str, media_type: str, digest: str, length: int
) -> None:
    """Hydrate an asset in place. Runtime-only: author code has no path into this.

    The facts come from the caller because the caller is the runtime and the runtime
    already established them under the grant. Re-reading the file here to recompute a
    digest — which is what this did — reads every input twice and produces a SECOND
    identity for one object, so a handler comparing `asset.digest` to the one the
    coordinator granted would find two different spellings of the same bytes.
    """
    if asset._attempt not in (_UNBOUND, attempt):
        raise CapabilityError(
            f"asset {asset.ref!r} belongs to attempt {asset._attempt!r}", code="foreign_asset"
        )
    asset._local = local
    asset._attempt = attempt
    asset.media_type = media_type
    asset.size_bytes = length
    asset.digest = digest


def bind_tree(tree: Tree, *, root: Path, digest: str, attempt: str) -> None:
    """Materialize an input tree in place. Runtime-only, like `bind`."""
    if tree._attempt not in (_UNBOUND, attempt):
        raise CapabilityError(
            f"tree {tree.ref!r} belongs to attempt {tree._attempt!r}", code="foreign_tree"
        )
    tree._root = root
    tree.digest = digest
    tree._attempt = attempt
