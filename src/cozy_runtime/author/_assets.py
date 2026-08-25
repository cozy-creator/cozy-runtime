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


def asset_dec_hook(target: type, obj: Any) -> Any:
    """msgspec `dec_hook` for asset request fields: a bare ref STRING, never a record.

    Anything else — a mapping carrying `local`, a pre-filled digest — is refused, so the
    request cannot hand itself a hydrated handle or dictate identity-bearing fields.
    """
    if isinstance(target, type) and issubclass(target, Asset):
        if not isinstance(obj, str):
            raise TypeError(
                f"{target.__name__} takes a reference string; "
                f"got {type(obj).__name__} — hydration state is never wire data"
            )
        return target(obj)
    raise NotImplementedError(f"no decoder for {target!r}")


def digest_bytes(data: bytes) -> str:
    return "blake2b:" + hashlib.blake2b(data, digest_size=16).hexdigest()


def bind(asset: Asset, *, local: Path, attempt: str) -> None:
    """Hydrate an asset in place. Runtime-only: author code has no path into this."""
    if asset._attempt not in (_UNBOUND, attempt):
        raise CapabilityError(
            f"asset {asset.ref!r} belongs to attempt {asset._attempt!r}", code="foreign_asset"
        )
    asset._local = local
    asset._attempt = attempt
    data = local.read_bytes()
    asset.size_bytes = len(data)
    asset.digest = digest_bytes(data)
