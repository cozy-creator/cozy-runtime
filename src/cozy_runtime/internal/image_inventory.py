"""The placed worker image's pinned inventory (`tensorhub.image_inventory/1`).

The pinned requirements list a base worker image is built from (xs-009) is the truth about
what the image provides. Tensorhub stages that document as `image-inventory.json` beside the
package manifest (th-112), and pod preparation builds the package venv by NAME against it:
an inventory-owned distribution is served by the image's site-packages copy, never by a lock
wheel (cr-070). The live base observation is demoted to a cross-check — a disagreement
falsifies the inventory rather than silently rewiring the pod.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import msgspec
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

INVENTORY_FORMAT = "tensorhub.image_inventory/1"
INVENTORY_FILENAME = "image-inventory.json"


class ImageInventoryRefusal(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Inventory:
    profile: str
    python: str
    distributions: tuple[tuple[str, str], ...]
    interpreters: tuple[tuple[str, str], ...] = ()
    provisionable_minors: tuple[str, ...] = ()

    @property
    def versions(self) -> dict[str, str]:
        return dict(self.distributions)


class _Distribution(msgspec.Struct, frozen=True):
    name: str
    version: str


class _Interpreter(msgspec.Struct, frozen=True):
    version: str
    abi: str


_Minor = Annotated[str, msgspec.Meta(pattern=r"^3\.[0-9]+$")]


class _Document(msgspec.Struct, frozen=True):
    format: str
    profile: Annotated[str, msgspec.Meta(min_length=1)]
    python: str
    distributions: Annotated[tuple[_Distribution, ...], msgspec.Meta(min_length=1)]
    interpreters: Annotated[tuple[_Interpreter, ...], msgspec.Meta(max_length=3)] = ()
    provisionable_minors: Annotated[tuple[_Minor, ...], msgspec.Meta(max_length=3)] = ()


def _version(spelled: str, what: str) -> Version:
    try:
        return Version(spelled)
    except InvalidVersion as exc:
        raise ImageInventoryRefusal(f"{what} {spelled!r}") from exc


def read_bytes(raw: bytes) -> Inventory:
    """Read required inventory facts while tolerating additive producer metadata."""

    try:
        document = msgspec.json.decode(raw, type=_Document)
    except msgspec.DecodeError as exc:
        raise ImageInventoryRefusal(f"invalid inventory: {exc}") from exc
    if document.format != INVENTORY_FORMAT:
        raise ImageInventoryRefusal(f"format {document.format!r}")
    _version(document.python, "python")
    prior = ""
    for row in document.distributions:
        if canonicalize_name(row.name) != row.name or row.name <= prior:
            raise ImageInventoryRefusal(f"distribution {row.name!r}")
        _version(row.version, f"{row.name} version")
        prior = row.name
    interpreters = []
    for interpreter in document.interpreters:
        version = _version(interpreter.version, "interpreter version")
        if len(version.release) != 3 or interpreter.abi != "cp" + "".join(
            map(str, version.release[:2])
        ):
            raise ImageInventoryRefusal("interpreter version and ABI disagree")
        interpreters.append((str(version), interpreter.abi))
    if interpreters != sorted(set(interpreters), key=lambda row: Version(row[0])):
        raise ImageInventoryRefusal("interpreters must be sorted and unique")
    minors = document.provisionable_minors
    if list(minors) != sorted(set(minors), key=Version):
        raise ImageInventoryRefusal("provisionable Python minors must be sorted and unique")
    return Inventory(
        profile=document.profile,
        python=document.python,
        distributions=tuple((row.name, row.version) for row in document.distributions),
        interpreters=tuple(interpreters),
        provisionable_minors=minors,
    )


def disagreement(inventory: Inventory, observed_versions: dict[str, str]) -> str | None:
    """The cross-check that demoted the observation: every inventory row must be observed.

    Returns the first disagreement as `distribution=NAME inventory=X image=Y|absent`, or
    None when the image actually provides what its inventory promises.
    """

    for name, version in inventory.distributions:
        observed = observed_versions.get(name)
        if observed is None:
            return f"distribution={name} inventory={version} image=absent"
        try:
            agreed = Version(observed) == Version(version)
        except InvalidVersion:
            agreed = observed == version
        if not agreed:
            return f"distribution={name} inventory={version} image={observed}"
    return None
