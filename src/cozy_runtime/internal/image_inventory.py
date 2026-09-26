"""The placed worker image's pinned inventory (`tensorhub.image_inventory/1`).

The pinned requirements list a base worker image is built from (xs-009) is the truth about
what the image provides. Tensorhub stages that document as `image-inventory.json` beside the
package manifest (th-112), and pod preparation builds the package venv by NAME against it:
an inventory-owned distribution is served by the image's site-packages copy, never by a lock
wheel (cr-070). The live base observation is demoted to a cross-check — a disagreement
falsifies the inventory rather than silently rewiring the pod.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

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


def read_bytes(raw: bytes) -> Inventory:
    """Read required inventory facts while tolerating additive producer metadata."""

    try:
        value: Any = json.loads(raw)
    except ValueError as exc:
        raise ImageInventoryRefusal(f"not JSON: {exc}") from exc
    if not isinstance(value, dict) or not {
        "distributions",
        "format",
        "profile",
        "python",
    }.issubset(value):
        raise ImageInventoryRefusal("wrong fields")
    if value["format"] != INVENTORY_FORMAT:
        raise ImageInventoryRefusal(f"format {value['format']!r}")
    profile = value["profile"]
    python = value["python"]
    rows = value["distributions"]
    if not isinstance(profile, str) or not profile or not isinstance(rows, list) or not rows:
        raise ImageInventoryRefusal("profile or distributions absent")
    try:
        Version(str(python))
    except InvalidVersion as exc:
        raise ImageInventoryRefusal(f"python {python!r}") from exc
    distributions: list[tuple[str, str]] = []
    prior = ""
    for row in rows:
        if not isinstance(row, dict) or not {"name", "version"}.issubset(row):
            raise ImageInventoryRefusal("distribution row fields")
        name = row["name"]
        version = row["version"]
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or canonicalize_name(name) != name
            or name <= prior
        ):
            raise ImageInventoryRefusal(f"distribution {name!r}")
        try:
            Version(version)
        except InvalidVersion as exc:
            raise ImageInventoryRefusal(f"{name} version {version!r}") from exc
        prior = name
        distributions.append((name, version))
    return Inventory(
        profile=profile,
        python=str(python),
        distributions=tuple(distributions),
        interpreters=_interpreters(value.get("interpreters", [])),
        provisionable_minors=_provisionable_minors(value.get("provisionable_minors", [])),
    )


def from_wire(profile: str, python: str, rows: Any, interpreters: Any = ()) -> Inventory:
    """Admit the request-carried inventory rows (wire 30); same rules as the staged document."""

    if not isinstance(profile, str) or not profile:
        raise ImageInventoryRefusal("profile absent")
    try:
        Version(str(python))
    except InvalidVersion as exc:
        raise ImageInventoryRefusal(f"python {python!r}") from exc
    distributions: list[tuple[str, str]] = []
    prior = ""
    for row in rows:
        name = str(getattr(row, "distribution", ""))
        version = str(getattr(row, "version", ""))
        if canonicalize_name(name) != name or name <= prior:
            raise ImageInventoryRefusal(f"distribution {name!r}")
        try:
            Version(version)
        except InvalidVersion as exc:
            raise ImageInventoryRefusal(f"{name} version {version!r}") from exc
        prior = name
        distributions.append((name, version))
    return Inventory(
        profile=profile,
        python=str(python),
        distributions=tuple(distributions),
        interpreters=_interpreters(
            [{"version": row.version, "abi": row.abi} for row in interpreters]
        ),
    )


def _interpreters(rows: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(rows, list) or len(rows) > 3:
        raise ImageInventoryRefusal("interpreter inventory must contain at most three versions")
    result = []
    for row in rows:
        if not isinstance(row, dict) or not {"version", "abi"}.issubset(row):
            raise ImageInventoryRefusal("interpreter inventory row fields")
        try:
            version = Version(row["version"])
        except (InvalidVersion, TypeError) as exc:
            raise ImageInventoryRefusal("interpreter version is invalid") from exc
        if len(version.release) != 3 or row["abi"] != "cp" + "".join(map(str, version.release[:2])):
            raise ImageInventoryRefusal("interpreter version and ABI disagree")
        result.append((str(version), row["abi"]))
    if result != sorted(set(result), key=lambda row: Version(row[0])):
        raise ImageInventoryRefusal("interpreters must be sorted and unique")
    return tuple(result)


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


def _provisionable_minors(rows: Any) -> tuple[str, ...]:
    if (
        not isinstance(rows, list)
        or len(rows) > 3
        or any(not isinstance(row, str) or not re.fullmatch(r"3\.[0-9]+", row) for row in rows)
    ):
        raise ImageInventoryRefusal("invalid provisionable Python minors")
    if rows != sorted(set(rows), key=Version):
        raise ImageInventoryRefusal("provisionable Python minors must be sorted and unique")
    return tuple(rows)
