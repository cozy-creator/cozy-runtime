"""The callee sources a project's `uv.lock` pins, for the static reader (cr-114).

A caller's signature may use a callee package's type, and the callee is read as source, never
run. A local package is read from the directory the lock names. A published one is read from
the exact wheel the lock pins: fetched through the egress boundary by its URL and sha256, and
unpacked as files. Nothing is built, installed or imported (#713).
"""

from __future__ import annotations

import re
import shutil
import tempfile
import tomllib
import weakref
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from cozy_runtime.internal import egress
from cozy_runtime.internal.package_environment import EnvironmentRefusal

#: A callee wheel is a package's source and its assets, never a framework.
WHEEL_CEILING = 64 << 20
UNPACKED_CEILING = 256 << 20
LOCK_CEILING = 16 << 20
_APPLICATION_GROUP = "[cozy.application]"


class LockedSourceRefusal(Exception):
    """A pinned callee wheel that could not be read as the lock pins it."""


@dataclass(frozen=True, slots=True)
class LockedWheel:
    distribution: str
    url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Locked:
    #: Local packages' source trees: `<tree>/src` and `<tree>`.
    roots: tuple[Path, ...] = ()
    #: Published packages' pure wheels, by the module name their distribution name spells.
    wheels: Mapping[str, LockedWheel] = field(default_factory=dict)


def read(project: Path) -> Locked:
    """What `project/uv.lock` pins beside the project: its `directory` and `editable`
    sources, and each registry package's pure (`none-any`) wheel. Reading TOML only."""
    lock = project / "uv.lock"
    try:
        if not lock.is_file() or lock.stat().st_size > LOCK_CEILING:
            return Locked()
        packages = tomllib.loads(lock.read_text()).get("package", [])
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return Locked()
    roots: list[Path] = []
    wheels: dict[str, LockedWheel] = {}
    for row in packages if isinstance(packages, list) else ():
        if not isinstance(row, dict) or not isinstance(source := row.get("source"), dict):
            continue
        where = source.get("directory") or source.get("editable")
        if isinstance(where, str):
            tree = (project / where).resolve()
            if tree != project and tree.is_dir():
                roots += [tree / "src", tree]
        elif "registry" in source and isinstance(name := row.get("name"), str):
            pure = [w for w in row.get("wheels", ()) if isinstance(w, dict) and _pure(w)]
            if len(pure) == 1 and str(pure[0].get("hash", "")).startswith("sha256:"):
                module = re.sub(r"[-_.]+", "_", name).lower()
                wheels[module] = LockedWheel(name, str(pure[0]["url"]), str(pure[0]["hash"]))
    return Locked(tuple(dict.fromkeys(r for r in roots if r.is_dir())), wheels)


def _pure(wheel: dict[str, object]) -> bool:
    filename = unquote(PurePosixPath(urlsplit(str(wheel.get("url", ""))).path).name)
    tags = filename.removesuffix(".whl").split("-")
    return filename.endswith(".whl") and len(tags) >= 5 and tags[-2:] == ["none", "any"]


class Wheels:
    """The pinned wheels, each fetched and unpacked once, on the first module that needs it,
    into a directory removed with this object."""

    def __init__(self, wheels: Mapping[str, LockedWheel]) -> None:
        self.pending = dict(wheels)
        self.directory = Path(tempfile.mkdtemp(prefix="cozy-locked-wheels-"))
        weakref.finalize(self, shutil.rmtree, self.directory, True)

    def root(self, module: str) -> Path | None:
        """The unpacked wheel pinned for top-level `module`, if it is a cozy package's."""
        wheel = self.pending.pop(module, None)
        if wheel is None:
            return None
        target = self.directory / wheel.sha256.removeprefix("sha256:")
        archive = target.with_suffix(".whl")
        try:
            egress.fetch_public_wheel(archive, wheel.url, wheel.sha256, WHEEL_CEILING)
        except EnvironmentRefusal as exc:
            raise LockedSourceRefusal(
                f"{wheel.distribution}: the wheel its uv.lock pins ({wheel.url}) could not be "
                f"read as pinned: {exc}"
            ) from exc
        with zipfile.ZipFile(archive) as unpacked:
            members = [m for m in unpacked.infolist() if not m.is_dir()]
            if sum(m.file_size for m in members) > UNPACKED_CEILING:
                raise LockedSourceRefusal(f"{wheel.distribution}: its wheel unpacks too large")
            names = [PurePosixPath(m.filename) for m in members]
            if any(n.is_absolute() or ".." in n.parts for n in names):
                raise LockedSourceRefusal(f"{wheel.distribution}: its wheel names unsafe paths")
            entry_points = [
                m
                for m, n in zip(members, names, strict=True)
                if len(n.parts) == 2
                and n.parts[0].endswith(".dist-info")
                and n.name == "entry_points.txt"
            ]
            if not any(_APPLICATION_GROUP in unpacked.read(m).decode() for m in entry_points):
                return None  # not a cozy package: its types are outside the vocabulary anyway
            unpacked.extractall(target, members)
        return target
