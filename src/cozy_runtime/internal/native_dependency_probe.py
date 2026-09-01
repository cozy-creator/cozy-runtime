"""Banked loader/import operator for an admitted native dependency-wheel set."""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

MAX_REQUEST_BYTES = 1 << 20
MAX_MEMBERS = 4096
MAX_ROOTS = 129
_ROOT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def probe(raw: str) -> dict[str, Any]:
    """Load every native member, then import every wheel-owned Python root.

    The request arrives as this fixture's one argument. It is not trusted for being passed:
    every member is re-checked to be a relative path inside the admitted environment before
    the loader is asked to open it.
    """

    if not raw or len(raw.encode()) > MAX_REQUEST_BYTES:
        raise ValueError("native dependency proof request is absent or oversized")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"members", "roots", "site_packages"}:
        raise ValueError("native dependency proof request fields are not closed")
    members = value["members"]
    roots = value["roots"]
    site_raw = value["site_packages"]
    if (
        not isinstance(members, list)
        or not 1 <= len(members) <= MAX_MEMBERS
        or members != sorted(set(members))
        or not all(isinstance(item, str) for item in members)
        or not isinstance(roots, list)
        or not 1 <= len(roots) <= MAX_ROOTS
        or roots != sorted(set(roots))
        or not all(isinstance(item, str) and _ROOT.fullmatch(item) for item in roots)
        or not isinstance(site_raw, str)
    ):
        raise ValueError("native dependency proof request values are invalid")
    site = Path(site_raw).resolve(strict=True)
    if not site.is_absolute() or not site.is_dir():
        raise ValueError("native dependency site-packages is invalid")
    for member in members:
        relative = PurePosixPath(member)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != member:
            raise ValueError(f"native dependency member is invalid: {member}")
        path = (site / member).resolve(strict=True)
        if not path.is_relative_to(site) or not path.is_file():
            raise ValueError(f"native dependency member escapes its environment: {member}")
        ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL | os.RTLD_NOW)
    importlib.invalidate_caches()
    for root in roots:
        module = importlib.import_module(root)
        candidates = []
        filename = getattr(module, "__file__", None)
        if isinstance(filename, str):
            candidates.append(Path(filename).resolve(strict=True))
        locations = getattr(getattr(module, "__spec__", None), "submodule_search_locations", None)
        if locations is not None:
            candidates.extend(Path(item).resolve(strict=True) for item in locations)
        if not candidates or not any(path.is_relative_to(site) for path in candidates):
            raise ImportError(f"{root} was not imported from the admitted package environment")
    return {"members": members, "roots": roots}
