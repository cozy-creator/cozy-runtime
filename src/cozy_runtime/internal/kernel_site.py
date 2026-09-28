"""The image kernel site executors before Runtime 0.18.70 read, filled from the machine store.

A package's executor runs the cozy-runtime its package locked. One before 0.18.70 compiles
nothing: at entry it places `ROOT/site` on its import path when `ROOT/manifest.json` names its
own torch, CUDA and Python, and at each selection it imports an attention kernel from there by
its installed distribution, asking again while one is absent. So the worker writes the
manifest at boot, before any executor starts (`open_site`), and links each tree the machine
compiles into the site as it becomes ready (`publish`): such an executor serves it from its
next selection, its next construction, with no relock. Newer executors read the store itself.

    ROOT/manifest.json   {"site": str, "torch": str, "cuda": str, "python_abi": str,
                          "kernels": {artifact: {"key": digest}}}
    ROOT/site/<name>     a link to one top-level entry of a ready store tree
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path

from cozy_runtime.internal import kernel_cache, kernel_sources

ROOT = Path("/opt/cozy/kernels")
MANIFEST = "manifest.json"


def open_site(root: Path = ROOT) -> str:
    """The manifest for this environment's torch and an empty site; empty, or why not."""
    release, cuda = kernel_sources.torch_build()
    if not release or not cuda:
        return "this environment has no CUDA torch"
    try:
        (root / "site").mkdir(mode=0o755, parents=True, exist_ok=True)
        manifest = _manifest(root)
        if manifest.get("site") not in (None, str(root / "site")):
            return f"{root / MANIFEST} names another site: {manifest['site']}"
        python = f"cp{sys.version_info.major}{sys.version_info.minor}"
        facts = {"site": str(root / "site"), "torch": release, "cuda": cuda, "python_abi": python}
        _write(root, {"kernels": {}, **manifest, **facts})
    except OSError as exc:
        return f"{root} is not writable here: {exc}"
    return ""


def publish(entry: Path, key: kernel_cache.Key, root: Path = ROOT) -> None:
    """Link every top-level entry of a ready tree into the site. Each link appears whole (a
    rename), so an executor never imports half of one."""
    tree = entry / kernel_cache.SITE
    items = list(tree.iterdir())
    # The CuTe DSL's packages are placed by a .pth, which an executor already running never
    # reads again: link them beside it.
    dsl = tree / "nvidia_cutlass_dsl" / "dsl_packages"
    if dsl.is_dir():
        items += list(dsl.iterdir())
    site = root / "site"
    for item in items:
        link = site / item.name
        if link.is_symlink() and os.readlink(link) == str(item):
            continue
        if link.exists() and not link.is_symlink():
            continue  # a tree the image itself carries
        scratch = site / f".{item.name}.{secrets.token_hex(4)}"
        os.symlink(item, scratch)
        os.replace(scratch, link)
    manifest = _manifest(root)
    kernels = manifest.get("kernels")
    published = dict(kernels) if isinstance(kernels, dict) else {}
    _write(root, {**manifest, "kernels": {**published, key.kernel: {"key": key.digest}}})


def _manifest(root: Path) -> dict[str, object]:
    try:
        found = json.loads((root / MANIFEST).read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _write(root: Path, manifest: dict[str, object]) -> None:
    scratch = root / f".{MANIFEST}.{secrets.token_hex(4)}"
    scratch.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    scratch.chmod(0o644)
    os.replace(scratch, root / MANIFEST)
