"""Model slots this machine resolves itself: the caller's explicit choice, else the owner's
binding, else the callee's authored default. The rung is chosen for this machine's devices
before anything is read, so a slot costs one public catalog resolution of that rung's lane.
Nothing here is cached: a rebinding is seen by the next selection."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from cozy_runtime.author import ConformanceError
from cozy_runtime.author._model_defaults import default_ladder, reference
from cozy_runtime.internal import egress, fill
from cozy_runtime.protocol import documents

from .workspace import WorkspaceRefusal

MAX_METADATA = 1 << 20
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

type Fits = Callable[[Mapping[str, Any]], bool]


class Catalog:
    """Anonymous public reads at one catalog origin; a package's bindings are read once."""

    def __init__(self, origin: str) -> None:
        parts = urlsplit(origin)
        self.origin, self.scheme = origin, parts.scheme
        self.host, self.port = parts.hostname or "", parts.port
        self.bindings: dict[str, dict[str, Any]] = {}

    def get(self, path: str) -> Any:
        try:
            status, raw = egress.request_catalog_metadata(
                self.host, self.port, self.scheme == "https", path, limit=MAX_METADATA
            )
        except egress.EgressRefusal as exc:
            raise WorkspaceRefusal(f"model catalog read failed: {exc.code}") from exc
        if status == 404:
            return None
        if status != 200:
            raise WorkspaceRefusal(f"model catalog answered {status} for {path[:200]}")
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise WorkspaceRefusal("model catalog answered invalid JSON") from exc

    def binding(self, package: str, slot: str) -> dict[str, Any] | None:
        """The owner's current binding of one slot, or None."""
        if package not in self.bindings:
            org, _, name = package.partition("/")
            value = self.get(f"/v1/packages/{quote(org)}/{quote(name)}/bindings") or {}
            rows = value.get("bindings", []) if isinstance(value, dict) else None
            if not isinstance(rows, list):
                raise WorkspaceRefusal("model catalog answered invalid bindings")
            self.bindings[package] = {
                row["slot"]: row for row in rows if isinstance(row, dict) and "slot" in row
            }
        return self.bindings[package].get(slot)

    def resolve(self, model: str, release: str, lane: str) -> dict[str, Any]:
        """One lane's exact checkpoint: its manifest digest and length."""
        spec = model + ("@" + release if release else "")
        query = {"ref": spec, **({"lane": lane} if lane else {})}
        value = self.get("/v1/models/resolve?" + urlencode(query))
        if not isinstance(value, dict) or value.get("model") != model:
            raise WorkspaceRefusal(f"model catalog cannot resolve {spec}/{lane}")
        digest, length = value.get("manifest_id"), value.get("manifest_length")
        try:
            documents.raw(str(digest))
        except documents.DocumentError as exc:
            raise WorkspaceRefusal(f"model catalog returned no manifest for {spec}/{lane}") from exc
        if type(length) is not int or length <= 0:
            length = self.measure(model, str(digest))
        return {
            "manifest": {"digest": digest, "length": length},
            "release": str(value.get("release") or release),
            "lane": str(value.get("lane") or lane),
        }

    def measure(self, model: str, digest: str) -> int:
        """An unstated manifest length, measured from the exact bytes its digest names."""
        org, _, name = model.partition("/")
        path = f"/v1/models/{quote(org)}/{quote(name)}/checkpoints/{digest}"
        try:
            status, length, sha256 = egress.measure_catalog_document(
                self.host,
                self.port,
                self.scheme == "https",
                path,
                limit=fill.tensorfs_module().manifest_max_bytes(),
            )
        except egress.EgressRefusal as exc:
            raise WorkspaceRefusal(f"model catalog manifest read failed: {exc.code}") from exc
        if status != 200:
            raise WorkspaceRefusal(f"model catalog answered {status} for manifest {digest}")
        if sha256 != documents.raw(digest):
            raise WorkspaceRefusal(f"model catalog manifest bytes do not match {digest}")
        return length


def _ladder(
    catalog: Catalog, package: str, declaration: Mapping[str, Any]
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """The owner's binding, else the authored default: (model, release, rungs)."""
    if package and not package.startswith("local/"):
        row = catalog.binding(package, str(declaration["path"]))
        if row is not None:
            rungs = row.get("ladder")
            if not isinstance(rungs, list) or not rungs:
                raise WorkspaceRefusal(f"the owner's binding of {declaration['path']} has no lanes")
            return str(row.get("model", "")), str(row.get("release", "")), rungs
    authored = declaration.get("default_ladder")
    if not authored:
        return None
    try:
        rows = default_ladder(authored)
        model, release, _ = reference(rows[0][2])
    except ConformanceError as exc:
        raise WorkspaceRefusal(f"{declaration['path']} has an invalid authored default") from exc
    if "/" not in model:
        org = package.partition("/")[0]
        if not org or org == "local":
            raise WorkspaceRefusal(f"{declaration['path']} names its owner's model with no owner")
        model = org + "/" + model
    return (
        model,
        release,
        [{"gpu": gpu, "gpus": gpus, "lane": reference(lane)[2]} for gpu, gpus, lane in rows],
    )


def slot(
    catalog: Catalog,
    package: str,
    declaration: Mapping[str, Any],
    parameter: str,
    choice: Mapping[str, Any] | None,
    fits: Fits,
) -> dict[str, Any]:
    """One slot's exact checkpoint for this machine, in the captured default row shape."""
    base = {"parameter": parameter, "public_origin": catalog.origin}
    choice = choice or {}
    manifest = choice.get("manifest") or {}
    if manifest.get("digest"):
        digest, repository = str(manifest["digest"]), str(choice.get("repository", ""))
        documents.raw(digest)
        pinned: dict[str, Any] = {"digest": digest, "length": int(manifest.get("length", 0))}
        if pinned["length"] <= 0:
            pinned = catalog.resolve(repository, digest, "")["manifest"]
        return {
            **base,
            "gpu": "*",
            "gpus": 0,
            "repository": repository,
            "manifest": pinned,
            "release": choice.get("release", ""),
            "lane": choice.get("lane", ""),
        }
    ladder = _ladder(catalog, package, declaration)
    if choice.get("repository"):
        rungs = (
            [{"gpu": "*", "gpus": 0, "lane": choice["lane"]}]
            if choice.get("lane")
            else (ladder[2] if ladder else [])
        )
        if not rungs:
            raise WorkspaceRefusal(f"choose a lane for {parameter}: its slot declares none")
        ladder = str(choice["repository"]), str(choice.get("release", "")), rungs
    if ladder is None:
        raise WorkspaceRefusal(f"{parameter} has no owner binding or authored default")
    model, release, rungs = ladder
    if _REPOSITORY.fullmatch(model) is None or model.startswith("local/"):
        raise WorkspaceRefusal(f"{parameter} names no public catalog repository")
    degrees = set(declaration.get("sequence_parallel", {}).get("degrees", []))
    fitting = [
        rung
        for rung in rungs
        if isinstance(rung, dict)
        and int(rung.get("gpus", 0) or 0) in {0, 1, *degrees}
        and fits(rung)
    ]
    if not fitting:
        raise WorkspaceRefusal(f"{parameter} has no rung for this accelerator")
    # The widest rung this machine holds: the width itself is the declared degree.
    rung = max(fitting, key=lambda candidate: int(candidate.get("gpus", 0) or 0))
    resolved = catalog.resolve(model, release, str(rung["lane"]))
    return {
        **base,
        "gpu": str(rung.get("gpu", "*")),
        "gpus": int(rung.get("gpus", 0) or 0),
        "repository": model,
        **resolved,
    }
