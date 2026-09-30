"""The construction census, computed once per content and cached beside the prepare cache.

A slot's census (the components its factory constructs and every tensor's name, dtype and
shape) plus its fit against one CozyTensors header is a pure function of the installed
environment, the application, the slot path and the header. It is derived once, in one
`derive_child` process for every missing slot, and then read from disk on every later prepare.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime.internal import canonical, derive_child, fill

FORMAT = "cozy.runtime.ConstructionCensus/1"


class CensusRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class SlotInput:
    """One serving slot bound to one checkpoint. `assets` is read only on a cache miss."""

    slot: str
    config: bytes
    tensor_dtypes: Mapping[str, str]
    header: bytes
    encoded_leaves: bool
    assets: Callable[[], Mapping[str, bytes]]
    adapters: bytes = b""


@dataclass(frozen=True, slots=True)
class Census:
    components: tuple[str, ...]
    rows: tuple[derive_child.RequirementRow, ...]
    fit_ok: bool
    fit_detail: str


def key(environment: str, application: str, row: SlotInput) -> str:
    """The header digest covers the config, dtypes and asset identities it declares."""

    return hashlib.sha256(
        canonical.write(
            {
                "application": application,
                "config": hashlib.sha256(row.config).hexdigest(),
                "dtypes": dict(sorted(row.tensor_dtypes.items())),
                "encoded_leaves": row.encoded_leaves,
                "env": environment,
                "format": FORMAT,
                "header": hashlib.sha256(row.header).hexdigest(),
                "slot": row.slot,
            }
        )
    ).hexdigest()


def ensure(
    root: Path,
    *,
    python: Path,
    environment: str,
    application: str,
    slots: Sequence[SlotInput],
) -> dict[str, Census]:
    """Every slot's census: cached rows as-is, all misses in ONE derive child."""

    if not application:
        raise CensusRefusal("package_prepare_interface_invalid", "application")
    found: dict[str, Census] = {}
    missing: list[tuple[str, SlotInput]] = []
    for row in slots:
        name = key(environment, application, row)
        held = _read(root / f"{name}.json")
        if held is None:
            missing.append((name, row))
        else:
            found[row.slot] = held
    if not missing:
        return found
    missing.sort(key=lambda item: item[1].slot)
    try:
        answers = derive_child.derive_in(
            python,
            derive_child.DeriveRequest(
                tuple(
                    derive_child.SlotRequest(
                        row.slot,
                        row.config,
                        dict(row.assets()),
                        dict(row.tensor_dtypes),
                        row.adapters,
                    )
                    for _name, row in missing
                ),
                application=application,
            ),
        )
    except derive_child.DeriveRefusal as exc:
        slots_named = ", ".join(row.slot for _name, row in missing)
        raise CensusRefusal(
            "package_prepare_derive_refused", f"{slots_named}: {exc.code}: {exc.detail}"
        ) from exc
    tensorfs = fill.tensorfs_module()
    for (name, row), answer in zip(missing, answers, strict=True):
        if isinstance(answer, derive_child.SourceSlot):
            raise CensusRefusal(
                "package_prepare_derive_refused",
                f"{row.slot}: derive_slot_not_serving: a job source, not a serving model",
            )
        requirements = [
            tensorfs.TensorRequirement(
                component=component,
                key=tensor,
                shape=list(shape),
                logical_dtype=fill.tensorfs_requirement_dtype(dtype),
            )
            for component, tensor, dtype, shape in answer.rows
        ]
        try:
            verdict = dict(
                tensorfs.fit(
                    requirements,
                    row.header,
                    custody="canonical",
                    encoded_leaves=row.encoded_leaves,
                )
            )
        except tensorfs.errors.Refusal as exc:
            raise CensusRefusal(
                "package_prepare_model_header_invalid", f"{row.slot}: {exc.code}: {exc}"
            ) from exc
        census = Census(
            components=answer.components,
            rows=answer.rows,
            fit_ok=bool(verdict.get("ok")),
            fit_detail=""
            if verdict.get("ok")
            else f"{verdict.get('code')} - {verdict.get('detail')}",
        )
        _write(root, f"{name}.json", census)
        found[row.slot] = census
    return found


def _read(path: Path) -> Census | None:
    try:
        body = json.loads(path.read_bytes())
        if body.get("format") != FORMAT:
            return None
        fit = body["fit"]
        return Census(
            components=tuple(str(c) for c in body["components"]),
            rows=tuple(
                (str(component), str(tensor), None if dtype is None else str(dtype), tuple(shape))
                for component, tensor, dtype, shape in body["rows"]
            ),
            fit_ok=bool(fit["ok"]),
            fit_detail=str(fit["detail"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None  # Absent, torn or foreign entries are misses; the rewrite replaces them.


def _write(root: Path, name: str, census: Census) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    raw = canonical.write(
        {
            "components": list(census.components),
            "fit": {"detail": census.fit_detail, "ok": census.fit_ok},
            "format": FORMAT,
            "rows": [
                [component, tensor, dtype, list(shape)]
                for component, tensor, dtype, shape in census.rows
            ],
        }
    )
    temporary = root / f".census-{uuid.uuid4().hex}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        with contextlib.suppress(OSError):
            temporary.chmod(0o444)
        os.replace(temporary, root / name)
    finally:
        temporary.unlink(missing_ok=True)
