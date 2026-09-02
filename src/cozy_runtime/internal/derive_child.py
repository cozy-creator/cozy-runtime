"""The bounded derive child: request/response documents and the launcher.

`cozy-runtime derive` runs the meta-device derivation in ONE bounded child of the package's
own interpreter — no network, no weight bytes, no writes, no subprocesses, no CUDA, a memory
and CPU cap — exactly the harness the publish memo has always used. This module is the seam
every caller shares: the verb, the memo writer, the local placement and the worker's package
preparation all hand a request here and read a response back. The derive machinery itself
is imported by the CHILD (`cozy_runtime.cli.derive --derive-child`), never by a caller, so
the worker's preparation plane stays derive-free by import while `Slot.components` is still
what the derive constructed (model-code-fit D9).
"""

from __future__ import annotations

import base64
import hashlib
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical, sandbox

REQUEST_FORMAT = "cozy.runtime.derive-request/1"
RESULT_PREFIX = "COZY_DERIVE="
MEMORY_CAP_BYTES = 24 << 30
CPU_CAP_SECONDS = 600
WALL_CAP_SECONDS = 900


class DeriveRefusal(Exception):
    """One stable derive refusal, carried across the child boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ConfigRef:
    """The construction document a slot is derived against: exact bytes at a path."""

    path: Path
    digest: str
    length: int

    def render(self) -> dict[str, Any]:
        return {"digest": self.digest, "length": self.length, "path": str(self.path)}


@dataclass(frozen=True, slots=True)
class SlotRequest:
    slot: str
    config: ConfigRef


@dataclass(frozen=True, slots=True)
class DeriveRequest:
    """What one child derives: the package (a project tree, or an installed application)
    and one config document per slot path. Sorted unique by slot."""

    slots: tuple[SlotRequest, ...]
    project: Path | None = None
    application: str = ""

    def render(self) -> dict[str, Any]:
        package: dict[str, Any] = (
            {"application": self.application}
            if self.application
            else {"project": str(self.project)}
        )
        return {
            "format": REQUEST_FORMAT,
            "package": package,
            "slots": [{"slot": s.slot, "config": s.config.render()} for s in self.slots],
        }

    def write(self, path: Path) -> Path:
        path.write_bytes(canonical.write(self.render()))
        return path


@dataclass(frozen=True, slots=True)
class DerivedTopology:
    """One slot's CodeTopology: canonical TensorFS bytes and their identity."""

    slot: str
    config_digest: str
    document: bytes

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.document).hexdigest()


def constructed_components(document: bytes) -> tuple[str, ...]:
    """The construction order a CodeTopology records (D9), read back off its bytes."""
    value = canonical.parse_canonical(document)
    if not isinstance(value, dict):
        raise DeriveRefusal("derive_child_invalid", "a CodeTopology is an object")
    order = value.get("constructed_components")
    if not isinstance(order, list) or not order or not all(isinstance(c, str) for c in order):
        raise DeriveRefusal(
            "derive_child_invalid", "CodeTopology.constructed_components is a non-empty name list"
        )
    return tuple(order)


def config_ref(path: Path) -> ConfigRef:
    """Bind a config document by its exact bytes: the identity the derive is keyed on."""
    raw = read_exact(path, "config")
    return ConfigRef(path, "sha256:" + hashlib.sha256(raw).hexdigest(), len(raw))


def read_exact(path: Path, name: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as err:
        raise DeriveRefusal("derive_input_unreadable", f"{name}: {path}: {err.strerror}") from err
    if not stat.S_ISREG(info.st_mode):
        raise DeriveRefusal("derive_input_invalid", f"{name}: {path} is not a regular file")
    if not 0 < info.st_size <= canonical.DOC_MAX_BYTES:
        raise DeriveRefusal("derive_input_invalid", f"{name}: {path} has length {info.st_size}")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise DeriveRefusal("derive_input_invalid", f"{name}: {path} changed while it was read")
    return raw


def read_request(raw: bytes) -> DeriveRequest:
    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as err:
        raise DeriveRefusal("derive_request_invalid", str(err)) from err
    if not isinstance(value, dict) or set(value) != {"format", "package", "slots"}:
        raise DeriveRefusal("derive_request_invalid", "request field set")
    if value["format"] != REQUEST_FORMAT:
        raise DeriveRefusal("derive_request_invalid", f"format {value['format']!r}")
    package = value["package"]
    if not isinstance(package, dict) or len(package) != 1:
        raise DeriveRefusal("derive_request_invalid", "package names a project or an application")
    project: Path | None = None
    application = ""
    if "project" in package:
        spelled = package["project"]
        if not isinstance(spelled, str) or not Path(spelled).is_absolute():
            raise DeriveRefusal("derive_request_invalid", "project is an absolute path")
        project = Path(spelled)
    elif "application" in package:
        application = str(package["application"])
        if application.count(":") != 1 or not all(p.strip() for p in application.split(":")):
            raise DeriveRefusal("derive_request_invalid", "application is <module>:<attribute>")
    else:
        raise DeriveRefusal("derive_request_invalid", "package names a project or an application")
    rows = value["slots"]
    if not isinstance(rows, list) or not rows:
        raise DeriveRefusal("derive_request_invalid", "slots is a non-empty array")
    slots: list[SlotRequest] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"slot", "config"}:
            raise DeriveRefusal("derive_request_invalid", "slot row field set")
        slot = row["slot"]
        ref = row["config"]
        if (
            not isinstance(slot, str)
            or not slot
            or not isinstance(ref, dict)
            or set(ref) != {"digest", "length", "path"}
            or not isinstance(ref["digest"], str)
            or canonical.DIGEST_RE.fullmatch(ref["digest"]) is None
            or not isinstance(ref["length"], int)
            or isinstance(ref["length"], bool)
            or ref["length"] <= 0
            or not isinstance(ref["path"], str)
            or not Path(ref["path"]).is_absolute()
        ):
            raise DeriveRefusal("derive_request_invalid", f"slot {slot!r}")
        slots.append(SlotRequest(slot, ConfigRef(Path(ref["path"]), ref["digest"], ref["length"])))
    names = [s.slot for s in slots]
    if names != sorted(set(names)):
        raise DeriveRefusal("derive_request_invalid", "slots are sorted unique by path")
    return DeriveRequest(tuple(slots), project=project, application=application)


def derive_in(python: Path, request: DeriveRequest, scratch: Path) -> tuple[DerivedTopology, ...]:
    """Run the bounded child in `python`'s environment and read back one CodeTopology per
    requested slot. `scratch` holds the request document the child reads."""
    scratch.mkdir(parents=True, exist_ok=True)
    request_path = request.write(scratch / f"derive-request-{os.getpid()}.json")
    child_env = sandbox.child_environment({})
    child_env.update(
        {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}
    )
    try:
        result = subprocess.run(
            [
                str(python),
                "-I",
                # -I implies -E, which DISCARDS PYTHONDONTWRITEBYTECODE from the child
                # environment; -B is the flag form the isolation cannot strip. Without it,
                # importing any module whose bytecode cache is stale trips the sandbox's
                # no_filesystem_writes fence on the .pyc write (found live, cr-067 e2e).
                "-B",
                "-m",
                "cozy_runtime.cli.derive",
                "--derive-child",
                str(request_path),
            ],
            cwd="/",
            env=child_env,
            capture_output=True,
            text=True,
            timeout=WALL_CAP_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise DeriveRefusal("derive_child_failed", f"{type(err).__name__}: {err}") from err
    finally:
        request_path.unlink(missing_ok=True)
    encoded = next(
        (
            line.removeprefix(RESULT_PREFIX)
            for line in reversed(result.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        "",
    )
    if not encoded:
        stderr = result.stderr.strip()
        if "No module named cozy_runtime" in stderr or "No module named 'cozy_runtime'" in stderr:
            raise DeriveRefusal(
                "derive_runtime_absent",
                f"{python}: its cozy-runtime carries no `derive` verb (the package's lock "
                "selects an older runtime than the one preparing it); relock the package to "
                "a runtime that carries it",
            )
        raise DeriveRefusal(
            "derive_child_failed",
            stderr[-1200:] or f"child exited {result.returncode} without a result",
        )
    try:
        response = canonical.parse_canonical(base64.b64decode(encoded, validate=True))
    except (ValueError, canonical.CanonicalError) as err:
        raise DeriveRefusal("derive_child_invalid", str(err)) from err
    if not isinstance(response, dict):
        raise DeriveRefusal("derive_child_invalid", "child result is not an object")
    if "error" in response:
        raise DeriveRefusal(
            str(response.get("error") or "derive_child_failed"),
            str(response.get("detail") or "refused"),
        )
    if result.returncode != 0:
        raise DeriveRefusal("derive_child_failed", f"child exited {result.returncode}")
    return read_response(response, request)


def read_response(
    response: Mapping[str, Any], request: DeriveRequest
) -> tuple[DerivedTopology, ...]:
    if set(response) != {"topologies"}:
        raise DeriveRefusal("derive_child_invalid", "child result has an open field set")
    rows = response["topologies"]
    if not isinstance(rows, list):
        raise DeriveRefusal("derive_child_invalid", "topologies is not an array")
    out: list[DerivedTopology] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "slot",
            "config_digest",
            "digest",
            "length",
            "bytes",
        }:
            raise DeriveRefusal("derive_child_invalid", "topology row field set")
        try:
            raw = base64.b64decode(str(row["bytes"]), validate=True)
        except ValueError as err:
            raise DeriveRefusal("derive_child_invalid", "topology bytes base64") from err
        held = DerivedTopology(str(row["slot"]), str(row["config_digest"]), raw)
        if held.digest != row["digest"] or len(raw) != row["length"]:
            raise DeriveRefusal("derive_child_invalid", "topology stored-byte identity")
        out.append(held)
    if [(t.slot, t.config_digest) for t in out] != [
        (s.slot, s.config.digest) for s in request.slots
    ]:
        raise DeriveRefusal("derive_child_invalid", "rows do not answer the requested slots")
    return tuple(out)


def emit(value: Mapping[str, Any]) -> None:
    """The child's one output line: a canonical document, base64, behind the prefix."""
    print(RESULT_PREFIX + base64.b64encode(canonical.write(dict(value))).decode())


def render(topologies: Sequence[DerivedTopology]) -> dict[str, Any]:
    return {
        "topologies": [
            {
                "slot": t.slot,
                "config_digest": t.config_digest,
                "digest": t.digest,
                "length": len(t.document),
                "bytes": base64.b64encode(t.document).decode(),
            }
            for t in topologies
        ]
    }
