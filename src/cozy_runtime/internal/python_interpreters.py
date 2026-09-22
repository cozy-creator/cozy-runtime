"""The rolling Python window and installed interpreter selection for every worker.

Discovery never installs Python. Image builders and local clients consume this same
policy through ``cozy-runtime --json python-interpreters``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from cozy_runtime._interpreter import supported_minors as supported_minors
from cozy_runtime.internal.config import package_install_environment, package_python_paths


class InterpreterRefusal(Exception):
    def __init__(self, detail: str) -> None:
        self.code = "package_environment_python_unavailable"
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


@dataclass(frozen=True)
class Interpreter:
    executable: Path
    version: str
    abi: str

    def document(self) -> dict[str, str]:
        return {"executable": str(self.executable), "version": self.version, "abi": self.abi}


def probe(python: Path) -> Interpreter:
    if not python.is_absolute() or not python.is_file():
        raise InterpreterRefusal(f"installed Python executable is absent: {python}")
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-S",
            "-c",
            "import json,platform,sys,sysconfig; "
            "print(json.dumps([platform.python_version(),sys.implementation.cache_tag,"
            "sysconfig.get_config_var('Py_GIL_DISABLED')]))",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        version, tag, free_threaded = json.loads(result.stdout)
        match = re.fullmatch(r"cpython-([0-9]{3})", tag)
        if result.returncode or not match or free_threaded or Version(version).is_prerelease:
            raise ValueError("requires standard release CPython with the GIL")
        return Interpreter(python, version, "cp" + match.group(1))
    except (ValueError, TypeError) as exc:
        raise InterpreterRefusal(f"cannot observe supported CPython at {python}: {exc}") from exc


def available(default: Path = Path(sys.executable)) -> tuple[Interpreter, ...]:
    """Return supported, installed interpreters in deterministic lowest-version order."""
    configured = package_python_paths()
    candidates = (
        [Path(item) for item in configured.split(os.pathsep)]
        if configured is not None
        else [default]
    )
    if configured is None:
        uv = shutil.which("uv")
        if uv:
            for minor in supported_minors():
                found = subprocess.run(
                    [
                        uv,
                        "python",
                        "find",
                        "--no-project",
                        "--no-config",
                        "--no-python-downloads",
                        minor,
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=package_install_environment(),
                )
                if found.returncode == 0:
                    candidates.append(Path(found.stdout.strip()))
    observed: dict[str, Interpreter] = {}
    for path in candidates:
        interpreter = probe(path)
        minor = ".".join(interpreter.version.split(".")[:2])
        if minor not in supported_minors():
            if configured is not None:
                raise InterpreterRefusal(
                    f"configured Python {interpreter.version} is outside "
                    f"the supported window {', '.join(supported_minors())}"
                )
            continue
        observed.setdefault(str(path.resolve()), interpreter)
    return tuple(
        sorted(observed.values(), key=lambda item: (Version(item.version), str(item.executable)))
    )


def select(
    requires: str, version: str = "", *, default: Path = Path(sys.executable)
) -> Interpreter:
    try:
        constraint = SpecifierSet(requires)
        pin = SpecifierSet("==" + version if version else "")
    except InvalidSpecifier as exc:
        raise InterpreterRefusal(f"invalid Python requirement: {exc}") from exc
    choices = available(default)
    for candidate in choices:
        if constraint.contains(candidate.version) and pin.contains(candidate.version):
            return candidate
    raise InterpreterRefusal(
        f"requires Python {requires or 'any'}"
        f"{(' with captured version ' + version) if version else ''}; "
        f"supported window: {', '.join(supported_minors())}; "
        f"installed: {', '.join(item.version for item in choices) or 'none'}; "
        "install a compatible supported interpreter or upgrade the package"
    )


def document() -> dict[str, object]:
    return {
        "format": "cozy.python-interpreters/1",
        "supported_minors": list(supported_minors()),
        "interpreters": [item.document() for item in available()],
    }
