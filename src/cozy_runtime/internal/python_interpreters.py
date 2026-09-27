"""The rolling Python window and installed interpreter selection for every worker.

Discovery never installs Python. Image builders and local clients consume this same
policy through ``cozy-runtime --json python-interpreters``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
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


def available(
    default: Path = Path(sys.executable), *, root: Path | None = None
) -> tuple[Interpreter, ...]:
    """Return supported, installed interpreters in deterministic lowest-version order."""
    configured = package_python_paths()
    if configured is not None and not configured.strip():
        raise InterpreterRefusal("configured Python interpreter list is empty")
    candidates = (
        [Path(item) for item in configured.split(os.pathsep)]
        if configured is not None
        else [default]
    )
    if configured is None:
        uv = shutil.which("uv")
        if uv:
            found = subprocess.run(
                [
                    uv,
                    "python",
                    "list",
                    "--only-installed",
                    "--managed-python",
                    "--output-format",
                    "json",
                    "--no-config",
                    "--offline",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                env=package_install_environment(),
            )
            try:
                rows = json.loads(found.stdout) if found.returncode == 0 else []
                candidates.extend(
                    Path(row["path"]).absolute()
                    for row in (rows if isinstance(rows, list) else [])
                    if isinstance(row, dict)
                    and row.get("implementation") == "cpython"
                    and row.get("variant") == "default"
                    and isinstance(row.get("path"), str)
                    and row["path"]
                )
            except (ValueError, KeyError, TypeError):
                # Optional discovery metadata cannot invalidate an executable
                # already supplied by this host. Selection probes actual paths.
                pass
            # Preserve installed PATH/system discovery as well as all managed patches.
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
                    candidates.append(Path(found.stdout.strip()).absolute())
    if root is not None:
        pattern = "cpython-*/python.exe" if sys.platform == "win32" else "cpython-*/bin/python*.*"
        candidates.extend(sorted((root / "managed").glob(pattern)))
    observed: dict[str, Interpreter] = {}
    for path in candidates:
        if path.name.endswith("-config"):
            continue
        try:
            interpreter = probe(path)
        except (InterpreterRefusal, OSError):
            # A stale sibling installation says nothing about the interpreter
            # selected for this operation. If no usable candidate remains,
            # select/ensure reports that requested requirement's failure.
            continue
        minor = ".".join(interpreter.version.split(".")[:2])
        if minor not in supported_minors():
            continue
        observed.setdefault(str(path.resolve()), interpreter)
    return tuple(
        sorted(observed.values(), key=lambda item: (Version(item.version), str(item.executable)))
    )


def select(
    requires: str,
    version: str = "",
    *,
    default: Path = Path(sys.executable),
    root: Path | None = None,
) -> Interpreter:
    constraint = _constraint(requires, version)
    choices = available(default, root=root)
    chosen = _choose(choices, constraint, version)
    if chosen is not None:
        return chosen
    raise InterpreterRefusal(
        f"requires Python {requires or 'any'}"
        f"{(' with captured version ' + version) if version else ''}; "
        f"supported window: {', '.join(supported_minors())}; "
        f"installed: {', '.join(item.version for item in choices) or 'none'}; "
        "install a compatible supported interpreter or upgrade the package"
    )


def document(*, root: Path | None = None) -> dict[str, object]:
    return {
        "format": "cozy.python-interpreters/1",
        **({"managed_root": str(root)} if root is not None else {}),
        "supported_minors": list(supported_minors()),
        "interpreters": [item.document() for item in available(root=root)],
        "provisionable_minors": list(provisionable_minors()),
    }


def provisionable_minors() -> tuple[str, ...]:
    """Policy, not installed facts. Discovery never creates a managed installation."""
    return supported_minors() if shutil.which("uv") else ()


def _constraint(requires: str, version: str) -> SpecifierSet:
    """Authored requirements stay exact; a captured version selects only its ABI minor."""
    if version and not re.fullmatch(r"3\.[0-9]+(?:\.[0-9]+)?", version):
        raise InterpreterRefusal(f"invalid stable CPython version: {version}")
    minor = ".".join(version.split(".")[:2]) + ".*" if version else ""
    try:
        return SpecifierSet(",".join(filter(None, (requires, "==" + minor if minor else ""))))
    except InvalidSpecifier as exc:
        raise InterpreterRefusal(f"invalid Python requirement: {exc}") from exc


def _choose(
    choices: tuple[Interpreter, ...], constraint: SpecifierSet, version: str
) -> Interpreter | None:
    """The captured patch when installed, otherwise the lowest compatible interpreter."""
    matching = [candidate for candidate in choices if constraint.contains(candidate.version)]
    exact = [candidate for candidate in matching if candidate.version == version]
    return next(iter(exact or matching), None)


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise InterpreterRefusal("Python provisioning cancelled")


def _run_uv(command: list[str], *, env: dict[str, str], cancel: Callable[[], bool] | None) -> str:
    """Drain both pipes and reap the child on cooperative cancellation."""
    _check_cancel(cancel)
    # Linux retains parent-death containment. Other platforms use ordinary process
    # launch with the same cooperative cancellation and child reaping below.
    if sys.platform.startswith("linux"):
        command = [
            sys.executable,
            "-I",
            "-c",
            "import sys; from cozy_runtime.internal.trampoline import main; main(sys.argv[1:])",
            "--expect-parent",
            str(os.getpid()),
            "--oom-adj",
            str(max(500, int(Path("/proc/self/oom_score_adj").read_text()))),
            "--uid",
            "-1",
            "--gid",
            "-1",
            "--scope-backend",
            "inherit",
            "--",
            *command,
        ]
    with subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    ) as child:
        try:
            while True:
                _check_cancel(cancel)
                try:
                    stdout, stderr = child.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            child.kill()
            child.communicate()
            raise
        if child.returncode:
            raise InterpreterRefusal(stderr.strip()[-2000:] or f"uv exited {child.returncode}")
        return stdout


def _download_version(
    uv: str,
    constraint: SpecifierSet,
    cancel: Callable[[], bool] | None = None,
    preferred: str = "",
) -> str:
    # uv's bundled catalogue is read offline. Filter every patch, so <3.13.5 does
    # not download the latest 3.13 only to discover that an older patch was needed.
    raw = _run_uv(
        [
            uv,
            "python",
            "list",
            "--only-downloads",
            "--all-versions",
            "--output-format",
            "json",
            "--no-config",
            "--offline",
        ],
        env=package_install_environment(),
        cancel=cancel,
    )
    try:
        versions = {
            Version(row["version"])
            for row in json.loads(raw)
            if row.get("implementation") == "cpython"
            and row.get("variant") == "default"
            and not Version(row["version"]).is_prerelease
            and ".".join(row["version"].split(".")[:2]) in supported_minors()
            and constraint.contains(row["version"])
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise InterpreterRefusal(f"invalid uv Python catalogue: {exc}") from exc
    if not versions:
        raise InterpreterRefusal(
            f"requires Python {constraint}; supported window: {', '.join(supported_minors())}; "
            "no matching stable CPython download in uv's catalogue"
        )
    if preferred and Version(preferred) in versions:
        return preferred
    minor = min((v.major, v.minor) for v in versions)
    return str(max(v for v in versions if (v.major, v.minor) == minor))


def ensure(
    requires: str,
    version: str = "",
    *,
    root: Path,
    default: Path = Path(sys.executable),
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Interpreter:
    """Reuse installed CPython, otherwise provision one owned interpreter with uv.

    Only the trusted preparation parent calls this; executors retain their network
    seal. Python is shared under the host's install root, while Torch and every
    other dependency still belong to the existing exact package closure cache.
    """
    from cozy_runtime.internal import storage_admission

    constraint = _constraint(requires, version)
    _check_cancel(cancel)
    chosen = _choose(available(default, root=root), constraint, version)
    if chosen is not None:
        return chosen
    uv = shutil.which("uv")
    if uv is None:
        raise InterpreterRefusal("uv is required to provision a supported Python interpreter")
    selected = _download_version(uv, constraint, cancel, version)
    _check_cancel(cancel)
    # Reserve archive + extraction + cache before creating any owned directories.
    # CPython's stripped standalone build fits well below this conservative bound.
    with storage_admission.admit(storage_admission.Write(root, 1 << 30, 20000)):
        _require_owned_root(root)
        # Recheck after admission. uv itself locks its managed directory across
        # the installed-version check and atomic publication, including races here.
        chosen = _choose(available(default, root=root), constraint, version)
        if chosen is not None:
            return chosen
        if progress is not None:
            progress(f"Installing CPython {selected}")
        managed = root / "managed"
        cache = root / "cache"
        env = {
            **package_install_environment(),
            "UV_PYTHON_INSTALL_DIR": str(managed),
            "UV_CACHE_DIR": str(cache),
            "TMPDIR": str(root),
            "TEMP": str(root),
            "TMP": str(root),
            "UV_CONCURRENT_DOWNLOADS": "1",
            "UV_CONCURRENT_INSTALLS": "1",
            "UV_CONCURRENT_BUILDS": "1",
        }
        _run_uv(
            [
                uv,
                "python",
                "install",
                "--no-config",
                "--no-bin",
                "--no-registry",
                "--install-dir",
                str(managed),
                "--cache-dir",
                str(cache),
                "cpython@" + selected,
            ],
            env=env,
            cancel=cancel,
        )
        # A package executor may traverse/read its interpreter but never write it.
        if sys.platform != "win32":
            managed.chmod(0o711)
        installed = select(requires, selected, default=default, root=root)
        if progress is not None:
            progress(f"CPython {installed.version} ready")
        return installed


def _require_owned_root(root: Path) -> None:
    # POSIX ownership is checked explicitly. Windows uses the caller-owned user
    # directory ACL; do not pretend st_uid or Unix mode bits represent that ACL.
    if not root.is_absolute() or root.is_symlink() or root.is_junction():
        raise InterpreterRefusal(f"invalid managed Python root: {root}")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or (sys.platform != "win32" and info.st_uid != os.geteuid()):
        raise InterpreterRefusal(f"managed Python root owner mismatch: {root}")
    if sys.platform != "win32":
        root.chmod(0o711)
