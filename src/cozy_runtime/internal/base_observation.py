"""Actual interpreter and protected platform facts used for package admission."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.version import InvalidVersion, Version

_NAME_RUNS = re.compile(r"[-_.]+")
_ROOT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UV_VERSION = re.compile(r"uv ([0-9]+(?:\.[0-9]+){1,3})(?: \([^\n]+\))?")
_PROTECTED = frozenset(
    {
        "cozy-runtime",
        "cozy-runtime-cuda-kernels",
        "tensorfs",
        "torch",
        "torchaudio",
        "torchvision",
        "triton",
    }
)


class ObservationRefusal(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Observation:
    python_abi: str
    os_arch: str
    libc: str
    accelerator_backend: str
    accelerator_build: str
    torch_release: str
    cuda_version: str
    uv_version: str
    distributions: tuple[tuple[str, str], ...]
    import_roots: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def distribution_versions(self) -> dict[str, str]:
        return dict(self.distributions)

    @property
    def root_providers(self) -> dict[str, tuple[str, ...]]:
        return dict(self.import_roots)

    @property
    def interpreter_identity(self) -> dict[str, str]:
        return {"libc": self.libc, "os_arch": self.os_arch, "python_abi": self.python_abi}


@lru_cache(maxsize=1)
def current() -> Observation:
    """Measure this process's real environment once."""

    cache_tag = sys.implementation.cache_tag or ""
    match = re.fullmatch(r"cpython-([0-9]{2,3})", cache_tag)
    if sys.implementation.name != "cpython" or match is None:
        raise ObservationRefusal(f"unsupported Python implementation: {cache_tag or 'unknown'}")
    machine = platform.machine().lower()
    if sys.platform != "linux" or machine not in {"amd64", "x86_64"}:
        raise ObservationRefusal(
            f"package execution requires Linux x86-64, got {sys.platform}/{machine}"
        )
    libc_name, libc_version = platform.libc_ver()
    if libc_name != "glibc" or re.fullmatch(r"[0-9]+\.[0-9]+", libc_version) is None:
        raise ObservationRefusal(f"glibc version is unavailable: {libc_name} {libc_version}")

    distributions, roots = _ownership()
    installed = dict(distributions)
    torch_version = installed.get("torch")
    if torch_version is None:
        backend, build, torch_release, cuda = "none", "cpu", "none", "none"
    else:
        try:
            import torch
        except Exception as exc:
            raise ObservationRefusal(f"installed Torch cannot be imported: {exc}") from exc
        if str(torch.__version__) != torch_version:
            raise ObservationRefusal(
                f"Torch metadata/import versions differ: {torch_version}/{torch.__version__}"
            )
        try:
            release = Version(torch_version).release
        except InvalidVersion as exc:
            raise ObservationRefusal(
                f"installed Torch version is invalid: {torch_version}"
            ) from exc
        torch_release = ".".join(str(part) for part in release[:3])
        cuda = str(torch.version.cuda or "none")
        if cuda == "none":
            backend, build = "none", "cpu"
        else:
            backend = "cuda"
            build = "cu" + cuda.replace(".", "")

    return Observation(
        python_abi="cp" + match.group(1),
        os_arch="linux/amd64",
        libc="glibc" + libc_version,
        accelerator_backend=backend,
        accelerator_build=build,
        torch_release=torch_release,
        cuda_version=cuda,
        uv_version=_uv_version(),
        distributions=distributions,
        import_roots=roots,
    )


def from_python(python: Path) -> Observation:
    """Measure the interpreter that will execute a preinstalled local package."""

    if not python.is_absolute() or not python.is_file():
        raise ObservationRefusal(f"package environment Python is absent: {python}")
    if python == Path(sys.executable):
        return current()
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "from cozy_runtime.internal.base_observation import print_current; print_current()",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ObservationRefusal(
            result.stderr.strip()[-2000:] or f"base observation exited {result.returncode}"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ObservationRefusal("base observation returned invalid JSON") from exc
    return _parse(value)


def print_current() -> None:
    """Internal subprocess bridge; output is transient and never an identity document."""

    print(json.dumps(_body(current()), sort_keys=True, separators=(",", ":")))


def _ownership() -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    versions: dict[str, str] = {}
    providers: dict[str, set[str]] = {}
    grouped: dict[str, list[importlib.metadata.Distribution]] = {}
    for distribution in importlib.metadata.distributions():
        name = _normalize(distribution.metadata.get("Name", ""))
        if not _protected(name):
            continue
        if not distribution.version:
            raise ObservationRefusal(f"protected distribution {name} has no version")
        grouped.setdefault(name, []).append(distribution)
    for name, candidates in grouped.items():
        ranked = sorted(
            ((_path_rank(candidate), candidate) for candidate in candidates), key=lambda row: row[0]
        )
        rank, selected = ranked[0]
        if any(
            other_rank == rank and other.version != selected.version
            for other_rank, other in ranked[1:]
        ):
            raise ObservationRefusal(f"protected distribution {name} has conflicting versions")
        versions[name] = selected.version
        for root in _distribution_roots(selected):
            providers.setdefault(root, set()).add(name)
    for root in sys.stdlib_module_names:
        try:
            present = importlib.util.find_spec(root) is not None
        except (ImportError, AttributeError, ValueError):
            present = False
        if present and _ROOT.fullmatch(root):
            providers.setdefault(root, set()).add("python-stdlib")
    return (
        tuple((name, versions[name]) for name in sorted(versions)),
        tuple((root, tuple(sorted(providers[root]))) for root in sorted(providers)),
    )


def _distribution_roots(distribution: importlib.metadata.Distribution) -> set[str]:
    roots: set[str] = set()
    for entry in distribution.files or ():
        parts = PurePosixPath(str(entry)).parts
        if not parts:
            continue
        first = parts[0]
        if first.endswith((".dist-info", ".egg-info")):
            continue
        if first.endswith(".data"):
            if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
                continue
            first = parts[2]
        if first == "__pycache__" or first.endswith(".libs"):
            continue
        if first.endswith((".py", ".pyi")):
            first = first.rsplit(".", 1)[0]
        elif "." in first:
            first = first.split(".", 1)[0]
        if _ROOT.fullmatch(first):
            roots.add(first)
    return roots


def _path_rank(distribution: importlib.metadata.Distribution) -> int:
    root = Path(str(distribution.locate_file(""))).resolve()
    for index, entry in enumerate(sys.path):
        if entry and Path(entry).resolve() == root:
            return index
    raise ObservationRefusal(
        f"protected distribution {distribution.metadata.get('Name', '')} is outside sys.path"
    )


def _uv_version() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise ObservationRefusal("uv is absent from the package-execution environment")
    result = subprocess.run(
        [uv, "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    match = _UV_VERSION.fullmatch(result.stdout.strip())
    if result.returncode != 0 or match is None:
        raise ObservationRefusal(
            result.stderr.strip()[-1000:] or f"uv --version exited {result.returncode}"
        )
    return match.group(1)


def _protected(name: str) -> bool:
    return name in _PROTECTED or name.startswith(("cuda-", "nvidia-"))


def _normalize(value: str) -> str:
    return _NAME_RUNS.sub("-", value.strip().lower())


def _body(observation: Observation) -> dict[str, Any]:
    return {
        "accelerator_backend": observation.accelerator_backend,
        "accelerator_build": observation.accelerator_build,
        "cuda_version": observation.cuda_version,
        "distributions": [list(row) for row in observation.distributions],
        "import_roots": [[root, list(providers)] for root, providers in observation.import_roots],
        "libc": observation.libc,
        "os_arch": observation.os_arch,
        "python_abi": observation.python_abi,
        "torch_release": observation.torch_release,
        "uv_version": observation.uv_version,
    }


def _parse(value: object) -> Observation:
    fields = {
        "accelerator_backend",
        "accelerator_build",
        "cuda_version",
        "distributions",
        "import_roots",
        "libc",
        "os_arch",
        "python_abi",
        "torch_release",
        "uv_version",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ObservationRefusal("base observation fields are invalid")
    try:
        observation = Observation(
            python_abi=str(value["python_abi"]),
            os_arch=str(value["os_arch"]),
            libc=str(value["libc"]),
            accelerator_backend=str(value["accelerator_backend"]),
            accelerator_build=str(value["accelerator_build"]),
            torch_release=str(value["torch_release"]),
            cuda_version=str(value["cuda_version"]),
            uv_version=str(value["uv_version"]),
            distributions=tuple(
                (str(name), str(version)) for name, version in value["distributions"]
            ),
            import_roots=tuple(
                (str(root), tuple(str(provider) for provider in providers))
                for root, providers in value["import_roots"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ObservationRefusal("base observation rows are invalid") from exc
    if _body(observation) != value:
        raise ObservationRefusal("base observation is not canonical internal data")
    return observation
