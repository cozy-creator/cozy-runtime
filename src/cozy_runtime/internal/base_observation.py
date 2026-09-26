"""Actual interpreter and protected platform facts used for package admission.

On pods the delivered image inventory (`image_inventory`) is the authority on what the placed
image provides; this live observation is demoted to its cross-check there (cr-070). The
protected-family roster below stays the ONE names authority everywhere.
"""

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
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

_NAME_RUNS = re.compile(r"[-_.]+")
_ROOT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UV_VERSION = re.compile(r"uv ([0-9]+(?:\.[0-9]+){1,3})(?: \([^\n]+\))?")
# The worker-image-owned families and the import roots each one owns. This mapping is the
# ONE list; `base-distributions.json` is generated from it (checks/architecture.py) and
# cozy-creator's upload-side prune list is generated from that, so the client and the worker
# cannot hold two different opinions about what the image provides.
#
# numpy and pillow belong here for the reason msgspec already did. numpy shares a compiled
# ABI with the torch family the image pins; pillow is cozy-runtime's own public dependency
# (`pillow>=12,<13`), so a package copy of either is loaded by image code, not only by the
# package. Membership is narrow on purpose: it does not promote other incidental image
# packages, which stay package-owned.
# Torch requires setuptools at runtime. Its distutils startup hook belongs to
# the trusted interpreter prefix; package wheels may not replace that hook.
_PROTECTED_IMPORT_ROOTS = {
    "cozy-runtime": ("cozy", "cozy_runtime"),
    "msgspec": ("msgspec",),
    "numpy": ("numpy",),
    "pillow": ("PIL",),
    "setuptools": ("_distutils_hack", "setuptools"),
    "tensorfs": ("tensorfs",),
    "torch": ("torch",),
    "torchaudio": ("torchaudio",),
    "torchvision": ("torchvision",),
    "triton": ("triton",),
}
_PROTECTED = frozenset(_PROTECTED_IMPORT_ROOTS)
# The accelerator runtime subtree the image installs beside Torch. It is a prefix rather
# than a roster because its member names carry the CUDA major in them.
_PROTECTED_PREFIXES = ("cuda-", "nvidia-")


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
    python_full_version: str

    @property
    def distribution_versions(self) -> dict[str, str]:
        return dict(self.distributions)

    @property
    def root_providers(self) -> dict[str, tuple[str, ...]]:
        return dict(self.import_roots)

    @property
    def interpreter_identity(self) -> dict[str, str]:
        return {
            "libc": self.libc,
            "os_arch": self.os_arch,
            "python_abi": self.python_abi,
            "python_full_version": self.python_full_version,
        }


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
    torch_metadata_version = installed.get("torch")
    if torch_metadata_version is None:
        backend, build, torch_release, cuda = "none", "cpu", "none", "none"
    else:
        try:
            import torch
        except Exception as exc:
            raise ObservationRefusal(f"installed Torch cannot be imported: {exc}") from exc
        backend, build, torch_release, cuda = _torch_identity(
            torch_metadata_version,
            str(torch.__version__),
            str(torch.version.cuda) if torch.version.cuda is not None else None,
        )

    return Observation(
        python_abi="cp" + match.group(1),
        python_full_version=platform.python_version(),
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


def _torch_identity(
    metadata_version: str, import_version: str, cuda_version: str | None
) -> tuple[str, str, str, str]:
    """Validate Torch's public release independently from its backend build tag."""

    try:
        metadata = Version(metadata_version)
    except InvalidVersion as exc:
        raise ObservationRefusal(
            f"installed Torch metadata version is invalid: {metadata_version}"
        ) from exc
    try:
        imported = Version(import_version)
    except InvalidVersion as exc:
        raise ObservationRefusal(f"imported Torch version is invalid: {import_version}") from exc
    if Version(metadata.public) != Version(imported.public):
        raise ObservationRefusal(
            f"Torch metadata/import public versions differ: {metadata.public}/{imported.public}"
        )

    cuda = cuda_version or "none"
    if cuda == "none":
        backend, build = "none", "cpu"
    else:
        if re.fullmatch(r"[0-9]+\.[0-9]+", cuda) is None:
            raise ObservationRefusal(f"Torch CUDA version is invalid: {cuda}")
        backend, build = "cuda", "cu" + cuda.replace(".", "")

    if imported.local != build:
        raise ObservationRefusal(
            f"Torch import build tag does not match its backend: {import_version}/{backend}-{build}"
        )
    if metadata.local not in {None, build}:
        raise ObservationRefusal(
            "Torch metadata build tag does not match its backend: "
            f"{metadata_version}/{backend}-{build}"
        )

    torch_release = ".".join(str(part) for part in metadata.release[:3])
    return backend, build, torch_release, cuda


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
            # Measure the target patch even when its installed Runtime predates this field.
            "import json, platform; "
            "from cozy_runtime.internal.base_observation import current, _body; "
            "print(json.dumps({**_body(current()), "
            "'python_full_version': platform.python_version()}))",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        bare = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.metadata,json,platform,sys; "
                "print(json.dumps({'version':platform.python_version(),"
                "'tag':sys.implementation.cache_tag,'machine':platform.machine(),"
                "'libc':platform.libc_ver(),'distributions':"
                "[d.metadata['Name'] for d in importlib.metadata.distributions()]}))",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            measured = json.loads(bare.stdout)
            if (
                bare.returncode == 0
                and set(measured["distributions"]) <= {"pip"}
                and measured["machine"] in {"x86_64", "amd64"}
                and measured["libc"][0] == "glibc"
                and re.fullmatch(r"cpython-[0-9]{3}", measured["tag"])
            ):
                return Observation(
                    python_abi=measured["tag"].replace("cpython-", "cp"),
                    python_full_version=measured["version"],
                    os_arch="linux/amd64",
                    libc="glibc" + measured["libc"][1],
                    accelerator_backend="none",
                    accelerator_build="cpu",
                    torch_release="none",
                    cuda_version="none",
                    uv_version=_uv_version(),
                    distributions=(),
                    import_roots=(),
                )
        except (ValueError, KeyError, TypeError):
            pass
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
        if not name:
            continue
        if not distribution.version:
            raise ObservationRefusal(f"base distribution {name} has no version")
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
        if _protected(name):
            for root in _distribution_roots(name):
                providers.setdefault(root, set()).add(name)
    for root in sys.stdlib_module_names:
        if _ROOT.fullmatch(root):
            providers.setdefault(root, set()).add("python-stdlib")
    return (
        tuple((name, versions[name]) for name in sorted(versions)),
        tuple((root, tuple(sorted(providers[root]))) for root in sorted(providers)),
    )


def _distribution_roots(distribution: str) -> tuple[str, ...]:
    roots = _PROTECTED_IMPORT_ROOTS.get(distribution)
    if roots is None:
        if distribution.startswith("cuda-"):
            roots = ("cuda",)
        elif distribution.startswith("nvidia-"):
            roots = ("nvidia",)
        else:  # pragma: no cover - every admitted name is closed above
            raise ObservationRefusal(f"protected distribution {distribution} has no root contract")
    for root in roots:
        try:
            present = importlib.util.find_spec(root) is not None
        except (ImportError, AttributeError, ValueError) as exc:
            raise ObservationRefusal(
                f"protected distribution {distribution} root {root} is invalid: {exc}"
            ) from exc
        if not present:
            raise ObservationRefusal(
                f"protected distribution {distribution} does not provide required root {root}"
            )
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
    return name in _PROTECTED or name.startswith(_PROTECTED_PREFIXES)


def protected_distribution(name: str) -> bool:
    """Whether a normalized distribution belongs exclusively to the worker base."""

    return _protected(_normalize(name))


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
        "python_full_version": observation.python_full_version,
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
        "python_full_version",
        "torch_release",
        "uv_version",
    }
    if not isinstance(value, dict) or not fields.issubset(value):
        raise ObservationRefusal("base observation fields are invalid")
    try:
        observation = Observation(
            python_abi=str(value["python_abi"]),
            python_full_version=str(value["python_full_version"]),
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
    # Another installed Runtime may add diagnostics; only consumed facts are canonical.
    if _body(observation) != {name: value[name] for name in fields}:
        raise ObservationRefusal("base observation is not canonical internal data")
    return observation
