"""The offline native-wheel proof's inspection and host qualification (cr-093).

ONE consumer: `cli/native_wheel_proof.py`, an authoring/qualification tool run on purpose —
never on a cold pod's prepare. `inspect_installed` reads the installed generation's ELF shape,
RPATH/RUNPATH containment, base-library shadowing, and CUDA image inventory; `qualify_host`
joins that to the actual node's loader inventory and GPU compute capability; and
`qualify_operator` is the proof that matters — it EXECUTES the banked operator fixture in a
bounded disposable process and compares a result digest. We install our own packages and
`--require-hashes`-verified PyPI wheels, so none of this is a defence against a hostile wheel;
it is evidence about an exact prebuilt wheel/base/hardware tuple, banked by its author.
"""

from __future__ import annotations

import contextlib
import csv
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import base_observation, canonical, liveness

MAX_ELF_SECTIONS = 16_384
MAX_NATIVE_MEMBERS = 4_096
MAX_STRING_TABLE_BYTES = 64 << 20
MAX_SYMBOLS = 2_000_000
MAX_HOST_LIBRARIES = 200_000
MAX_TOOL_OUTPUT = 4 << 20
MAX_OPERATOR_REQUEST_BYTES = 1 << 20
_SEAT_FIELDS = {
    "cuda_available",
    "distributions",
    "implementation",
    "libc_name",
    "libc_version",
    "machine",
    "python_abi",
    "system",
    "torch_cuda",
    "torch_version",
}

_ELF_MAGIC = b"\x7fELF"
_ELF64_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_ELF64_SECTION = struct.Struct("<IIQQQQIIQQ")
_ELF64_DYNAMIC = struct.Struct("<qQ")
_ELF64_SYMBOL = struct.Struct("<IBBHQQ")
_CUDA_SECTION = re.compile(r"(?:^|\.)(?:nv|cuda)(?:_|\.)", re.IGNORECASE)
_SM = re.compile(r"(?:^|[^a-z0-9])sm[_-]?([0-9]{2,3})(?:[^0-9]|$)", re.IGNORECASE)
_COMPUTE = re.compile(r"(?:^|[^a-z0-9])compute[_-]?([0-9]{2,3})(?:[^0-9]|$)", re.IGNORECASE)
_PROTECTED_LIBRARY = re.compile(
    r"^(?:lib)?(?:c10|cublas|cublaslt|cuda|cudart|cudnn|cufft|cufile|curand|"
    r"cusolver|cusparse|cusparselt|nccl|nvidia|nvjitlink|nvrtc|nvshmem|"
    r"nvtoolsext|nvtx|torch|triton)"
    r"(?:[_+.-].*)?\.so(?:\..*)?$",
    re.IGNORECASE,
)
_VIRTUAL_LIBRARIES = frozenset({"linux-vdso.so.1"})


class NativeWheelRefusal(Exception):
    """A fail-closed native-wheel verdict with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class GPUObservation(TypedDict):
    device_index: int
    driver_version: str
    name: str
    sm: int


@dataclass(frozen=True, slots=True)
class ELFMember:
    member: str
    soname: str
    needed: tuple[str, ...]
    runpaths: tuple[str, ...]
    defined_symbols: tuple[str, ...]
    required_symbols: tuple[str, ...]
    cuda_sections: tuple[str, ...]
    cubin_sms: tuple[int, ...]
    ptx_compute: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Inspection:
    members: tuple[ELFMember, ...]

    def document(self) -> dict[str, object]:
        return {
            "members": [
                {
                    "cubin_sms": list(item.cubin_sms),
                    "cuda_sections": list(item.cuda_sections),
                    "defined_symbols_digest": canonical.digest(list(item.defined_symbols)),
                    "member": item.member,
                    "needed": list(item.needed),
                    "ptx_compute": list(item.ptx_compute),
                    "required_symbols_digest": canonical.digest(list(item.required_symbols)),
                    "runpaths": list(item.runpaths),
                    "soname": item.soname,
                }
                for item in self.members
            ],
        }


@dataclass(frozen=True, slots=True)
class _Section:
    name: str
    kind: int
    offset: int
    size: int
    link: int
    entry_size: int


class _ELF:
    def __init__(self, path: Path, *, require_shared: bool) -> None:
        self.path = path
        self._size = path.stat().st_size
        if self._size < _ELF64_HEADER.size:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{path}: truncated ELF"
            )
        header = _ELF64_HEADER.unpack(self._read(0, _ELF64_HEADER.size))
        identity, elf_type, machine = header[0], header[1], header[2]
        if identity[:4] != _ELF_MAGIC or identity[4] != 2 or identity[5] != 1:
            raise NativeWheelRefusal(
                "package_environment_native_elf_unsupported",
                f"{path}: requires ELF64 little-endian",
            )
        if machine != 62 or (require_shared and elf_type != 3) or elf_type not in {2, 3}:
            raise NativeWheelRefusal(
                "package_environment_native_elf_unsupported",
                f"{path}: e_type={elf_type} e_machine={machine}, want x86_64 shared object",
            )
        section_offset, section_entry_size = header[6], header[11]
        section_count, names_index = header[12], header[13]
        if section_entry_size != _ELF64_SECTION.size or section_count > MAX_ELF_SECTIONS:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{path}: section table"
            )
        if section_count == 0:
            section_zero = _ELF64_SECTION.unpack(self._read(section_offset, section_entry_size))
            section_count = section_zero[5]
        if section_count <= 0 or section_count > MAX_ELF_SECTIONS:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{path}: section count"
            )
        table_size = section_count * section_entry_size
        table = self._read(section_offset, table_size)
        raw_sections = [
            _ELF64_SECTION.unpack_from(table, index * section_entry_size)
            for index in range(section_count)
        ]
        if names_index == 0xFFFF:
            names_index = raw_sections[0][6]
        if names_index >= section_count:
            raise NativeWheelRefusal("package_environment_native_elf_invalid", f"{path}: shstrndx")
        names_raw = self._section_bytes(raw_sections[names_index], "section names")
        self.sections = tuple(
            _Section(
                self._string(names_raw, row[0], "section name"),
                row[1],
                row[4],
                row[5],
                row[6],
                row[9],
            )
            for row in raw_sections
        )

    def _read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset > self._size or length > self._size - offset:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: file bounds"
            )
        with self.path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: short read"
            )
        return data

    def _section_bytes(self, row: tuple[int, ...], label: str) -> bytes:
        size = row[5]
        if size > MAX_STRING_TABLE_BYTES:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: oversized {label}"
            )
        return self._read(row[4], size)

    def _bytes(self, section: _Section, label: str) -> bytes:
        if section.size > MAX_STRING_TABLE_BYTES:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: oversized {label}"
            )
        return self._read(section.offset, section.size)

    def _string(self, table: bytes, offset: int, label: str) -> str:
        if offset < 0 or offset >= len(table):
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: {label}"
            )
        end = table.find(b"\0", offset)
        if end < 0:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: {label}"
            )
        try:
            return table[offset:end].decode("utf-8")
        except UnicodeError as exc:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: non-UTF-8 {label}"
            ) from exc

    def dynamic(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        dynamic = next((section for section in self.sections if section.kind == 6), None)
        if dynamic is None:
            return "", (), ()
        if dynamic.link >= len(self.sections) or dynamic.entry_size not in {0, 16}:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: dynamic table"
            )
        strings = self._bytes(self.sections[dynamic.link], "dynamic strings")
        raw = self._bytes(dynamic, "dynamic table")
        if len(raw) % _ELF64_DYNAMIC.size:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: dynamic size"
            )
        needed: list[str] = []
        runpaths: list[str] = []
        soname = ""
        for offset in range(0, len(raw), _ELF64_DYNAMIC.size):
            tag, value = _ELF64_DYNAMIC.unpack_from(raw, offset)
            if tag == 0:
                break
            if tag == 1:
                needed.append(self._string(strings, value, "DT_NEEDED"))
            elif tag == 14:
                soname = self._string(strings, value, "DT_SONAME")
            elif tag in {15, 29}:
                runpaths.extend(self._string(strings, value, "RPATH").split(":"))
        return soname, tuple(sorted(set(needed))), tuple(sorted(set(runpaths)))

    def symbols(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        table = next((section for section in self.sections if section.kind == 11), None)
        if table is None:
            return (), ()
        entry_size = table.entry_size or _ELF64_SYMBOL.size
        if entry_size != _ELF64_SYMBOL.size or table.link >= len(self.sections):
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: dynsym table"
            )
        count = table.size // entry_size
        if table.size % entry_size or count > MAX_SYMBOLS:
            raise NativeWheelRefusal(
                "package_environment_native_elf_invalid", f"{self.path}: dynsym size"
            )
        strings = self._bytes(self.sections[table.link], "symbol strings")
        raw = self._read(table.offset, table.size)
        defined: set[str] = set()
        required: set[str] = set()
        for offset in range(0, len(raw), entry_size):
            name_offset, info, _other, section_index, _value, _size = _ELF64_SYMBOL.unpack_from(
                raw, offset
            )
            binding = info >> 4
            if name_offset == 0 or binding not in {1, 2}:
                continue
            name = self._string(strings, name_offset, "symbol name")
            if not name:
                continue
            if section_index == 0:
                if binding == 1:
                    required.add(name)
            else:
                defined.add(name)
        return tuple(sorted(defined)), tuple(sorted(required))


def inspect_tree(root: Path, members: tuple[str, ...]) -> Inspection:
    """Inspect exact extracted native members before their immutable pool is published."""

    root = root.resolve(strict=True)
    # Collision is judged against the DYNAMIC LINKER's namespace only: DT_NEEDED
    # resolution is process-global, so a wheel bundling an ldconfig name can hijack
    # it. A Python extension in some environment's site-packages is imported by
    # path, never resolved by name — an isolated venv carrying the same filename
    # shadows nothing (cr-067 residual, ruled by the xs-008 self-contained-venv
    # design). Resolution below still sees the full inventory.
    host_libraries = frozenset(name for name, _ in _linker_library_rows())
    parsed: list[ELFMember] = []
    for member in members:
        path = (root / member).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise NativeWheelRefusal("package_environment_native_member_invalid", member)
        elf = _ELF(path, require_shared=True)
        soname, needed, runpaths = elf.dynamic()
        defined, required = elf.symbols()
        sections = tuple(sorted(section.name for section in elf.sections if section.name))
        cuda_sections = tuple(name for name in sections if _CUDA_SECTION.search(name))
        _require_safe_libraries(root, path, soname, needed, runpaths, host_libraries)
        cubins: tuple[int, ...] = ()
        ptx: tuple[int, ...] = ()
        if cuda_sections:
            cubins, ptx = _cuda_images(path)
            if not cubins and not ptx:
                raise NativeWheelRefusal(
                    "package_environment_native_cuda_images_absent",
                    f"{member}: CUDA sections contain no inspectable cubin or PTX target",
                )
        parsed.append(
            ELFMember(
                member,
                soname,
                needed,
                runpaths,
                defined,
                required,
                cuda_sections,
                cubins,
                ptx,
            )
        )
    return Inspection(tuple(parsed))


def inspect_installed(root: Path) -> Inspection:
    """Re-read native members from one immutable installed generation for host proof."""

    resolved = root.resolve(strict=True)
    members: list[str] = []
    visited = 0
    for path in resolved.rglob("*"):
        visited += 1
        if visited > 200_000:
            raise NativeWheelRefusal("package_environment_native_tree_oversized", str(visited))
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as handle:
            magic = handle.read(4)
        if magic == _ELF_MAGIC:
            members.append(path.relative_to(resolved).as_posix())
            if len(members) > MAX_NATIVE_MEMBERS:
                raise NativeWheelRefusal(
                    "package_environment_native_member_count", str(len(members))
                )
    if not members:
        raise NativeWheelRefusal("package_environment_native_members_absent", str(root))
    return inspect_tree(resolved, tuple(sorted(members)))


def _require_safe_libraries(
    root: Path,
    path: Path,
    soname: str,
    needed: tuple[str, ...],
    runpaths: tuple[str, ...],
    host_libraries: frozenset[str],
) -> None:
    for library in (path.name, soname):
        if library and _PROTECTED_LIBRARY.fullmatch(library):
            raise NativeWheelRefusal(
                "package_environment_native_base_library_shadow", f"{path}: bundles {library}"
            )
        if library and library in host_libraries:
            raise NativeWheelRefusal(
                "package_environment_native_base_library_shadow",
                f"{path}: {library} already belongs to the selected base",
            )
    for library in needed:
        if not library or "/" in library or "\\" in library or "$" in library:
            raise NativeWheelRefusal(
                "package_environment_native_needed_invalid", f"{path}: DT_NEEDED={library!r}"
            )
    for value in runpaths:
        normalized = value.replace("${ORIGIN}", "$ORIGIN")
        if not normalized or not normalized.startswith("$ORIGIN") or "$" in normalized[7:]:
            raise NativeWheelRefusal(
                "package_environment_native_rpath_forbidden", f"{path}: RPATH={value!r}"
            )
        suffix = normalized[7:]
        if suffix and not suffix.startswith("/"):
            raise NativeWheelRefusal(
                "package_environment_native_rpath_forbidden", f"{path}: RPATH={value!r}"
            )
        target = Path(os.path.normpath(str(path.parent) + suffix))
        if not target.is_relative_to(root):
            raise NativeWheelRefusal(
                "package_environment_native_rpath_forbidden",
                f"{path}: RPATH escapes wheel: {value!r}",
            )


def _cuda_images(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tool = shutil.which("cuobjdump")
    if tool is None:
        raise NativeWheelRefusal("package_environment_native_gpu_inspector_unavailable", str(path))
    output = bytearray()
    for option in ("--list-elf", "--list-ptx"):
        try:
            result = subprocess.run(
                [tool, option, str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise NativeWheelRefusal(
                "package_environment_native_gpu_inspection_failed", f"{path}: {type(exc).__name__}"
            ) from exc
        output.extend(result.stdout[: MAX_TOOL_OUTPUT + 1])
        if result.returncode != 0 or len(output) > MAX_TOOL_OUTPUT:
            raise NativeWheelRefusal(
                "package_environment_native_gpu_inspection_failed", f"{path}: cuobjdump {option}"
            )
    text = output.decode("utf-8", "replace")
    return (
        tuple(sorted({int(value) for value in _SM.findall(text)})),
        tuple(sorted({int(value) for value in _COMPUTE.findall(text)})),
    )


def qualify_host(inspection: Inspection, *, gpu_sm: int | None) -> dict[str, object]:
    """Join static evidence to the actual loader and GPU before any package import."""

    wheel_defined = {symbol for item in inspection.members for symbol in item.defined_symbols}
    libraries = _host_libraries()
    loaded_symbols: set[str] = set()
    try:
        executable = _ELF(Path(sys.executable), require_shared=False)
        executable_defined, _executable_required = executable.symbols()
        loaded_symbols.update(executable_defined)
    except (OSError, NativeWheelRefusal):
        pass
    visited: set[Path] = set()
    missing_libraries: set[str] = set()
    wheel_libraries = {
        name
        for item in inspection.members
        for name in (Path(item.member).name, item.soname)
        if name
    }
    pending = [library for item in inspection.members for library in item.needed]
    while pending:
        library = pending.pop()
        if library in _VIRTUAL_LIBRARIES or library in wheel_libraries:
            continue
        path = libraries.get(library)
        if path is None:
            missing_libraries.add(library)
            continue
        if path in visited:
            continue
        visited.add(path)
        try:
            elf = _ELF(path, require_shared=False)
            _soname, needed, _runpaths = elf.dynamic()
            defined, _required = elf.symbols()
        except (OSError, NativeWheelRefusal):
            missing_libraries.add(library)
            continue
        loaded_symbols.update(defined)
        pending.extend(needed)
    if missing_libraries:
        raise NativeWheelRefusal(
            "package_environment_native_dependency_unavailable", ",".join(sorted(missing_libraries))
        )
    unresolved = sorted(
        {
            symbol
            for item in inspection.members
            for symbol in item.required_symbols
            if symbol not in wheel_defined and symbol not in loaded_symbols
        }
    )
    if unresolved:
        raise NativeWheelRefusal(
            "package_environment_native_symbol_unavailable", ",".join(unresolved[:16])
        )
    gpu_members = [item for item in inspection.members if item.cuda_sections]
    if gpu_members and gpu_sm is None:
        raise NativeWheelRefusal(
            "package_environment_native_gpu_unavailable", "CUDA wheel on a CPU host"
        )
    if gpu_sm is not None:
        unsupported = [
            item.member
            for item in gpu_members
            if gpu_sm not in item.cubin_sms
            and not any(compute <= gpu_sm for compute in item.ptx_compute)
        ]
        if unsupported:
            raise NativeWheelRefusal(
                "package_environment_native_gpu_unsupported",
                f"sm_{gpu_sm}: no matching cubin or forward PTX in {','.join(unsupported)}",
            )
    return {
        "gpu_sm": gpu_sm,
        "host_library_count": len(visited),
        "inspection_digest": canonical.digest(inspection.document()),
        "qualification": "static-host-compatible",
    }


def observe_seat(python: Path, environment: Mapping[str, str]) -> dict[str, object]:
    """Ask the selected base interpreter to report its actual installed/runtime facts.

    The probe imports torch, which is a cold import of a gigabyte of shared objects on a
    first boot; it ends when the child exits, or when the child has PROVABLY stopped working
    (`_run_working`) — never at a number of seconds a warm box happened to fit inside.
    """

    try:
        returncode, stdout, stderr = _run_working(
            [str(python), "-I", "-m", "cozy_runtime.internal.native_seat_probe"],
            environment,
            "native_seat_probe",
        )
    except OSError as exc:
        raise NativeWheelRefusal(
            "package_environment_native_seat_probe_failed", type(exc).__name__
        ) from exc
    except _Wedged as exc:
        raise NativeWheelRefusal("package_environment_native_seat_probe_wedged", str(exc)) from exc
    if returncode != 0 or len(stdout) > MAX_TOOL_OUTPUT or len(stderr) > MAX_TOOL_OUTPUT:
        detail = stderr.decode("utf-8", "replace").strip()[:1000]
        raise NativeWheelRefusal(
            "package_environment_native_seat_probe_failed", detail or f"exit {returncode}"
        )
    try:
        value = canonical.parse_canonical(stdout)
    except canonical.CanonicalError as exc:
        raise NativeWheelRefusal("package_environment_native_seat_probe_invalid", str(exc)) from exc
    if not isinstance(value, dict) or set(value) != _SEAT_FIELDS:
        raise NativeWheelRefusal("package_environment_native_seat_probe_invalid", "wrong fields")
    return value


def verify_seat(
    base: base_observation.Observation,
    observation: Mapping[str, object],
    *,
    gpu: GPUObservation | None,
    venv_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Verify a native proof process still matches its observed package base.

    `venv_versions` names the base distributions the package venv itself carries — its
    own wheels and boot-installs — at the versions it installed. A `--system-site-packages`
    venv shadows those image copies by ordinary precedence (cr-070), so the probe run under
    the venv interpreter must measure the venv's version there and the image's everywhere
    else; either disagreement names the distribution.
    """

    if (
        observation.get("implementation") != "cpython"
        or observation.get("python_abi") != base.python_abi
    ):
        raise NativeWheelRefusal(
            "package_environment_native_python_abi_mismatch",
            f"measured={observation.get('implementation')}:{observation.get('python_abi')} "
            f"required=cpython:{base.python_abi}",
        )
    machine = observation.get("machine")
    measured_arch = "linux/amd64" if machine in {"amd64", "x86_64"} else ""
    if observation.get("system") != "linux" or measured_arch != base.os_arch:
        raise NativeWheelRefusal(
            "package_environment_native_os_cpu_mismatch",
            f"measured={observation.get('system')}/{machine} required={base.os_arch}",
        )
    libc_match = re.fullmatch(r"glibc([0-9]+\.[0-9]+)", base.libc)
    if observation.get("libc_name") != "glibc" or libc_match is None:
        raise NativeWheelRefusal(
            "package_environment_native_libc_mismatch",
            f"measured={observation.get('libc_name')}{observation.get('libc_version')} "
            f"required={base.libc}",
        )
    try:
        libc_ok = Version(str(observation.get("libc_version", ""))) >= Version(libc_match.group(1))
    except InvalidVersion as exc:
        raise NativeWheelRefusal("package_environment_native_libc_mismatch", str(exc)) from exc
    if not libc_ok:
        raise NativeWheelRefusal(
            "package_environment_native_libc_mismatch",
            f"measured=glibc{observation.get('libc_version')} minimum={base.libc}",
        )
    rows = observation.get("distributions")
    measured_distributions: list[tuple[str, str]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"distribution", "version"}:
                raise NativeWheelRefusal(
                    "package_environment_native_base_inventory_mismatch",
                    "invalid measured distribution row",
                )
            name, version = row["distribution"], row["version"]
            if not isinstance(name, str) or not isinstance(version, str):
                raise NativeWheelRefusal(
                    "package_environment_native_base_inventory_mismatch",
                    "invalid measured distribution",
                )
            measured_distributions.append((name, version))
    measured_versions = dict(measured_distributions)
    shadowed = venv_versions or {}
    for name, image_version in base.distributions:
        expected = shadowed.get(name, image_version)
        measured = measured_versions.get(name)
        if measured != expected:
            origin = "venv" if name in shadowed else "image"
            raise NativeWheelRefusal(
                "package_environment_native_base_inventory_mismatch",
                f"distribution={name} measured={measured or 'absent'} "
                f"expected={expected} ({origin}) image_version={image_version}",
            )
    torch_version = str(observation.get("torch_version", ""))
    if base.torch_release == "none":
        torch_release_ok = not torch_version
    else:
        try:
            torch_release_ok = (
                Version(torch_version).release[:3] == Version(base.torch_release).release[:3]
            )
        except InvalidVersion as exc:
            raise NativeWheelRefusal(
                "package_environment_native_torch_mismatch", torch_version
            ) from exc
    if not torch_release_ok:
        raise NativeWheelRefusal(
            "package_environment_native_torch_mismatch",
            f"measured={torch_version} required={base.torch_release}",
        )
    torch_cuda = str(observation.get("torch_cuda", ""))
    if base.accelerator_backend == "none":
        if torch_cuda or bool(observation.get("cuda_available")) or gpu is not None:
            raise NativeWheelRefusal(
                "package_environment_native_accelerator_mismatch",
                f"CPU profile measured torch_cuda={torch_cuda!r} available="
                f"{observation.get('cuda_available')} gpu={gpu is not None}",
            )
    else:
        required_cuda = base.cuda_version
        if (
            torch_cuda != required_cuda
            or not bool(observation.get("cuda_available"))
            or gpu is None
        ):
            raise NativeWheelRefusal(
                "package_environment_native_accelerator_mismatch",
                f"measured torch_cuda={torch_cuda!r} available="
                f"{observation.get('cuda_available')} gpu={gpu is not None}; "
                f"required={required_cuda}",
            )
    return dict(observation)


class _Wedged(Exception):
    """A child that provably stopped working, with the measurement that says so."""


def _run_working(
    command: list[str], environment: Mapping[str, str], what: str
) -> tuple[int, bytes, bytes]:
    """Run one disposable child to EXIT, ending it early only on a proven wedge.

    No wall clock runs here. The child is sampled on `liveness.SAMPLE_SECONDS` — a poll
    cadence, not a bound — and killed with its group only when its own work meter
    (`proctree.progress_burn`: CPU plus bytes moved) has been flat for longer than
    `liveness.Pace` allows against the pauses this same child has shown. A cold import that
    is paging a gigabyte of shared objects is moving bytes and is left alone; a child stuck
    on a lock it will never get is not, and is ended with the measurement in hand. An
    unreadable meter decides nothing: the child then ends only by exiting.
    """
    pace = liveness.Pace()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=dict(environment),
            start_new_session=True,
        )
        try:
            while True:
                try:
                    returncode = process.wait(timeout=liveness.SAMPLE_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    pace.observe(liveness.burn(process.pid))
                    if pace.wedged(liveness.noise_floor()):
                        raise _Wedged(f"{what}: {pace.verdict(liveness.noise_floor())}") from None
        finally:
            _kill_process_group(process)
            process.wait()
        stdout.seek(0)
        stderr.seek(0)
        return returncode, stdout.read(MAX_TOOL_OUTPUT + 1), stderr.read(MAX_TOOL_OUTPUT + 1)


def qualify_operator(
    *,
    python: Path,
    fixture: str,
    expected_result_digest: str,
    environment: Mapping[str, str],
    request: str = "",
    cpu_seconds: int = 60,
    max_address_space_bytes: int = 16 << 30,
) -> dict[str, object]:
    """Execute one banked fixture in a bounded disposable Python process.

    `request` is the fixture's one argument and travels in argv, not in the environment: the
    child's environment is the SEAL, and a name smuggled into it after `spawn.seal` is a name
    the allowlist never reviewed.

    `cpu_seconds` and `max_address_space_bytes` are the RUNAWAY FENCE, imposed by the child on
    itself as rlimits before the package imports (`native_operator_runner`): a fixture that
    spins or balloons is stopped by the kernel. Neither is a clock — a fixture that is merely
    slow, paging a cold wheel, burns few CPU-seconds and runs to its answer — and the wait for
    that answer ends on the child's exit or on its proven wedge, never on wall time.
    """

    if (
        not python.is_absolute()
        or not python.is_file()
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", fixture)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_result_digest)
        or not 0 < cpu_seconds <= 600
        or not 256 << 20 <= max_address_space_bytes <= 128 << 30
        or len(request.encode()) > MAX_OPERATOR_REQUEST_BYTES
        or "\x00" in request
    ):
        raise NativeWheelRefusal("package_environment_native_operator_request_invalid", fixture)
    command = [
        str(python),
        "-I",
        "-m",
        "cozy_runtime.internal.native_operator_runner",
        fixture,
        str(cpu_seconds),
        str(max_address_space_bytes),
        str(MAX_TOOL_OUTPUT),
        request,
    ]
    try:
        returncode, result_stdout, result_stderr = _run_working(command, environment, fixture)
    except _Wedged as exc:
        raise NativeWheelRefusal("package_environment_native_operator_wedged", str(exc)) from exc
    except OSError as exc:
        raise NativeWheelRefusal(
            "package_environment_native_operator_failed", type(exc).__name__
        ) from exc
    if len(result_stdout) > MAX_TOOL_OUTPUT or len(result_stderr) > MAX_TOOL_OUTPUT:
        raise NativeWheelRefusal("package_environment_native_operator_output_oversized", fixture)
    if returncode != 0:
        detail = result_stderr.decode("utf-8", "replace").strip()[:1000]
        code = (
            "package_environment_native_no_kernel_image"
            if "no kernel image" in detail.lower()
            else "package_environment_native_operator_failed"
        )
        raise NativeWheelRefusal(code, f"{fixture}: {detail or f'exit {returncode}'}")
    try:
        value = canonical.parse_canonical(result_stdout)
    except canonical.CanonicalError as exc:
        raise NativeWheelRefusal(
            "package_environment_native_operator_result_invalid", f"{fixture}: {exc}"
        ) from exc
    result_digest = canonical.digest(value)
    if result_digest != expected_result_digest:
        raise NativeWheelRefusal(
            "package_environment_native_operator_numerical_mismatch",
            f"{fixture}: result {result_digest}, expected {expected_result_digest}",
        )
    return {
        "expected_result_digest": expected_result_digest,
        "fixture": fixture,
        "result_digest": result_digest,
    }


def observe_gpu(device_index: int) -> GPUObservation:
    """Measure one qualification device; never infer driver or SM from wheel bytes."""

    if not 0 <= device_index < 64:
        raise NativeWheelRefusal(
            "package_environment_native_gpu_request_invalid", str(device_index)
        )
    tool = shutil.which("nvidia-smi")
    if tool is None:
        raise NativeWheelRefusal(
            "package_environment_native_gpu_unavailable", "nvidia-smi is absent"
        )
    try:
        result = subprocess.run(
            [
                tool,
                f"--id={device_index}",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativeWheelRefusal(
            "package_environment_native_gpu_unavailable", type(exc).__name__
        ) from exc
    output = result.stdout.decode("utf-8", "replace").strip()
    try:
        rows = list(csv.reader(output.splitlines()))
    except csv.Error:
        rows = []
    if result.returncode != 0 or len(rows) != 1 or len(rows[0]) != 3:
        raise NativeWheelRefusal(
            "package_environment_native_gpu_unavailable", output[:500] or "probe failed"
        )
    name, driver, capability = (item.strip() for item in rows[0])
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", capability)
    # NVIDIA driver versions are two OR three components (595.84 and 580.126.20
    # are both real); requiring three refused every two-component driver series.
    if not name or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", driver) is None or match is None:
        raise NativeWheelRefusal("package_environment_native_gpu_unavailable", output[:500])
    return {
        "device_index": device_index,
        "driver_version": driver,
        "name": name,
        "sm": int(match.group(1)) * 10 + int(match.group(2)),
    }


def observe_gpu_sm(device_index: int) -> int:
    return observe_gpu(device_index)["sm"]


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _host_libraries() -> dict[str, Path]:
    return {name: Path(path) for name, path in _host_library_rows()}


# glibc installs its loader-cache tool under /sbin, which the pod worker's fixed PATH
# (spawn-allowlists row D: /usr/local/bin:/usr/bin:/bin) deliberately omits. The tool is
# resolved at its own home rather than through PATH, so the sealed PATH stays narrow and no
# PATH entry can substitute a different ldconfig.
_LDCONFIG_HOMES = ("/sbin/ldconfig", "/usr/sbin/ldconfig")


def _ldconfig() -> str | None:
    for candidate in _LDCONFIG_HOMES:
        if os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("ldconfig")


@lru_cache(maxsize=1)
def _linker_library_rows() -> tuple[tuple[str, str], ...]:
    libraries: dict[str, Path] = {}
    tool = _ldconfig()
    if tool is not None:
        try:
            result = subprocess.run(
                [tool, "-p"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0 and len(result.stdout) <= MAX_TOOL_OUTPUT:
            for line in result.stdout.decode("utf-8", "replace").splitlines():
                left, separator, right = line.partition("=>")
                if not separator:
                    continue
                name = left.strip().split(" ", 1)[0]
                path = Path(right.strip())
                if name and path.is_absolute() and path.is_file():
                    libraries.setdefault(name, path)
    return tuple(sorted((name, str(path)) for name, path in libraries.items()))


@lru_cache(maxsize=1)
def _host_library_rows() -> tuple[tuple[str, str], ...]:
    libraries: dict[str, Path] = {name: Path(path) for name, path in _linker_library_rows()}
    roots = {
        Path(value)
        for value in sysconfig.get_paths().values()
        if value and Path(value).is_absolute()
    }
    roots.update({Path(value) for value in sys.path if value and Path(value).is_absolute()})
    count = 0
    for root in sorted(roots):
        if not root.is_dir():
            continue
        for path in root.rglob("*.so*"):
            count += 1
            if count > MAX_HOST_LIBRARIES:
                raise NativeWheelRefusal(
                    "package_environment_native_host_inventory_oversized", str(MAX_HOST_LIBRARIES)
                )
            if path.is_file() and not path.is_symlink():
                libraries.setdefault(path.name, path)
    executable = Path(sys.executable)
    if executable.is_file():
        libraries.setdefault(executable.name, executable)
    return tuple(sorted((name, str(path)) for name, path in libraries.items()))
