"""Bounded native-wheel inspection and host qualification.

Wheel extraction calls :func:`inspect_tree` before publishing any native bytes.  That pass is
purely structural: ELF shape, RPATH/RUNPATH containment, base-library shadowing, and CUDA image
inventory.  It does not infer a driver floor from bytes and does not import the extension.

The separately authorized hardware proof calls :func:`qualify_host`.  That pass joins the same
static evidence to the actual node's loader inventory and GPU compute capability.  The final
operator fixture remains the executable proof; these checks make malformed or hostile wheels fail
before Python's dynamic loader sees them.
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
from typing import Protocol, TypedDict

from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import canonical

MAX_ELF_SECTIONS = 16_384
MAX_NATIVE_MEMBERS = 4_096
MAX_STRING_TABLE_BYTES = 64 << 20
MAX_SYMBOLS = 2_000_000
MAX_HOST_LIBRARIES = 200_000
MAX_TOOL_OUTPUT = 4 << 20
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


class BaseSeat(Protocol):
    @property
    def compatibility_profile(self) -> tuple[tuple[str, str], ...]: ...

    @property
    def platform_target(self) -> tuple[tuple[str, str], ...]: ...

    @property
    def base_distributions(self) -> tuple[tuple[str, str], ...]: ...

    @property
    def minimum_cuda_version(self) -> str: ...

    @property
    def minimum_driver_version(self) -> str: ...


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
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{path}: truncated ELF")
        header = _ELF64_HEADER.unpack(self._read(0, _ELF64_HEADER.size))
        identity, elf_type, machine = header[0], header[1], header[2]
        if identity[:4] != _ELF_MAGIC or identity[4] != 2 or identity[5] != 1:
            raise NativeWheelRefusal(
                "overlay_native_elf_unsupported", f"{path}: requires ELF64 little-endian"
            )
        if machine != 62 or (require_shared and elf_type != 3) or elf_type not in {2, 3}:
            raise NativeWheelRefusal(
                "overlay_native_elf_unsupported",
                f"{path}: e_type={elf_type} e_machine={machine}, want x86_64 shared object",
            )
        section_offset, section_entry_size = header[6], header[11]
        section_count, names_index = header[12], header[13]
        if section_entry_size != _ELF64_SECTION.size or section_count > MAX_ELF_SECTIONS:
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{path}: section table")
        if section_count == 0:
            section_zero = _ELF64_SECTION.unpack(self._read(section_offset, section_entry_size))
            section_count = section_zero[5]
        if section_count <= 0 or section_count > MAX_ELF_SECTIONS:
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{path}: section count")
        table_size = section_count * section_entry_size
        table = self._read(section_offset, table_size)
        raw_sections = [
            _ELF64_SECTION.unpack_from(table, index * section_entry_size)
            for index in range(section_count)
        ]
        if names_index == 0xFFFF:
            names_index = raw_sections[0][6]
        if names_index >= section_count:
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{path}: shstrndx")
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
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{self.path}: file bounds")
        with self.path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{self.path}: short read")
        return data

    def _section_bytes(self, row: tuple[int, ...], label: str) -> bytes:
        size = row[5]
        if size > MAX_STRING_TABLE_BYTES:
            raise NativeWheelRefusal(
                "overlay_native_elf_invalid", f"{self.path}: oversized {label}"
            )
        return self._read(row[4], size)

    def _bytes(self, section: _Section, label: str) -> bytes:
        if section.size > MAX_STRING_TABLE_BYTES:
            raise NativeWheelRefusal(
                "overlay_native_elf_invalid", f"{self.path}: oversized {label}"
            )
        return self._read(section.offset, section.size)

    def _string(self, table: bytes, offset: int, label: str) -> str:
        if offset < 0 or offset >= len(table):
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{self.path}: {label}")
        end = table.find(b"\0", offset)
        if end < 0:
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{self.path}: {label}")
        try:
            return table[offset:end].decode("utf-8")
        except UnicodeError as exc:
            raise NativeWheelRefusal(
                "overlay_native_elf_invalid", f"{self.path}: non-UTF-8 {label}"
            ) from exc

    def dynamic(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        dynamic = next((section for section in self.sections if section.kind == 6), None)
        if dynamic is None:
            return "", (), ()
        if dynamic.link >= len(self.sections) or dynamic.entry_size not in {0, 16}:
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{self.path}: dynamic table")
        strings = self._bytes(self.sections[dynamic.link], "dynamic strings")
        raw = self._bytes(dynamic, "dynamic table")
        if len(raw) % _ELF64_DYNAMIC.size:
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{self.path}: dynamic size")
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
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{self.path}: dynsym table")
        count = table.size // entry_size
        if table.size % entry_size or count > MAX_SYMBOLS:
            raise NativeWheelRefusal("overlay_native_elf_invalid", f"{self.path}: dynsym size")
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
    host_libraries = frozenset(_host_libraries())
    parsed: list[ELFMember] = []
    for member in members:
        path = (root / member).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise NativeWheelRefusal("overlay_native_member_invalid", member)
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
                    "overlay_native_cuda_images_absent",
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
            raise NativeWheelRefusal("overlay_native_tree_oversized", str(visited))
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as handle:
            magic = handle.read(4)
        if magic == _ELF_MAGIC:
            members.append(path.relative_to(resolved).as_posix())
            if len(members) > MAX_NATIVE_MEMBERS:
                raise NativeWheelRefusal("overlay_native_member_count", str(len(members)))
    if not members:
        raise NativeWheelRefusal("overlay_native_members_absent", str(root))
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
                "overlay_native_base_library_shadow", f"{path}: bundles {library}"
            )
        if library and library in host_libraries:
            raise NativeWheelRefusal(
                "overlay_native_base_library_shadow",
                f"{path}: {library} already belongs to the selected base",
            )
    for library in needed:
        if not library or "/" in library or "\\" in library or "$" in library:
            raise NativeWheelRefusal(
                "overlay_native_needed_invalid", f"{path}: DT_NEEDED={library!r}"
            )
    for value in runpaths:
        normalized = value.replace("${ORIGIN}", "$ORIGIN")
        if not normalized or not normalized.startswith("$ORIGIN") or "$" in normalized[7:]:
            raise NativeWheelRefusal("overlay_native_rpath_forbidden", f"{path}: RPATH={value!r}")
        suffix = normalized[7:]
        if suffix and not suffix.startswith("/"):
            raise NativeWheelRefusal("overlay_native_rpath_forbidden", f"{path}: RPATH={value!r}")
        target = Path(os.path.normpath(str(path.parent) + suffix))
        if not target.is_relative_to(root):
            raise NativeWheelRefusal(
                "overlay_native_rpath_forbidden", f"{path}: RPATH escapes wheel: {value!r}"
            )


def _cuda_images(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tool = shutil.which("cuobjdump")
    if tool is None:
        raise NativeWheelRefusal("overlay_native_gpu_inspector_unavailable", str(path))
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
                "overlay_native_gpu_inspection_failed", f"{path}: {type(exc).__name__}"
            ) from exc
        output.extend(result.stdout[: MAX_TOOL_OUTPUT + 1])
        if result.returncode != 0 or len(output) > MAX_TOOL_OUTPUT:
            raise NativeWheelRefusal(
                "overlay_native_gpu_inspection_failed", f"{path}: cuobjdump {option}"
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
    except OSError, NativeWheelRefusal:
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
        except OSError, NativeWheelRefusal:
            missing_libraries.add(library)
            continue
        loaded_symbols.update(defined)
        pending.extend(needed)
    if missing_libraries:
        raise NativeWheelRefusal(
            "overlay_native_dependency_unavailable", ",".join(sorted(missing_libraries))
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
        raise NativeWheelRefusal("overlay_native_symbol_unavailable", ",".join(unresolved[:16]))
    gpu_members = [item for item in inspection.members if item.cuda_sections]
    if gpu_members and gpu_sm is None:
        raise NativeWheelRefusal("overlay_native_gpu_unavailable", "CUDA wheel on a CPU host")
    if gpu_sm is not None:
        unsupported = [
            item.member
            for item in gpu_members
            if gpu_sm not in item.cubin_sms
            and not any(compute <= gpu_sm for compute in item.ptx_compute)
        ]
        if unsupported:
            raise NativeWheelRefusal(
                "overlay_native_gpu_unsupported",
                f"sm_{gpu_sm}: no matching cubin or forward PTX in {','.join(unsupported)}",
            )
    return {
        "gpu_sm": gpu_sm,
        "host_library_count": len(visited),
        "inspection_digest": canonical.digest(inspection.document()),
        "qualification": "static-host-compatible",
    }


def observe_seat(python: Path, environment: Mapping[str, str]) -> dict[str, object]:
    """Ask the selected base interpreter to report its actual installed/runtime facts."""

    try:
        result = subprocess.run(
            [str(python), "-I", "-m", "cozy_runtime.internal.native_seat_probe"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=dict(environment),
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativeWheelRefusal("overlay_native_seat_probe_failed", type(exc).__name__) from exc
    if (
        result.returncode != 0
        or len(result.stdout) > MAX_TOOL_OUTPUT
        or len(result.stderr) > MAX_TOOL_OUTPUT
    ):
        detail = result.stderr.decode("utf-8", "replace").strip()[:1000]
        raise NativeWheelRefusal(
            "overlay_native_seat_probe_failed", detail or f"exit {result.returncode}"
        )
    try:
        value = canonical.parse_canonical(result.stdout)
    except canonical.CanonicalError as exc:
        raise NativeWheelRefusal("overlay_native_seat_probe_invalid", str(exc)) from exc
    if not isinstance(value, dict) or set(value) != _SEAT_FIELDS:
        raise NativeWheelRefusal("overlay_native_seat_probe_invalid", "wrong fields")
    return value


def verify_seat(
    base: BaseSeat,
    observation: Mapping[str, object],
    *,
    gpu: GPUObservation | None,
) -> dict[str, object]:
    """Verify measured interpreter/OS/libc/Torch/CUDA/driver facts against one WHM."""

    profile = dict(base.compatibility_profile)
    target = dict(base.platform_target)
    if observation.get("implementation") != "cpython" or observation.get(
        "python_abi"
    ) != target.get("python_abi"):
        raise NativeWheelRefusal(
            "overlay_native_python_abi_mismatch",
            f"measured={observation.get('implementation')}:{observation.get('python_abi')} "
            f"required=cpython:{target.get('python_abi')}",
        )
    machine = observation.get("machine")
    measured_arch = "linux/amd64" if machine in {"amd64", "x86_64"} else ""
    if observation.get("system") != "linux" or measured_arch != target.get("os_arch"):
        raise NativeWheelRefusal(
            "overlay_native_os_cpu_mismatch",
            f"measured={observation.get('system')}/{machine} required={target.get('os_arch')}",
        )
    libc_match = re.fullmatch(r"glibc([0-9]+\.[0-9]+)", str(target.get("libc", "")))
    if observation.get("libc_name") != "glibc" or libc_match is None:
        raise NativeWheelRefusal(
            "overlay_native_libc_mismatch",
            f"measured={observation.get('libc_name')}{observation.get('libc_version')} "
            f"required={target.get('libc')}",
        )
    try:
        libc_ok = Version(str(observation.get("libc_version", ""))) >= Version(libc_match.group(1))
    except InvalidVersion as exc:
        raise NativeWheelRefusal("overlay_native_libc_mismatch", str(exc)) from exc
    if not libc_ok:
        raise NativeWheelRefusal(
            "overlay_native_libc_mismatch",
            f"measured=glibc{observation.get('libc_version')} minimum={target.get('libc')}",
        )
    rows = observation.get("distributions")
    measured_distributions: list[tuple[str, str]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"distribution", "version"}:
                raise NativeWheelRefusal(
                    "overlay_native_base_inventory_mismatch", "invalid measured distribution row"
                )
            name, version = row["distribution"], row["version"]
            if not isinstance(name, str) or not isinstance(version, str):
                raise NativeWheelRefusal(
                    "overlay_native_base_inventory_mismatch", "invalid measured distribution"
                )
            measured_distributions.append((name, version))
    if tuple(measured_distributions) != base.base_distributions:
        raise NativeWheelRefusal(
            "overlay_native_base_inventory_mismatch",
            f"measured={len(measured_distributions)} required={len(base.base_distributions)}",
        )
    torch_version = str(observation.get("torch_version", ""))
    try:
        torch_release_ok = (
            Version(torch_version).release[:3] == Version(profile["torch_release"]).release[:3]
        )
    except InvalidVersion as exc:
        raise NativeWheelRefusal("overlay_native_torch_mismatch", torch_version) from exc
    if not torch_release_ok:
        raise NativeWheelRefusal(
            "overlay_native_torch_mismatch",
            f"measured={torch_version} required={profile['torch_release']}",
        )
    build = profile["accelerator_build"]
    torch_cuda = str(observation.get("torch_cuda", ""))
    if build == "cpu":
        if torch_cuda or bool(observation.get("cuda_available")) or gpu is not None:
            raise NativeWheelRefusal(
                "overlay_native_accelerator_mismatch",
                f"CPU profile measured torch_cuda={torch_cuda!r} available="
                f"{observation.get('cuda_available')} gpu={gpu is not None}",
            )
    else:
        required_cuda = {"cu126": "12.6", "cu130": "13.0"}.get(build)
        if required_cuda is None:
            raise NativeWheelRefusal("overlay_native_accelerator_mismatch", build)
        if (
            torch_cuda != required_cuda
            or not bool(observation.get("cuda_available"))
            or gpu is None
        ):
            raise NativeWheelRefusal(
                "overlay_native_accelerator_mismatch",
                f"measured torch_cuda={torch_cuda!r} available="
                f"{observation.get('cuda_available')} gpu={gpu is not None}; "
                f"required={required_cuda}",
            )
        if Version(torch_cuda) < Version(base.minimum_cuda_version):
            raise NativeWheelRefusal(
                "overlay_native_cuda_minimum_mismatch",
                f"measured={torch_cuda} minimum={base.minimum_cuda_version}",
            )
        if Version(gpu["driver_version"]) < Version(base.minimum_driver_version):
            raise NativeWheelRefusal(
                "overlay_native_driver_minimum_mismatch",
                f"measured={gpu['driver_version']} minimum={base.minimum_driver_version}",
            )
    return dict(observation)


def qualify_operator(
    *,
    python: Path,
    overlay: Path,
    fixture: str,
    expected_result_digest: str,
    environment: Mapping[str, str],
    timeout_seconds: float = 60.0,
    max_address_space_bytes: int = 16 << 30,
) -> dict[str, object]:
    """Execute one banked no-argument fixture in a bounded disposable Python process."""

    if (
        not python.is_absolute()
        or not python.is_file()
        or not overlay.is_absolute()
        or not overlay.is_dir()
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", fixture)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_result_digest)
        or not 0 < timeout_seconds <= 600
        or not 256 << 20 <= max_address_space_bytes <= 128 << 30
    ):
        raise NativeWheelRefusal("overlay_native_operator_request_invalid", fixture)
    command = [
        str(python),
        "-I",
        "-m",
        "cozy_runtime.internal.native_operator_runner",
        str(overlay),
        fixture,
        str(int(timeout_seconds) + 1),
        str(max_address_space_bytes),
        str(MAX_TOOL_OUTPUT),
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
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
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _kill_process_group(process)
                process.wait()
                raise NativeWheelRefusal("overlay_native_operator_timeout", fixture) from exc
            finally:
                _kill_process_group(process)
            stdout.seek(0)
            stderr.seek(0)
            result_stdout = stdout.read(MAX_TOOL_OUTPUT + 1)
            result_stderr = stderr.read(MAX_TOOL_OUTPUT + 1)
    except NativeWheelRefusal:
        raise
    except OSError as exc:
        raise NativeWheelRefusal("overlay_native_operator_failed", type(exc).__name__) from exc
    if len(result_stdout) > MAX_TOOL_OUTPUT or len(result_stderr) > MAX_TOOL_OUTPUT:
        raise NativeWheelRefusal("overlay_native_operator_output_oversized", fixture)
    if returncode != 0:
        detail = result_stderr.decode("utf-8", "replace").strip()[:1000]
        code = (
            "overlay_native_no_kernel_image"
            if "no kernel image" in detail.lower()
            else "overlay_native_operator_failed"
        )
        raise NativeWheelRefusal(code, f"{fixture}: {detail or f'exit {returncode}'}")
    try:
        value = canonical.parse_canonical(result_stdout)
    except canonical.CanonicalError as exc:
        raise NativeWheelRefusal(
            "overlay_native_operator_result_invalid", f"{fixture}: {exc}"
        ) from exc
    result_digest = canonical.digest(value)
    if result_digest != expected_result_digest:
        raise NativeWheelRefusal(
            "overlay_native_operator_numerical_mismatch",
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
        raise NativeWheelRefusal("overlay_native_gpu_request_invalid", str(device_index))
    tool = shutil.which("nvidia-smi")
    if tool is None:
        raise NativeWheelRefusal("overlay_native_gpu_unavailable", "nvidia-smi is absent")
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
        raise NativeWheelRefusal("overlay_native_gpu_unavailable", type(exc).__name__) from exc
    output = result.stdout.decode("utf-8", "replace").strip()
    try:
        rows = list(csv.reader(output.splitlines()))
    except csv.Error:
        rows = []
    if result.returncode != 0 or len(rows) != 1 or len(rows[0]) != 3:
        raise NativeWheelRefusal("overlay_native_gpu_unavailable", output[:500] or "probe failed")
    name, driver, capability = (item.strip() for item in rows[0])
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", capability)
    if not name or re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", driver) is None or match is None:
        raise NativeWheelRefusal("overlay_native_gpu_unavailable", output[:500])
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


@lru_cache(maxsize=1)
def _host_library_rows() -> tuple[tuple[str, str], ...]:
    libraries: dict[str, Path] = {}
    tool = shutil.which("ldconfig")
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
        except OSError, subprocess.TimeoutExpired:
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
                    "overlay_native_host_inventory_oversized", str(MAX_HOST_LIBRARIES)
                )
            if path.is_file() and not path.is_symlink():
                libraries.setdefault(path.name, path)
    executable = Path(sys.executable)
    if executable.is_file():
        libraries.setdefault(executable.name, executable)
    return tuple(sorted((name, str(path)) for name, path in libraries.items()))
