"""The code a native module runs on one GPU architecture: the cubins its fat binary holds for that
`sm`, read from the ELF file itself (no CUDA, no toolkit). A kernel verdict binds to this, so a
change to one family's kernels never voids another family's facts. Imports nothing but stdlib:
the release gate reads wheels with it.
"""

from __future__ import annotations

import base64
import hashlib
import struct

#: `fatBinaryHeader.magic`, and the entry kind of a cubin (1 is PTX).
MAGIC, CUBIN = 0xBA55ED50, 2


def _section(elf: bytes, wanted: bytes) -> bytes:
    """An ELF64 little-endian file's section by name, or b"" (a PE or Mach-O module has none)."""
    if elf[:4] != b"\x7fELF" or elf[4] != 2:
        return b""
    (offset,) = struct.unpack_from("<Q", elf, 0x28)
    size, count, names = struct.unpack_from("<HHH", elf, 0x3A)
    headers = [struct.unpack_from("<IIQQQQ", elf, offset + i * size) for i in range(count)]
    table = headers[names][4]
    for name, _, _, _, start, length in headers:
        if elf[table + name : table + name + len(wanted) + 1] == wanted + b"\0":
            return elf[start : start + length]
    return b""


def cubins(module: bytes, sm: int) -> str:
    """The sha256 (RECORD's urlsafe base64 form) of every cubin entry, header and payload, built
    for `sm`, in file order; "" where the module carries none."""
    section, digest, found, at = _section(module, b".nv_fatbin"), hashlib.sha256(), False, 0
    while at + 16 <= len(section):
        magic, _, header, size = struct.unpack_from("<IHHQ", section, at)
        if magic != MAGIC:  # alignment padding between one translation unit's binary and the next
            at += 8
            continue
        entry, end = at + header, at + header + size
        while entry < end:
            kind, _, length, padded = struct.unpack_from("<HHIQ", section, entry)
            (arch,) = struct.unpack_from("<I", section, entry + 28)
            if kind == CUBIN and arch == sm:
                digest.update(section[entry : entry + length + padded])
                found = True
            entry += length + padded
        at = end
    return (
        "sha256:" + base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode() if found else ""
    )
