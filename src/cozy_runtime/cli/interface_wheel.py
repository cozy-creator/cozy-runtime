"""Creator's bounded pure interface generation seat; imports no target package."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli.io import CliError, Result
from cozy_runtime.internal import canonical, interface_wheel
from cozy_runtime.internal.exits import Exit


def run(args: list[str]) -> Result:
    if args:
        raise CliError(
            "usage",
            "interface-wheel accepts its exact generation input on stdin",
            "pipe one canonical JSON object to cozy-runtime --json interface-wheel",
            Exit.usage,
        )
    try:
        raw = sys.stdin.buffer.read((64 << 10) + 1)
        if len(raw) > 64 << 10:
            raise ValueError("interface generation input exceeds 64 KiB")
        request = canonical.parse(raw.rstrip(b"\n"))
        fields = {
            "interface_path",
            "distribution",
            "version",
            "output_directory",
            "implementation_wheel",
            "implementation_wheel_digest",
        }
        if not isinstance(request, dict) or not all(
            isinstance(request.get(name), str) and request[name] for name in fields
        ):
            raise ValueError("interface generation input is missing a field")
        path = Path(request["interface_path"])
        if not path.is_file():
            raise ValueError("interface input is not a regular file")
        with path.open("rb") as stream:
            interface = stream.read(canonical.DOC_MAX_BYTES + 1)
        if len(interface) > canonical.DOC_MAX_BYTES:
            raise ValueError("interface input exceeds its document bound")
        source = Path(request["implementation_wheel"])
        with source.open("rb") as stream:
            implementation = stream.read(interface_wheel.MAX_IMPLEMENTATION_WHEEL + 1)
        if (
            len(implementation) > interface_wheel.MAX_IMPLEMENTATION_WHEEL
            or "sha256:" + hashlib.sha256(implementation).hexdigest()
            != request["implementation_wheel_digest"]
        ):
            raise ValueError("implementation wheel changed or exceeds its bound")
        filename, wheel = interface_wheel.build(
            interface,
            distribution=request["distribution"],
            version=request["version"],
            implementation_wheel=implementation,
            implementation_filename=source.name,
        )
        output = Path(request["output_directory"]).resolve()
        output.mkdir(parents=True, exist_ok=True)
        target = output / filename
        if target.exists():
            if not target.is_file() or target.read_bytes() != wheel:
                raise ValueError("interface wheel output exists with different immutable bytes")
        else:
            with target.open("xb") as stream:
                stream.write(wheel)
                stream.flush()
                os.fsync(stream.fileno())
        return Result(
            document={
                "filename": filename,
                "path": str(target),
                "digest": "sha256:" + hashlib.sha256(wheel).hexdigest(),
                "length": len(wheel),
                "generator_abi": interface_wheel.GENERATOR_ABI,
            }
        )
    except (OSError, ValueError, ConformanceError) as exc:
        raise CliError(
            "interface_generation_refused",
            str(exc),
            "supply one exact described interface and retained implementation revision",
            Exit.validation,
        ) from exc
