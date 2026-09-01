"""Child half of one registered native-operator qualification fixture."""

from __future__ import annotations

import importlib
import resource
import sys

from cozy_runtime.internal import canonical


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 5:
        print(
            "native fixture runner requires "
            "MODULE:CALLABLE CPU_SECONDS AS_BYTES OUTPUT_BYTES REQUEST",
            file=sys.stderr,
        )
        return 2
    fixture, cpu_raw, address_raw, output_raw, request = args
    module_name, separator, callable_name = fixture.partition(":")
    if not separator or not module_name or not callable_name or "." in callable_name:
        print("invalid native fixture name", file=sys.stderr)
        return 2
    try:
        cpu_seconds = int(cpu_raw)
        address_bytes = int(address_raw)
        output_bytes = int(output_raw)
        if not 1 <= cpu_seconds <= 601 or not 256 << 20 <= address_bytes <= 128 << 30:
            raise ValueError
        if not 1 << 20 <= output_bytes <= 16 << 20:
            raise ValueError
    except ValueError:
        print("invalid native fixture resource limits", file=sys.stderr)
        return 2
    # Tensorhub's networkless VM/cgroup remains the qualification-seat authority. These local
    # limits bound the child even inside that seat and are imposed before package import.
    _cap(resource.RLIMIT_CORE, 0)
    _cap(resource.RLIMIT_CPU, cpu_seconds)
    _cap(resource.RLIMIT_AS, address_bytes)
    _cap(resource.RLIMIT_FSIZE, output_bytes)
    _cap(resource.RLIMIT_NOFILE, 256)
    try:
        candidate = getattr(importlib.import_module(module_name), callable_name)
        if not callable(candidate):
            raise TypeError("registered fixture is not callable")
        result = candidate(request) if request else candidate()
        sys.stdout.buffer.write(canonical.write(result))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def _cap(kind: int, wanted: int) -> None:
    _soft, hard = resource.getrlimit(kind)
    limit = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    resource.setrlimit(kind, (limit, limit))


if __name__ == "__main__":
    raise SystemExit(main())
