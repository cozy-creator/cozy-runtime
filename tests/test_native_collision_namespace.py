"""Collision is a linker-namespace fact, not a site-packages coincidence.

Live incident (se-022 verification, 2026-09-01): the host runtime tool env
carried hf_xet.abi3.so, so any package whose venv legitimately contained the
same extension refused `package_environment_native_base_library_shadow:
already belongs to the selected base`. An isolated venv's extension module is
imported by path and shadows nothing; only ldconfig names are process-global.
"""

from __future__ import annotations

from pathlib import Path

from cozy_runtime.internal.native_wheel import (
    NativeWheelRefusal,
    _linker_library_rows,
    _require_safe_libraries,
)


def _refusal(library: str) -> NativeWheelRefusal | None:
    try:
        _require_safe_libraries(
            root=Path("."),
            path=Path(library),
            soname="",
            needed=(),
            runpaths=(),
            host_libraries=frozenset(name for name, _ in _linker_library_rows()),
        )
    except NativeWheelRefusal as caught:
        return caught
    return None


def test_a_python_extension_name_is_not_a_linker_collision() -> None:
    assert _refusal("hf_xet.abi3.so") is None


def test_a_linker_namespace_name_still_refuses() -> None:
    caught = _refusal("libssl.so.3")
    assert caught is not None
    assert caught.code == "package_environment_native_base_library_shadow"
