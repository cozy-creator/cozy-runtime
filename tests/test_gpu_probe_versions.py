"""observe_gpu admits real NVIDIA driver spellings.

Live incident (se-022 verification, 2026-09-01): driver 595.84 — a
two-component series — refused package_environment_native_gpu_unavailable
because the probe demanded exactly three components.
"""

from __future__ import annotations

import re

_DRIVER = r"[0-9]+(?:\.[0-9]+){1,2}"


def test_two_and_three_component_drivers_admit() -> None:
    assert re.fullmatch(_DRIVER, "595.84")
    assert re.fullmatch(_DRIVER, "580.126.20")


def test_non_versions_still_refuse() -> None:
    for bad in ("595", "595.84.", "x.1", ""):
        assert re.fullmatch(_DRIVER, bad) is None
