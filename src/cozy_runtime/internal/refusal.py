"""Typed structural refusals shared by Runtime worker boot and materialization."""

from __future__ import annotations


class LaunchRefusal(Exception):
    """A fail-closed worker launch or artifact-join verdict with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
