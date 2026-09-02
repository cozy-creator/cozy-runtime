"""_validate_result over the real walker: an Optional result field may be None.

The recorded incident (se-022 verification, 2026-09-01): cozy-video-assembly's
`master_audio_digest: str | None` returned None on the segments path and the
validator refused `result_schema: ... is NoneType, declared str` — the walker
unwraps Optional into (annotation, optional) and the validator read only the
annotation. Both arms below drive the REAL walk() and _validate_result.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import OutputError
from cozy_runtime.author._invoke import _validate_result
from cozy_runtime.author._services import Attempt
from cozy_runtime.author._signature import Surface


class _Result(msgspec.Struct):
    digest: str
    master_audio_digest: str | None


def _surface() -> Surface:
    return Surface(
        name="assemble",
        kind="entrypoint",
        fn=lambda: None,
        is_async=False,
        params=(),
        payload_type=object,
        result_type=_Result,
        capabilities=frozenset(),
    )


def _attempt(tmp_path: Path) -> Attempt:
    return Attempt(request_id="req-test", spool=tmp_path)


def test_an_optional_result_field_may_be_none(tmp_path: Path) -> None:
    result = _Result(digest="sha256:ab", master_audio_digest=None)
    _validate_result(_surface(), result, _attempt(tmp_path))


def test_a_required_result_field_still_refuses_none(tmp_path: Path) -> None:
    bad = _Result(digest=None, master_audio_digest="x")  # type: ignore[arg-type]
    with pytest.raises(OutputError) as caught:
        _validate_result(_surface(), bad, _attempt(tmp_path))
    assert caught.value.code == "result_schema"
