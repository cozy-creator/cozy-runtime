"""Request-start selection among an author's supported numerical decode modes."""

from collections.abc import Callable
from typing import cast

from cozy_runtime.author._errors import CapabilityError


def select_decode_mode(
    module: object, candidates: tuple[int, ...], probe: Callable[[int], None]
) -> int:
    """Measure legal decode modes before request state or random inputs are created.

    Candidates are output-pixel tile sides, largest first, optionally preceded by
    zero for the author's default mode. The probe executes the actual managed
    decode with disposable inputs, releases its output, and restores mode attrs
    and dtype on every exit. Runtime owns admission and capacity refusals.

    The returned mode must remain fixed throughout this request and its retries.
    """
    choose = getattr(module, "_cozy_select_decode_mode", None)
    if not callable(choose):
        raise CapabilityError(
            "decode mode selection requires the current Runtime memory planner",
            code="decode_mode_unavailable",
        )
    return cast(Callable[[tuple[int, ...], Callable[[int], None]], int], choose)(candidates, probe)
