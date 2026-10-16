"""The GPU's row (precision-tiers §2.1): its MMA family and native 8- and 4-bit formats, from its
compute capability (`sm`, e.g. 90). Hardware fact, not policy: every kernel choice reads the row,
never a card name or a list of cards.

A cc no row names inherits (§2.6): it tries the newest family whose build works for its own
target (tcgen05, then wgmma; mma.sync always runs), learned by the 8-bit op's build
(`int8_linear`); its columns are int8, fp8 from 8.9 and fp4 from 10.0 (not on mma.sync), and
each kernel is verified on it before it serves. Every run on it says so once (`inherited`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

MMA_SYNC, WGMMA, TCGEN05 = "mma.sync", "wgmma", "tcgen05"
#: Oldest first; the lowest cc each family can be.
FAMILIES = ((MMA_SYNC, 80), (WGMMA, 90), (TCGEN05, 100))


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    family: str  # "" for no tensor-core family (CPU, <8.0)
    int8: bool = False
    fp8: bool = False
    fp4: bool = False
    inherited: bool = False

    def formats(self) -> frozenset[str]:
        return frozenset(
            f for f, on in (("int8", self.int8), ("fp8", self.fp8), ("fp4", self.fp4)) if on
        )


NONE = Row("none", "")
_AMPERE = Row("Ampere", MMA_SYNC, int8=True)
_RTX = Row("Blackwell RTX", MMA_SYNC, int8=True, fp8=True, fp4=True)  # block-scaled mma.sync
ROWS: dict[int, Row] = {
    80: _AMPERE,
    86: _AMPERE,
    87: _AMPERE,
    89: Row("Ada", MMA_SYNC, int8=True, fp8=True),
    90: Row("Hopper", WGMMA, int8=True, fp8=True),
    100: Row("Blackwell", TCGEN05, int8=True, fp8=True, fp4=True),
    103: Row("Blackwell Ultra", TCGEN05, fp8=True, fp4=True),
    120: _RTX,
    121: _RTX,
}
#: cc -> the family an unnamed card's 8-bit op built for (`learn`)
_LEARNED: dict[int, str] = {}


def families(sm: int) -> tuple[str, ...]:
    """The families a card tries, newest first: its row's, or every one its cc could hold."""
    if sm in ROWS:
        return (ROWS[sm].family,)
    return tuple(family for family, floor in reversed(FAMILIES) if sm >= floor)


def row(sm: int, family: str = "") -> Row:
    """The card's row; an unnamed cc's inherited one, as `family` (else as learned, else its
    newest candidate)."""
    if sm in ROWS:
        return ROWS[sm]
    tried = families(sm)
    if not tried:
        return NONE
    chosen = family or _LEARNED.get(sm, tried[0])
    fp4 = sm >= 100 and chosen != MMA_SYNC  # an inherited mma.sync is not block-scaled
    return Row(f"sm_{sm}", chosen, int8=True, fp8=sm >= 89, fp4=fp4, inherited=True)


def learn(sm: int, family: str) -> None:
    _LEARNED[sm] = family


def common(rows: Iterable[Row]) -> Row:
    """A set of GPUs: each column ANDed, the oldest family."""
    found = list(rows)
    order = [family for family, _ in FAMILIES]
    families_ = [r.family for r in found]
    family = "" if "" in families_ or not found else min(families_, key=order.index)
    return replace(
        found[0] if found else NONE,
        name="+".join(sorted({r.name for r in found})),
        family=family,
        int8=all(r.int8 for r in found),
        fp8=all(r.fp8 for r in found),
        fp4=all(r.fp4 for r in found),
        inherited=any(r.inherited for r in found),
    )


def inherited(sm: int, name: str) -> str:
    """The run warning for a card no row names; empty for a named one."""
    found = row(sm)
    if not found.inherited:
        return ""
    return (
        f"sm_{sm} ({name}) has no row; served as {found.family} with "
        f"{' '.join(sorted(found.formats()))}, each kernel verified before it serves"
    )
