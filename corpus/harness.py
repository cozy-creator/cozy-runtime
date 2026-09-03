"""The adversarial-corpus harness: one hostile construct per case, three legal verdicts.

Every case is a REAL package module — real torch, real `nn.Module`s, a real `Model`
subclass whose real `load()` the derivation executes. Nothing is mocked, and the scoreboard
is produced by running them. Decision 691 allows test suites again (it repeals #160/#614);
what it does not allow is a suite that agrees with a double, so this stays as it is — few,
and real.

The verdicts:

* **derived-correct** — the derived census equals a census of the SAME construction
  performed for real on CPU. This is the only verdict that requires the ground-truth diff,
  and the diff is the point: a refusal is visible, but a wrong derivation is not.
* **typed-refusal** — the harness named the construct and refused. The package changes.
* **typed-harness-gap** — the harness named ITSELF as unable to see the construct. This
  file changes.
* **silent-wrong** — the derivation succeeded and disagrees with reality. The red arm of
  red arms; any occurrence fails the corpus.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from cozy_runtime.author import Artifact, Config, Model, TensorSpec
from cozy_runtime.internal.derive import RichCensus, component_members, walk

Verdict = Literal["derived-correct", "typed-refusal", "typed-harness-gap", "silent-wrong"]


#: Every case builds one of these, so the component table is the same shape a real
#: diffusers pipeline presents and the loader's own member walk finds it unchanged.
class Pipeline:
    def __init__(self, components: Mapping[str, object]) -> None:
        self.components = dict(components)


@dataclass(frozen=True)
class Case:
    """One hostile construct."""

    name: str
    construct: str
    build: Callable[[Config], Pipeline]
    expect: Verdict
    names: str = ""
    """The construct a refusal MUST name. Asserting only "something refused" passes for the
    wrong reason — cr-003 learned that when a red arm refused over a struct field order."""
    config: Mapping[str, object] = field(default_factory=dict)
    component_use: Mapping[str, Sequence[str]] = field(default_factory=dict)
    ground_truth: bool = True
    """False when the construct cannot be built for real either (a CUDA-only case on a
    fenced box) — the diff is then skipped and the case says so."""
    note: str = ""


def model_for(case: Case) -> Model[Pipeline]:
    """A real `Model` subclass over the case's factory — the object the harness loads."""

    class CaseModel(Model[Pipeline]):
        pipe: Pipeline

        def load(self, loader: Any) -> None:
            self.pipe = loader.construct(Pipeline, factory=case.build)

    CaseModel.__qualname__ = f"corpus.{case.name}"
    return CaseModel()


def ground_truth_census(case: Case) -> RichCensus:
    """Construct the SAME factory for real, on CPU, and census it with the SAME walk."""
    built = case.build(Config(dict(case.config)))
    return walk(component_members([built]))


def weights_for(case: Case, truth: RichCensus) -> Artifact:
    """A checkpoint that supplies exactly what the real construction demands.

    Built from ground truth on purpose: the corpus is testing DERIVATION, so the match must
    not be the thing that fails. Match-direction arms are separate (`scripts/verify.py`).
    """
    tensor_schema = {
        t.key: TensorSpec(t.shape, t.dtype or "bf16", nbytes=_nbytes(t.shape, t.dtype))
        for t in truth.logical
    }
    return Artifact(
        snapshot=f"corpus/{case.name}@1",
        tensor_schema=tensor_schema,
        config=Config(dict(case.config)),
        custody="canonical",
    )


def _nbytes(shape: Sequence[int], dtype: str | None) -> int:
    width = {"f32": 4, "i64": 8, "i32": 4, "u8": 1, "bool": 1}.get(dtype or "", 2)
    total = width
    for dim in shape:
        total *= dim
    return total


def diff(derived: Sequence[Any], truth: Sequence[Any]) -> list[str]:
    """Where a derived census and a real one disagree. Empty means derived-correct."""
    left = {row.key: row for row in derived}
    right = {row.key: row for row in truth}
    out = [f"derive-only {k}" for k in sorted(set(left) - set(right))]
    out += [f"real-only {k}" for k in sorted(set(right) - set(left))]
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        for attribute in ("shape", "kind", "module_class", "alias_group", "dtype"):
            x, y = getattr(a, attribute, None), getattr(b, attribute, None)
            if x != y:
                out.append(f"{key}.{attribute}: derive={x!r} real={y!r}")
    return out


# ------------------------------------------------------------------ shared fixtures


def linear_block(width: int, *, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """An ordinary transformer-ish leaf set — the SHAPE every case varies one thing from."""
    block = torch.nn.Sequential(
        torch.nn.Linear(width, width),
        torch.nn.LayerNorm(width),
        torch.nn.Linear(width, width * 2),
    )
    return block.to(dtype)
