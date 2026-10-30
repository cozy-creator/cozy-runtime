"""Portable exact model references. Identity is data; the owner grants custody."""

from __future__ import annotations

import math
import re

import msgspec

from cozy_runtime.author._errors import ConformanceError

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_REQUEST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_SLOT = re.compile(r"[a-z][a-z0-9_-]{0,63}")


class ObjectRef(msgspec.Struct, frozen=True):
    digest: str
    length: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.digest, str)
            or _DIGEST.fullmatch(self.digest) is None
            or type(self.length) is not int
            or not 0 < self.length <= (1 << 53) - 1
        ):
            raise ConformanceError(
                "artifact reference requires an exact digest and positive length",
                code="artifact_reference",
            )


class ModelArtifact(msgspec.Struct, frozen=True):
    """A produced model: the checkpoint (its header ref) an output committed, the request and
    output that produced it, and the derivation's replay key (`transaction`).

    Reusing it never rewrites the producer. Neither this value nor its digest grants access
    by itself.
    """

    producer_request_id: str
    output_slot: str
    checkpoint: str
    transaction: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.producer_request_id, str)
            or _REQUEST.fullmatch(self.producer_request_id) is None
            or not isinstance(self.output_slot, str)
            or _SLOT.fullmatch(self.output_slot) is None
            or not isinstance(self.checkpoint, str)
            or _DIGEST.fullmatch(self.checkpoint) is None
            or not isinstance(self.transaction, str)
            or _DIGEST.fullmatch(self.transaction) is None
        ):
            raise ConformanceError(
                "model artifact requires an exact checkpoint, producer, output and transaction",
                code="model_artifact",
            )


class Lora(msgspec.Struct, frozen=True, omit_defaults=True):
    """A LoRA, named as `cozy run --lora` names one: `[org/]name[@release]`,
    `name@sha256:<commit>`, `local/<alias>`, `hf://…` or `civitai://…`."""

    ref: str
    strength: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ref, str)
            or not self.ref.strip()
            or not isinstance(self.strength, (int, float))
            or isinstance(self.strength, bool)
            or not math.isfinite(self.strength)
        ):
            raise ConformanceError("a LoRA needs a ref and a finite strength", code="model_choice")


class ModelChoice(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """A call's model for one slot of its callee.

    `ref` names the model as `cozy run model.<slot>=` does, or is a ModelArtifact; it replaces the
    slot's model and its LoRAs. Without it the call keeps the model its parent runs for the slot,
    with that model's LoRAs (a request's `--lora`), and `lora` stacks after them: the same LoRA
    again takes the new strength, and strength 0 drops it. The machine resolves each name under
    the run's access, as it resolves the run's own.
    """

    ref: str | ModelArtifact | None = None
    lora: tuple[Lora, ...] = ()

    def __post_init__(self) -> None:
        if not (
            self.ref is None
            or isinstance(self.ref, ModelArtifact)
            or (isinstance(self.ref, str) and self.ref.strip())
        ) or not (
            isinstance(self.lora, (tuple, list)) and all(isinstance(x, Lora) for x in self.lora)
        ):
            raise ConformanceError(
                "a model choice takes a ref (a model name or ModelArtifact) and Lora values",
                code="model_choice",
            )
