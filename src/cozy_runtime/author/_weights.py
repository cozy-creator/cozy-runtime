"""Model-output declarations, receipts, and `Weights[M]`: raw checkpoint data, never constructed."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args, get_origin

from cozy_runtime.author._errors import CapabilityError, ConformanceError
from cozy_runtime.author._model import Model

if TYPE_CHECKING:
    from cozy_runtime.weights_store import Checkpoint, TensorMetadata

MAX_WEIGHTS_OUTPUTS = 16
MAX_WEIGHTS_SOURCE_READ = 64 << 20
MAX_WEIGHTS_NEW_BYTES = (1 << 53) - 1
_OUTPUT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_REF = re.compile(r"^sha256:[0-9a-f]{64}$")

ReadValues = Callable[["Checkpoint", str, str, int, memoryview], None]


@dataclass(frozen=True, slots=True)
class WeightsOutput:
    """One package-interface model output and its new-byte ceiling."""

    name: str
    max_new_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _OUTPUT_NAME.fullmatch(self.name) is None:
            raise ConformanceError(
                f"weights output {self.name!r} is not a portable name",
                code="weights_output_name",
                fields=[self.name],
            )
        if (
            not isinstance(self.max_new_bytes, int)
            or isinstance(self.max_new_bytes, bool)
            or not 0 <= self.max_new_bytes <= MAX_WEIGHTS_NEW_BYTES
        ):
            raise ConformanceError(
                f"weights output {self.name} max_new_bytes must be in 0..={MAX_WEIGHTS_NEW_BYTES}",
                code="weights_output_bound",
                fields=[self.name],
            )


class Weights[M: Model[Any]]:
    """The raw checkpoint of `M`'s layout (bare `Weights`: any layout). Never constructed.

    Read it here, or pass it to `ctx.tensorfs_source` and `derive.quantize_artifact`.
    Reads are bound to the attempt that received it.
    """

    __slots__ = ("_open", "_source", "_values", "checkpoint_ref")

    def __init__(self, checkpoint_ref: str) -> None:
        if _REF.fullmatch(checkpoint_ref) is None:
            raise ConformanceError(
                "Weights requires one checkpoint's header ref (sha256:<hex>)",
                code="weights_ref_invalid",
            )
        self.checkpoint_ref = checkpoint_ref
        self._open: Callable[[], Checkpoint] | None = None
        self._values: ReadValues | None = None
        self._source: Checkpoint | None = None

    def _bind(self, open: Callable[[], Checkpoint] | None, values: ReadValues | None) -> None:
        """Runtime: this attempt's source access; `_bind(None, None)` ends it."""
        self._open, self._values, self._source = open, values, None

    def _view(self) -> Checkpoint:
        if self._source is None:
            if self._open is None:
                raise CapabilityError(
                    "these weights are not bound to a running attempt",
                    code="weights_source_ungranted",
                )
            self._source = self._open()
        return self._source

    def components(self) -> tuple[str, ...]:
        """The components that hold tensors, in header order."""
        return tuple(name for name, held in self._view().components.items() if held.tensors)

    def keys(self, component: str) -> tuple[str, ...]:
        """The component's tensors in construction order."""
        held = self._view().components.get(component)
        if held is None or not held.tensors:
            raise CapabilityError("model component is absent", code="weights_reader_tensor")
        return tuple(held.tensors)

    def tensor(self, component: str, key: str) -> TensorMetadata:
        """The tensor's logical dtype, shape and encoding, without physical locations."""
        try:
            return self._view().components[component].tensors[key]
        except KeyError:
            raise CapabilityError("model tensor is absent", code="weights_reader_tensor") from None

    def read_into(self, component: str, key: str, role: str, offset: int, into: Any) -> None:
        """Stored bytes of one part (`""`: a plain tensor's), at most MAX_WEIGHTS_SOURCE_READ
        per call."""
        from cozy_runtime.weights_store import read_into

        read_into(self._view(), component, key, role, offset, into)

    def read_values_into(self, component: str, key: str, offset: int, into: Any) -> None:
        """Decode a flat logical element range into a contiguous writable float32 buffer."""
        source = self._view()
        if type(offset) is not int or not 0 <= offset <= MAX_WEIGHTS_NEW_BYTES:
            raise CapabilityError("model read offset is invalid", code="weights_reader_bounds")
        if self._values is None:
            raise CapabilityError(
                "logical model decoder is unavailable", code="weights_reader_decoder_unavailable"
            )
        self._values(source, component, key, offset, _float32(into))


def _float32(value: Any) -> memoryview:
    try:
        target = memoryview(value)
    except TypeError:
        raise CapabilityError(
            "model read needs a writable buffer", code="weights_reader_buffer"
        ) from None
    if (
        target.readonly
        or not target.c_contiguous
        or not 0 < target.nbytes <= MAX_WEIGHTS_SOURCE_READ
        or target.itemsize != 4
        or target.format not in {"f", "=f", "<f"}
    ):
        raise CapabilityError(
            "model read buffer is readonly, noncontiguous, oversized or not float32",
            code="weights_reader_buffer",
        )
    return target.cast("B")


def weights_class(annotation: object) -> type[Model[Any]] | None:
    """`M` of a `Weights[M]` annotation (`Model` for `Weights` or `Weights[Any]`), else None."""
    if annotation is not Weights and get_origin(annotation) is not Weights:
        return None
    args = get_args(annotation)
    cls = Model if not args or args[0] is Any else get_origin(args[0]) or args[0]
    if not (isinstance(cls, type) and issubclass(cls, Model)):
        raise ConformanceError(f"Weights[{cls!r}]: M must be a Model class", code="weights_class")
    return cls
