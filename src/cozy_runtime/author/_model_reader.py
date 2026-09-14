"""Attempt-scoped read access to exact model inputs, without a model output."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tensorfs.derived import SourceCapability, SourceInspection, Tensor

from cozy_runtime.author._errors import Cancelled, CapabilityError
from cozy_runtime.author._model import Model
from cozy_runtime.author._weights import MAX_WEIGHTS_SOURCE_READ

ReadValues = Callable[[SourceCapability, SourceInspection, str, str, int, memoryview], None]


class WeightsView:
    """A live read lease over one model's canonical geometry and encoded tensors.

    Identities are opaque equality facts, never object or location capabilities.
    Logical reads fill caller-owned float32 buffers in bounded element ranges.
    Quantizer-time statistics, including saturation counts, are not inferred here.
    """

    def __init__(self, reader: WeightsReader, host: SourceCapability) -> None:
        self._reader, self._host, self._closed = reader, host, False
        self._structure = host.inspect()

    def _check(self) -> None:
        self._reader._check()
        if self._closed:
            raise CapabilityError("model view is closed", code="weights_reader_closed")
        try:
            self._host.check()
        except BaseException:
            self._reader._check()
            raise

    def __enter__(self) -> WeightsView:
        self._check()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._host.close()

    def components(self) -> tuple[str, ...]:
        self._check()
        return tuple(self._structure.components)

    def keys(self, component: str) -> tuple[str, ...]:
        self._check()
        rows = tuple(self._structure.components.get(component, {}))
        if not rows:
            raise CapabilityError("model component is absent", code="weights_reader_tensor")
        return rows

    def tensor(self, component: str, key: str) -> Tensor:
        self._check()
        try:
            return self._structure.components[component][key]
        except KeyError:
            raise CapabilityError("model tensor is absent", code="weights_reader_tensor") from None

    def identity(self, component: str, key: str) -> str:
        self.tensor(component, key)
        return self._structure.identities[component, key]

    def read_part_into(self, component: str, key: str, role: str, offset: int, into: Any) -> None:
        """Read an encoded role's byte range, at most MAX_WEIGHTS_SOURCE_READ bytes."""
        self._check()
        self._host.read_part_into(component, key, role, offset, into)
        self._check()

    def read_values_into(self, component: str, key: str, offset: int, into: Any) -> None:
        """Decode a flat logical element range into a contiguous writable float32 buffer.

        The destination bounds each read; callers can fill a larger tensor in chunks.
        This performs no model inference and changes no source or output artifact.
        """
        self._check()
        target = _buffer(into, logical=True)
        _offset(offset)
        if self._reader._values is None:
            raise CapabilityError(
                "logical model decoder is unavailable", code="weights_reader_decoder_unavailable"
            )
        self._reader._values(self._host, self._structure, component, key, offset, target)
        self._check()


def _offset(value: int) -> None:
    if type(value) is not int or not 0 <= value <= (1 << 53) - 1:
        raise CapabilityError("model read offset is invalid", code="weights_reader_bounds")


def _buffer(value: Any, *, logical: bool) -> memoryview:
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
        or (logical and (target.itemsize != 4 or target.format not in {"f", "=f", "<f"}))
    ):
        raise CapabilityError(
            "model read buffer is readonly, noncontiguous, oversized or not float32",
            code="weights_reader_buffer",
        )
    return target.cast("B")


class WeightsReader:
    """Job service granting read-only views of this attempt's exact model inputs."""

    def __init__(
        self,
        attempt: Any,
        models: Mapping[str, Model[object]],
        open_view: Callable[[Model[object]], SourceCapability] | None,
        cancel: Callable[[], bool] = lambda: False,
        read_values: ReadValues | None = None,
    ) -> None:
        self._attempt, self._models = attempt, dict(models)
        self._open, self._cancel, self._values = open_view, cancel, read_values

    def _check(self) -> None:
        self._attempt.check_open("WeightsReader")
        if self._cancel():
            raise Cancelled("model read was canceled")

    def open(self, source: Model[object]) -> WeightsView:
        """Bind an already granted Model; a bare manifest or ModelArtifact is not a grant.

        Ordinary invocable calls accept ModelArtifacts for their typed Model arguments;
        the owner validates provenance and installs this exact input before execution.
        """
        self._check()
        if not any(source is model for model in self._models.values()):
            raise CapabilityError("model input is not granted", code="weights_source_ungranted")
        if self._open is None:
            raise CapabilityError("model reader is unavailable", code="weights_reader_unavailable")
        host = self._open(source)
        try:
            self._check()
            return WeightsView(self, host)
        except BaseException:
            host.close()
            raise
