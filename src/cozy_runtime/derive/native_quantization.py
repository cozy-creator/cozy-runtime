"""One request-owned native Writer conversion; model policy stays in the derivation."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, TimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import tensorfs

from cozy_runtime.author import Context, Telemetry, UnsupportedInput
from cozy_runtime.author._loader import TENSORFS_NAMES
from cozy_runtime.author._prepared import Progress
from cozy_runtime.weights_store import Checkpoint, Output

from .quantization import (
    _ENCODINGS,
    ArtifactQuantizationPlan,
    QuantizationStats,
    QuantizationTensor,
    _pool,
)

if TYPE_CHECKING:
    from tensorfs.api import QuantizeTensor

_STRATEGIES = {"int8.convrot": "int8-convrot256/1", "nvfp4": "nvfp4-min-sse/1"}


def supported(source: Checkpoint, selection: ArtifactQuantizationPlan, encoding: str) -> bool:
    """The first checkpoint route is explicit: all selected inputs must be plain floats."""
    return bool(selection.tensors) and all(
        (row.encoding or encoding) in _STRATEGIES
        and source.components[row.component].tensors[row.key].encoding is None
        and source.components[row.component].tensors[row.key].dtype in {"F16", "BF16", "F32"}
        for row in selection.tensors
    )


class _Execution:
    def __init__(
        self,
        future: Future[dict[str, object]],
        control: tensorfs.ConversionControl,
        context: Context | None,
    ) -> None:
        self.future, self.control, self.context = future, control, context

    def poll(self) -> Progress:
        _, completed, _ = self.control.progress()
        return Progress(completed, self.control.canceled, self.future.done())

    def cancel(self) -> None:
        self.control.cancel()
        self.future.cancel()  # A queued operation has not touched its Writer yet.

    def _join(self) -> tuple[object, ...]:
        try:
            return (self.future.result(),)
        except CancelledError:
            if self.control.canceled:
                return ()
            raise
        except tensorfs.NetworkError as error:
            if (
                self.control.canceled
                and error.detail == "the conversion was stopped before publication"
            ):
                return ()
            raise
        except tensorfs.DecodeError as error:
            # This prefix is the native Writer's explicit quality-refusal contract;
            # unrelated format, storage and device failures retain their own families.
            if error.detail.startswith("quantization_tripwire:"):
                raise UnsupportedInput(error.detail, code="quantization_tripwire") from error
            raise

    def wait(self) -> tuple[object, ...]:
        while not self.future.done():
            if self.context is not None and self.context.cancelled:
                self.cancel()
            try:
                self.future.result(timeout=0.05)
            except TimeoutError:
                continue
            except Exception:
                break  # The checked join below preserves/classifies the actual result.
        result = self._join()
        if self.context is not None:
            self.context.raise_if_cancelled()
        return result

    def close(self) -> None:
        if not self.future.done():
            self.cancel()
        self._join()  # Joining is mandatory even while an outer exception unwinds.


@dataclass(frozen=True)
class _Work:
    convert: Callable[[tensorfs.ConversionControl], dict[str, object]]

    def submit(self, context: Context | None = None) -> _Execution:
        if context is not None:
            context.raise_if_cancelled()
        control = tensorfs.ConversionControl()
        return _Execution(_pool().submit(self.convert, control), control, context)


def quantize(
    source: Checkpoint,
    output: Output,
    selection: ArtifactQuantizationPlan,
    encoding: str,
    logical_dtype: Callable[[QuantizationTensor], str],
    *,
    ctx: Context,
    tel: Telemetry,
    max_relative_frobenius: float | None,
) -> QuantizationStats:
    rows: list[QuantizeTensor] = []
    for row in selection.tensors:
        scheme = row.encoding or encoding
        rows.append(
            {
                "component": row.component,
                "key": row.key,
                "source_dtype": source.components[row.component].tensors[row.key].dtype,
                "shape": list(row.shape),
                "dtype": TENSORFS_NAMES[logical_dtype(row)],
                "encoding": {"scheme": scheme, "params": dict(_ENCODINGS[scheme].params)},
                "strategy": _STRATEGIES[scheme],
            }
        )
    convert, pending = output.prepare_quantization(
        source, rows, max_relative_frobenius=max_relative_frobenius
    )
    execution = ctx.plan(_Work(convert))
    observed = -1
    try:
        while True:
            ctx.raise_if_cancelled()
            progress = execution.poll()
            if progress.completed_steps != observed:
                observed = progress.completed_steps
                tel.progress(observed / max(1, pending), stage="native quantization")
            if progress.done:
                break
            time.sleep(0.05)  # Observation never advances native reads/encoding/publication.
        (result,) = execution.wait()
        stats = msgspec.convert(result, type=QuantizationStats)
        tel.progress(1.0, stage="native quantization")
        tel.metric("native_quantized_tensors", float(stats.encoded_keys))
        return stats
    finally:
        # ctx.output can exit before the outer request scope. Join before it closes Output.
        execution.close()
