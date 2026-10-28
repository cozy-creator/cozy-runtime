"""One scoped native Writer conversion; model policy stays in the derivation."""

from __future__ import annotations

import functools
import tempfile
import time
import zlib
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import msgspec
import tensorfs

from cozy_runtime._cpus import cpus
from cozy_runtime.author import Context, Telemetry, UnsupportedInput
from cozy_runtime.author._loader import TENSORFS_NAMES
from cozy_runtime.author._prepared import Progress
from cozy_runtime.weights_store import Checkpoint, Output

from .microscale import CONVROT_GROUP, NVFP4_BLOCK, W4A8_GROUP

if TYPE_CHECKING:
    from tensorfs.api import QuantizeTensor

    from .quantization import ArtifactQuantizationPlan, QuantizationTensor
    from .svdq import Calibration

SVDQ_STRATEGY = "svdq-nvfp4-r32-faer/1"
#: The host bytes one calibrated (svdq.nvfp4) fit may hold: an allowance, not an allocation. Its
#: workspace is about 8 bytes per weight element (FLUX.3 Klein's 18432x3072 needs 526 MiB), so 2 GiB
#: fits a DiT linear of about 250M elements.
FIT_WORKSPACE_BYTES = 2 << 30
_STRATEGIES = {
    "int8.convrot": "int8-convrot256/1",
    "nvfp4": "nvfp4-min-sse/1",
    "svdq.nvfp4": SVDQ_STRATEGY,
    "fp8.rowwise": "fp8-rowwise-rne/1",
    "w4a8.convrot": "w4a8-convrot256-fixed/1",
}
_NVFP4 = {
    "block_size": NVFP4_BLOCK,
    "global_scale": "multiply",
    "nibble_order": "low_first",
    "scale_layout": "row_major",
}
#: Each lane scheme's stored params, written alike by the Writer and the Python encoders.
PARAMS: dict[str, dict[str, object]] = {
    "fp8.rowwise": {},
    "int8.convrot": {"rotation": "kron_regular_h4", "rotation_size": CONVROT_GROUP},
    "nvfp4": _NVFP4,
    "w4a8.convrot": {
        "group_size": W4A8_GROUP,
        "rotation": "kron_regular_h4",
        "rotation_size": CONVROT_GROUP,
    },
    "svdq.nvfp4": dict(_NVFP4),
}
#: Encodings a derivation reads through the serving decoder (`model_values`): per-row scaled
#: codes, whose exact values a lane is written from (Ideogram 4 ships only fp8.rowwise).
PER_ROW = frozenset({"fp8.rowwise", "int8.rowwise"})


class QuantizationStats(msgspec.Struct):
    encoded_keys: int
    reused_keys: int
    source_bytes_read: int
    new_bytes_written: int
    saturated_elements: int
    worst_relative_frobenius: float | None
    tensors: list[dict[str, Any]] = msgspec.field(default_factory=list)


@functools.cache
def _pool() -> ThreadPoolExecutor:
    """One process-wide encoder pool: callers that shard their own work share its bound."""
    return ThreadPoolExecutor(cpus(), thread_name_prefix="cozy-quantize")


def supported(source: Checkpoint, selection: ArtifactQuantizationPlan, encoding: str) -> bool:
    """One native selection of plain or rowwise floats; TensorFS validates stored parts."""
    for row in selection.tensors:
        info = source.components[row.component].tensors[row.key]
        if (
            (row.encoding or encoding) not in _STRATEGIES
            or info.dtype not in {"F16", "BF16", "F32"}
            or len(info.shape) != 2
        ):
            return False
        if info.encoding is not None and (
            info.encoding.scheme not in PER_ROW or dict(info.encoding.params)
        ):
            return False
    return bool(selection.tensors)


def resolved(
    source: Checkpoint,
    selection: ArtifactQuantizationPlan,
    encoding: str,
    logical_dtype: Callable[[QuantizationTensor], str],
) -> list[QuantizeTensor]:
    """Exact native tensor declarations, including the explicit per-key sketch seed."""
    rows: list[QuantizeTensor] = []
    for row in selection.tensors:
        scheme = row.encoding or encoding
        spec: QuantizeTensor = {
            "component": row.component,
            "key": row.key,
            "source_dtype": source.components[row.component].tensors[row.key].dtype,
            "shape": list(row.shape),
            "dtype": TENSORFS_NAMES[logical_dtype(row)],
            "encoding": {"scheme": scheme, "params": dict(PARAMS[scheme])},
            "strategy": _STRATEGIES[scheme],
        }
        if scheme == "svdq.nvfp4":
            spec["fit"] = {
                "seed": zlib.crc32(row.key.encode()),
                "component": row.component,
                "xmax": row.key + ".xmax",
                "rows": row.key + ".rows",
            }
        rows.append(spec)
    return rows


def provenance(
    source: Checkpoint, rows: list[QuantizeTensor], calibration: Checkpoint | None
) -> dict[str, Any]:
    """Bind the completed calibration and native producer before Output replay lookup."""
    return {
        "native_quantization": {
            "source": source.id,
            **({"calibration": calibration.id} if calibration is not None else {}),
            "selection": rows,
            "implementation": tensorfs.raw.quantization_implementation(),
        }
    }


@contextmanager
def snapshot(
    source: Checkpoint,
    selection: ArtifactQuantizationPlan | None,
    encoding: str | None,
    calibration: Calibration | None,
    ctx: Context,
) -> Iterator[Checkpoint | None]:
    """Serialize finished model hooks once; no live tensors cross the native worker boundary.

    Writer borrows each contiguous CPU view only for its synchronous put. The ordinary
    checkpoint and its directory outlive the scoped conversion and are closed after join.
    """
    fitted = (
        []
        if selection is None
        else [row for row in selection.tensors if (row.encoding or encoding) == "svdq.nvfp4"]
    )
    if not fitted:
        yield None
        return
    if calibration is None or not calibration.completed:
        raise UnsupportedInput(
            "svdq.nvfp4 requires completed model calibration before conversion",
            code="quantization_calibration",
        )
    torch = calibration.torch
    with tempfile.TemporaryDirectory(prefix="cozy-svdq-calibration-") as directory:
        writer = tensorfs.Writer(Path(directory) / "calibration.cozytensors")
        writer.metadata["provenance"] = {
            "source": source.id,
            "collection": "runtime-linear-reservoir/1",
            "rows_per_linear": 512,
            "rows_per_call": 32,
            "sampling_seed": 0,
            "selection": [[row.component, row.key, list(row.shape)] for row in fitted],
        }
        components: dict[str, tensorfs.ComponentWriter] = {}
        try:
            for row in fitted:
                ctx.raise_if_cancelled()
                stats = calibration.stats(row.component, row.key)
                component = components.get(row.component)
                if component is None:
                    component = components[row.component] = writer.component(row.component)
                for suffix, value, dtype in (
                    ("xmax", stats.xmax, "F32"),
                    ("rows", stats.rows, "BF16"),
                ):
                    # Calibration already supplies CPU F32 maxima and CPU BF16 rows.
                    wanted = torch.float32 if dtype == "F32" else torch.bfloat16
                    if value.dtype != wanted or value.device.type != "cpu":
                        raise UnsupportedInput("calibration snapshot has unexpected dtype/device")
                    held = value.detach().contiguous()
                    view = memoryview(held.view(torch.uint8).numpy())
                    component.put(row.key + "." + suffix, view, dtype=dtype, shape=list(held.shape))
            checkpoint = writer.commit()
        except BaseException:
            writer.abandon()
            raise
        try:
            yield checkpoint
        finally:
            checkpoint.close()


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

    def close(self, original: BaseException | None = None) -> None:
        secondary: BaseException | None = None
        try:
            if not self.future.done():
                self.cancel()
        except BaseException as error:
            secondary = error
        try:
            self._join()  # A failed cancel still must join the actual Writer owner.
        except BaseException as error:
            if not self.future.done():
                raise  # An interrupted join did not prove completion.
            secondary = secondary or error
        if secondary is not None:
            if original is None:
                raise secondary
            if secondary is not original:
                original.add_note(
                    f"Native conversion completed with a secondary failure: {secondary}"
                )


@dataclass(frozen=True)
class _Work:
    convert: Callable[[tensorfs.ConversionControl], dict[str, object]]

    def submit(self, context: Context | None = None) -> _Execution:
        if context is not None:
            context.raise_if_cancelled()
        control = tensorfs.ConversionControl()
        future = _pool().submit(self.convert, control)
        try:
            return _Execution(future, control, context)
        except BaseException:
            control.cancel()
            future.cancel()
            with suppress(Exception):
                future.result()  # A signal cannot orphan native work owning the Output.
            raise


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
    calibration: Calibration | None = None,
    calibration_checkpoint: Checkpoint | None = None,
    fit_workspace_bytes: int = FIT_WORKSPACE_BYTES,
    component: str | None = None,
) -> QuantizationStats:
    with (
        snapshot(source, selection, encoding, calibration, ctx)
        if calibration_checkpoint is None
        else nullcontext(calibration_checkpoint) as made
    ):
        held = calibration_checkpoint if calibration_checkpoint is not None else made
        rows = resolved(source, selection, encoding, logical_dtype)
        if any("fit" in row for row in rows) and held is None:
            raise UnsupportedInput("svdq.nvfp4 requires a completed calibration checkpoint")
        convert, pending = output.prepare_quantization(
            source,
            rows,
            max_relative_frobenius=max_relative_frobenius,
            calibration=held,
            max_buffer_bytes=fit_workspace_bytes if held is not None else 8 << 20,
            component=component,
        )
        return _run(convert, pending, ctx, tel)


def _run(
    convert: Callable[[tensorfs.ConversionControl], dict[str, object]],
    pending: int,
    ctx: Context,
    tel: Telemetry,
) -> QuantizationStats:
    stats = msgspec.convert(
        _run_native(convert, pending, ctx, tel, stage="native quantization"),
        type=QuantizationStats,
    )
    for row in stats.tensors:
        if row.get("fit") is not None:
            tel.log(
                "native fitted tensor", component=row["component"], key=row["key"], **row["fit"]
            )
    tel.metric("native_quantized_tensors", float(stats.encoded_keys))
    return stats


def _run_native(
    convert: Callable[[tensorfs.ConversionControl], dict[str, object]],
    pending: int,
    ctx: Context,
    tel: Telemetry,
    *,
    stage: str,
) -> dict[str, object]:
    # This helper joins before returning, like the existing bounded encoding loop.
    # It must not consume Context.plan's one whole GPU program slot: an author may
    # convert several outputs and then submit that program in the same request.
    execution = _Work(convert).submit(ctx)
    observed = -1
    failure: BaseException | None = None
    try:
        while True:
            ctx.raise_if_cancelled()
            progress = execution.poll()
            if progress.completed_steps != observed:
                observed = progress.completed_steps
                tel.progress(observed / max(1, pending), stage=stage)
            if progress.done:
                break
            time.sleep(0.05)  # Observation never advances native reads/encoding/publication.
        (result,) = execution.wait()
        tel.progress(1.0, stage=stage)
        return cast("dict[str, object]", result)
    except BaseException as error:
        failure = error
        raise
    finally:
        # ctx.output can exit before the outer request scope. Join before it closes Output.
        execution.close(failure)
