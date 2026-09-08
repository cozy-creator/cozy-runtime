"""One ordinary render's activation evidence; no second model or replayed trajectory."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextvars import Token
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author._capture import ActivationCapture, _capture_clock, _CaptureClock
from cozy_runtime.author._errors import InvalidRequest
from cozy_runtime.author._model import Model, component_use
from cozy_runtime.probe._sketch import ProbeConfig, TapStepAccumulator
from cozy_runtime.probe._taps import MAX_TAPS, resolve_taps

MAX_CAPTURE_BYTES = 128 << 20
CAPTURE_OUTPUT = "runtime.capture"


def validate_components(options: ActivationCapture, classes: list[type[Model[Any]]]) -> None:
    declared = {name for cls in classes for names in component_use(cls).values() for name in names}
    missing = set(options.components) - declared
    if missing:
        raise InvalidRequest(
            f"capture components are not declared by the entrypoint's Models: {sorted(missing)}",
            code="capture_component_undeclared",
        )


def _tensors(value: Any) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensors(item)]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in _tensors(item)]
    return []


class CaptureSession:
    """Hooks only existing component modules; retains sketches, never live activations."""

    def __init__(self, options: ActivationCapture, roots: Mapping[str, Any]) -> None:
        self.options = options
        self.roots = roots
        self.config = ProbeConfig()
        self.handles: list[Any] = []
        self.token: Token[_CaptureClock | None] | None = None
        self.total: int | None = None
        self.step = 0
        self.completed_steps = 0
        self.accumulators: dict[str, TapStepAccumulator] = {}
        self.taps: dict[str, str] = {}
        self.order: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self.blobs: list[bytes] = []
        self.bytes = 0

    def __enter__(self) -> CaptureSession:
        try:
            for component in self.options.components:
                root = self.roots.get(component)
                if root is None:
                    raise InvalidRequest(
                        f"capture component {component!r} has no constructed root",
                        code="capture_component_undeclared",
                    )
                plan = resolve_taps(root, component=component)
                for tap in plan.taps:
                    if tap.kind != "module":
                        raise InvalidRequest(
                            f"capture component {component!r} does not expose module hooks",
                            code="capture_component_unsupported",
                        )
                    if len(self.taps) >= MAX_TAPS:
                        raise InvalidRequest("capture exceeds the tap bound", code="capture_bound")
                    self.taps[tap.path] = tap.kind
                    module = root.get_submodule(plan.module_path(tap))

                    def hook(_module: Any, _args: Any, output: Any, path: str = tap.path) -> None:
                        self._observe(path, output)

                    self.handles.append(module.register_forward_hook(hook))
            self.token = _capture_clock.set(self)
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.token is not None:
            _capture_clock.reset(self.token)
            self.token = None

    def schedule(self, total: int) -> None:
        if type(total) is not int or not 0 < total <= 1_000_000_000:
            raise InvalidRequest("capture schedule has an invalid total", code="capture_steps")
        if self.rows or self.accumulators:
            raise InvalidRequest(
                "capture requires one measured schedule for its selected components",
                code="capture_steps",
            )
        if self.options.steps and self.options.steps[-1] >= total:
            raise InvalidRequest("capture step exceeds the measured schedule", code="capture_steps")
        if (
            not self.options.steps
            and total * max(1, len(self.taps)) * self.config.sketch_dim * 4 > MAX_CAPTURE_BYTES
        ):
            raise InvalidRequest(
                "complete capture exceeds the byte bound; select fewer steps", code="capture_bound"
            )
        self.total, self.step, self.completed_steps = total, 0, 0

    def completed(self, step: int) -> None:
        if step != self.step or self.total is None or step >= self.total:
            raise InvalidRequest(
                "capture schedule skipped or repeated a step", code="capture_steps"
            )
        self._flush()
        self.completed_steps = step + 1
        self.step = step + 1

    def _observe(self, path: str, output: Any) -> None:
        if self.total is not None and self.step >= self.total:
            raise InvalidRequest(
                "component fired beyond the capture schedule", code="capture_steps"
            )
        if self.options.steps and self.step not in self.options.steps:
            return
        tensors = _tensors(output)
        if not tensors:
            raise InvalidRequest(f"capture tap {path!r} returned no tensors", code="capture_output")
        if path not in self.order:
            self.order.append(path)
        accumulator = self.accumulators.setdefault(path, TapStepAccumulator(self.config))
        for tensor in tensors:
            accumulator.fold(path, tensor)

    def _flush(self) -> None:
        for path in self.order:
            accumulator = self.accumulators.pop(path, None)
            if accumulator is None:
                continue
            observed = accumulator.finalize()
            blob = observed.sketch.numpy().astype("<f4", copy=False).tobytes()
            self.bytes += len(blob)
            if self.bytes > MAX_CAPTURE_BYTES:
                raise InvalidRequest("capture exceeds the byte bound", code="capture_bound")
            self.rows.append(
                {
                    "tap": path,
                    "step": self.step,
                    "shapes": observed.shapes,
                    "numel": observed.numel,
                    "nan": observed.nan,
                    "inf": observed.inf,
                    "max_abs": observed.max_abs,
                    "mean": observed.mean,
                    "std": observed.std,
                }
            )
            self.blobs.append(blob)

    def write(self, spool: Path) -> dict[str, Any]:
        """Finish a successful attempt's tree, returning facts for worker-owned settlement."""
        if self.handles:
            raise RuntimeError("capture must close its hooks before publication")
        if self.total is not None and self.completed_steps != self.total:
            raise InvalidRequest("capture schedule did not complete", code="capture_steps")
        if self.total is None and self.options.steps not in ((), (0,)):
            raise InvalidRequest("capture has no measured schedule", code="capture_steps")
        self._flush()
        if not self.rows:
            raise InvalidRequest("capture measured no activations", code="capture_empty")
        steps = self.options.steps or tuple(range(self.total or 1))
        manifest = canonical_json.encode(
            {
                "format": "cozy.capture/1",
                "components": self.options.components,
                "taps": [{"path": path, "kind": self.taps[path]} for path in self.order],
                "steps": steps,
                "sketch": {
                    "kind": "countsketch",
                    "dim": self.config.sketch_dim,
                    "seed_domain": "cozy.probe.sketch/1",
                },
                "rows": self.rows,
            }
        )
        blob = b"".join(self.blobs)
        if len(manifest) + len(blob) > MAX_CAPTURE_BYTES:
            raise InvalidRequest("capture exceeds the byte bound", code="capture_bound")
        root = spool / CAPTURE_OUTPUT
        root.mkdir()
        (root / "capture.json").write_bytes(manifest)
        (root / "sketches.f32").write_bytes(blob)
        return {
            "output_id": CAPTURE_OUTPUT,
            "content_digest": "sha256:" + hashlib.sha256(manifest + blob).hexdigest(),
            "root": str(root),
            "length": len(manifest) + len(blob),
        }
