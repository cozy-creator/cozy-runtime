"""The teacher-forced two-pass engine and its thin driver, ``run_probe``.

A naive two-run diff diverges after the first denoising step — trajectories separate,
diffs explode, signal dies. So ``record_trajectory`` runs the reference lane's own
free-running loop ONCE and snapshots every model invocation's inputs (exact, on CPU), and
every later pass drives a lane with those recorded inputs, so each step's diff isolates
that step's error. The floor is measured, never assumed: the reference lane is probed
against ITSELF (two teacher-forced passes) and every candidate stat is judged — by the
gate, never here — as a multiple of that floor.

Sequential two-pass, never two resident models: the reference lane is built, recorded,
passed twice and RELEASED before the candidate lane is built. Each lane runs in its own
encoding's natural dtype; the evidence is fp32 either way (``_sketch``).
"""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Callable
from typing import Any, NamedTuple

from cozy_runtime.probe._report import PassCapture, build_report
from cozy_runtime.probe._sketch import ProbeConfig, TapStepAccumulator
from cozy_runtime.probe._taps import ProbeRefusal, Tap, TapPlan, resolve_taps

#: One recorded model invocation: its positional and keyword inputs, snapshotted to CPU.
RecordedCall = tuple[tuple[Any, ...], dict[str, Any]]


class Lane(NamedTuple):
    """One probe lane: a label for the report, and a builder for its constructed model —
    an nn.Module for block taps, or any callable (degraded component-output taps)."""

    label: str
    build: Callable[[], Any]


class Trajectory(NamedTuple):
    """The reference trajectory: one RecordedCall per model invocation of the free run."""

    steps: tuple[RecordedCall, ...]


def record_trajectory(call: Any, drive: Callable[[Any], None]) -> Trajectory:
    """Run ``drive`` — the lane's own free-running loop — against a recording wrapper of
    ``call``, and return the exact per-step inputs it saw."""
    import torch

    steps: list[RecordedCall] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        step = len(steps)
        steps.append(
            (
                tuple(_snapshot(a, where=f"step {step} arg {i}") for i, a in enumerate(args)),
                {k: _snapshot(v, where=f"step {step} kwarg {k!r}") for k, v in kwargs.items()},
            )
        )
        with torch.inference_mode():
            return call(*args, **kwargs)

    drive(recording)
    if not steps:
        raise ProbeRefusal(
            "the drive loop never invoked the model: nothing was recorded, so there is "
            "nothing to teacher-force",
            code="empty_trajectory",
        )
    return Trajectory(tuple(steps))


def teacher_forced_pass(
    model: Any, plan: TapPlan, trajectory: Trajectory, config: ProbeConfig
) -> PassCapture:
    """Drive one lane with the recorded inputs and capture every tap — sketches plus exact
    scalars per (tap, step), accumulated in fp32 whatever dtype the lane runs in."""
    import torch

    device = _lane_device(model)
    accumulators: dict[tuple[str, int], TapStepAccumulator] = {}
    order: list[str] = []
    fired: set[str] = set()
    step = [0]

    def fold(path: str, value: Any) -> None:
        leaves = _tensor_leaves(value)
        if not leaves:
            return
        if path not in fired:
            fired.add(path)
            order.append(path)
        accumulator = accumulators.get((path, step[0]))
        if accumulator is None:
            accumulator = accumulators[(path, step[0])] = TapStepAccumulator(config)
        for leaf in leaves:
            accumulator.fold(path, leaf)

    module_taps = [tap for tap in plan.taps if tap.kind == "module"]
    output_taps = [tap for tap in plan.taps if tap.kind == "component_output"]
    handles: list[Any] = []
    was_training = isinstance(model, torch.nn.Module) and model.training
    if isinstance(model, torch.nn.Module):
        model.eval()
    try:
        for tap in module_taps:

            def hook(_m: Any, _inputs: Any, output: Any, *, path: str = tap.path) -> None:
                fold(path, output)

            handles.append(_target(model, plan, tap).register_forward_hook(hook))
        with torch.inference_mode():
            for index, (args, kwargs) in enumerate(trajectory.steps):
                step[0] = index
                output = model(
                    *(_restore(a, device) for a in args),
                    **{k: _restore(v, device) for k, v in kwargs.items()},
                )
                for tap in output_taps:
                    fold(tap.path, output)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    observations = {key: accumulator.finalize() for key, accumulator in accumulators.items()}
    return PassCapture(observations=observations, order=tuple(order))


def run_probe(
    reference: Lane,
    candidate: Lane,
    drive: Callable[[Any], None],
    *,
    component: str = "model",
    zoom: str | None = None,
    config: ProbeConfig | None = None,
) -> dict[str, Any]:
    """One whole probe, sequentially: build the reference lane, record its free run, pass
    it twice (baseline + floor), release it, pass the candidate lane once, and return the
    ``cozy.probe.report/1`` document. Raw stats only — no verdict lives here."""
    config = config or ProbeConfig()
    reference_model: Any = reference.build()
    plan = resolve_taps(reference_model, component=component, zoom=zoom)
    trajectory = record_trajectory(reference_model, drive)
    baseline = teacher_forced_pass(reference_model, plan, trajectory, config)
    floor = teacher_forced_pass(reference_model, plan, trajectory, config)
    reference_model = None
    gc.collect()
    candidate_model: Any = candidate.build()
    candidate_plan = resolve_taps(candidate_model, component=component, zoom=zoom)
    if candidate_plan != plan:
        raise ProbeRefusal(
            f"lane {candidate.label!r} resolved a different tap plan than "
            f"{reference.label!r} — lanes change weights, never topology",
            code="topology_divergence",
        )
    candidate_pass = teacher_forced_pass(candidate_model, plan, trajectory, config)
    candidate_model = None
    gc.collect()
    return build_report(
        reference=reference.label,
        candidate=candidate.label,
        plan=plan,
        steps=len(trajectory.steps),
        baseline=baseline,
        floor=floor,
        candidate_pass=candidate_pass,
        config=config,
        zoom=zoom,
    )


def _target(model: Any, plan: TapPlan, tap: Tap) -> Any:
    relative = plan.module_path(tap)
    if not relative:
        return model
    try:
        return model.get_submodule(relative)
    except AttributeError as exc:
        raise ProbeRefusal(
            f"tap {tap.path!r} names no module in this lane: {exc}", code="tap_unknown_path"
        ) from exc


def _lane_device(model: Any) -> Any:
    import torch

    if isinstance(model, torch.nn.Module):
        for parameter in model.parameters():
            return parameter.device
        for buffer in model.buffers():
            return buffer.device
    return torch.device("cpu")


def _snapshot(value: Any, *, where: str) -> Any:
    import torch

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, tuple):
        return tuple(_snapshot(v, where=f"{where}[{i}]") for i, v in enumerate(value))
    if isinstance(value, list):
        return [_snapshot(v, where=f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: _snapshot(v, where=f"{where}.{k}") for k, v in value.items()}
    raise ProbeRefusal(
        f"{where}: a recorded model input is a tensor, a JSON scalar, or a list/tuple/dict "
        f"of them — not a {type(value).__name__}",
        code="unrecordable_input",
    )


def _restore(value: Any, device: Any) -> Any:
    """A recorded input back onto the lane, COPIED so no pass can corrupt the recording."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, copy=True)
    if isinstance(value, tuple):
        return tuple(_restore(v, device) for v in value)
    if isinstance(value, list):
        return [_restore(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _restore(v, device) for k, v in value.items()}
    return value


def _tensor_leaves(value: Any) -> list[Any]:
    import torch

    leaves: list[Any] = []

    def walk(node: Any) -> None:
        if isinstance(node, torch.Tensor):
            leaves.append(node)
        elif isinstance(node, (tuple, list)):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                walk(getattr(node, field.name))

    walk(value)
    return leaves
