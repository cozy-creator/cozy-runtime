"""``cozy.probe.report/1`` — raw stats only; the runtime never emits a verdict.

One row per (tap, step) between the reference and candidate lanes, the ref-vs-ref floor
rows beside them, and per-tap aggregates (worst step + mean). Every sequence rides in
FIRING order — the order taps fired in the reference pass — so "the FIRST tap over
K x floor" is a well-defined read. Thresholds (the K's) live in gate-owned config, never
here. Undefined stats are ``null``; a NaN or infinity never enters the document.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from cozy_runtime.probe._sketch import Observation, ProbeConfig, compare
from cozy_runtime.probe._taps import ProbeRefusal, Tap, TapPlan

REPORT_FORMAT = "cozy.probe.report/1"


class PassCapture(NamedTuple):
    """One teacher-forced pass's finalized evidence: observations per (tap path, step),
    and the tap paths in the order they first fired."""

    observations: dict[tuple[str, int], Observation]
    order: tuple[str, ...]


def build_report(
    *,
    reference: str,
    candidate: str,
    plan: TapPlan,
    steps: int,
    baseline: PassCapture,
    floor: PassCapture,
    candidate_pass: PassCapture,
    config: ProbeConfig,
    zoom: str | None = None,
) -> dict[str, Any]:
    rows = _rows(baseline, candidate_pass, steps)
    floor_rows = _rows(baseline, floor, steps)
    return {
        "format": REPORT_FORMAT,
        "reference": reference,
        "candidate": candidate,
        "component": plan.component,
        "zoom": zoom,
        "steps": steps,
        "sketch": {"kind": "countsketch", "dim": config.sketch_dim},
        "outlier_sigma": config.outlier_sigma,
        "taps": [
            {"path": tap.path, "kind": tap.kind} for tap in _ordered_taps(plan, baseline.order)
        ],
        "rows": rows,
        "floor": floor_rows,
        "aggregates": _aggregates(rows),
        "floor_aggregates": _aggregates(floor_rows),
    }


def _ordered_taps(plan: TapPlan, order: tuple[str, ...]) -> list[Tap]:
    by_path = {tap.path: tap for tap in plan.taps}
    ordered = [by_path[path] for path in order if path in by_path]
    ordered += [tap for tap in plan.taps if tap.path not in set(order)]
    return ordered


def _rows(reference: PassCapture, other: PassCapture, steps: int) -> list[dict[str, Any]]:
    fired = set(reference.order)
    stray = [path for path in other.order if path not in fired]
    if stray:
        raise ProbeRefusal(
            f"taps {stray} fired in one lane and never in the reference — lanes change "
            "weights, never topology",
            code="firing_divergence",
        )
    rows: list[dict[str, Any]] = []
    for path in reference.order:
        for step in range(steps):
            ours = reference.observations.get((path, step))
            theirs = other.observations.get((path, step))
            if ours is None and theirs is None:
                continue
            if ours is None or theirs is None:
                raise ProbeRefusal(
                    f"tap {path!r} step {step}: fired in one lane and not the other — lanes "
                    "change weights, never topology",
                    code="firing_divergence",
                )
            stats = compare(path, step, ours, theirs)
            rows.append(
                {
                    "tap": path,
                    "step": step,
                    "rel_l2": stats.rel_l2,
                    "cosine": stats.cosine,
                    "snr_db": stats.snr_db,
                    "max_abs_delta": stats.max_abs_delta,
                    "outlier_frac": stats.outlier_frac,
                    "nan_inf": {"ref": stats.nan_inf_ref, "cand": stats.nan_inf_cand},
                }
            )
    return rows


#: Whether a stat's WORST step is its maximum (True) or its minimum (False).
_WORST_IS_MAX = {
    "rel_l2": True,
    "cosine": False,
    "snr_db": False,
    "max_abs_delta": True,
    "outlier_frac": True,
}


def _aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-tap aggregate = worst step + mean, over the defined (non-null) steps."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["tap"], []).append(row)
    aggregates: list[dict[str, Any]] = []
    for tap, tap_rows in grouped.items():
        worst: dict[str, float | None] = {}
        mean: dict[str, float | None] = {}
        for stat, worst_is_max in _WORST_IS_MAX.items():
            defined = [row[stat] for row in tap_rows if row[stat] is not None]
            if defined:
                worst[stat] = max(defined) if worst_is_max else min(defined)
                mean[stat] = sum(defined) / len(defined)
            else:
                worst[stat] = None
                mean[stat] = None
        aggregates.append(
            {
                "tap": tap,
                "worst": worst,
                "mean": mean,
                "nan_inf": {
                    "ref": sum(row["nan_inf"]["ref"] for row in tap_rows),
                    "cand": sum(row["nan_inf"]["cand"] for row in tap_rows),
                },
            }
        )
    return aggregates
