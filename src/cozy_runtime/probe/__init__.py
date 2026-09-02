"""The activation probe — tier 2 of the fidelity ladder (cr-075).

Teacher-forced per-layer diffs across lanes: record the reference trajectory once, drive
both lanes with those recorded inputs, and name WHICH block a candidate lane degraded. The
runtime owns the MECHANISM (stats); the gate owns the POLICY (thresholds) — nothing here
emits a verdict.

Import-fenced like ``cozy_runtime.derive``: never the worker planes, the CLI, or TensorFS.
torch is reached only inside functions, so importing this package stays torch-free (§1.0).
"""

from cozy_runtime.probe._engine import (
    Lane,
    Trajectory,
    record_trajectory,
    run_probe,
    teacher_forced_pass,
)
from cozy_runtime.probe._report import REPORT_FORMAT, PassCapture, build_report
from cozy_runtime.probe._sketch import DiffStats, Observation, ProbeConfig, compare
from cozy_runtime.probe._taps import MAX_TAPS, ProbeRefusal, Tap, TapPlan, resolve_taps

__all__ = [
    "MAX_TAPS",
    "REPORT_FORMAT",
    "DiffStats",
    "Lane",
    "Observation",
    "PassCapture",
    "ProbeConfig",
    "ProbeRefusal",
    "Tap",
    "TapPlan",
    "Trajectory",
    "build_report",
    "compare",
    "record_trajectory",
    "resolve_taps",
    "run_probe",
    "teacher_forced_pass",
]
