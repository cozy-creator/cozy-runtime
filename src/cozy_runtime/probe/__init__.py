"""Block-level activation tap and deterministic sketch primitives.

Captures ride ordinary invocations. Comparison and admission policy belong to cozy-eval.
"""

from cozy_runtime.probe._sketch import Observation, ProbeConfig
from cozy_runtime.probe._taps import MAX_TAPS, ProbeRefusal, Tap, TapPlan, resolve_taps

__all__ = [
    "MAX_TAPS",
    "Observation",
    "ProbeConfig",
    "ProbeRefusal",
    "Tap",
    "TapPlan",
    "resolve_taps",
]
