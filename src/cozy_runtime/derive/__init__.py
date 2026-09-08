"""Carrier-free structural derivation: FP8/MXFP8 encoding, safetensors IO, identity.

Run-to-completion job packages import these modules directly
(``cozy_runtime.derive.quantization``) or just the cr-076 facade re-exported here
(``derive.plan`` / ``derive.quantize``); nothing in this package is imported by the
resident worker planes. numpy is required at module import and is the consumer's declared
dependency — the base Runtime package stays numpy-free.
"""

from cozy_runtime.derive.facade import (
    QuantizePlan,
    QuantizeResult,
    plan,
    quantize,
    quantize_artifact,
)

__all__ = ["QuantizePlan", "QuantizeResult", "plan", "quantize", "quantize_artifact"]
