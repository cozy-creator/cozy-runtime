"""Carrier-free structural derivation: FP8/MXFP8 encoding, safetensors IO, identity.

Run-to-completion job packages import these modules directly
(``cozy_runtime.derive.quantization``); nothing here is imported by the resident
worker planes. numpy is required at module import and is the consumer's declared
dependency — the base Runtime package stays numpy-free.
"""
