"""Prepared FP16 biased linear; callers retain every tensor through completion."""
from typing import Any

API: int
NOT_SUPPORTED: int

class Error(RuntimeError):
    domain: str
    status: int

class Context:
    def __init__(self, device: int, stream: int) -> None: ...
    def close(self, stream: int) -> None: ...
    def query(
        self, rows: int, inputs: int, outputs: int, bias: int, bias_capacity: int,
        half_accumulation: bool, reduction: int, sm_target: int,
        workspace_limit: int, stream: int,
    ) -> Plan: ...

class Plan:
    def info(self) -> dict[str, Any]: ...
    def validate(
        self, x: int, x_capacity: int, weight: int, weight_capacity: int,
        output: int, output_capacity: int, workspace: int,
        workspace_capacity: int, stream: int,
    ) -> None: ...
    def execute(
        self, x: int, x_capacity: int, weight: int, weight_capacity: int,
        output: int, output_capacity: int, workspace: int,
        workspace_capacity: int, stream: int,
    ) -> None: ...
