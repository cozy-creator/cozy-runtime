"""The rank command wire: what a mirrored model call may carry across the rank seam.

The writer is rank 0, the process that runs TENANT handler code and marshals whatever the
handler passed to a component-use method. The seam between ranks is the same length-framed
canonical-JSON control seam the worker drives an executor with (`internal/seam.py`), so the
frame is control and never pickle: a follower that unpickled would be a deserialization
gadget on tenant-reachable input. The value vocabulary is CLOSED - JSON scalars, list,
tuple, str-keyed dict, `torch.Tensor` and `torch.Generator` as its seed - and anything else
is a typed refusal raised on rank 0 at send time, with the group still coherent.

Bytes never ride the frame. A tensor's payload lands in the attempt SPOOL, the one road
data takes in this runtime, as raw little-endian storage under a name rank 0 minted; the
frame carries the name, dtype and shape, and a follower reads the file into ITS device.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cozy_runtime.internal.parallel.plan import GroupRefusal

_TAG = "__sp__"
_VAL = "v"


class UncrossableArgument(GroupRefusal):
    """A model-call argument outside the wire's value vocabulary."""

    default_code = "uncrossable_argument"


class TensorSpool:
    """Where one mirrored call's tensor payloads live: a directory rank 0 writes and every
    follower reads. Files are numbered per call and removed by the caller with the call."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.written = 0

    def write(self, tensor: Any) -> str:
        import torch

        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"tensor-{self.written}.raw"
        self.written += 1
        flat = tensor.detach().to("cpu").contiguous()
        payload = bytes(flat.view(torch.uint8).untyped_storage()) if flat.numel() else b""
        (self.directory / name).write_bytes(payload)
        return name

    def read(self, name: str, dtype: str, shape: list[int], device: Any) -> Any:
        import torch

        if "/" in name or not name.startswith("tensor-"):
            raise UncrossableArgument(f"malformed tensor payload name {name!r}")
        raw = bytearray((self.directory / name).read_bytes())
        kind = getattr(torch, dtype, None)
        if not isinstance(kind, torch.dtype):
            raise UncrossableArgument(f"unknown tensor dtype {dtype!r} on the wire")
        if not raw:
            return torch.empty(shape, dtype=kind, device=device)
        return torch.frombuffer(raw, dtype=kind).reshape(shape).to(device)

    def clear(self) -> None:
        if not self.directory.is_dir():
            return
        for entry in self.directory.iterdir():
            entry.unlink(missing_ok=True)
        self.written = 0


def marshal(value: Any, spool: TensorSpool, *, path: str = "") -> Any:
    """One call argument -> JSON-native tagged data, or refuse by name."""
    import torch

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return {
            _TAG: "tensor",
            _VAL: spool.write(value),
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
        }
    if isinstance(value, torch.Generator):
        return {_TAG: "generator", _VAL: int(value.initial_seed())}
    if isinstance(value, tuple):
        return {
            _TAG: "tuple",
            _VAL: [marshal(v, spool, path=f"{path}[{i}]") for i, v in enumerate(value)],
        }
    if isinstance(value, list):
        return [marshal(v, spool, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UncrossableArgument(
                    f"{path or 'argument'}: dict key {key!r} is {type(key).__name__}, and only "
                    "str keys cross the rank seam"
                )
            if key == _TAG:
                raise UncrossableArgument(
                    f"{path or 'argument'}: {_TAG!r} is the wire's own tag key and cannot be a "
                    "model-call dict key"
                )
            out[key] = marshal(item, spool, path=f"{path}.{key}" if path else key)
        return out
    raise UncrossableArgument(
        f"{path or 'argument'} is a {type(value).__name__}, which cannot cross the rank seam. "
        "A mirrored model call carries JSON scalars, lists, tuples, str-keyed dicts, torch "
        "tensors (through the attempt spool) and torch.Generator (as its seed); a closure, a "
        "callback, a request view or a live handle cannot be broadcast to follower ranks"
    )


def unmarshal(value: Any, spool: TensorSpool, *, device: Any) -> Any:
    """Wire data -> the argument, on THIS rank's device."""
    import torch

    if isinstance(value, dict):
        tag = value.get(_TAG)
        if tag is None:
            return {k: unmarshal(v, spool, device=device) for k, v in value.items()}
        payload = value.get(_VAL)
        if tag == "tensor" and isinstance(payload, str):
            return spool.read(
                payload, str(value.get("dtype")), [int(n) for n in value.get("shape") or []], device
            )
        if tag == "generator" and isinstance(payload, int):
            return torch.Generator(device=device).manual_seed(payload)
        if tag == "tuple" and isinstance(payload, list):
            return tuple(unmarshal(v, spool, device=device) for v in payload)
        raise UncrossableArgument(
            f"malformed wire envelope: tag={tag!r} payload={type(payload).__name__}"
        )
    if isinstance(value, list):
        return [unmarshal(v, spool, device=device) for v in value]
    return value


def run_command(
    method: str, args: tuple[Any, ...], kwargs: Mapping[str, Any], spool: TensorSpool
) -> dict[str, Any]:
    """Marshal one model-method call as the `run` command every follower receives."""
    return {
        "cmd": "run",
        "method": method,
        "args": [marshal(a, spool, path=f"args[{i}]") for i, a in enumerate(args)],
        "kwargs": {k: marshal(v, spool, path=k) for k, v in kwargs.items()},
    }


def run_call(
    command: Mapping[str, Any], spool: TensorSpool, *, device: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """A follower's side of `run_command`."""
    args = tuple(unmarshal(a, spool, device=device) for a in command.get("args") or [])
    kwargs = {
        str(k): unmarshal(v, spool, device=device) for k, v in (command.get("kwargs") or {}).items()
    }
    return args, kwargs
