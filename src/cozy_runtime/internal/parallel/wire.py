"""Transport sharded component arguments as JSON control and tensor snapshots.

Tensor bytes live in the call's shared spool. Followers read the exact dtype and
shape onto their own device. Values are snapshots; aliases, generators, callbacks
and other Python state are not transported. A module can cross only as the identity
of a prepared component root whose owning scope is active on every rank.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
import sys
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout
from cozy_runtime.internal.parallel.plan import GroupRefusal

_TAG = "__sp__"
_VAL = "v"


class UncrossableArgument(GroupRefusal):
    """A model-call argument outside the wire's value vocabulary."""

    default_code = "uncrossable_argument"


class TensorSpool:
    """Where one mirrored call's tensor payloads live: a directory rank 0 writes and every
    follower reads. Files are numbered per call and removed by the caller with the call."""

    def __init__(
        self, directory: Path, components: Mapping[tuple[str, str], Any] | None = None
    ) -> None:
        self.directory = directory
        self.written = 0
        # References identify already prepared local roots, never a serialized module graph.
        self.components = dict(components or {})
        self.component_names = {id(root): key for key, root in self.components.items()}

    def write(self, tensor: Any) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"tensor-{self.written}.raw"
        self.written += 1
        # data_ptr includes a slice's offset; storage bytes do not. Keep the tensor alive
        # while copying exactly its logical bytes, including scalars and empty tensors.
        flat = tensor.detach().to("cpu").resolve_conj().resolve_neg().contiguous()
        payload = ctypes.string_at(flat.data_ptr(), flat.numel() * flat.element_size())
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
        expected = math.prod(shape) * torch.empty((), dtype=kind).element_size()
        if len(raw) != expected:
            raise UncrossableArgument(
                f"tensor payload {name!r} has {len(raw)} bytes, expected {expected}"
            )
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
    if isinstance(value, torch.nn.Module):
        key = spool.component_names.get(id(value))
        if key is None:
            raise UncrossableArgument(
                f"{path or 'argument'} is not a prepared component in an active model scope"
            )
        return {_TAG: "component", _VAL: list(key)}
    if isinstance(value, OrderedDict) and dataclasses.is_dataclass(type(value)):
        # A model output (transformers' `ModelOutput`): its class crosses by name and is
        # rebuilt only from a module the receiving rank has already imported.
        kind = type(value)
        return {
            _TAG: "output",
            "class": f"{kind.__module__}:{kind.__qualname__}",
            _VAL: {k: marshal(v, spool, path=f"{path}.{k}") for k, v in value.items()},
        }
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
        "tensors (through the attempt spool); a generator, closure, "
        "callback, a request view or a live handle cannot be broadcast to follower ranks"
    )


def unmarshal(value: Any, spool: TensorSpool, *, device: Any) -> Any:
    """Wire data -> the argument, on THIS rank's device."""

    if isinstance(value, dict):
        tag = value.get(_TAG)
        if tag is None:
            return {k: unmarshal(v, spool, device=device) for k, v in value.items()}
        payload = value.get(_VAL)
        if tag == "tensor" and isinstance(payload, str):
            return spool.read(
                payload, str(value.get("dtype")), [int(n) for n in value.get("shape") or []], device
            )
        if tag == "component":
            if (
                set(value) != {_TAG, _VAL}
                or not isinstance(payload, list)
                or len(payload) != 2
                or not all(isinstance(part, str) for part in payload)
                or (payload[0], payload[1]) not in spool.components
            ):
                raise UncrossableArgument("component reference has no prepared active owner")
            return spool.components[(payload[0], payload[1])]
        if tag == "tuple" and isinstance(payload, list):
            return tuple(unmarshal(v, spool, device=device) for v in payload)
        if tag == "output" and isinstance(payload, dict):
            module, _, qualname = str(value.get("class", "")).partition(":")
            kind = getattr(sys.modules.get(module), qualname, None)
            if not (
                isinstance(kind, type)
                and dataclasses.is_dataclass(kind)
                and issubclass(kind, OrderedDict)
            ):
                raise UncrossableArgument(
                    f"model output class {value.get('class')!r} is not loaded"
                )
            return kind(**{k: unmarshal(v, spool, device=device) for k, v in payload.items()})
        raise UncrossableArgument(
            f"malformed wire envelope: tag={tag!r} payload={type(payload).__name__}"
        )
    if isinstance(value, list):
        return [unmarshal(v, spool, device=device) for v in value]
    return value


#: dtypes a point-to-point result may carry, by position in its header
_P2P_DTYPES = ("float32", "float16", "bfloat16", "float64", "int64", "int32", "uint8", "bool")
_P2P_RANK = 8
#: header codes for a call that sends no tensor: declined (the rank had no room) or failed
_P2P_DECLINED = -1
_P2P_FAILED = -2


def send_tensor(tensor: Any, dst: int, pg: Any) -> None:
    """One tensor to rank `dst` over the group: a fixed header (dtype, shape), then the data.
    A decoded video clip is ~200 MB; the attempt spool would copy it through the host twice."""
    import torch

    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _P2P_DTYPES or tensor.ndim > _P2P_RANK:
        raise UncrossableArgument(f"a {dtype} tensor of rank {tensor.ndim} has no p2p header")
    _send_head([_P2P_DTYPES.index(dtype), tensor.ndim, *tensor.shape], dst, pg, tensor.device)
    torch.distributed.send(tensor.contiguous(), dst=dst, group=pg)


def send_nothing(dst: int, pg: Any, device: Any, *, failed: bool) -> None:
    """A header with no tensor behind it, so a receiver waiting on this rank never hangs."""
    _send_head([_P2P_FAILED if failed else _P2P_DECLINED, 0], dst, pg, device)


def _send_head(head: list[int], dst: int, pg: Any, device: Any) -> None:
    import torch

    head += [0] * (_P2P_RANK + 2 - len(head))
    torch.distributed.send(torch.tensor(head, dtype=torch.int64, device=device), dst=dst, group=pg)


def recv_tensor(src: int, pg: Any, device: Any) -> Any:
    """The tensor rank `src` sent, or None when it sent a header alone (its reply says why)."""
    import torch

    head = torch.empty(_P2P_RANK + 2, dtype=torch.int64, device=device)
    torch.distributed.recv(head, src=src, group=pg)
    code, ndim, *shape = head.tolist()
    if code < 0:
        return None
    out = torch.empty(shape[:ndim], dtype=getattr(torch, _P2P_DTYPES[code]), device=device)
    torch.distributed.recv(out, src=src, group=pg)
    return out


def run_command(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], spool: TensorSpool
) -> dict[str, Any]:
    """Marshal one component call as the `run` command every follower receives."""
    return {
        "cmd": "run",
        "attention_layout": asdict(layout) if (layout := _ACTIVE_LAYOUT.get()) else None,
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


@contextmanager
def attention_context(command: Mapping[str, Any]) -> Iterator[None]:
    """Recreate only the leader's closed attention layout, never ambient Python context."""
    value = command.get("attention_layout")
    layout = None
    if value is not None:
        if not isinstance(value, dict) or set(value) != {
            "live_tokens",
            "protected_prefix",
            "step",
            "dense_until_step",
            "dense_paths",
        }:
            raise UncrossableArgument("malformed mirrored attention layout")
        paths = value["dense_paths"]
        if not isinstance(paths, (list, tuple)):
            raise UncrossableArgument("mirrored attention dense_paths must be a sequence")
        try:
            layout = AttentionLayout(**{**value, "dense_paths": tuple(paths)})
        except (TypeError, ValueError) as exc:
            raise UncrossableArgument(f"malformed mirrored attention layout: {exc}") from exc
    token = _ACTIVE_LAYOUT.set(layout)
    try:
        yield
    finally:
        _ACTIVE_LAYOUT.reset(token)
