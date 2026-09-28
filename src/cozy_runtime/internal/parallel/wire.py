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
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout
from cozy_runtime.internal import execution_evidence
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.parallel.plan import GroupRefusal

_TAG = "__sp__"


class UncrossableArgument(GroupRefusal):
    """A model-call argument outside the wire's value vocabulary."""

    default_code = "uncrossable_argument"


class _Tensor(msgspec.Struct, frozen=True, tag_field=_TAG, tag="tensor"):
    """A tensor's bytes in the spool file `v`, with the exact dtype and shape."""

    v: str
    dtype: str
    shape: list[int]


class _Component(msgspec.Struct, frozen=True, tag_field=_TAG, tag="component"):
    """A prepared component root, by (model, component)."""

    v: tuple[str, str]


class _Tuple(msgspec.Struct, frozen=True, tag_field=_TAG, tag="tuple"):
    v: list[Json]


class _Output(msgspec.Struct, frozen=True, tag_field=_TAG, tag="output"):
    """A model output (transformers' `ModelOutput`): rebuilt only from a class the
    receiving rank has already imported."""

    v: dict[str, Json]
    kind: str = msgspec.field(name="class")


_Envelope = _Tensor | _Component | _Tuple | _Output


class Scope(msgspec.Struct, frozen=True):
    """One component scope active on the leader, as a follower re-enters it. The residency
    facts are present exactly when the leader's model has a residency plane."""

    model: str
    method: str
    components: tuple[str, ...]
    placement: str | msgspec.UnsetType = msgspec.UNSET
    headroom_bytes: int | msgspec.UnsetType = msgspec.UNSET
    scope_headroom_bytes: dict[str, int] | msgspec.UnsetType = msgspec.UNSET
    measured_scopes: list[str] | msgspec.UnsetType = msgspec.UNSET


class RunCommand(msgspec.Struct, frozen=True, tag_field="cmd", tag="run"):
    """One component call as every follower receives it: a sharded forward, a hosted module
    call (`module`, `results`) or a spread method call (`method`, `p2p`, `spare`)."""

    component: tuple[str, str]
    spool: str
    payloads: str
    scopes: tuple[Scope, ...]
    grad_enabled: bool
    inference_mode: bool
    #: device kind -> autocast dtype name
    autocast: dict[str, str]
    #: the leader's closed attention layout, never ambient Python context
    attention_layout: AttentionLayout | None
    args: list[Json]
    kwargs: dict[str, Json]
    module: str | msgspec.UnsetType = msgspec.UNSET
    results: str | msgspec.UnsetType = msgspec.UNSET
    method: str | msgspec.UnsetType = msgspec.UNSET
    p2p: bool | msgspec.UnsetType = msgspec.UNSET
    spare: bool | msgspec.UnsetType = msgspec.UNSET

    @classmethod
    def read(cls, command: Mapping[str, object]) -> RunCommand:
        try:
            return msgspec.convert(command, cls)
        except msgspec.ValidationError as exc:
            raise UncrossableArgument(f"malformed mirrored call: {exc}") from exc


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


def marshal(value: object, spool: TensorSpool, *, path: str = "") -> Json:
    """One call argument -> JSON-native tagged data, or refuse by name."""
    import torch

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        dtype = str(value.dtype).removeprefix("torch.")
        return msgspec.to_builtins(_Tensor(spool.write(value), dtype, list(value.shape)))
    if isinstance(value, torch.nn.Module):
        key = spool.component_names.get(id(value))
        if key is None:
            raise UncrossableArgument(
                f"{path or 'argument'} is not a prepared component in an active model scope"
            )
        return msgspec.to_builtins(_Component(key))
    if isinstance(value, OrderedDict) and dataclasses.is_dataclass(type(value)):
        kind = type(value)
        items = {k: marshal(v, spool, path=f"{path}.{k}") for k, v in value.items()}
        return msgspec.to_builtins(_Output(items, f"{kind.__module__}:{kind.__qualname__}"))
    if isinstance(value, tuple):
        members = [marshal(v, spool, path=f"{path}[{i}]") for i, v in enumerate(value)]
        return msgspec.to_builtins(_Tuple(members))
    if isinstance(value, list):
        return [marshal(v, spool, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out: dict[str, Json] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UncrossableArgument(
                    f"{path or 'argument'}: dict key {key!r} is {type(key).__name__}, and only "
                    "str keys cross to the group's other GPUs"
                )
            if key == _TAG:
                raise UncrossableArgument(
                    f"{path or 'argument'}: {_TAG!r} is the wire's own tag key and cannot be a "
                    "model-call dict key"
                )
            out[key] = marshal(item, spool, path=f"{path}.{key}" if path else key)
        return out
    raise UncrossableArgument(
        f"{path or 'argument'} is a {type(value).__name__}, which cannot cross to the group's "
        "other GPUs. "
        "A mirrored model call carries JSON scalars, lists, tuples, str-keyed dicts, torch "
        "tensors (through the attempt spool); a generator, closure, "
        "callback, a request view or a live handle cannot be sent to another GPU's process"
    )


def unmarshal(value: Json, spool: TensorSpool, *, device: Any) -> object:
    """Wire data -> the argument, on THIS rank's device."""
    if isinstance(value, list):
        return [unmarshal(v, spool, device=device) for v in value]
    if not isinstance(value, dict):
        return value
    if _TAG not in value:
        return {k: unmarshal(v, spool, device=device) for k, v in value.items()}
    try:
        envelope = msgspec.convert(value, _Envelope)
    except msgspec.ValidationError as exc:
        raise UncrossableArgument(f"malformed wire envelope: {exc}") from exc
    if isinstance(envelope, _Tensor):
        return spool.read(envelope.v, envelope.dtype, envelope.shape, device)
    if isinstance(envelope, _Component):
        if envelope.v not in spool.components:
            raise UncrossableArgument("component reference has no prepared active owner")
        return spool.components[envelope.v]
    if isinstance(envelope, _Tuple):
        return tuple(unmarshal(v, spool, device=device) for v in envelope.v)
    module, _, qualname = envelope.kind.partition(":")
    kind = getattr(sys.modules.get(module), qualname, None)
    if not (
        isinstance(kind, type) and dataclasses.is_dataclass(kind) and issubclass(kind, OrderedDict)
    ):
        raise UncrossableArgument(f"model output class {envelope.kind!r} is not loaded")
    return kind(**{k: unmarshal(v, spool, device=device) for k, v in envelope.v.items()})


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


class RunReply(msgspec.Struct, frozen=True):
    """A follower's answer to one `RunCommand`."""

    ok: bool
    code: str = ""
    detail: str = ""
    #: Sol attention calls by kind, on a sharded forward
    sol_calls: dict[str, int] = {}
    working_peak_bytes: int = 0
    evidence: execution_evidence.RankEvidence | None = None
    #: a hosted call's marshalled result
    result: Json = None
    #: why a spread call was declined; "" when it ran
    declined: str = ""


def run_call(
    command: Mapping[str, object], spool: TensorSpool, *, device: Any
) -> tuple[tuple[object, ...], dict[str, object]]:
    """A follower's side of a `RunCommand`: its arguments on THIS rank's device."""
    run = RunCommand.read(command)
    args = tuple(unmarshal(a, spool, device=device) for a in run.args)
    return args, {k: unmarshal(v, spool, device=device) for k, v in run.kwargs.items()}


@contextmanager
def attention_context(command: Mapping[str, object]) -> Iterator[None]:
    """Recreate only the leader's closed attention layout, never ambient Python context."""
    token = _ACTIVE_LAYOUT.set(RunCommand.read(command).attention_layout)
    try:
        yield
    finally:
        _ACTIVE_LAYOUT.reset(token)
