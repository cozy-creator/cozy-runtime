"""The values one GPU of a group hands the others (`spmd.Ranks.cross`): JSON for the structure,
the tensors apart, each crossing by one broadcast from the GPU that holds it.

A value is JSON scalars, lists, tuples, str-keyed dicts, tensors and model outputs
(transformers' `ModelOutput`); anything else is refused by name.
"""

from __future__ import annotations

import dataclasses
import sys
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import msgspec

from cozy_runtime.author._errors import Origin, Terminal
from cozy_runtime.internal import execution_evidence
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.parallel.plan import GroupRefusal

_TAG = "__sp__"


class UncrossableArgument(GroupRefusal):
    """A value outside the wire's vocabulary."""

    default_code = "uncrossable_argument"


class _Tensor(msgspec.Struct, frozen=True, tag_field=_TAG, tag="tensor"):
    """The `v`th tensor of the value; `host`: it lives in host memory on the GPU that sent it."""

    v: int
    dtype: str
    shape: list[int]
    host: bool


class _Tuple(msgspec.Struct, frozen=True, tag_field=_TAG, tag="tuple"):
    v: list[Json]


class _Output(msgspec.Struct, frozen=True, tag_field=_TAG, tag="output"):
    """A model output, rebuilt only from a class the receiving GPU has already imported."""

    v: dict[str, Json]
    kind: str = msgspec.field(name="class")


_Envelope = _Tensor | _Tuple | _Output


def marshal(value: object, tensors: list[Any], *, path: str = "") -> Json:
    """`value` as JSON-native tagged data; its tensors are appended to `tensors`."""
    import torch

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        tensors.append(value)
        dtype = str(value.dtype).removeprefix("torch.")
        host = value.device.type == "cpu"
        return msgspec.to_builtins(_Tensor(len(tensors) - 1, dtype, list(value.shape), host))
    if isinstance(value, OrderedDict) and dataclasses.is_dataclass(type(value)):
        kind = type(value)
        items = {k: marshal(v, tensors, path=f"{path}.{k}") for k, v in value.items()}
        return msgspec.to_builtins(_Output(items, f"{kind.__module__}:{kind.__qualname__}"))
    if isinstance(value, tuple):
        return msgspec.to_builtins(
            _Tuple([marshal(v, tensors, path=f"{path}[{i}]") for i, v in enumerate(value)])
        )
    if isinstance(value, list):
        return [marshal(v, tensors, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out: dict[str, Json] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key == _TAG:
                raise UncrossableArgument(
                    f"{path or 'value'}: dict key {key!r} cannot cross to the group's other GPUs"
                )
            out[key] = marshal(item, tensors, path=f"{path}.{key}" if path else key)
        return out
    raise UncrossableArgument(
        f"{path or 'value'} is a {type(value).__name__}, which cannot cross to the group's other "
        "GPUs: a value that crosses is JSON scalars, lists, tuples, str-keyed dicts, tensors "
        "and model outputs"
    )


def specs(value: Json) -> list[_Tensor]:
    """The tensor envelopes in wire data, in the order `marshal` met them."""
    return sorted(_tensors(value), key=lambda t: t.v)


def _tensors(value: Json) -> Iterator[_Tensor]:
    if isinstance(value, list):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        if value.get(_TAG) == "tensor":
            yield msgspec.convert(value, _Tensor)
        else:
            for item in value.values():
                yield from _tensors(item)


def unmarshal(value: Json, tensors: list[Any]) -> object:
    """Wire data with its tensors back in place."""
    if isinstance(value, list):
        return [unmarshal(v, tensors) for v in value]
    if not isinstance(value, dict):
        return value
    if _TAG not in value:
        return {k: unmarshal(v, tensors) for k, v in value.items()}
    try:
        envelope = msgspec.convert(value, _Envelope)
    except msgspec.ValidationError as exc:
        raise UncrossableArgument(f"malformed wire envelope: {exc}") from exc
    if isinstance(envelope, _Tensor):
        return tensors[envelope.v]
    if isinstance(envelope, _Tuple):
        return tuple(unmarshal(v, tensors) for v in envelope.v)
    module, _, qualname = envelope.kind.partition(":")
    kind = getattr(sys.modules.get(module), qualname, None)
    if not (
        isinstance(kind, type) and dataclasses.is_dataclass(kind) and issubclass(kind, OrderedDict)
    ):
        raise UncrossableArgument(f"model output class {envelope.kind!r} is not loaded")
    return kind(**{k: unmarshal(v, tensors) for k, v in envelope.v.items()})


#: dtypes a spread result may carry (`spread`: one tensor per call)
DTYPES: tuple[str, ...] = ("float32", "float16", "bfloat16", "float64", "int64", "int32")
DTYPES += ("uint8", "bool", "int8", "int16", "float8_e4m3fn", "float8_e5m2")


def one_tensor(value: object) -> None:
    """A spread call returns one tensor (`author.spread`)."""
    dtype = str(getattr(value, "dtype", "")).removeprefix("torch.")
    if dtype not in DTYPES or not hasattr(value, "shape"):
        raise UncrossableArgument(
            f"a spread call returns one tensor, not {type(value).__name__} {dtype}".rstrip()
        )


class RunReply(msgspec.Struct, frozen=True):
    """A follower's answer to its `run`: how its handler ended, and its part of the record."""

    ok: bool
    code: str = ""
    detail: str = ""
    terminal: Terminal = "failed"
    origin: Origin = "runtime"
    #: the shape of its result (`spmd.signature`); GPU 0 compares it with its own
    signature: str = ""
    #: its last store meeting, so GPU 0 can clear the attempt's keys
    meetings: int = 0
    #: Sol attention calls by kind
    sol_calls: dict[str, int] = {}
    working_peak_bytes: int = 0
    evidence: execution_evidence.GpuEvidence | None = None
    #: its own varena request's record (`executor_replies.GpuMemory`)
    memory: Json = None
    #: its kernels not serving yet (`kernel_ready.pending`), and those it began serving
    kernels: dict[str, str] = {}
    began: list[str] = []
