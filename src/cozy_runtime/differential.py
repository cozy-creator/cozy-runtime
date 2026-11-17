"""Unstable: for the release gate only, outside the stable API; it may change or go in any release.

The varena suite's differential arms (cozy-creator `release-gate/suites/varena/differential`): a
gate's package copy calls `arm` once at import. The Runtime then digests every component root
call's outputs, records each split decision and, by arm, turns on varena's checks, presses the card
before the second request, or runs every component root as written. The rows go in the execution
record as `loops`. A package can't import the Runtime's internals (cr-329: Runtime 0.48's contract
refused the hook that did), so the arms live here. Not the author API: inert unless the machine is a
dev machine (`dev = true` in its config.toml, its executors' Start).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from types import ModuleType

    from tensorfs.varena._facts import SplitSelection
    from torch import Tensor
    from torch.nn import Module

#: A split site's run of one decision: [level, basis, calls].
type Mark = list[str | int]
type Menu = Sequence[tuple[str, Sequence[int]]]
type Press = Literal["", "refill", "hold", "eager"]


class Row(TypedDict, total=False):
    """A request's row of `loops`: each component's output digest, each split site's decisions."""

    steps: int
    digests: dict[str, str]
    split_trace: dict[str, list[Mark]]
    checks: dict[str, int]


#: What a press leaves free: less than SDXL's UNet, so its weights can't all stay; a block fits.
LEFT = 4 << 30

_armed: list[tuple[dict[str, bool], Press]] = []
_sums: dict[str, dict[str, hashlib._Hash]] = {}
_traces: dict[str, dict[str, list[Mark]]] = {}
_rows: dict[str, list[Row]] = {}
_requests: list[str] = []
_held: list[Tensor] = []


def arm(checks: Mapping[str, bool] | None = None, press: Press = "") -> None:
    """This process's arm, once, on a dev machine only (elsewhere nothing). `checks` names
    varena's (`unmap_idle`, `perturb`, `guards`); `press` is what the second request meets:
    "refill", the card filled and let go before it, so a kept graph's weights left and come back;
    "hold", kept filled through it; "eager", every component root runs as written, no graph."""
    from cozy_runtime import graphs

    if not graphs.AUDIT:  # set from the executor's Start: the machine's `dev = true`
        return
    import tensorfs.varena.torch as vt
    import torch
    from cozy_runtime.author._errors import DoesNotFit
    from cozy_runtime.internal import executor
    from cozy_runtime.internal.execution import catch

    if _armed:
        raise RuntimeError(f"differential.arm was already called: {_armed[0]}")
    wanted = dict(checks or {})
    _armed.append((wanted, press))
    vt.checks(**wanted)
    dispatch, record, select = graphs._dispatch, executor.Executor._execution, catch.select

    def digested(
        owner: ModuleType,
        module: Module,
        call: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        _press(torch, module, press)
        result = dispatch(owner, module, call, args, kwargs)  # type: ignore[arg-type]
        capturing = torch.cuda.is_current_stream_capturing()
        if not (graphs.BYPASS.get() or graphs._WARMING.get() or capturing):
            _add(torch, vt, graphs._name(module), result)
        return result

    def selected(device: int, key: str, name: str, menu: Menu, stream: int) -> SplitSelection:
        try:
            selection = select(device, key, name, menu, stream)
        except DoesNotFit:
            _mark(catch, key, len(menu) - 1, "refused")
            raise
        _mark(catch, key, int(selection["choice_id"]), str(selection["basis"]))
        return selection

    def recorded(*args: object, **kwargs: object) -> object:
        """The execution record with this request's rows as `loops`; a held press lets go."""
        _held.clear()
        document = record(*args, **kwargs)  # type: ignore[arg-type]
        rows = [row for held in _rows.values() for row in held]
        _rows.clear()
        return {**document, "loops": rows} if rows else document

    def written(
        torch: ModuleType,
        module: Module,
        recorder: object,
        call: graphs._Call,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        return call(module, *args, **kwargs)

    graphs._dispatch = digested  # type: ignore[assignment]
    catch.select = selected
    executor.Executor._execution = recorded  # type: ignore[assignment, method-assign]
    if press == "eager":
        graphs._region_call = written  # type: ignore[assignment]


def _bytes(torch: ModuleType, tensor: Tensor) -> bytes:
    held = tensor.detach().cpu()
    wide = held.to(torch.float64) if held.is_floating_point() else held
    return bytes(wide.numpy().tobytes())


def _add(torch: ModuleType, vt: ModuleType, name: str, value: object) -> None:
    from cozy_runtime.internal.residency import _REQUEST

    request = _REQUEST.get()
    if request is None:
        return
    rows = _rows.setdefault(request.id, [])
    row = next((row for row in rows if "digests" in row), None)
    if row is None:
        row = Row(steps=0, digests={}, split_trace=_traces.setdefault(request.id, {}))
        rows.append(row)
        _sums[request.id] = {}
    digest = _sums[request.id].setdefault(name, hashlib.sha256())
    for leaf in torch.utils._pytree.tree_leaves(value):
        if isinstance(leaf, torch.Tensor):
            digest.update(_bytes(torch, leaf))
    row["digests"][name] = digest.hexdigest()
    row["checks"] = vt.check_counts()


def _press(torch: ModuleType, module: Module, press: Press) -> None:
    """Before the second request's first component call: the card filled, and let go ("refill")
    or kept ("hold")."""
    from cozy_runtime.internal.contract import unsealed
    from cozy_runtime.internal.residency import _REQUEST

    request = _REQUEST.get()
    if request is None or request.id in _requests:
        return
    _requests.append(request.id)
    if press in ("refill", "hold") and len(_requests) == 2:
        device = next(module.parameters()).device
        size = unsealed(torch, "mem_get_info")(device)[1] - LEFT  # all but LEFT of the card
        held = torch.empty(size, dtype=torch.uint8, device=device)
        held.fill_(1)
        if press == "hold":
            _held.append(held)


def _mark(catch: ModuleType, key: str, level: int, basis: str) -> None:
    request = catch.CURRENT.get()
    if request is not None:
        trace = _traces.setdefault(request.id, {}).setdefault(key, [])
        mark: Mark = [level, basis]
        if trace and trace[-1][:2] == mark:
            trace[-1][2] = int(trace[-1][2]) + 1  # a mark's third item is its count
        else:
            trace.append([*mark, 1])
