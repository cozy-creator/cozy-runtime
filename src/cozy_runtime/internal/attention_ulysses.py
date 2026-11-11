"""One Ulysses wrapper that makes every Runtime-local attention kernel context-parallel.

After Diffusers' unequal-length all-to-all each rank holds the WHOLE sequence for its head
subset, so a forward-only kernel whose reductions stay within one head computes exactly what
one GPU computes for those heads. `function` makes a `Local` kernel's backend call: without a
parallel config it calls the kernel, with Runtime's Ulysses config it exchanges around it. A
Runtime site (`attention_site`) calls it directly; `member` registers it with Diffusers. Ring
attention needs LSE merging and is refused. A kernel with a cross-head reduction
(`head_local=False`) is registered without context parallelism.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from diffusers.models.attention_dispatch import AttentionBackendName
    from torch import Tensor
    from torch.distributed import ProcessGroup

    class PartialOptions(TypedDict, total=False):
        """Forwarding adapters may omit options and use the concrete kernel's defaults."""

        attn_mask: Tensor | None
        dropout_p: float
        is_causal: bool
        scale: float | None
        enable_gqa: bool


from cozy_runtime.internal.attention_registry import register


class Options(TypedDict):
    """The keyword options every `Local.forward` takes; `attn_mask` is a tensor or None."""

    attn_mask: Tensor | None
    dropout_p: float
    is_causal: bool
    scale: float | None
    enable_gqa: bool


@dataclass(frozen=True, slots=True)
class Local:  # type: ignore[explicit-any]  # `forward`'s (below)
    """A forward-only kernel over [batch, sequence, heads, dim].

    `forward(query, key, value, *, attn_mask, dropout_p, is_causal, scale, enable_gqa)`
    validates what it serves; `impl(query, key, value, options)` names the concrete
    implementation serving those operands, e.g. the SDPA backend torch dispatches. `rows`: the
    query block a call may be cut at and compute each row as whole (memory's `attend` cut);
    0 where the kernel needs every query row at once (a scale or mean over the sequence).
    `refusal`: why its row does not serve a call, or empty; such a call runs on `instead`, a
    (backend, kernel) tried the same way, SDPA last. `narrow`: it quantizes Q, K and V to int8 or
    narrower itself, so under Ulysses they cross as int8 (`internal.parallel.ulysses.exchange`).
    `rungs`: a ladder (`attention.Pick`), whose call takes the rung of its `min(Lq, Lk)`: each
    (upper token bound, 0 for none, backend, kernel); its own `forward` is never called. `band`:
    the label a ladder records a call of N tokens under (`attention_table.Table.band`).
    """

    #: `Options`' keywords, and a sparse kernel's `dense=` rows until `attention` binds them
    forward: Callable[..., Tensor]  # type: ignore[explicit-any]  # signatures vary until bound
    impl: Implementation
    head_local: bool = True
    rows: int = 0
    refusal: Callable[[Tensor, Tensor, Options], str] | None = None
    instead: tuple[str, Local] | None = None
    narrow: bool = False
    rungs: tuple[tuple[int, str, Local], ...] = ()
    band: Callable[[int], str] | None = None


#: `impl(query, key, value, options)`: the concrete implementation serving those operands.
type Implementation = Callable[[Tensor, Tensor, Tensor, Options], str]
#: Ulysses' exchange around a computation (`internal.parallel.ulysses.exchange`).
type Exchange = Callable[
    [
        Tensor,
        Tensor,
        Tensor,
        ProcessGroup,
        Callable[[Tensor, Tensor, Tensor], Tensor],
        str,
        bool | Callable[[int], bool],
    ],
    Tensor,
]


class BackendCall(Protocol):
    """A backend's call, as Diffusers' dispatch and a Runtime site make it (`function`)."""

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        _parallel_config: object = None,
    ) -> Tensor: ...


class Seen(dict[str, set[str]]):
    """Backend -> implementations served in an observed scope on this rank; `refused`: per
    (reason, backend that took the call), the calls a kernel's row refused, which an `instead`
    served; `bands`: per (backend a site dispatched to, the call's band), the backends that
    served it."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: dict[tuple[str, str], int] = {}
        self.bands: dict[tuple[str, str], set[str]] = {}


#: what served in the observed scope on this rank, and the operand signatures already named
#: there (an implementation is named once per signature, not per call)
_SEEN: ContextVar[tuple[Seen, set[object]] | None] = ContextVar("cozy_attention_seen", default=None)
#: The dev-only attention capture's hook (`attention_capture`), called with each call's site
#: backend, band, the backend that serves it, Q/K/V and options, right after the band pick.
CAPTURE: ContextVar[Callable[[str, str, str, Tensor, Tensor, Tensor, Options], None] | None] = (
    ContextVar("cozy_attention_capture", default=None)
)
_FUNCTIONS: dict[str, BackendCall] = {}
_MEMBERS: dict[str, AttentionBackendName] = {}
#: backend -> its kernel, as `function` made its call
_LOCALS: dict[str, Local] = {}


class Execution(Protocol):
    """The execution layer's side (`internal.execution`, installed with the memory): the
    attention cut."""

    def attend(
        self,
        call: Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        *,
        rows: int = 0,
        heads: bool = True,
    ) -> Tensor: ...


_EXECUTION: Execution | None = None


def install(execution: Execution | None) -> None:
    global _EXECUTION
    _EXECUTION = execution


#: the torch side, served only after the executor imports torch.
_GROUP: Callable[[object], ProcessGroup] | None = None
_EXCHANGE: Exchange | None = None


def serve(group_of: Callable[[object], ProcessGroup], exchange_of: Exchange) -> None:
    global _GROUP, _EXCHANGE
    _GROUP, _EXCHANGE = group_of, exchange_of


@contextmanager
def observing() -> Iterator[Seen]:
    """Collect what served on this rank in this scope."""
    seen = Seen()
    token = _SEEN.set((seen, set()))
    try:
        yield seen
    finally:
        _SEEN.reset(token)


def function(backend: str, local: Local) -> BackendCall:
    """`backend`'s call, made once per process: Diffusers' backend signature."""
    found = _FUNCTIONS.get(backend)
    if found is None:
        found = _FUNCTIONS[backend] = _dispatch(backend, local)
        _LOCALS[backend] = local
    return found


def _rung(local: Local, tokens: int) -> tuple[str, Local]:
    """A ladder's rung for a call of TOKENS (`min(Lq, Lk)`): the first whose bound is above."""
    return next((name, rung) for upper, name, rung in local.rungs if not upper or tokens < upper)


def serves(backend: str) -> tuple[str, ...]:
    """The backends a site dispatching to BACKEND serves on, by rung: a ladder's, else itself."""
    local = _LOCALS.get(backend)
    if local is None or not local.rungs:
        return (backend,)
    return tuple(dict.fromkeys(name for _, name, _ in local.rungs))


def rung(backend: str, tokens: int) -> tuple[str, str]:
    """The backend a site dispatching to BACKEND serves a call of TOKENS on, and its band."""
    local = _LOCALS.get(backend)
    if local is None or not local.rungs:
        return backend, ""
    return _rung(local, tokens)[0], local.band(tokens) if local.band is not None else ""


def opened(backend: str) -> BackendCall:
    """The call of a `backend` this process has made (`function`)."""
    found = _FUNCTIONS.get(backend)
    if found is None:
        raise ValueError(f"attention backend {backend!r} is not open in this process")
    return found


def member(backend: str, local: Local) -> AttentionBackendName:
    """The Diffusers member for `backend`, registered once per process."""
    found = _MEMBERS.get(backend)
    if found is None:
        found = register(
            backend, function(backend, local), supports_context_parallel=local.head_local
        )
        _MEMBERS[backend] = found
    return found


def _dispatch(backend: str, local: Local) -> BackendCall:
    def attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        _parallel_config: object = None,
    ) -> Tensor:
        if return_lse:
            raise ValueError(f"{backend} does not return log-sum-exp")
        options = Options(
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        if _parallel_config is None:
            return _call(backend, local, query, key, value, options)
        if not local.head_local:
            raise ValueError(f"{backend} reduces across heads and does not serve Ulysses")
        if attn_mask is not None or is_causal or query.shape[2] != key.shape[2]:
            raise ValueError(
                f"{backend} under Ulysses serves unmasked noncausal attention with equal heads"
            )
        return exchange(
            query,
            key,
            value,
            group(_parallel_config),
            lambda q, k, v: _call(backend, local, q, k, v, options),
            narrow=_narrow(local, query, options),
        )

    return attention


def _narrow(local: Local, query: Tensor, options: Options) -> Callable[[int], bool]:
    """Whether a Ulysses call of `total` tokens crosses as int8: the kernel it takes (a ladder's
    rung for `total`, the full sequence) quantizes Q, K and V itself, and so does whichever of
    its `instead` serves the call when it refuses it (`_call`)."""

    def narrow(total: int) -> bool:
        taken = _rung(local, total)[1] if local.rungs else local
        if not taken.narrow:
            return False
        whole = query.new_empty((query.shape[0], total, 1, query.shape[3]), device="meta")
        while taken.refusal is not None and taken.instead is not None:
            if not taken.refusal(whole, whole, options):
                break
            taken = taken.instead[1]
        return taken.narrow

    return narrow


def _call(
    backend: str, local: Local, query: Tensor, key: Tensor, value: Tensor, options: Options
) -> Tensor:
    # A ladder's rung first, by the call's whole length: under Ulysses this runs after the
    # exchange, so N GPUs take the rung one takes, and memory's row cut never changes it.
    site, band = backend, ""
    if local.rungs:
        tokens = min(int(query.shape[1]), int(key.shape[1]))
        band = local.band(tokens) if local.band is not None else ""
        backend, local = _rung(local, tokens)
    why = local.refusal(query, key, options) if local.refusal is not None else ""
    first, refusing = why, local.instead is not None
    while why and local.instead is not None:
        backend, local = local.instead
        why = local.refusal(query, key, options) if local.refusal is not None else ""
    if first and refusing:
        refused(first, 1, backend)
    capture = CAPTURE.get()
    if capture is None:
        return _serve(site, band, backend, local, query, key, value, options)
    capture(site, band, backend, query, key, value, options)
    # The kernel's own nested calls (Sol's dense rows) are this site call's, not new ones.
    token = CAPTURE.set(None)
    try:
        return _serve(site, band, backend, local, query, key, value, options)
    finally:
        CAPTURE.reset(token)


def _serve(
    site: str,
    band: str,
    backend: str,
    local: Local,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    options: Options,
) -> Tensor:
    """`_call`'s kernel LOCAL (BACKEND, the rung SITE's call of BAND took), recorded and run."""
    scope = _SEEN.get()
    if scope is not None:
        masked, causal = options["attn_mask"] is not None, options["is_causal"]
        signature = (site, band, backend, query.shape, key.shape, query.dtype, masked, causal)
        seen, named = scope
        if signature not in named:
            named.add(signature)
            seen.setdefault(backend, set()).add(local.impl(query, key, value, options))
            seen.bands.setdefault((site, band), set()).add(backend)
    if _EXECUTION is None or options["dropout_p"] != 0.0:
        return local.forward(query, key, value, **options)
    return _EXECUTION.attend(
        lambda q, k, v, mask: local.forward(q, k, v, **{**options, "attn_mask": mask}),
        query,
        key,
        value,
        options["attn_mask"],
        rows=0 if options["is_causal"] else local.rows,
        heads=local.head_local,
    )


def group(config: object) -> ProcessGroup:
    """The process group of Runtime's initialized Ulysses configuration; ring is refused."""
    if _GROUP is None:
        raise ValueError(_UNSERVED)
    return _GROUP(config)


def exchange(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    compute: Callable[[Tensor, Tensor, Tensor], Tensor],
    site: str = "",
    narrow: bool | Callable[[int], bool] = False,
) -> Tensor:
    """Sequence shards -> full sequence per head subset -> `compute` -> back
    (`internal.parallel.ulysses.exchange`)."""
    if _EXCHANGE is None:
        raise ValueError(_UNSERVED)
    return _EXCHANGE(query, key, value, handle, compute, site, narrow)


_UNSERVED = "Ulysses runs where torch does: nothing imported internal.parallel.ulysses"


def describe(seen: dict[str, set[str]], names: dict[str, str]) -> str:
    """One implementation line: the bare implementation when one kernel served, else
    `kernel:impl` per kernel."""
    rows = {names.get(backend, backend): "+".join(sorted(impls)) for backend, impls in seen.items()}
    if len(rows) == 1:
        return next(iter(rows.values()))
    return ",".join(f"{name}:{impl}" for name, impl in sorted(rows.items()))


def refused(why: str, calls: int = 1, took: str = "") -> None:
    """Record CALLS a kernel's row refused for WHY, which backend TOOK served."""
    scope = _SEEN.get()
    if scope is not None:
        scope[0].refused[(why, took)] = scope[0].refused.get((why, took), 0) + calls


def record(backend: str, implementation: str, site: str = "", band: str = "") -> None:
    """Record a call that did not pass `_call`: a producer that projects before the ordinary
    Q/K/V dispatch seam (SITE: the backend its site dispatches to), a graph's replay."""
    scope = _SEEN.get()
    if scope is not None:
        scope[0].setdefault(backend, set()).add(implementation)
        if site:
            scope[0].bands.setdefault((site, band), set()).add(backend)


def banded(site: str, band: str, backend: str) -> None:
    """Record that BACKEND served a call of SITE's in BAND (a graph's replay)."""
    scope = _SEEN.get()
    if scope is not None:
        scope[0].bands.setdefault((site, band), set()).add(backend)
