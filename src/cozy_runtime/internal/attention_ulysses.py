"""One Ulysses wrapper that makes every Runtime-local attention kernel context-parallel.

After Diffusers' unequal-length all-to-all each rank holds the WHOLE sequence for its head
subset, so a forward-only kernel whose reductions stay within one head computes exactly what
one GPU computes for those heads. `member` registers a `Local` kernel as a Diffusers backend:
without a parallel config it calls the kernel, with Runtime's Ulysses config it exchanges
around it. Ring attention needs LSE merging and is refused. A kernel with a cross-head
reduction (`head_local=False`) is registered without context parallelism.
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
class Local:
    """A forward-only kernel over [batch, sequence, heads, dim].

    `forward(query, key, value, *, attn_mask, dropout_p, is_causal, scale, enable_gqa)`
    validates what it serves; `impl(query, key, value, options)` names the concrete
    implementation serving those operands, e.g. the SDPA backend torch dispatches. `rows`: the
    query block a call may be cut at and compute each row as whole (memory's `attend` cut);
    0 where the kernel needs every query row at once (a scale or mean over the sequence).
    """

    forward: Callable[..., Tensor]
    impl: Callable[..., str]
    head_local: bool = True
    rows: int = 0


#: backend -> implementations served in the observed scope on this rank, and the operand
#: signatures already named there (an implementation is named once per signature, not per call)
_SEEN: ContextVar[tuple[dict[str, set[str]], set[object]] | None] = ContextVar(
    "cozy_attention_seen", default=None
)
_MEMBERS: dict[str, AttentionBackendName] = {}


class Execution(Protocol):
    """The execution layer's side (`internal.execution`, installed with the memory): the
    attention cut, the pass's agreed exchange level and its exchanged sizes."""

    def attend(
        self,
        call: Callable[..., Tensor],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        *,
        rows: int = 0,
        heads: bool = True,
    ) -> Tensor: ...

    def agreed(self) -> int: ...

    def sizes(self, gather: Callable[[], list[int]]) -> list[int]: ...


_EXECUTION: Execution | None = None
#: the exchanges' one cut site (`execution.agree.SITE`)
SITE = "ulysses"


def install(execution: Execution | None) -> None:
    global _EXECUTION
    _EXECUTION = execution


#: the torch side, served only after the executor imports torch.
_GROUP: Callable[[object], ProcessGroup] | None = None
_EXCHANGE: Callable[..., Tensor] | None = None


def serve(group_of: Callable[[object], ProcessGroup], exchange_of: Callable[..., Tensor]) -> None:
    global _GROUP, _EXCHANGE
    _GROUP, _EXCHANGE = group_of, exchange_of


@contextmanager
def observing() -> Iterator[dict[str, set[str]]]:
    """Collect, per backend, the implementations that served on this rank in this scope."""
    seen: dict[str, set[str]] = {}
    token = _SEEN.set((seen, set()))
    try:
        yield seen
    finally:
        _SEEN.reset(token)


def member(backend: str, local: Local) -> AttentionBackendName:
    """The Diffusers member for `backend`, registered once per process."""
    found = _MEMBERS.get(backend)
    if found is None:
        found = register(
            backend, _dispatch(backend, local), supports_context_parallel=local.head_local
        )
        _MEMBERS[backend] = found
    return found


def _dispatch(backend: str, local: Local) -> Callable[..., Tensor]:
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
        )

    return attention


def _call(
    backend: str, local: Local, query: Tensor, key: Tensor, value: Tensor, options: Options
) -> Tensor:
    scope = _SEEN.get()
    if scope is not None:
        masked, causal = options["attn_mask"] is not None, options["is_causal"]
        signature = (backend, query.shape, key.shape, query.dtype, masked, causal)
        seen, named = scope
        if signature not in named:
            named.add(signature)
            seen.setdefault(backend, set()).add(local.impl(query, key, value, options))
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
) -> Tensor:
    """Sequence shards -> full sequence per head subset -> `compute` -> back
    (`internal.parallel.ulysses.exchange`)."""
    if _EXCHANGE is None:
        raise ValueError(_UNSERVED)
    return _EXCHANGE(query, key, value, handle, compute)


_UNSERVED = "Ulysses runs where torch does: nothing imported internal.parallel.ulysses"


def describe(seen: dict[str, set[str]], names: dict[str, str]) -> str:
    """One implementation line: the bare implementation when one kernel served, else
    `kernel:impl` per kernel."""
    rows = {names.get(backend, backend): "+".join(sorted(impls)) for backend, impls in seen.items()}
    if len(rows) == 1:
        return next(iter(rows.values()))
    return ",".join(f"{name}:{impl}" for name, impl in sorted(rows.items()))


def record(backend: str, implementation: str) -> None:
    """Record a producer that projects before the ordinary Q/K/V dispatch seam."""
    scope = _SEEN.get()
    if scope is not None:
        scope[0].setdefault(backend, set()).add(implementation)
