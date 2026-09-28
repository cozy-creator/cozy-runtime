"""One bridge for Runtime-owned backends into Diffusers' closed backend enum."""

from collections.abc import Callable, Sequence
from typing import Any

# Diffusers' backend functions and enum members are `Any`: diffusers is not in the check venv.


def register(
    name: str,
    function: Callable[..., Any],
    *,
    constraints: Sequence[Callable[..., Any]] = (),
    supports_context_parallel: bool = False,
) -> Any:
    """Register once without changing Diffusers' global active backend.

    Diffusers 0.40 accepts registry entries but converts every dispatch name through
    its closed enum. Extend that value lookup in one place; keep the normal registry
    argument/constraint/parallelism checks. Never replace an existing implementation.
    """
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

    found = AttentionBackendName._value2member_map_.get(name)
    if found is not None:
        if _AttentionBackendRegistry._backends.get(found) is not function:
            raise ValueError(f"attention backend {name!r} already has another implementation")
        return found
    member = str.__new__(AttentionBackendName, name)
    member._name_ = name.upper()
    member._value_ = name
    _AttentionBackendRegistry.register(
        member, constraints=list(constraints), supports_context_parallel=supports_context_parallel
    )(function)
    AttentionBackendName._value2member_map_[name] = member
    return member
