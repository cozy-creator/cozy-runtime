"""The processor a package's `AttentionProducer` serves a site through.

Runtime selects a producer by name like any backend (`attention.known`). At each site it is
selected for, the attention module's processor is wrapped: while the site's backend is the
producer's, the module's call reaches `producer.produce` with a `ProducerCall`; any other
backend (a request's pin, a later reselection) calls the original. Restoring the original
processor restores the site exactly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from cozy_runtime.author._attention import AttentionProducer, ProducerCall
from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.author.kernels import PreQuantized
from cozy_runtime.internal import attention_sol, attention_ulysses
from cozy_runtime.internal.encoding import leaves

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


class _Processor(Protocol):
    """A Diffusers attention processor: its backend and parallel config, and its call."""

    _attention_backend: object
    _parallel_config: object

    def __call__(self, attn: Module, *args: object, **kwargs: object) -> Tensor: ...


class Processor:
    """A site's processor while a producer may serve it."""

    def __init__(
        self,
        original: _Processor,
        producer: AttentionProducer,
        backend: str,
        kernel: str,
        component: str,
        path: str,
    ) -> None:
        self.original = original
        self.producer = producer
        self.backend = backend
        self.kernel = kernel
        self.component = component
        self.path = path

    @property
    def _attention_backend(self) -> object:
        return self.original._attention_backend

    @_attention_backend.setter
    def _attention_backend(self, value: object) -> None:
        self.original._attention_backend = value

    @property
    def _parallel_config(self) -> object:
        return self.original._parallel_config

    @_parallel_config.setter
    def _parallel_config(self, value: object) -> None:
        self.original._parallel_config = value

    def __call__(self, attn: Module, *args: object, **kwargs: object) -> Tensor:
        selected = self._attention_backend
        if getattr(selected, "value", str(selected)) != self.backend:
            return self.original(attn, *args, **kwargs)
        layout, state = _ACTIVE_LAYOUT.get(), self.producer.request_state()
        if layout is None or state is None:
            raise ValueError(
                f"{self.producer.name} requires the model's attention layout and a request scope"
            )
        config = self._parallel_config
        group = (
            attention_ulysses.group(config)
            if getattr(config, "context_parallel_config", None) is not None
            else None
        )

        def exchange(
            query: Tensor,
            key: Tensor,
            value: Tensor,
            compute: Callable[[Tensor, Tensor, Tensor], Tensor],
        ) -> Tensor:
            if group is None:
                raise ValueError(f"{self.producer.name}: no Ulysses group to exchange over")
            return attention_ulysses.exchange(query, key, value, group, compute)

        def served(implementation: str) -> None:
            attention_ulysses.record(self.backend, implementation)
            attention_sol._count("sparse")
            if layout.protected_prefix:
                attention_sol._count("dense_prefix")

        def original(*call_args: object, **call_kwargs: object) -> Tensor:
            return self.original(attn, *call_args, **call_kwargs)

        call = ProducerCall(
            module=attn,
            component=self.component,
            path=self.path,
            layout=layout,
            kernel=sys.modules.get(self.kernel),
            state=state,
            group=group,
            original=original,
            quantize=_quantize,
            exchange=exchange,
            served=served,
        )
        result: Tensor = self.producer.produce(call, *args, **kwargs)
        return result


def _quantize(x: Tensor) -> PreQuantized:
    import torch

    payload, scale = leaves.quantize_activation_rowwise(torch, x)
    return PreQuantized(payload, scale, tuple(x.shape))


def install(
    module: Module,
    producer: AttentionProducer,
    backend: str,
    kernel: str,
    component: str,
    path: str,
) -> None:
    """Wrap `module`'s processor for `producer`, once; another producer's wrapper is replaced.
    `kernel` names the base's module, which resolving the base imported."""
    current = module.processor
    if isinstance(current, Processor) and current.producer is producer:
        return
    original = current.original if isinstance(current, Processor) else current
    module.set_processor(Processor(original, producer, backend, kernel, component, path))
