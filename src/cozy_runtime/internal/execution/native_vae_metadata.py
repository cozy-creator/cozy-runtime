"""Contiguous activation metadata for preparation before an activation exists."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def contiguous(cls, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> TensorSpec:
        if any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("activation shape must be positive and concrete")
        return cls(shape, tuple(math.prod(shape[i + 1:]) for i in range(len(shape))), dtype, device)

    @classmethod
    def read(cls, value: Any) -> TensorSpec | None:
        if type(value) is cls:
            return value
        if (type(value) is not Tensor or value.layout != torch.strided or value.requires_grad
                or value.is_conj() or value.is_neg()):
            return None
        return cls(tuple(value.shape), tuple(value.stride()), value.dtype, value.device)

    def is_contiguous(self) -> bool:
        stride = 1
        for size, actual in zip(reversed(self.shape), reversed(self.strides), strict=True):
            if size != 1 and actual != stride:
                return False
            stride *= size
        return True

    @property
    def metadata(self) -> tuple[Any, ...]:
        return self.shape, self.strides, self.dtype, self.device
