from collections.abc import Callable
from typing import Protocol

class _Buffer(Protocol):
    def data_ptr(self) -> int: ...

NOT_SUPPORTED: int
OUT_OF_MEMORY: int
UNKNOWN: int
KERNELS: tuple[str, str, str, str]
UPSTREAM: str

def error_name(code: int) -> str: ...
def error() -> str: ...
def kernel(device: int, type: int, K: int, N: int, M: int, w: int) -> int: ...
def matmul(
    device: int,
    type: int,
    w: int,
    x: int,
    out: int,
    K: int,
    N: int,
    M: int,
    allocate: Callable[[int], _Buffer] | None,
    stream: int,
) -> int: ...
