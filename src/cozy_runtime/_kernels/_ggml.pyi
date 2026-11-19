from collections.abc import Callable
from typing import Protocol

class _Buffer(Protocol):
    def data_ptr(self) -> int: ...

NOT_SUPPORTED: int
OUT_OF_MEMORY: int
UNKNOWN: int
KERNELS: tuple[str, str, str, str]
ACTIVATIONS: tuple[str, str, str]
UPSTREAM: str

def error_name(code: int) -> str: ...
def error() -> str: ...
def kernel(device: int, type: int, K: int, N: int, M: int, w: int) -> int: ...
def matmul(
    device: int,
    type: int,
    w: int,
    x: int,
    x_type: int,
    out: int,
    K: int,
    N: int,
    M: int,
    allocate: Callable[[int], _Buffer] | None,
    stream: int,
) -> int: ...
def quantized_bytes(K: int, M: int) -> int: ...
def quantize(device: int, x: int, x_type: int, q8_1: int, K: int, M: int, stream: int) -> int: ...
def matmul_quantized(
    device: int,
    type: int,
    w: int,
    q8_1: int,
    out: int,
    K: int,
    N: int,
    M: int,
    allocate: Callable[[int], _Buffer] | None,
    stream: int,
) -> int: ...
def ids_batch(device: int, type: int) -> int: ...
def matmul_ids(
    device: int,
    type: int,
    table: int,
    x: int,
    rows: int,
    ids: int,
    out: int,
    K: int,
    N: int,
    T: int,
    k: int,
    allocate: Callable[[int], _Buffer] | None,
    stream: int,
) -> int: ...
