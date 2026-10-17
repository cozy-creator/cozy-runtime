NOT_SUPPORTED: int
OUT_OF_MEMORY: int
GEMV_ROWS: int
MAX_K: int
#: Whether CUTLASS's TMA GEMMs are built in (not on Windows).
TMA: bool

def error_name(code: int) -> str: ...
def quantize(
    device: int,
    fp8: bool,
    x: int,
    x_half: bool,
    q: int,
    scales: int,
    M: int,
    K: int,
    scale: int,
    shift: int,
    scale_stride: int,
    shift_stride: int,
    rows_per_entry: int,
    norm: int,
    eps: float,
    activation: int,
    stream: int,
) -> int: ...
def gemv(
    device: int,
    x: int,
    x_half: bool,
    w: int,
    w_scale: int,
    bias: int,
    out: int,
    out_half: bool,
    M: int,
    N: int,
    K: int,
    stream: int,
) -> int: ...
def gemm(
    device: int,
    family: int,
    fp8: bool,
    q: int,
    scales: int,
    w: int,
    w_scale: int,
    bias: int,
    out: int,
    out_half: bool,
    M: int,
    N: int,
    K: int,
    config: int,
    allocate: object,
    stream: int,
) -> int: ...
