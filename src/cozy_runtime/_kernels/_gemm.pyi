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
    up: int,
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
def w4a8_gemm(
    device: int,
    q: int,
    scales: int,
    w: int,
    s: int,
    codebook: int,
    w_scale: int,
    bias: int,
    out: int,
    out_half: bool,
    M: int,
    N: int,
    K: int,
    config: int,
    stream: int,
) -> int: ...
def w4a8_unpack(
    device: int, w: int, s: int, codebook: int, out: int, N: int, K: int, stream: int
) -> int: ...
def svdq_quantize(
    device: int,
    x: int,
    x_half: bool,
    p: int,
    g: int,
    down: int,
    act: int,
    ascales: int,
    la: int,
    M: int,
    K: int,
    R: int,
    stream: int,
) -> int: ...
def svdq_gemm(
    device: int,
    act: int,
    ascales: int,
    w: int,
    wscales: int,
    gx: int,
    gw: int,
    c: int,
    la: int,
    up: int,
    bias: int,
    out: int,
    out_half: bool,
    M: int,
    N: int,
    K: int,
    R: int,
    stream: int,
) -> int: ...
