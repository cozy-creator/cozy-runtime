NOT_SUPPORTED: int

def error_name(code: int) -> str: ...
def fwd(
    device: int,
    q: int,
    k: int,
    v: int,
    out: int,
    lse: int,
    bf16: bool,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    heads: int,
    heads_k: int,
    head_dim: int,
    q_strides: tuple[int, int, int],
    k_strides: tuple[int, int, int],
    v_strides: tuple[int, int, int],
    o_strides: tuple[int, int, int],
    scale: float,
    causal: bool,
    semaphore: int,
    stream: int,
) -> int: ...
