from typing import Any

def rms_rope_split_half(
    q: Any,
    k: Any,
    freqs: Any,
    q_scale: Any,
    k_scale: Any,
    q_out: Any,
    k_out: Any,
    epsilon: float,
    stream_ptr: int,
    rot_dim: int = ...,
) -> None: ...
