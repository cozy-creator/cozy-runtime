from torch import Tensor

def rms_rope_split_half(
    q: Tensor,
    k: Tensor,
    freqs: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    q_out: Tensor,
    k_out: Tensor,
    epsilon: float,
    stream_ptr: int,
    rot_dim: int = ...,
) -> None: ...
