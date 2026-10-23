"""Every CUDA stream the Runtime makes. Creating one takes device memory the driver allocates
outside torch's allocator: the machine's driver headroom covers it. `torch` is passed in: this
module imports none, so a session without a GPU never does. Graph recording, the LM engine and the
executor's links all make theirs here: beside graphs, not in `internal`, which `cozy_runtime.lm`
may not import.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import device as Device
    from torch.cuda import Stream


def stream(torch: ModuleType, device: Device | str | int | None = None) -> Stream:
    """A new stream on `device`, the current GPU by default."""
    at = torch.device("cuda" if device is None else device)
    if at.index is None:
        at = torch.device("cuda", torch.cuda.current_device())
    made: Stream = torch.cuda.Stream(at)
    return made
