"""Every CUDA stream the Runtime makes. Creating one takes device memory the driver allocates
outside torch's allocator, so on a card filled to the byte it fails (`cuStreamCreateWithPriority`,
run 14188: the LM engine's graph stream beside five packages' resident executors) where an
allocation would have made room. Made through the memory's launch (`tensorfs.varena.cuts.launch`,
installed with the memory), it runs again with room made, as a launch its dispatch never sees:
rungs 1-2 (other processes' idle residents asked), then rung 4. `torch` is passed in: this module
imports none, so a session without a GPU never does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: the memory's launch, installed with the memory; without one a stream is made as written
_LAUNCH: Callable[..., Any] | None = None


def install(launch: Callable[..., Any] | None) -> None:
    global _LAUNCH
    _LAUNCH = launch


def stream(torch: Any, device: Any = None) -> Any:
    """A new stream on `device`, the current GPU by default."""
    at = torch.device("cuda" if device is None else device)
    if at.index is None:
        at = torch.device("cuda", torch.cuda.current_device())

    def make() -> Any:
        return torch.cuda.Stream(at)

    return make() if _LAUNCH is None else _LAUNCH(at, "a CUDA stream", make)
