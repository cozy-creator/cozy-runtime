#: A ring slot's header: its sequence number (0 while it is written), then the record's length.
HEADER: int

class Segment:
    """A captured step graph (`raw_cuda_graph()`) run as the body of a CUDA WHILE node: one launch runs
    steps until a step writes 0 to the int32 at device pointer `go`. The step is cloned in, so the
    caller's graph may be replayed alone; its memory must outlive the Segment."""

    def __init__(self, step: int, go: int, device: int) -> None: ...
    def launch(self, stream: int) -> None:
        """One launch on `stream`; the GIL is released and nothing waits on the GPU."""

    def close(self) -> None: ...
    @property
    def copies(self) -> list[tuple[int, int, int]]:
        """The body's 1-D copies as (dst, src, bytes), in the step graph's node order."""

    def set_copy(self, index: int, dst: int, src: int, nbytes: int) -> None:
        """Copy `index`'s operands and size for later launches, through the parent's executable."""

    def enable_copy(self, index: int, on: bool) -> None: ...

def report(
    ring: int, slots: int, slot_bytes: int, counter: int, src: int, nbytes: int, stream: int
) -> None:
    """Launch, on the current device (capturable as a step's last op): `nbytes` from device `src` into
    slot `*counter % slots` of a zeroed ring in pinned, device-mapped host memory, its sequence number
    `*counter + 1` written last, then `*counter += 1` (an int64 on the device)."""

class Reader:
    """One consumer of the records `report` writes to a ring (its host address)."""

    def __init__(self, ring: int, slots: int, slot_bytes: int) -> None: ...
    def poll(self) -> list[bytes]:
        """Every record since the last poll, in order. Raises OverflowError naming the records the
        device overwrote before they were read, once those before them are taken; the next poll
        resumes past them."""

    @property
    def next(self) -> int:
        """The sequence number of the next record to read (the first is 1)."""
