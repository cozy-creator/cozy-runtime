"""`preview(x0, vae)`: a small picture of what a loop is making, for a client watching its run (cr-282).

Optional, one line in a loop: after a step, the loop's estimate of the clean latents (`x0`, legible
from the first steps), or its latents as they are. With no client watching it returns at once and
computes nothing. With one, a preview is due at most every `INTERVAL_S`: the latents are projected
to RGB on their device by their VAE's linear factors, a few kilobytes come back without waiting on
the GPU, and they are encoded off the loop's thread. A picture is a WebP at the latents'
resolution; a video an animated WebP of its latent frames. The newest wins: nothing queues behind
a preview.

`x0` is a tensor, `[C, H, W]` or `[B, C, H, W]` for a picture and `[C, T, H, W]` or
`[B, C, T, H, W]` for a video (the first of a batch shows); or a sequence of tokens, `[N, C]` or
`[B, N, C]`, laid out row by row on `grid`, `(h, w)` or `(t, h, w)`; or a function of none
returning one, called only when a preview is due. Channels that carry each 2x2 patch (four times
the VAE's, as FLUX.2's, Qwen-Image's and H3's tokens do) are unpacked. `vae` names the latent
space (`FAMILIES`).
"""

from __future__ import annotations

import io
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cozy_runtime.author._context import CURRENT

if TYPE_CHECKING:
    from torch import Tensor

#: The least time between two previews of one attempt.
INTERVAL_S = 0.5
#: A preview's longest side, in latent pixels: larger latents are averaged down first.
LONGEST = 512
#: A video preview's most frames, spread over its latent frames.
FRAMES = 32
#: Each frame of a video preview, in ms.
FRAME_MS = 125

type RGB = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Family:
    """A VAE's latent space: its channels, whether it holds frames, and its linear map to RGB
    in -1..1 (ComfyUI's `latent_rgb_factors` and bias)."""

    channels: int
    factors: tuple[RGB, ...]
    bias: RGB = (0.0, 0.0, 0.0)
    video: bool = False


#: SDXL's 4-channel space (ComfyUI `SDXL`).
SDXL = Family(
    4,
    (
        (0.3651, 0.4232, 0.4341),
        (-0.2533, -0.0042, 0.1068),
        (0.1076, 0.1111, -0.0362),
        (-0.3165, -0.2492, -0.2188),
    ),
    (0.1084, -0.0175, -0.0011),
)

#: FLUX.2's 32 channels at /8 (ComfyUI `Flux2`): FLUX.2, FLUX.3 Klein, Fibo 2, Ideogram 4 and
#: MOSS-Image.
FLUX2 = Family(
    32,
    (
        (0.0058, 0.0113, 0.0073), (0.0495, 0.0443, 0.0836), (-0.0099, 0.0096, 0.0644),
        (0.2144, 0.3009, 0.3652), (0.0166, -0.0039, -0.0054), (0.0157, 0.0103, -0.0160),
        (-0.0398, 0.0902, -0.0235), (-0.0052, 0.0095, 0.0109), (-0.3527, -0.2712, -0.1666),
        (-0.0301, -0.0356, -0.0180), (-0.0107, 0.0078, 0.0013), (0.0746, 0.0090, -0.0941),
        (0.0156, 0.0169, 0.0070), (-0.0034, -0.0040, -0.0114), (0.0032, 0.0181, 0.0080),
        (-0.0939, -0.0008, 0.0186), (0.0018, 0.0043, 0.0104), (0.0284, 0.0056, -0.0127),
        (-0.0024, -0.0022, -0.0030), (0.1207, -0.0026, 0.0065), (0.0128, 0.0101, 0.0142),
        (0.0137, -0.0072, -0.0007), (0.0095, 0.0092, -0.0059), (0.0000, -0.0077, -0.0049),
        (-0.0465, -0.0204, -0.0312), (0.0095, 0.0012, -0.0066), (0.0290, -0.0034, 0.0025),
        (0.0220, 0.0169, -0.0048), (-0.0332, -0.0457, -0.0468), (-0.0085, 0.0389, 0.0609),
        (-0.0076, 0.0003, -0.0043), (-0.0111, -0.0460, -0.0614),
    ),
    (-0.0329, -0.0718, -0.0851),
)  # fmt: skip

#: Qwen-Image's 16 channels at /8, normalized (Wan 2.1's space, ComfyUI `Wan21`): Qwen-Image,
#: Anima and Krea 2.
QWEN_IMAGE = Family(
    16,
    (
        (-0.1299, -0.1692, 0.2932), (0.0671, 0.0406, 0.0442), (0.3568, 0.2548, 0.1747),
        (0.0372, 0.2344, 0.1420), (0.0313, 0.0189, -0.0328), (0.0296, -0.0956, -0.0665),
        (-0.3477, -0.4059, -0.2925), (0.0166, 0.1902, 0.1975), (-0.0412, 0.0267, -0.1364),
        (-0.1293, 0.0740, 0.1636), (0.0680, 0.3019, 0.1128), (0.0032, 0.0581, 0.0639),
        (-0.1251, 0.0927, 0.1699), (0.0060, -0.0633, 0.0005), (0.3477, 0.2275, 0.2950),
        (0.1984, 0.0913, 0.1861),
    ),
    (-0.1835, -0.0868, -0.3360),
)  # fmt: skip

#: Qwen-Image 2's 64 channels at /16, normalized (ComfyUI `QwenImage21`).
QWEN_IMAGE_2 = Family(
    64,
    (
        (-0.0158, -0.0115, -0.0174), (0.0030, 0.0120, 0.0027), (0.0637, 0.0470, -0.0127),
        (0.0360, 0.0661, -0.0030), (0.0159, 0.0181, 0.0082), (0.0132, 0.0326, 0.0169),
        (0.0191, 0.0261, 0.0136), (-0.0146, -0.0276, -0.0361), (0.0187, -0.0024, -0.0072),
        (-0.1059, -0.0090, 0.0350), (-0.0195, -0.0226, -0.0138), (-0.0295, 0.0024, -0.0215),
        (0.0191, -0.0393, -0.0001), (-0.0144, -0.0166, -0.0272), (0.0389, 0.0430, 0.0445),
        (-0.0153, -0.0336, 0.0031), (0.0339, 0.0122, 0.0220), (-0.0136, -0.0078, -0.0120),
        (-0.0340, -0.0282, -0.0245), (-0.0133, -0.0176, -0.0133), (0.0109, -0.0087, 0.0096),
        (-0.0010, 0.0044, 0.0016), (0.0301, 0.0053, 0.0361), (-0.0281, -0.0205, -0.0032),
        (-0.0725, 0.0002, 0.0160), (-0.0036, 0.0158, 0.0807), (0.0087, 0.0040, -0.0053),
        (-0.0260, 0.0183, -0.0077), (-0.0039, -0.0035, -0.0107), (-0.0026, 0.0172, 0.0237),
        (0.0088, 0.0078, 0.0078), (-0.0087, -0.0310, -0.0122), (-0.0027, 0.0018, 0.0094),
        (-0.0064, 0.0292, -0.0256), (0.0594, 0.1049, 0.1180), (0.0103, -0.0103, -0.0026),
        (-0.0091, 0.0025, -0.0015), (0.0178, 0.0243, 0.0292), (-0.0063, -0.0012, 0.0202),
        (0.0452, 0.0246, 0.0143), (0.0149, 0.0270, 0.0052), (0.1484, 0.0801, 0.0804),
        (-0.0120, 0.0040, 0.0010), (0.0181, 0.0051, -0.0021), (0.0132, 0.0050, 0.0019),
        (0.0291, 0.0020, 0.0092), (0.0066, -0.0410, -0.1314), (-0.1153, -0.0629, -0.0802),
        (0.0258, 0.0378, 0.0298), (0.0375, 0.1139, 0.0468), (-0.0142, -0.0126, -0.0276),
        (0.0339, 0.0153, 0.0138), (0.0346, 0.0211, 0.0267), (0.0369, -0.0431, -0.0993),
        (-0.0052, -0.0092, 0.0056), (-0.0279, 0.0410, -0.0357), (0.0036, 0.0017, -0.0083),
        (-0.0441, -0.0367, -0.0454), (-0.0001, -0.0092, -0.0001), (-0.0222, -0.0183, -0.0051),
        (0.0039, 0.0053, -0.0184), (-0.0094, -0.0075, -0.0143), (-0.0066, -0.0088, -0.0063),
        (0.0220, 0.0074, 0.0100),
    ),
    (-0.1228, -0.1869, -0.3083),
)  # fmt: skip

#: MiniMax H3's 24 channels at /16 and /4 in time, normalized (ComfyUI `MiniMaxH3Video`): its
#: videos, and P-H3-Image's pictures, one frame.
MINIMAX_H3 = Family(
    24,
    (
        (-0.018555, 0.024344, -0.017536), (0.150164, 0.137244, 0.129221),
        (0.027367, -0.050369, -0.208606), (-0.000793, -0.164622, -0.323161),
        (-0.048556, 0.013970, -0.074286), (0.011740, 0.014172, -0.006906),
        (0.061517, 0.061212, 0.110025), (0.035321, 0.086879, 0.110059),
        (-0.017426, 0.002997, 0.035356), (0.531539, 0.548819, 0.624404),
        (-0.024968, -0.040234, -0.034302), (-0.032549, -0.029096, -0.017221),
        (0.022609, 0.020286, 0.050661), (-0.084001, -0.038131, -0.020805),
        (-0.018830, 0.010412, 0.061120), (0.020777, 0.011196, -0.030994),
        (-0.008390, -0.012201, -0.025687), (-0.013281, -0.002924, 0.006331),
        (0.000260, 0.001833, -0.011038), (0.105471, 0.100482, 0.132106),
        (0.016529, 0.015213, 0.009999), (-0.014015, -0.017438, -0.019134),
        (-0.033787, -0.009984, -0.019725), (0.004224, 0.017284, 0.027196),
    ),
    (0.057426, -0.022078, -0.071449),
    video=True,
)  # fmt: skip

#: RGB in -1..1 itself: a model that denoises pixels (HiDream-O1).
PIXELS = Family(3, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

FAMILIES: dict[str, Family] = {
    "sdxl": SDXL,
    "flux2": FLUX2,
    "qwen-image": QWEN_IMAGE,
    "qwen-image-2": QWEN_IMAGE_2,
    "minimax-h3": MINIMAX_H3,
    "pixels": PIXELS,
}


def preview(
    x0: Tensor | Callable[[], Tensor], vae: str, *, grid: tuple[int, ...] | None = None
) -> None:
    """Show `x0`, a loop's latents in `vae`'s space (tokens on `grid`), to a client watching
    this run. Optional; a no-op when no client watches, outside a run, or on a group's other
    GPUs."""
    ctx = CURRENT.get()
    if ctx is not None and ctx._preview is not None:
        ctx._preview.offer(x0, vae, grid, ctx._step)


@dataclass(frozen=True, slots=True)
class PreviewFrame:
    """One encoded preview on the lossy lane: the step it shows, and its bytes."""

    position: int
    total: int
    media_type: str
    width: int
    height: int
    frames: int
    data: bytes


class Previews:
    """One attempt's previews: due when a client watches (`wanted`) and the last is older than
    `INTERVAL_S` and encoded; `sink` takes each encoded one, `say` a misuse, once."""

    def __init__(
        self,
        wanted: Callable[[], bool],
        sink: Callable[[PreviewFrame], None],
        say: Callable[[str], object],
    ) -> None:
        self._wanted, self._sink, self._say = wanted, sink, say
        self._last = -math.inf
        self._busy = False
        self._said: set[str] = set()
        self._pinned: dict[tuple[int, ...], Tensor] = {}
        self._factors: dict[tuple[int, str], tuple[Tensor, Tensor]] = {}

    def offer(
        self,
        x0: Tensor | Callable[[], Tensor],
        vae: str,
        grid: tuple[int, ...] | None,
        step: tuple[int, int],
    ) -> None:
        now = time.monotonic()
        if self._busy or now - self._last < INTERVAL_S or not self._wanted():
            return
        self._last = now
        family = FAMILIES.get(vae)
        if family is None:
            return self._misuse(f"no VAE family {vae!r}: one of {', '.join(sorted(FAMILIES))}")
        try:
            latents = x0() if callable(x0) else x0
            shot = self._shoot(latents, family, grid)
        except Exception as exc:  # a preview never fails the run it shows
            return self._misuse(f"{type(exc).__name__}: {exc}"[:300])
        if isinstance(shot, str):
            return self._misuse(shot)
        self._busy = True
        _ENCODER.put(lambda: self._encode(*shot, step))

    def _shoot(
        self, latents: Tensor, family: Family, grid: tuple[int, ...] | None
    ) -> tuple[Tensor, object] | str:
        """`latents` as uint8 RGB `[T, H, W, 3]` on the host, and what to wait on before reading
        it; or why they can't show."""
        import torch

        x = _frames(latents.detach(), family, grid)
        if isinstance(x, str):
            return x
        if x.shape[0] == family.channels * 4:  # each 2x2 patch in the channels
            c, t, h, w = family.channels, *x.shape[1:]
            x = x.reshape(c, 2, 2, t, h, w).permute(0, 3, 4, 1, 5, 2).reshape(c, t, h * 2, w * 2)
        if x.shape[0] != family.channels:
            return f"{x.shape[0]} channels, where this VAE's latents have {family.channels}"
        with torch.no_grad():
            x = x.float()
            if x.shape[1] > FRAMES:
                x = x[:, torch.linspace(0, x.shape[1] - 1, FRAMES, device=x.device).round().long()]
            longest = max(x.shape[-2:])
            if longest > LONGEST:
                x = torch.nn.functional.avg_pool2d(x, math.ceil(longest / LONGEST), ceil_mode=True)
            key = (id(family), str(x.device))
            if key not in self._factors:  # copied to the device once: a copy waits on its queue
                self._factors[key] = (
                    torch.tensor(family.factors, device=x.device),
                    torch.tensor(family.bias, device=x.device),
                )
            factors, bias = self._factors[key]
            rgb = torch.einsum("cthw,cr->thwr", x, factors).add_(bias)
            rgb = rgb.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8)
            if rgb.device.type != "cuda":
                return rgb.cpu(), None
            pinned = self._pinned.get(tuple(rgb.shape))
            if pinned is None:
                pinned = torch.empty(rgb.shape, dtype=torch.uint8, pin_memory=True)
                self._pinned = {tuple(rgb.shape): pinned}
            pinned.copy_(rgb, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            return pinned, done

    def _encode(self, rgb: Tensor, done: object, step: tuple[int, int]) -> None:
        from PIL import Image

        try:
            if done is not None:
                done.synchronize()  # type: ignore[attr-defined]  # a torch.cuda.Event
            frames, height, width, _ = rgb.shape
            pictures = [Image.fromarray(frame) for frame in rgb.numpy()]
            body = io.BytesIO()
            if frames == 1:
                pictures[0].save(body, format="WEBP", quality=80, method=0)
            else:
                pictures[0].save(
                    body, format="WEBP", quality=70, method=0, save_all=True,
                    append_images=pictures[1:], duration=FRAME_MS, loop=0,
                )  # fmt: skip
            self._sink(PreviewFrame(step[0], step[1], "image/webp", width, height, frames, body.getvalue()))
        except Exception as exc:  # a preview never fails the run it shows
            self._misuse(f"encode: {type(exc).__name__}: {exc}"[:300])
        finally:
            self._busy = False

    def _misuse(self, why: str) -> None:
        if why not in self._said:
            self._said.add(why)
            self._say(f"preview: {why}")


def _frames(x: Tensor, family: Family, grid: tuple[int, ...] | None) -> Tensor | str:
    """`x` as `[C, T, H, W]` (T 1 for a picture), or why it isn't one."""
    shape = tuple(x.shape)
    if grid is not None:
        x = x[0] if x.dim() == 3 else x
        if len(grid) not in (2, 3) or x.dim() != 2 or x.shape[0] != math.prod(grid):
            return f"latents of shape {shape} are not [B,] N, C tokens on a grid {grid}"
        x = x.reshape(*grid, x.shape[-1]).movedim(-1, 0)
        return x if x.dim() == 4 else x[:, None]
    if family.video:
        x = x[0] if x.dim() == 5 else x
        return x if x.dim() == 4 else f"latents of shape {shape} are not [B,] C, T, H, W"
    x = x[0] if x.dim() == 5 or (x.dim() == 4 and x.shape[1] != 1) else x
    x = x[:, 0] if x.dim() == 4 else x  # an image VAE's one frame
    return x[:, None] if x.dim() == 3 else f"latents of shape {shape} are not [B,] C, H, W"


class _Encoder:
    """One thread per process encodes previews. Each attempt has at most one in flight
    (`Previews._busy`), so what waits here is one per attempt running."""

    def __init__(self) -> None:
        self._jobs: deque[Callable[[], None]] = deque()
        self._ready = threading.Condition()
        self._thread: threading.Thread | None = None

    def put(self, job: Callable[[], None]) -> None:
        with self._ready:
            self._jobs.append(job)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="cozy-previews", daemon=True)
                self._thread.start()
            self._ready.notify()

    def _run(self) -> None:
        while True:
            with self._ready:
                self._ready.wait_for(lambda: bool(self._jobs))
                job = self._jobs.popleft()
            job()


_ENCODER = _Encoder()
