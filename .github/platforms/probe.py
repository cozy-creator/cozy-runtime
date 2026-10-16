"""What of an installed cozy-runtime imports and runs on this host (platforms.yaml).

Every leg prints SERVES or REFUSED with the exception; a refusal is a result, not an error.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import cozy_runtime
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import logging as diffusers_logging


def leg(name: str, fn: Callable[[], Any]) -> Any:
    try:
        value = fn()
        print(f"{name}: SERVES {value}", flush=True)
        return value
    except BaseException as e:  # noqa: BLE001 - a refusal of any kind is the result
        print(f"{name}: REFUSED {type(e).__name__}: {str(e)[:300]}", flush=True)
        return None


def cause(err: str) -> str:
    for key, label in (
        ("triton", "triton"),
        ("fcntl", "unix-only module"),
        ("resource", "unix-only module"),
        ("termios", "unix-only module"),
        ("pwd", "unix-only module"),
        ("grp", "unix-only module"),
        ("posix", "unix-only module"),
        ("cuda", "cuda"),
        ("CUDA", "cuda"),
        ("nvidia", "cuda"),
        ("/proc", "linux /proc"),
        ("cgroup", "linux cgroup"),
        ("tensorfs", "tensorfs"),
    ):
        if key in err:
            return label
    return "other"


def import_walk() -> None:
    failed: dict[str, str] = {}

    def onerror(name: str) -> None:
        failed[name] = f"{sys.exc_info()[0].__name__}: {sys.exc_info()[1]}"

    names = [m.name for m in pkgutil.walk_packages(cozy_runtime.__path__, "cozy_runtime.", onerror)]
    for name in names:
        try:
            importlib.import_module(name)
        except BaseException as e:  # noqa: BLE001
            failed[name] = f"{type(e).__name__}: {str(e)[:300]}"
    print(f"import walk: {len(names) - len(failed)}/{len(names)} modules import")
    for name, err in sorted(failed.items()):
        print(f"  IMPORT-FAIL [{cause(err)}] {name}: {err}")
    print("  causes:", dict(Counter(cause(e) for e in failed.values())))


def cli() -> None:
    exe = os.path.join(os.path.dirname(sys.executable), "cozy-runtime")
    with tempfile.TemporaryDirectory() as d:
        for argv in (
            ["version"],
            ["new", "smoke"],
            ["--dir", os.path.join(d, "smoke"), "describe"],
            ["--builtin", "operations", "describe"],
        ):
            r = subprocess.run([exe, *argv], cwd=d, capture_output=True, text=True)
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"cli {' '.join(argv)}: exit {r.returncode}: {' | '.join(tail)[:400]}")


def compare(name: str, ref: torch.Tensor, out: torch.Tensor) -> str:
    a, b = ref.double().flatten(), out.cpu().double().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / a.norm()).item()
    finite = bool(torch.isfinite(b).all())
    return f"{name}: cos={cos:.6f} rel={rel:.2e} finite={finite}"


def timed(fn: Callable[[], torch.Tensor], device: str, n: int = 3) -> tuple[torch.Tensor, float]:
    out = fn()
    sync(device)
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    sync(device)
    return out, (time.perf_counter() - t) / n


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def sdxl_unet(device: str) -> None:
    """SDXL widths and heads (320/640/1280, 2048-wide text, text_time conditioning), one
    resnet and one transformer per level so it fits a hosted runner. Random weights: this
    proves the ops serve and agree with CPU fp32, not image quality."""
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=64,
        layers_per_block=1,
        block_out_channels=(320, 640, 1280),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        transformer_layers_per_block=1,
        attention_head_dim=(5, 10, 20),
        cross_attention_dim=2048,
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=256,
        projection_class_embeddings_input_dim=2816,
    ).eval()
    print(f"unet params: {sum(p.numel() for p in unet.parameters()) / 1e6:.0f}M")
    x = torch.randn(1, 4, 64, 64)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, 77, 2048)
    added = {"text_embeds": torch.randn(1, 1280), "time_ids": torch.randn(1, 6)}

    def run(dev: str, dtype: torch.dtype) -> torch.Tensor:
        m = unet.to(dev, dtype)
        with torch.no_grad():
            args = (x.to(dev, dtype), t.to(dev), ctx.to(dev, dtype))
            kw = {k: v.to(dev, dtype) for k, v in added.items()}
            return m(*args, added_cond_kwargs=kw).sample.float()

    ref, cpu_s = timed(lambda: run("cpu", torch.float32), "cpu", n=1)
    print(f"unet cpu fp32: {cpu_s:.2f}s/forward")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        def go(dtype: torch.dtype = dtype) -> str:
            out, s = timed(lambda: run(device, dtype), device)
            mem = ""
            if device == "mps":
                mem = f" allocated={torch.mps.current_allocated_memory() >> 20}MiB driver={torch.mps.driver_allocated_memory() >> 20}MiB"
            return f"{s:.3f}s/forward{mem} " + compare("vs cpu fp32", ref, out)

        leg(f"unet {device} {dtype}", go)
    unet.to("cpu", torch.float32)
    if device == "mps":
        torch.mps.empty_cache()


def sdxl_vae(device: str) -> None:
    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=4,
        sample_size=1024,
    ).eval()
    z = torch.randn(1, 4, 64, 64)

    def run(dev: str, dtype: torch.dtype) -> torch.Tensor:
        with torch.no_grad():
            return vae.to(dev, dtype).decode(z.to(dev, dtype)).sample.float()

    ref, cpu_s = timed(lambda: run("cpu", torch.float32), "cpu", n=1)
    print(f"vae decode 512px cpu fp32: {cpu_s:.2f}s")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        def go(dtype: torch.dtype = dtype) -> str:
            out, s = timed(lambda: run(device, dtype), device, n=2)
            return f"{s:.3f}s " + compare("vs cpu fp32", ref, out)

        leg(f"vae decode 512px {device} {dtype}", go)
    vae.to("cpu", torch.float32)


def attention(device: str) -> None:
    """SDXL level-1 self-attention: 4096 tokens, 10 heads of 64 (a 1024px image)."""
    f = torch.nn.functional.scaled_dot_product_attention
    q, k, v = (torch.randn(1, 10, 4096, 64) for _ in range(3))
    ref = f(q, k, v)
    for dtype in (torch.float16, torch.bfloat16):
        def go(dtype: torch.dtype = dtype) -> str:
            a, b, c = (t.to(device, dtype) for t in (q, k, v))
            out, s = timed(lambda: f(a, b, c), device)
            return f"{s * 1e3:.1f}ms " + compare("vs cpu fp32", ref, out)

        leg(f"sdpa 4096x10x64 {device} {dtype}", go)


def mps_limits() -> None:
    """Weight-only int8, the OOM type, and whether SDPA materializes the score matrix: 16k
    tokens x 24 heads x 128 is 12 GiB of fp16 scores, past the runner's MPS budget."""
    x = torch.randn(64, 4096, dtype=torch.float16)
    w = torch.randint(-127, 127, (4096, 4096), dtype=torch.int8)
    scale = torch.rand(4096, dtype=torch.float16) / 64
    ref = x.float() @ (w.float() * scale.float()[:, None]).t()

    def int8_weight_only() -> str:
        out, s = timed(lambda: torch._weight_int8pack_mm(x.to("mps"), w.to("mps"), scale.to("mps")), "mps")
        return f"{s * 1e3:.2f}ms " + compare("vs dequant fp32", ref, out)

    leg("weight-only int8 _weight_int8pack_mm on mps", int8_weight_only)
    f = torch.nn.functional.scaled_dot_product_attention
    q = torch.randn(1, 24, 16384, 128, device="mps", dtype=torch.float16)

    def long_sdpa() -> str:
        out, s = timed(lambda: f(q, q, q), "mps", n=1)
        return f"{s:.2f}s driver={torch.mps.driver_allocated_memory() >> 20}MiB"

    leg("sdpa 16384x24x128 fp16 whole", long_sdpa)
    torch.mps.empty_cache()

    def chunked() -> str:
        out, s = timed(lambda: torch.cat([f(c, q, q) for c in q.split(2048, dim=2)], dim=2), "mps", n=1)
        return f"{s:.2f}s driver={torch.mps.driver_allocated_memory() >> 20}MiB"

    leg("sdpa 16384x24x128 fp16 in 2048-query chunks", chunked)
    torch.mps.empty_cache()

    def oom() -> str:
        try:
            torch.empty(64 << 30, dtype=torch.uint8, device="mps")
            return "64 GiB allocated"
        except BaseException as e:  # noqa: BLE001
            return f"{type(e).__module__}.{type(e).__qualname__} is torch.OutOfMemoryError: {isinstance(e, torch.OutOfMemoryError)}: {str(e)[:160]}"

    leg("mps oom type", oom)


def mps_facts() -> None:
    print("mps.is_built:", torch.backends.mps.is_built())
    print("mps.is_available:", torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        for n in ("recommended_max_memory", "driver_allocated_memory", "current_allocated_memory"):
            leg(f"mps.{n}", lambda n=n: f"{getattr(torch.mps, n)() >> 20} MiB")
        for n in ("float8_e4m3fn", "float8_e5m2"):
            leg(f"{n} tensor on mps", lambda n=n: torch.zeros(4, device="mps", dtype=getattr(torch, n)).dtype)
        leg("int8 matmul on mps", lambda: torch._int_mm(
            torch.ones(32, 32, dtype=torch.int8, device="mps"),
            torch.ones(32, 32, dtype=torch.int8, device="mps")).sum().item())


def main() -> None:
    diffusers_logging.set_verbosity_error()
    print("python:", sys.version.split()[0], "platform:", platform.platform(), platform.machine())
    print("torch:", torch.__version__, "cuda build:", torch.version.cuda, "cuda.is_available:", torch.cuda.is_available())
    print("cpu count:", os.cpu_count())
    leg("import tensorfs", lambda: importlib.import_module("tensorfs").__file__)
    leg("import tensorfs.plane", lambda: importlib.import_module("tensorfs.plane").__name__)
    import_walk()
    leg("cli", cli)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if sys.platform == "darwin":
        mps_facts()
    if device != "cpu":
        leg("attention", lambda: attention(device))
        leg("sdxl unet", lambda: sdxl_unet(device))
        leg("sdxl vae", lambda: sdxl_vae(device))
        leg("mps limits", mps_limits)


if __name__ == "__main__":
    main()
