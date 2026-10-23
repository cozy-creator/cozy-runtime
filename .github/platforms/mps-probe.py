"""Does this macOS runner expose an Apple GPU to torch, how much memory, and does real diffusion run?

Writes mps-probe.json and tiny-sd.png to the directory given as argv[1].
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
facts: dict[str, object] = {}


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def leg(name, fn):
    try:
        facts[name] = fn()
    except Exception as e:  # a refusal is a result here
        facts[name] = f"REFUSED {type(e).__name__}: {str(e)[:300]}"
    print(f"{name}: {facts[name]}", flush=True)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.cpu().double(), b.cpu().double()
    return float((a - b).norm() / b.norm())


def timed(fn, reps=5) -> float:
    fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / reps


leg("chip", lambda: sh("sysctl", "-n", "machdep.cpu.brand_string"))
leg("model", lambda: sh("sysctl", "-n", "hw.model"))
leg("ram_gib", lambda: round(int(sh("sysctl", "-n", "hw.memsize")) / 2**30, 2))
leg("cpus", lambda: int(sh("sysctl", "-n", "hw.ncpu")))
leg("macos", lambda: platform.mac_ver()[0])
leg("system_profiler_displays", lambda: sh("system_profiler", "SPDisplaysDataType")[:400] or "(empty)")
leg("torch", lambda: torch.__version__)
leg("mps_built", torch.backends.mps.is_built)
leg("mps_available", torch.backends.mps.is_available)
if not torch.backends.mps.is_available():
    (out / "mps-probe.json").write_text(json.dumps(facts, indent=2))
    sys.exit(0)

d = "mps"
leg("recommended_max_gib", lambda: round(torch.mps.recommended_max_memory() / 2**30, 2))
g = torch.Generator().manual_seed(0)
a, b = torch.randn(1024, 1024, generator=g), torch.randn(1024, 1024, generator=g)
ref = a @ b
for dt in (torch.float32, torch.float16, torch.bfloat16):
    n = str(dt).removeprefix("torch.")
    leg(f"matmul_{n}_rel", lambda dt=dt: rel((a.to(d, dt) @ b.to(d, dt)).float(), ref))
    x, y = torch.randn(4096, 4096, device=d, dtype=dt), torch.randn(4096, 4096, device=d, dtype=dt)
    leg(f"matmul_{n}_tflops", lambda x=x, y=y: round(2 * 4096**3 / timed(lambda: x @ y) / 1e12, 2))

conv = torch.nn.Conv2d(64, 64, 3, padding=1)
img = torch.randn(1, 64, 128, 128, generator=g)
with torch.no_grad():
    cref = conv(img)
    leg("conv2d_fp16_rel", lambda: rel(conv.to(d, torch.float16)(img.to(d, torch.float16)).float(), cref))
    for ch, side in ((320, 64), (640, 32), (1280, 16)):  # SD UNet widths
        c, x = torch.nn.Conv2d(ch, ch, 3, padding=1), torch.randn(2, ch, side, side, generator=g)
        leg(f"conv2d_{ch}_fp16_rel", lambda c=c, x=x: rel(c.to(d, torch.float16)(x.to(d, torch.float16)).float(), c.cpu().float()(x)))
    lin, x = torch.nn.Linear(320, 1280), torch.randn(2, 4096, 320, generator=g)
    leg("linear_fp16_rel", lambda: rel(lin.to(d, torch.float16)(x.to(d, torch.float16)).float(), lin.cpu().float()(x)))
q = torch.randn(1, 10, 4096, 64, generator=g)
leg("sdpa_fp16_rel", lambda: rel(torch.nn.functional.scaled_dot_product_attention(
    *[q.to(d, torch.float16)] * 3).float(), torch.nn.functional.scaled_dot_product_attention(q, q, q)))
leg("fp8_on_mps", lambda: str(torch.zeros(4, device=d, dtype=torch.float8_e4m3fn).dtype))
leg("float64_on_mps", lambda: str(torch.zeros(4, device=d, dtype=torch.float64).dtype))

# A tiny real diffusion model: segmind/tiny-sd (distilled SD 1.5, 323M-param UNet).
from diffusers import StableDiffusionPipeline  # noqa: E402

pipe = StableDiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32, safety_checker=None)
unet_cpu = pipe.unet.eval()
lat = torch.randn(2, 4, 64, 64, generator=g)
ctx = torch.randn(2, 77, 768, generator=g)
t = torch.tensor([500, 500])
with torch.no_grad():
    uref = unet_cpu(lat, t, ctx).sample
    pipe.to(d, torch.float16)
    leg("unet_fp16_rel", lambda: rel(pipe.unet(lat.to(d, torch.float16), t.to(d), ctx.to(d, torch.float16)).sample.float(), uref))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        leg("unet_fp16_math_attention_rel", lambda: rel(pipe.unet(
            lat.to(d, torch.float16), t.to(d), ctx.to(d, torch.float16)).sample.float(), uref))
    leg("unet_step_s", lambda: round(timed(lambda: pipe.unet(
        lat.to(d, torch.float16), t.to(d), ctx.to(d, torch.float16)), reps=3), 3))


def generate():
    s = time.perf_counter()
    image = pipe("a red fox sitting in fresh snow, photograph", num_inference_steps=20,
                 generator=torch.Generator().manual_seed(7)).images[0]
    torch.mps.synchronize()
    image.save(out / "tiny-sd.png")
    px = torch.tensor(list(image.getdata()), dtype=torch.float32)
    return {"seconds": round(time.perf_counter() - s, 2), "pixel_std": round(float(px.std()), 1),
            "driver_gib": round(torch.mps.driver_allocated_memory() / 2**30, 2)}


leg("tiny_sd_512_20_steps", generate)
# The verdict: fp16 within 1e-2 of CPU fp32 and a picture with contrast (a black or NaN image has none).
wrong = [k for k, v in facts.items() if k.endswith("_rel") and "bfloat16" not in k
         and not (isinstance(v, float) and v < 1e-2)]
image = facts["tiny_sd_512_20_steps"]
if not (isinstance(image, dict) and image["pixel_std"] > 20):
    wrong.append("tiny_sd_512_20_steps")
facts["wrong"] = wrong
(out / "mps-probe.json").write_text(json.dumps(facts, indent=2))
print("verdict:", "correct" if not wrong else f"WRONG {wrong}")
sys.exit(1 if wrong else 0)
