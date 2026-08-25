#!/usr/bin/env python3
"""Time + memory benchmarks for the CLI and the author import. Real runs, no harness."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time

RUNS = 20


# Peak RSS is measured in a FRESH helper process per case: ru_maxrss over children is
# cumulative, so measuring in-line would report the largest case for every later case.
HELPER = (
    "import resource,subprocess,sys;"
    "subprocess.run(sys.argv[1:],capture_output=True,check=True);"
    "print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)"
)


def timed(argv: list[str]) -> tuple[list[float], int]:
    """Wall-clock ms per run, plus that command's peak RSS in KiB."""
    samples: list[float] = []
    for _ in range(RUNS):
        start = time.perf_counter()
        subprocess.run(argv, capture_output=True, check=True)
        samples.append((time.perf_counter() - start) * 1000)
    probe = subprocess.run(
        [sys.executable, "-c", HELPER, *argv], capture_output=True, text=True, check=True
    )
    return samples, int(probe.stdout.strip())


def report(label: str, samples: list[float], rss_kib: int) -> None:
    print(
        f"{label:<34} min {min(samples):6.1f} ms  median {statistics.median(samples):6.1f} ms  "
        f"max {max(samples):6.1f} ms  peak RSS {rss_kib / 1024:5.1f} MiB"
    )


ITERATIONS = 3000

# The per-call cost of the author surface, measured in one process against the real
# endpoint in examples/. "kernel" is everything invoke() does around the handler: decode,
# ModelDefault overlay + digest, RequestFeatures normalization, preflight, injection,
# return validation and the output transaction. "handler" is the same handler called
# directly with the same arguments, so the DIFFERENCE is the surface's overhead.
KERNEL = f"""
import resource, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, 'examples')
import tiny_sdxl as ep
from cozy_runtime.author import Invocation, describe, invoke
from cozy_runtime.author.fakes import fake_attempt, fake_outputs, invoke_with_fakes

N = {ITERATIONS}
models = {{"model": ep.SdxlModel()}}
payload = ep.Txt2ImgInput(prompt="bench", num_inference_steps=4)
inv = Invocation(request_id="bench", spool=Path(tempfile.mkdtemp()),
                 deadline=time.monotonic() + 3600, models=models)


def bench(label, fn, n=N):
    fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    per = (time.perf_counter() - start) / n * 1e6
    print(f"  {{label:<46}} {{per:9.1f}} us/call  ({{n}})")
    return per


t0 = time.perf_counter()
describe(ep.app)
print(f"  {{'describe(app) — 4 entrypoints, once':<46}} {{(time.perf_counter()-t0)*1000:9.3f}} ms")

kernel = bench("invoke(plan): decode+overlay+features+inject+validate",
               lambda: invoke(ep.app.get('plan'), {{"prompt": "bench"}}, inv))
handler = bench("the same handler called directly, no kernel",
                lambda: ep.plan(payload, models['model']))
print(f"  {{'-> injection + validation overhead per call':<46}} {{kernel - handler:9.1f}} us/call")

faked = bench("invoke_with_fakes(plan) — adds a fresh temp spool",
              lambda: invoke_with_fakes(ep.app.get('plan'), {{"prompt": "bench"}}, models=models),
              n=N // 3)
print(f"  {{'-> the fakes overhead (mkdtemp per attempt)':<46}} {{faked - kernel:9.1f}} us/call")

bench("invoke(generate): 64x64 render + PNG + transaction",
      lambda: invoke_with_fakes(ep.app.get('generate'), {{"prompt": "b",
                                                         "num_inference_steps": 4}},
                                models=models), n=N // 20)
bench("invoke(generate_ref): + typed preflight + Settings",
      lambda: invoke_with_fakes(ep.app.get('generate_ref'), {{"prompt": "b"}},
                                models=models, settings=ep.RefSettings()), n=N // 20)

attempt = fake_attempt('bench')
out = fake_outputs(attempt)
frame = ep.TinyPipe()(prompt='b', width=64, height=64, steps=1, guidance=1.0,
                      generator=ep.random.Random(0))
bench("out.save_image -> real PNG encode (64x64)",
      lambda: out.save_image(frame, format='png'), n=N // 10)

print(f"  {{'peak RSS after these runs':<46}} "
      f"{{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:9.1f}} MiB")
"""


def main() -> int:
    py = sys.executable
    cases = [
        ("cozy-runtime version (cold start)", ["cozy-runtime", "version"]),
        ("cozy-runtime --help", ["cozy-runtime", "--help"]),
        ("import cozy_runtime.author", [py, "-c", "import cozy_runtime.author"]),
        ("import cozy_runtime.author.fakes", [py, "-c", "import cozy_runtime.author.fakes"]),
        ("import msgspec (the one dependency)", [py, "-c", "import msgspec"]),
        ("import cozy_runtime.cli.main", [py, "-c", "import cozy_runtime.cli.main"]),
        ("bare python interpreter", [py, "-c", "pass"]),
    ]
    print(f"{RUNS} runs each; peak RSS measured per case in a fresh helper process")
    for label, argv in cases:
        report(label, *timed(argv))
    print("\nauthor surface, in-process against examples/tiny_sdxl.py")
    subprocess.run([py, "-c", KERNEL], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
