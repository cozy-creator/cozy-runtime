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
# package in proofs/fixtures/tiny-models/. "kernel" is everything invoke() does around the
# handler: decode,
# ModelDefault overlay + digest, RequestFeatures normalization, preflight, injection,
# return validation and the output transaction. "handler" is the same handler called
# directly with the same arguments, so the DIFFERENCE is the surface's overhead.
KERNEL = f"""
import random, resource, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, 'proofs/fixtures/tiny-models')
import tiny_artifacts as art
import tiny_h3 as h3
import tiny_sdxl as ep
from tiny_diffusers import TinySdxlPipeline
from cozy_runtime.author import Invocation, ModelRegistry, describe, invoke
from cozy_runtime.author._loader import census
from cozy_runtime.author._services import Attempt
from cozy_runtime.author.fakes import (fake_attempt, fake_context, fake_outputs,
                                       invoke_with_fakes)
from cozy_runtime.author.fakes import StagedBackend

N = {ITERATIONS}


def registry(app_models):
    reg = ModelRegistry(release="bench", backend=StagedBackend())
    for path, (cls, artifact) in app_models.items():
        reg.acquire(path, cls, artifact)
    return reg


sdxl = registry({{"plan.models.model": (ep.SdxlModel, art.sdxl()),
                 "generate.models.model": (ep.SdxlModel, art.sdxl()),
                 "generate_ref.models.model": (ep.SdxlModel, art.sdxl())}})
h3_reg = registry({{"generate.models.model": (h3.Fl2VAModel, art.h3(task="fl2va"))}})
models = sdxl.models("plan")
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


for label, app in (("tiny_sdxl — 4 entrypoints", ep.app), ("tiny_h3 — 2 entrypoints", h3.app)):
    t0 = time.perf_counter()
    describe(app)
    print(f"  {{'describe(' + label + '), once':<46}} {{(time.perf_counter()-t0)*1000:9.3f}} ms")

kernel = bench("invoke(plan): decode+overlay+features+inject+validate",
               lambda: invoke(ep.app.get('plan'), {{"prompt": "bench"}}, inv,
                              Attempt("bench", inv.spool)))
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
                                models=sdxl.models('generate')), n=N // 20)
bench("invoke(generate_ref): + typed preflight + Settings",
      lambda: invoke_with_fakes(ep.app.get('generate_ref'), {{"prompt": "b"}},
                                models=sdxl.models('generate_ref'),
                                settings=ep.RefSettings()), n=N // 20)
bench("invoke(h3.generate): 6 component scopes + cache",
      lambda: invoke_with_fakes(h3.app.get('generate'),
                                {{"prompt": "b", "duration_s": 1}},
                                models=h3_reg.models('generate')), n=N // 20)

print()
pipe = h3.build_h3_pipeline(art.h3(task='fl2va').config, task='fl2va')
h3_artifact = art.h3(task="fl2va")
bench("census: walk one H3 construction (12 destinations)",
      lambda: census(pipe), n=N // 3)
bench("construct: factory + census + match + fill (H3)",
      lambda: ModelRegistry(release='b', backend=StagedBackend()).acquire(
          'g.models.m', h3.Fl2VAModel, h3_artifact), n=N // 10)
bench("construct: the SDXL whole pipeline through load(T)",
      lambda: ModelRegistry(release='b', backend=StagedBackend()).acquire(
          'g.models.m', ep.SdxlModel, art.sdxl()), n=N // 10)
bench("model.for_request -> RequestView + fresh scheduler",
      lambda: sdxl.models('generate')['model'].for_request(fake_context()))

attempt = fake_attempt('bench')
out = fake_outputs(attempt)
frame = ep._render(TinySdxlPipeline(art.sdxl().config)(
    prompt='b', negative_prompt=None, width=64, height=64, steps=1, guidance=1.0,
    generator=random.Random(0)))
bench("out.save_image -> real PNG encode (64x64)",
      lambda: out.save_image(frame, format='png'), n=N // 10)

print(f"  {{'peak RSS after these runs':<46}} "
      f"{{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:9.1f}} MiB")
"""


DESCRIPTOR = """
import statistics, time, sys, pathlib
sys.path.insert(0, 'proofs/fixtures/tiny-models')
from cozy_runtime.internal import canonical, descriptor
from cozy_runtime.internal.discovery import discover

project = pathlib.Path('proofs/fixtures/tiny-models')


def bench(label, fn, n):
    fn()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    median = statistics.median(samples)
    unit = f"{median:9.1f} us" if median < 1000 else f"{median / 1000:9.2f} ms"
    print(f"  {label:<46} {unit}  (n={n})")


found = discover(project)
body = descriptor.build(found)
raw = descriptor.canonical_bytes(body)

bench('descriptor.build (frozen app -> document)', lambda: descriptor.build(found), 200)
bench('descriptor.canonical_bytes (publication output)',
      lambda: descriptor.canonical_bytes(body), 500)
bench('package_descriptor_digest (canonical semantic hash)',
      lambda: descriptor.package_descriptor_digest(body), 500)
bench('canonical.parse (read publication bytes)', lambda: canonical.parse(raw), 500)
bench('compare (the server proof comparison)',
      lambda: descriptor.compare(body, body, ('a', 'b')), 500)
print(f"  {'canonical publication bytes':<46} {len(raw):9,} B")
print(f"  {'callables / model slots described':<46} "
      f"{descriptor.counts(body)['entrypoints'] + descriptor.counts(body)['jobs']:5} / "
      f"{descriptor.counts(body)['model_bindings']}")
"""


DESCRIBE = ("cozy-runtime", "describe", "--dir", "proofs/fixtures/tiny-models")


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
        # cr-003: what the build and CI actually pay per package.
        ("cozy-runtime describe (h3)", [*DESCRIBE]),
        ("cozy-runtime describe --json", [*DESCRIBE, "--json"]),
    ]
    print(f"{RUNS} runs each; peak RSS measured per case in a fresh helper process")
    for label, argv in cases:
        report(label, *timed(argv))
    print("\nauthor surface, in-process against tiny-model proof fixtures", flush=True)
    subprocess.run([py, "-c", KERNEL], check=True)
    print("\ndescriptor plane, in-process (the boundary, the writer, the digest)", flush=True)
    subprocess.run([py, "-c", DESCRIPTOR], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
