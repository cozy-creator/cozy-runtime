"""Compile a model and capture everything Inductor decided.

Writes, under --out:
  output_code/            every generated Triton/wrapper file Inductor emitted
  inductor_debug/         torch._inductor.config.trace debug dirs (fusion decisions, ir_post_fusion)
  logs.txt                the +inductor / +output_code / fusion TORCH_LOGS stream

Driven by env in the parent shell:
  TORCH_LOGS="+inductor,output_code,fusion"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["anima", "sdxl"], default="anima")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--latent-hw", type=int, default=128)
    ap.add_argument("--plan", choices=["raw", "op"], default=None)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "inductor_cache"
    debug = out / "inductor_debug"
    for path in (cache, debug):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)

    import torch._inductor.config as icfg

    icfg.trace.enabled = True
    icfg.trace.debug_dir = str(debug)
    icfg.trace.output_code = True
    icfg.trace.fx_graph = True
    icfg.trace.ir_post_fusion = True
    icfg.trace.graph_diagram = False
    icfg.force_disable_caches = True

    if args.model == "anima":
        import anima_model

        model = anima_model.build(num_layers=args.layers)
        if args.plan:
            anima_model.apply_fused_plan(model, via=args.plan, strict=args.layers == 28)
        kwargs = anima_model.inputs(latent_hw=args.latent_hw)
    else:
        import sdxl_model

        model = sdxl_model.build(layers=args.layers)
        kwargs = sdxl_model.inputs(batch=args.batch or sdxl_model.BATCH)

    compiled = torch.compile(model, mode=args.mode, fullgraph=True)
    with torch.no_grad():
        compiled(**kwargs)
        torch.cuda.synchronize()

    dest = out / "output_code"
    dest.mkdir(exist_ok=True)
    n = 0
    for path in cache.rglob("*.py"):
        text = path.read_text(errors="replace")
        if "triton" not in text and "async_compile" not in text:
            continue
        shutil.copy(path, dest / f"{path.parent.name}_{path.name}")
        n += 1
    print(f"captured {n} generated files into {dest}")
    print(f"debug dirs: {[p.name for p in debug.iterdir()]}")


if __name__ == "__main__":
    main()
