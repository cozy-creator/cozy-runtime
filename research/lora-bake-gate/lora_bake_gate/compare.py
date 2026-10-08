"""Run old and new helper on exactly the same synthetic stored operands."""

import hashlib
import importlib.metadata
import json
from pathlib import Path


def _software(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _stored(torch, weight, kind):
    if kind == "plain_bf16":
        return {"value": weight.to(torch.bfloat16)}
    if kind == "plain_fp32":
        return {"value": weight.clone()}
    if kind == "rowwise":
        scale = (weight.abs().amax(dim=1) / 448).clamp(min=1e-12)
        data = (weight / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return {"data": data, "scale": scale}
    rows, columns = weight.shape
    scale = torch.full((rows, columns // 32), 115, dtype=torch.uint8, device=weight.device)
    # The stored block grid is intentionally fixed; the production bake keeps it.
    return {"data": (weight * 4096).clamp(-448, 448).to(torch.float8_e4m3fn), "scale": scale}


def compare(seed):
    import torch

    from cozy_runtime.internal import lora_bake as candidate

    from . import baseline_lora_bake as baseline

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This explicit gate is for the SM120 benchmark worker")
    # H3's K5376/rank64; two whole default 2M-element row chunks plus a one-row tail.
    rows, columns, rank = 781, 5376, 64
    generator = torch.Generator(device="cuda").manual_seed(seed)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cases = []
    try:
        with torch.inference_mode():
            weight = torch.randn((rows, columns), generator=generator, device="cuda") * 0.02
            down = (torch.randn((rank, columns), generator=generator, device="cuda") * 0.02).to(
                torch.bfloat16
            )
            up = (torch.randn((rows, rank), generator=generator, device="cuda") * 0.02).to(
                torch.bfloat16
            )
            factor_before = down.clone(), up.clone()
            for kind in ("plain_bf16", "plain_fp32", "rowwise", "mx"):
                initial = _stored(torch, weight, kind)
                for statistics in (False, True):
                    old = {key: value.clone() for key, value in initial.items()}
                    new = {key: value.clone() for key, value in initial.items()}
                    before_stats = baseline.Stats() if statistics else None
                    after_stats = candidate.Stats() if statistics else None
                    old_rng = torch.Generator(device="cuda").manual_seed(seed + 1)
                    new_rng = torch.Generator(device="cuda").manual_seed(seed + 1)
                    baseline.bake(torch, old, down, up, generator=old_rng, stats=before_stats)
                    candidate.bake(torch, new, down, up, generator=new_rng, stats=after_stats)
                    torch.cuda.synchronize()
                    parts = {
                        key: bool(
                            torch.equal(old[key].view(torch.uint8), new[key].view(torch.uint8))
                        )
                        for key in initial
                    }
                    differing_bytes = {
                        key: int(
                            (old[key].view(torch.uint8) != new[key].view(torch.uint8)).sum().item()
                        )
                        for key in initial
                    }
                    previous = (
                        {
                            name: getattr(before_stats, name)
                            for name in before_stats.__slots__
                            if name != "seconds"
                        }
                        if statistics
                        else {}
                    )
                    following = (
                        {
                            name: getattr(after_stats, name)
                            for name in after_stats.__slots__
                            if name != "seconds"
                        }
                        if statistics
                        else {}
                    )
                    rng_equal = bool(torch.equal(old_rng.get_state(), new_rng.get_state()))
                    finite = all(bool(torch.isfinite(t.float()).all().item()) for t in new.values())
                    cases.append(
                        {
                            "kind": kind,
                            "statistics": statistics,
                            "part_bytes_equal": parts,
                            "differing_bytes": differing_bytes,
                            "statistics_equal": previous == following,
                            "baseline_statistics": previous,
                            "candidate_statistics": following,
                            "rng_state_equal": rng_equal,
                            "finite": finite,
                            "passed": all(parts.values())
                            and previous == following
                            and rng_equal
                            and finite,
                        }
                    )
            factors_unchanged = bool(
                torch.equal(down, factor_before[0]) and torch.equal(up, factor_before[1])
            )
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return {
        "passed": factors_unchanged and all(row["passed"] for row in cases),
        "seed": seed,
        "rows": rows,
        "columns": columns,
        "rank": rank,
        "chunks": baseline._chunks(rows, columns),
        "factors_unchanged": factors_unchanged,
        "baseline": json.loads(Path(__file__).with_name("reference.json").read_text()),
        "candidate_module_sha256": hashlib.sha256(
            Path(candidate.__file__).read_bytes()
        ).hexdigest(),
        "software": {name: _software(name) for name in ("cozy-runtime", "tensorfs", "torch")},
        "gpu": torch.cuda.get_device_name(),
        "cases": cases,
        "limits": [
            "Synthetic stored-weight equivalence gate; not video quality or performance",
            "TF32 disabled identically for both helpers",
            "Statistics seconds intentionally excluded",
            "No fallback for unsupported CUDA operation; it fails visibly",
        ],
    }
