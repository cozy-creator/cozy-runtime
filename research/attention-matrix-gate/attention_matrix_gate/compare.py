"""All new selectors dispatch through the same Diffusers registry as real inference."""

import hashlib
import importlib.metadata
from pathlib import Path


def compare(seed, index):
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    from cozy_runtime import _build_provenance
    from cozy_runtime.author import AttentionLayout, attention_scope
    from cozy_runtime.internal import (
        attention,
        attention_precision,
        attention_sol,
        attention_ulysses,
    )
    from cozy_runtime.internal import attention_fp8_sol_kernel as kernel
    from cozy_runtime.internal.encoding import DeviceFacts

    if torch.cuda.get_device_capability(index) != (12, 0):
        raise ValueError("This experiment is specifically SM120")
    device = torch.device("cuda", index)
    generator = torch.Generator(device=device).manual_seed(seed)
    facts = DeviceFacts("cuda", torch.cuda.get_device_name(index), 120, "", "", index)
    rows = []
    with torch.inference_mode():
        for name in attention_precision.NAMES:
            ready = attention.pinned(name, facts)[0]
            mixed = name.startswith("sol-")
            cases = [
                ("odd_tail", 257, 257, 0, 3, "blocks.2.attn"),
                ("multi_group", 4097, 4097, 65, 4, "blocks.2.attn"),
            ]
            if mixed:
                cases += [
                    ("protected_path", 257, 257, 65, 4, "blocks.1.attn"),
                    ("protected_all", 257, 257, 257, 4, "blocks.2.attn"),
                ]
            else:
                cases += [
                    ("prefix_unequal", 7, 257, 0, 4, "blocks.2.attn"),
                    ("dense_in_sparse_step", 65, 65, 0, 7, "blocks.2.attn"),
                ]
            for case, qn, kn, prefix, step, path in cases:
                tensors = [
                    torch.randn(1, n, 2, 128, generator=generator, device=device).bfloat16()
                    for n in (qn, kn, kn)
                ]
                before = [t.clone() for t in tensors]
                layout = AttentionLayout(qn, prefix, step, 4, ("blocks.1",))
                site = attention_sol._SITE.set(attention_sol.Site("dit", path))
                try:
                    with (
                        attention_scope(layout),
                        attention_sol.observing() as counts,
                        attention_ulysses.observing() as seen,
                    ):
                        got = dispatch_attention_fn(*tensors, backend=ready.member)
                    with attention_scope(layout), attention_sol.observing():
                        repeat = dispatch_attention_fn(*tensors, backend=ready.member)
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA]
                    ) as profiler:
                        with attention_scope(layout), attention_sol.observing():
                            dispatch_attention_fn(*tensors, backend=ready.member)
                        torch.cuda.synchronize(device)
                finally:
                    attention_sol._SITE.reset(site)
                cuda_names = sorted(
                    {
                        e.name
                        for e in profiler.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA
                    }
                )
                q, k, v = tensors
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
                ).transpose(1, 2)
                relative = float((got.float() - expected).norm() / expected.norm().clamp_min(1e-8))
                finite = bool(torch.isfinite(got).all())
                repeat_equal = bool(torch.equal(got, repeat))
                unchanged = all(torch.equal(a, b) for a, b in zip(tensors, before, strict=True))
                sparse = mixed and step >= 4 and path != "blocks.1.attn"
                wanted = dict(
                    sparse=int(sparse),
                    dense_step=int(mixed and step < 4),
                    dense_path=int(mixed and path == "blocks.1.attn"),
                    dense_prefix=int(sparse and prefix > 0),
                )
                limit = None if sparse else (0.08 if "fp8" in name else 0.01)
                passed = (
                    finite
                    and repeat_equal
                    and unchanged
                    and counts == wanted
                    and (limit is None or relative < limit)
                    and bool(cuda_names)
                )
                rows.append(
                    dict(
                        selector=name,
                        case=case,
                        shape=list(q.shape),
                        kv_tokens=kn,
                        step=step,
                        prefix=prefix,
                        counts=counts,
                        wanted_counts=wanted,
                        finite=finite,
                        repeat_identical=repeat_equal,
                        inputs_unchanged=unchanged,
                        relative_l2_fp32=relative,
                        relative_l2_gate=limit,
                        actual_cuda_kernels=cuda_names,
                        implementations=attention.implementations(seen),
                        passed=passed,
                    )
                )
        # Inspect the exact compiled dense FP8 MMA, independently of its friendly selector label.
        q = torch.ones(1, 65, 2, 128, device=device).to(torch.float8_e4m3fn)
        scales = torch.ones(1, 2, 2, device=device)
        out = torch.empty(1, 65, 2, 128, device=device, dtype=torch.bfloat16)
        compiled = kernel.dense_forward_kernel[(2, 2)](
            q,
            q,
            q,
            scales,
            scales,
            scales,
            out,
            128**-0.5,
            65,
            65,
            H=2,
            D=128,
            num_warps=4,
            num_stages=1,
        )
        mma = sorted(
            {
                line.strip()
                for line in compiled.asm["ptx"].splitlines()
                if "mma.sync" in line or "tcgen" in line
            }
        )
        fp16_mma = any("f16.e4m3.e4m3.f16" in line for line in mma)
        constant = attention_precision.fp8_dense(
            torch.ones_like(out), torch.ones_like(out), torch.ones_like(out)
        )
        constant_exact = bool(torch.equal(constant, torch.ones_like(constant)))
    modules = {
        name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
        for name, mod in [
            ("attention", attention),
            ("precision", attention_precision),
            ("sol", attention_sol),
            ("fp8_kernel", kernel),
        ]
    }
    return dict(
        passed=all(row["passed"] for row in rows) and fp16_mma and constant_exact,
        runtime=importlib.metadata.version("cozy-runtime"),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(index),
        capability=list(torch.cuda.get_device_capability(index)),
        module_sha256=modules,
        provenance={
            k: v
            for k, v in vars(_build_provenance).items()
            if k.isupper() and isinstance(v, (str, int, bool))
        },
        rows=rows,
        fp8_dense_ptx_mma=mma,
        fp16_mma_observed=fp16_mma,
        constant_value_exact=constant_exact,
        quality_claim=False,
    )
