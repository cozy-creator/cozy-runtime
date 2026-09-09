"""The runtime picks the fastest attention kernel it has (cr-124).

The table arms run everywhere. The CUDA arm runs `testdata/attention_selection_proof.py` in a
child process against real diffusers 0.40.0 modules on the box's own card, and skips naming
whichever of those this host lacks.

This file also holds the cr-122 observation-cap arms. They were written for the attestation
this issue deletes and they OUTLIVE it: the rule they prove — telemetry may not fail the thing
it observes — is what made a 1,744-character record a truncation instead of an infinite retry
loop, and nothing about that depends on what produced the record.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from cozy.worker.v1 import worker_pb2 as pb
from cozy_runtime.author._observations import (
    NAME_CAP,
    TEXT_CAP,
    EventRing,
    ObservationRefusal,
)
from cozy_runtime.cli import host
from cozy_runtime.cli import run as cli_run
from cozy_runtime.cli.io import CliError
from cozy_runtime.internal import attention, attention_fp8, child_env, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.local import LocalRequest, development_source_digest, run_slice
from cozy_runtime.internal.worker import attempts
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.plan import DeclaredBinding, ModelBinding
from cozy_runtime.internal.worker.session import WorkerOptions, executor_prepare_command
from cozy_runtime.protocol import documents
from test_end_to_end import NO_EXECUTOR

PACKAGE = Path(__file__).resolve().parent.parent / "examples" / "marco-polo"
needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


def _device(sm: int, kind: str = "cuda") -> DeviceFacts:
    return DeviceFacts(kind=kind, name="card", sm=sm, driver="580", configuration="", index=0)


def _refusal(run: Any) -> attention.AttentionRefusal:
    with pytest.raises(attention.AttentionRefusal) as refused:
        run()
    return refused.value


# ------------------------------------------------------------------------- the static rule


def test_the_ranked_table_is_ordered_by_a_measurement_and_ends_in_the_floor() -> None:
    """A rank is a measurement or it is not a rank. The floor is last and takes everything."""
    assert [c.name for c in attention.CANDIDATES] == ["flash-attn3", "sdpa"]
    for candidate in attention.CANDIDATES:
        assert "MEASURED" in candidate.why or "torch's own" in candidate.why, candidate.name
    floor = attention.CANDIDATES[-1]
    assert floor.name == "sdpa"
    assert floor.min_sm == 0 and not floor.dtypes and floor.max_head_dim == 0
    assert attention._admits(floor, "float32", 512) and attention._admits(floor, "", 0)


def test_every_kernel_declares_the_numeric_budget_it_is_held_to() -> None:
    """cr-127. A kernel is admitted only if it AGREES with exact fp32 attention, so every entry
    has to say how far it is allowed to drift. The two classes are visible in the numbers: an
    exact kernel is held near bf16's own 2.3e-3 floor, and a numerics-changing one is allowed
    its quantisation (h3a-013 measured 3.7-4.0e-2) and nothing like the 311 a dead kernel
    returned."""
    for candidate in (*attention.CANDIDATES, *attention.PINNABLE):
        assert candidate.rel_l2_max > 0, candidate.name
        assert candidate.rel_l2_max < 1, candidate.name
    # flash-attn4 is EXACT: bf16/fp16 in, no quantisation anywhere in it. Its
    # `torch2cute_dtype_map` is float16/bfloat16/float32 and its interface has no fp8 or fp4
    # path at all (h3a-024), so it is held to the same floor as flash-attn3 and is not the
    # 4-bit story h3a-022's Blackwell case rests on.
    exact = {"flash-attn3", "flash-attn4", "sdpa", "cudnn"}
    for candidate in (*attention.CANDIDATES, *attention.PINNABLE):
        budget = candidate.rel_l2_max
        assert (budget <= 2e-2) is (candidate.name in exact), (candidate.name, budget)
    assert attention.PROBE == (1, 256, 4, 128)


def test_fp8_attention_is_measured_pinnable_and_deliberately_not_ranked() -> None:
    """h3a-013 timed both 8-bit arms on an H100 and NEITHER is ranked. The reason is recorded
    in the module, because a table that only records speed would have promoted them: flash-attn3
    fp8 is 1.31x rank 1 from the .so we already ship. It is not ranked because it is INEXACT
    (~16x bf16's floor) and no gate we own can separate 'degrades quality' from 'changes the
    sample' — and because at 2.68 c/s it misses the 2.5 target anyway. cr-136 makes it
    NAMEABLE: a pin is the instrument that produces the quality evidence a promotion needs."""
    assert [c.name for c in attention.CANDIDATES] == ["flash-attn3", "sdpa"]
    fp8 = attention.BY_NAME["flash-attn3-fp8"]
    assert fp8 in attention.PINNABLE and fp8 not in attention.CANDIDATES
    # The SAME baked snapshot as rank 1, reached through OUR backend rather than diffusers':
    # 0.40.0's `_flash_3_hub` hard-codes `q_descale=k_descale=v_descale=None`, so the fp8 path
    # is not an `AttentionBackendName` diffusers ships and the row names the one we register.
    assert fp8.baked == attention.BY_NAME["flash-attn3"].baked == "flash-attn3"
    assert fp8.backend == attention_fp8.BACKEND == "_flash_3_hub_fp8"
    assert fp8.backend not in {c.backend for c in attention.PINNABLE if c is not fp8}
    # e4m3 is instantiated for head dim 128 only in the baked build; bf16 serves 256.
    assert (fp8.max_head_dim, attention.BY_NAME["flash-attn3"].max_head_dim) == (128, 256)
    # Inexact, and the budget says by how much: h3a-013 measured 3.9e-2, the budget is 1e-1,
    # and a kernel that returns V unchanged measures 16 — the budget is between them.
    assert 2e-2 < fp8.rel_l2_max < 1
    doc = attention.__doc__ or ""
    assert "1.31x" in doc and "2.68 c/s" in doc and "PINNABLE" in doc
    assert "q_descale=k_descale=v_descale=None" in doc
    assert attention_fp8.E4M3_TARGET < attention_fp8.E4M3_MAX


def test_the_fp8_quantiser_maps_each_head_onto_e4m3_and_hands_back_its_descale() -> None:
    """cr-136. The eager path is the executable floor of the fused one and what a CPU probe
    runs: per-(batch, head) descales in FA3's shape, every code finite, the largest magnitude
    of every head landing on 440 (not 448, which a rounded product could overshoot into NaN),
    and a dequantised tensor within e4m3's own precision of the input. An all-zero head is
    zeros, not a division by zero."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    import torch

    x = torch.randn(2, 96, 3, 32, generator=torch.Generator().manual_seed(1)) * 0.6
    x[1, :, 2] = 0.0
    codes, descale = attention_fp8.quantise_eager(x.to(torch.bfloat16))
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == x.shape
    assert descale.shape == (2, 3) and descale.dtype == torch.float32
    assert torch.isfinite(codes.float()).all()
    top = codes.float().abs().amax(dim=(1, 3))
    assert torch.allclose(
        top[:, :2], torch.full_like(top[:, :2], attention_fp8.E4M3_TARGET), rtol=0.02
    )
    assert float(top[1, 2]) == 0.0
    assert float(descale[1, 2]) == pytest.approx(attention_fp8.SCALE_FLOOR)
    back = codes.float() * descale[:, None, :, None]
    error = float((back - x.to(torch.bfloat16).float()).norm() / x.norm())
    assert 1e-2 < error < 6e-2, error
    # A CUDA-less box has no fused path, and asks for it exactly once.
    assert attention_fp8.quantise(x.to(torch.bfloat16))[1].shape == (2, 3)


def test_every_nameable_kernel_has_one_name_and_one_backend() -> None:
    names = [c.name for c in (*attention.CANDIDATES, *attention.PINNABLE)]
    backends = [c.backend for c in (*attention.CANDIDATES, *attention.PINNABLE)]
    assert len(names) == len(set(names)) == len(set(backends))
    assert set(attention.BY_NAME) == set(names)
    # A numerics-changing kernel is admitted and NAMEABLE, and it is not ranked, because
    # nothing has timed it here. That is the whole reason `PINNABLE` exists.
    assert "sageattention" in {c.name for c in attention.PINNABLE}
    assert "sageattention" not in {c.name for c in attention.CANDIDATES}


@pytest.mark.parametrize(
    ("name", "dtype", "head_dim", "admits"),
    [
        ("flash-attn3", "bfloat16", 128, True),
        ("flash-attn3", "bfloat16", 256, True),
        ("flash-attn3", "bfloat16", 512, False),
        ("flash-attn3", "float32", 128, False),
        ("cudnn", "float16", 128, True),
        ("cudnn", "float16", 256, False),
        ("cudnn", "bfloat16", 0, False),
        ("sdpa", "float32", 512, True),
    ],
)
def test_a_site_is_admitted_by_its_own_dtype_and_head_dimension(
    name: str, dtype: str, head_dim: int, admits: bool
) -> None:
    """Admission is per SITE and per KERNEL, decided once at the pin. An unknown head
    dimension admits only the unbounded floor: the runtime does not guess a shape into a
    kernel that would fault on it."""
    assert attention._admits(attention.BY_NAME[name], dtype, head_dim) is admits


def test_the_device_alone_decides_which_ranked_kernels_are_even_candidates() -> None:
    """The pure half of the rule: no import, no image, just the card. A device below a
    kernel's floor never sees it, and every device sees the floor last.

    A BAKED kernel's ceiling is not here, and that is deliberate: it lives in the artifact,
    which this half is not allowed to read. So sm_100 and sm_120 still list `flash-attn3`
    here and are refused one step later, at resolution, by the binary's own declaration."""
    for device in (_device(0, kind="cpu"), _device(70), _device(89)):
        assert [c.name for c in attention.for_device(device)] == ["sdpa"]
    assert [c.name for c in attention.for_device(_device(90))] == ["flash-attn3", "sdpa"]
    assert [c.name for c in attention.for_device(_device(100))] == ["flash-attn3", "sdpa"]
    # Resolution then drops what the image does not ship, in order and without raising.
    for device in (_device(0, kind="cpu"), _device(89), _device(90)):
        offered = [entry.name for entry in attention.available(device)]
        ranked = [candidate.name for candidate in attention.for_device(device)]
        assert offered == [name for name in ranked if name in offered]


# ----------------------------------------------------------- the range is bounded ABOVE too


def test_every_candidate_states_the_architectures_it_serves_and_where_that_came_from() -> None:
    """h3a-022. The defect was an OPEN-ENDED range: `min_sm=90` and nothing above it, on an
    entry whose `.so` carries `sm_90a` and dies from C++ anywhere else. So every entry now
    answers "and up to what?", and the answer comes from one of exactly two places.

    A BAKED entry answers with its artifact and declares no `max_sm` at all — a static copy of
    a binary's arch list is the thing that drifted. Everything else declares one, and a 0 is a
    CLAIM about a dispatcher that owns its own fallback, not a field nobody filled in."""
    for candidate in (*attention.CANDIDATES, *attention.PINNABLE):
        # A COMPILED artifact is bounded by its own metadata and may not restate it. A `jit`
        # snapshot is source and has no metadata to be bounded by, so it MUST declare the
        # ceiling instead — the one case where `baked` and `max_sm` belong together.
        assert candidate.jit or not (candidate.baked and candidate.max_sm), candidate.name
        assert not candidate.jit or candidate.max_sm, candidate.name
        assert not candidate.max_sm or candidate.max_sm >= candidate.min_sm, candidate.name
    # The two shapes, named. `flash-attn3` is the artifact-bounded one; `sageattention` is
    # bounded here because its architecture is IN ITS BACKEND NAME and so cannot drift; the
    # two torch backends are unbounded because torch, not this image, builds them.
    assert attention.BY_NAME["flash-attn3"].baked and not attention.BY_NAME["flash-attn3"].max_sm
    # h3a-024. The third shape: a snapshot that is SOURCE. It declares a ceiling where
    # `flash-attn3` may not, because there is no arch list to drift from — FA4 ships device
    # code for no card and compiles for the one it finds.
    fa4 = attention.BY_NAME["flash-attn4"]
    assert fa4.jit and fa4.baked and (fa4.min_sm, fa4.max_sm) == (90, 110)
    assert not attention.BY_NAME["flash-attn3"].jit
    assert attention.BY_NAME["sageattention"].max_sm == 90
    assert attention.BY_NAME["sageattention"].backend.endswith("sm90")
    assert attention.BY_NAME["cudnn"].max_sm == 0 and attention.BY_NAME["sdpa"].max_sm == 0
    # The floor may never be bounded above: a card it refused would hold no attention at all.
    assert attention.CANDIDATES[-1].name == "sdpa"
    assert attention.CANDIDATES[-1].min_sm == 0 and attention.CANDIDATES[-1].max_sm == 0


@pytest.mark.parametrize(
    ("declared", "sm", "covers"),
    [
        # The baked snapshot's own two entries, against every card the published ladder binds.
        ("9.0a", 90, True),  # H100/H200 — the only card this image's FA3 has an image for
        ("9.0a", 100, False),  # B200
        ("9.0a", 120, False),  # RTX PRO 6000 Blackwell / RTX 5090 — the abort h3a-022 measured
        ("8.0", 90, False),  # never across a major generation, however close the numbers
        ("8.0", 120, False),
        # NVIDIA's binary-compatibility rule: X.y runs on X.z for z >= y, same major only.
        ("8.0", 80, True),
        ("8.0", 86, True),
        ("8.0", 89, True),
        ("8.6", 80, False),
        # An ACCELERATED target is architecture-conditional and does not even carry forward to
        # a higher minor of its own generation; a FAMILY target does.
        ("12.0a", 121, False),
        ("12.0f", 121, True),
        ("12.0", 121, True),
        ("12.1a", 121, True),
        # Spellings a kernel-builder artifact and a torch arch list actually use.
        ("sm_120", 120, True),
        ("compute_90a", 90, True),
        # UNREADABLE IS NOT COVERED. A declaration this runtime cannot parse must never admit
        # a kernel it could not check, because the failure it is checking for is unrecoverable.
        ("", 90, False),
        ("hopper", 90, False),
        ("9.x", 90, False),
    ],
)
def test_device_code_covers_a_card_by_nvidias_rule_and_never_by_optimism(
    declared: str, sm: int, covers: bool
) -> None:
    assert attention._covers(declared, sm) is covers


def test_a_static_ceiling_refuses_the_card_by_name_instead_of_reaching_the_kernel() -> None:
    """RED ARM, no plant and no device: `sageattention`'s declared range excludes sm_120, so a
    pin naming it there is refused BY NAME rather than admitted and probed.

    Admitting it is what the old table did, and probing it is what would then have run the
    kernel: h3a-022 measured `cudaErrorNoKernelImageForDevice` out of this extension on an
    sm_120 card, from an image that builds it `arch=compute_90a,code=sm_90a`."""
    refused = _refusal(lambda: attention.pinned("sageattention", _device(120)))
    assert refused.code == "attention_kernel_unsupported"
    assert "sageattention serves sm90 only" in str(refused)
    assert "sm120" in str(refused)
    # The refusal is also the discoverability (cr-125): it says what this card CAN run.
    assert "This image runs" in str(refused)
    # And the same entry on the card it does claim gets past the device gate, so the ceiling
    # is a bound and not a blanket refusal — sm90 reaches resolution and fails on absence.
    assert _refusal(lambda: attention.pinned("sageattention", _device(90))).code == (
        "attention_kernel_absent"
    )
    # The device gate is applied wherever a candidate is offered, not only at the pin.
    assert "sageattention" not in attention.image_kernels(_device(120))


def test_a_baked_kernels_ceiling_is_read_from_the_artifact_and_refuses_the_card_by_name(
    tmp_path: Path,
) -> None:
    """THE DEFECT, closed against the REAL baked metadata. `backend.archs` below is verbatim
    from `kernels-community/flash-attn3` at revision `7cb368cf…`, variant
    `torch-stable-abi29-cu130-x86_64-linux` — the exact snapshot the cu130 worker image bakes
    at `/opt/cozy/kernels/flash-attn3` and the exact one h3a-022 measured aborting on sm_120.

    The whole point of reading it is that the runtime never had to be told: the table said
    `min_sm=90` with no ceiling, the binary said `9.0a`, and the two disagreed for as long as
    nobody compared them. Now the artifact is the authority.

    No `kernels` distribution, no diffusers and no CUDA device are needed to reach this — the
    gate runs before any of them, because a card the binary has no image for must be refused
    wherever this code runs, not only where the rest of the stack happens to be installed."""
    variant = "torch-stable-abi29-cu130-x86_64-linux"
    build = tmp_path / "flash-attn3" / "build" / variant
    build.mkdir(parents=True)
    (build / "metadata.json").write_text(
        json.dumps(
            {
                "name": "flash-attn3",
                "id": "_flash_attn3_cuda_0f52d9d",
                "version": 1,
                "license": "BSD-3-Clause",
                "backend": {"type": "cuda", "archs": ["8.0", "9.0a"]},
            }
        )
    )
    real = attention.IMAGE_KERNELS_DIR
    attention.IMAGE_KERNELS_DIR = tmp_path
    try:
        flash = attention.BY_NAME["flash-attn3"]
        assert attention._baked_archs(flash) == {variant: ["8.0", "9.0a"]}

        # B200 and the 5090 rung of the published `paul/minimax-h3` ladder. Both are refused
        # by name, and neither reaches the kernel — which is the only outcome that helps,
        # because the kernel's answer is `exit(1)` and not an exception.
        for sm in (100, 120):
            refused = _refusal(lambda card=sm: attention.pinned("flash-attn3", _device(card)))
            assert refused.code == "attention_kernel_unsupported"
            assert variant in str(refused) and "'9.0a'" in str(refused)
            assert f"sm{sm}" in str(refused)
            assert "exits the process from C++" in str(refused)
            # AUTO drops it rather than failing the prepare, and never calls it either.
            assert "flash-attn3" not in [e.name for e in attention.available(_device(sm))]
            assert "flash-attn3" not in attention.image_kernels(_device(sm))

        # GREEN: the card the artifact DOES declare gets through the arch gate and fails
        # further in, on what this environment actually lacks. A gate that refused everything
        # would pass the red arms above while shipping nothing.
        assert _refusal(lambda: attention.pinned("flash-attn3", _device(90))).code == (
            "attention_kernel_absent"
        )

        # SILENCE IS NOT ADMISSION. A baked variant that declares no architecture is refused
        # on every card, because a check that read nothing must not look like one that passed.
        (build / "metadata.json").write_text(json.dumps({"name": "flash-attn3", "version": 1}))
        assert attention._baked_archs(flash) == {variant: []}
        assert _refusal(lambda: attention.pinned("flash-attn3", _device(90))).code == (
            "attention_kernel_unsupported"
        )
    finally:
        attention.IMAGE_KERNELS_DIR = real
    # And with no snapshot baked at all the answer is ABSENT, as it always was: an image that
    # ships no kernel is not an image that ships one for the wrong card.
    assert _refusal(lambda: attention.pinned("flash-attn3", _device(90))).code == (
        "attention_kernel_absent"
    )


def test_a_candidate_that_fails_to_resolve_for_any_reason_drops_out_of_auto() -> None:
    """PLANTED RED ARM. Auto-selection has no refusal arm — the floor takes everything — so a
    kernel that fails to resolve for ANY reason, not only a typed one, must drop out of the
    ranking rather than fail a prepare that was only trying to go faster. The plant is the
    shape that would really happen: a baked snapshot built for another ABI, whose loader
    raises whatever it raises. A PIN is the opposite and keeps its typed refusal."""
    broken = attention.BY_NAME["flash-attn3"]
    real = attention._resolve

    def planted(candidate: attention.Candidate, device: DeviceFacts) -> attention.Ready:
        if candidate is broken:
            raise RuntimeError("ImportError: undefined symbol: _ZN3c105ErrorC1E")
        return real(candidate, device)

    attention._resolve = planted
    try:
        assert "flash-attn3" not in [entry.name for entry in attention.available(_device(90))]
        assert "flash-attn3" not in attention.image_kernels(_device(90))
        # The pin does NOT swallow it: a caller who named it is told, not quietly served
        # something else.
        with pytest.raises(RuntimeError, match="undefined symbol"):
            attention.pinned("flash-attn3", _device(90))
    finally:
        attention._resolve = real


def test_a_pin_this_runtime_does_not_know_refuses_and_names_what_it_knows() -> None:
    refused = _refusal(lambda: attention.pinned("flash_attention_2", _device(90)))
    assert refused.code == "attention_kernel_unknown"
    assert "flash-attn3" in str(refused) and "sageattention" in str(refused)


def test_a_pin_this_device_cannot_serve_refuses_naming_the_device() -> None:
    refused = _refusal(lambda: attention.pinned("flash-attn3", _device(89)))
    assert refused.code == "attention_kernel_unsupported"
    assert "sm90" in str(refused) and "sm89" in str(refused)
    assert _refusal(lambda: attention.pinned("cudnn", _device(0, "cpu"))).code == (
        "attention_kernel_unsupported"
    )


def test_the_group_boot_line_renders_from_merged_facts_alone() -> None:
    """A GROUP prepare has ONE boot note, so its `attention` fact is merged from every
    model's and rendered by the same function — not by an `Applied` it does not have. Same
    sentence, namespaced components, offered set unioned in rank order."""
    single = attention.Applied(
        hosts={"dit": {"flash-attn3": 52}},
        locations={},
        offered=("flash-attn3", "sdpa"),
    )
    assert single.line() == attention.line(single.totals(), single.hosts, list(single.offered), "")
    merged = attention.line(
        {"flash-attn3": 104, "sdpa": 36},
        {
            "fl2va/dit": {"flash-attn3": 52},
            "ref2va/dit": {"flash-attn3": 52},
            "fl2va/vae": {"sdpa": 36},
        },
        ["flash-attn3", "sdpa"],
        "",
    )
    assert merged.startswith("flash-attn3 x104, sdpa x36 (auto over ['flash-attn3', 'sdpa'])")
    assert "fl2va/vae: sdpa" in merged and "ref2va/dit: flash-attn3" in merged
    # A construction with no diffusers attention site says so rather than saying nothing.
    assert attention.line({}, {}, [], "").startswith("NONE")


def test_a_pinned_group_prepare_renders_pinned_from_its_models_own_facts() -> None:
    """THE RED ARM (cr-135). `_prepare_many` merges its models' `attention` facts into the
    one boot record through `attention.merge`; the merge it replaced hardcoded an empty pin,
    so a PINNED two-model prepare printed `auto over [...]` — indistinguishable in the record
    from an unpinned one, which is exactly the A/B h3a-013 cannot afford."""
    pinned = {
        parameter: attention.Applied(
            hosts={"dit": {"cudnn": 52}}, locations={}, offered=("cudnn",), pin="cudnn"
        ).document()
        for parameter in ("fl2va", "ref2va")
    }
    merged = attention.merge(pinned)
    assert merged is not None
    assert merged["pin"] == "cudnn"
    assert merged["line"] == "cudnn x104 (PINNED cudnn) — fl2va/dit: cudnn; ref2va/dit: cudnn"
    assert merged["hosts"] == {"fl2va/dit": {"cudnn": 52}, "ref2va/dit": {"cudnn": 52}}
    assert merged["kernels"] == {"cudnn": 104} and merged["offered"] == ["cudnn"]

    # Unpinned models merge to `auto`, the offered set unioned in rank order.
    auto = attention.merge(
        {
            "fl2va": attention.Applied(
                hosts={"dit": {"flash-attn3": 52}}, locations={}, offered=("flash-attn3", "sdpa")
            ).document(),
            "ref2va": attention.Applied(
                hosts={"vae": {"sdpa": 36}}, locations={}, offered=("sdpa",)
            ).document(),
        }
    )
    assert auto is not None and auto["pin"] == ""
    assert auto["line"].startswith("flash-attn3 x52, sdpa x36 (auto over ['flash-attn3', 'sdpa'])")

    # A model with no site contributes nothing; a set with no site at all is no fact.
    partial = attention.merge({"fl2va": pinned["fl2va"], "vae": {}})
    assert partial is not None
    assert partial["line"] == "cudnn x52 (PINNED cudnn) — fl2va/dit: cudnn"
    assert attention.merge({"a": {}, "b": {}}) is None


# ------------------------------------------------------------------------------ the CUDA arm


def _cuda_proof_skip() -> str:
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        installed = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return "diffusers is not installed"
    if installed != "0.40.0":
        return f"diffusers is {installed}, the proof needs 0.40.0"
    return ""


def test_the_fastest_admitted_kernel_is_pinned_on_real_attention_sites() -> None:
    why = _cuda_proof_skip()
    if why:
        pytest.skip(why)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "testdata/attention_selection_proof.py")],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    out = json.loads(child.stdout.splitlines()[-1])

    # The head dimension and dtype come off the REAL modules, not from a declaration.
    assert out["head_dims"] == {"dit": 128, "vae": 256}
    assert out["dtypes"] == {"dit": "bfloat16", "vae": "float16"}

    # AUTO lands on real processors, and the boot line says what it did in one line.
    assert out["auto"]["offered"] == out["offered"]
    assert set(out["auto"]["kernels"]) <= set(out["offered"])
    for backends in out["auto_backends"].values():
        assert backends and all(b for b in backends)
    assert out["auto"]["line"].startswith(next(iter(out["auto"]["kernels"])))

    # PER-SITE ADMISSION on one construction: cuDNN serves head dim 128 and not 256, so the
    # two sites take different kernels and neither falls back per call.
    assert out["mixed"]["hosts"] == {"dit": {"cudnn": 1}, "vae": {"sdpa": 1}}
    assert out["mixed_backends"] == {"dit": ["_native_cudnn"], "vae": ["native"]}

    # A PIN IS NEVER QUIETLY REPLACED. The same 256-dim site under a cuDNN-only pin refuses.
    assert out["pin_unserved"]["code"] == "attention_kernel_unsupported"
    assert "head dimension 256" in out["pin_unserved"]["detail"]
    assert out["pin_served"]["pin"] == "cudnn"
    assert out["pin_served"]["line"] == "cudnn x1 (PINNED cudnn) — dit: cudnn"
    assert out["pin_backends"] == {"dit": ["_native_cudnn"]}

    # cr-135: the pin SURVIVES THE GROUP MERGE. Two real constructions pinned to cuDNN, read
    # back off their own processors, render PINNED from the one merged record; unpinned they
    # render auto over what this card offers.
    assert out["group_pinned"]["pin"] == "cudnn"
    assert out["group_pinned"]["line"] == (
        "cudnn x2 (PINNED cudnn) — fl2va/dit: cudnn; ref2va/dit: cudnn"
    )
    assert out["group_auto"]["pin"] == ""
    assert out["group_auto"]["offered"] == out["offered"]
    assert sum(out["group_auto"]["kernels"].values()) == 2
    assert "(auto over " in out["group_auto"]["line"]

    # The refusals double as discoverability: each names what would have to be true.
    assert out["unknown"]["code"] == "attention_kernel_unknown"
    assert out["too_old"]["code"] == "attention_kernel_unsupported"
    assert out["absent"]["code"] == "attention_kernel_absent"
    assert "sdpa" in out["image_kernels"]
    # A device refusal names what this image and card DO run, so the developer who pinned a
    # kernel that cannot serve here is not left guessing (cr-125's discoverability rule).
    assert "This image runs" in out["too_old"]["detail"]
    # h3a-022, through the real registry: the range is bounded ABOVE as well as below, and the
    # card above the ceiling is refused by name rather than probed.
    assert out["too_new"]["code"] == "attention_kernel_unsupported"
    assert "sageattention serves sm90 only" in out["too_new"]["detail"]

    # And the pin computes: one real forward through the pinned site.
    assert out["forward"] == {"shape": [1, 64, 1024], "dtype": "torch.bfloat16"}

    # cr-127. Every kernel this card offers agrees with exact fp32 attention well inside its
    # declared budget, and the number is the bf16 floor h3a-013 measured on an H100
    # (2.32-2.33e-3) — the same arithmetic on a different card, so the budget is calibrated
    # against a measurement rather than picked.
    assert set(out["selfcheck_rel_l2"]) >= {"sdpa"}
    for name, disagreement in out["selfcheck_rel_l2"].items():
        assert 1e-3 < disagreement < out["selfcheck_budgets"][name], (name, disagreement)
    assert all(budget > 0 for budget in out["selfcheck_budgets"].values())

    # THE PLANTED DEFECT, three shapes, through diffusers' own registry and dispatch. `empty`
    # is what upstream's meta stub did; `nan` is the h3a-023 hazard, which a bare ratio test
    # would PASS because NaN compares False; `wrong` is finite, right-shaped and the right
    # magnitude — the arm no finiteness check and no implausible-rate check can catch.
    for kind in ("empty", "nan", "wrong"):
        for name in ("sdpa", "cudnn"):
            refusal = out[f"dead_{kind}_{name}_pin"]
            assert refusal["code"] == "attention_kernel_wrong", (kind, name, refusal)
            assert name in refusal["detail"] and f"sm{out['sm']}" in refusal["detail"]
        # A PIN refuses; AUTO drops. With the floor itself dead, auto-selection offers
        # NOTHING rather than silently serving a kernel that returns uninitialized memory —
        # which is exactly what would have shipped corrupted video.
        assert out[f"dead_{kind}_auto"] == []
        assert "sdpa" not in out[f"dead_{kind}_image_kernels"]
    # The finite arm names the disagreement it measured, so triage reads a number.
    assert "rel L2 15.99" in out["dead_wrong_sdpa_pin"]["detail"]
    # And the green arm holds again once the plants are removed: a passing red arm is not a
    # registry left broken.
    assert out["selfcheck_restored"] == out["offered"]

    # cr-136. The fp8 path is NAMEABLE on any box, and what it does depends on what the box
    # ships: without the baked snapshot the pin refuses `absent` by name; with it, the pin
    # resolves through OUR registered backend, measures inside its budget (quantisation, not
    # the 16 of a kernel that returns V), lands on a real 128-dim site and computes, and
    # leaves the bf16 entry's Hub slot untouched.
    fp8 = out["fp8"]
    if fp8["code"]:
        assert fp8["code"] == "attention_kernel_absent" and "flash-attn3-fp8" in fp8["detail"]
    else:
        assert 1e-2 < fp8["rel_l2"] < out["selfcheck_budgets"]["flash-attn3-fp8"]
        assert fp8["backend"] == "_flash_3_hub_fp8"
        assert fp8["line"] == "flash-attn3-fp8 x1 (PINNED flash-attn3-fp8) — dit: flash-attn3-fp8"
        assert fp8["forward"] == {"shape": [1, 64, 1024], "dtype": "torch.bfloat16"}
        assert fp8["bf16_slot_untouched"] is True


# ------------------------------------------------- the durable record is BOUNDED (cr-122)


def test_an_over_cap_observation_truncates_and_never_faults_the_attempt() -> None:
    """cr-122 RED ARM, KEPT. This exact value faulted the executor on the H100: the emit
    boundary raised `observation_too_large`, the refusal escaped device release, the worker
    abandoned the attempt EXECUTOR_INVALIDATED, rebuilt and ran the same 30-step generation
    again — three complete attempts before an operator cancelled the fourth. A record that
    cannot be held is truncated and counted. It is never a fault."""
    ring = EventRing(subject="req#1")
    row = ring.emit("confession", "attention", "x" * 1744, subject="req#1")
    assert isinstance(row.value, str)
    assert len(row.value) == TEXT_CAP and row.value.endswith("…")
    assert ring.caps()["truncated"] == 1 and ring.caps()["refused"] == 0
    assert ring.caps()["kept"] == 1 and ring.caps()["admitted"] == 1
    # Field values are payload too, and a field of 8 KiB is the same defect wearing a hat.
    ring.emit("log", "note", "short", detail="y" * 8192)
    assert len(str(ring.events[-1].fields["detail"])) == TEXT_CAP
    assert ring.caps()["truncated"] == 2 and ring.caps()["refused"] == 0
    # A NAME still refuses: it is the record's identity, and clipping one would make two
    # different records the same record.
    with pytest.raises(ObservationRefusal) as refused:
        ring.emit("log", "n" * (NAME_CAP + 1))
    assert refused.value.code == "observation_too_large"


def test_the_attempt_confession_of_a_huge_value_is_admitted() -> None:
    """The real worker record, through the real `confess` the release path calls."""
    attempt = AttemptRecord(request_id="req", attempt=1, digest=b"", spec={})
    attempt.confess("attention", "z" * 4096)
    assert attempt.ring.caps()["refused"] == 0 and attempt.ring.caps()["truncated"] == 1
    assert [row["kind"] for row in attempt.confessions] == ["attention"]
    assert len(str(attempt.ring.events[-1].value)) == TEXT_CAP


# ------------------------------------------ the execution-path override, threaded (cr-125)


def _binding(models: int = 1) -> DeclaredBinding:
    """One resolved binding, exactly the shape the boot path hands the prepare builder."""
    slots = tuple(
        ModelBinding(
            model_class="pkg:Dit",
            model_binding_path=f"m{index}",
            model_parameter_name=f"p{index}",
            store="store",
            variant="fp8",
            reference_snapshot="sha256:aa",
            components=("dit",),
            snapshots={"dit": "sha256:aa"},
        )
        for index in range(models)
    )
    return DeclaredBinding(
        entrypoint_binding_digest="sha256:bb",
        entrypoint="fl2va",
        model_class=slots[0].model_class,
        model_binding_path=slots[0].model_binding_path,
        model_parameter_name=slots[0].model_parameter_name,
        release="rel-1",
        models=slots if models > 1 else (),
        components=slots[0].components,
        snapshots=slots[0].snapshots,
        store=slots[0].store,
        variant=slots[0].variant,
        reference_snapshot=slots[0].reference_snapshot,
    )


def test_a_serving_prepare_carries_no_pin_at_all() -> None:
    """PRODUCTION IS KNOB-FREE. The override is absent by default and the serving prepare is
    byte-identical to what it was before cr-125 — the key is not present, not empty."""
    served = executor_prepare_command(_binding(), devices="0", authorized_device_limit_bytes=7)
    assert "attention_pin" not in served
    # And nothing can put one there from a `serve` launch: the flag set is CLOSED and does
    # not carry it, so the guard is the absence of a way to say it rather than a condition.
    assert not any("attention" in flag for flag in host._SERVE_FLAGS)
    assert WorkerOptions(root=Path("/tmp")).attention_pin == ""


def test_a_pinned_prepare_names_one_kernel_for_every_model_of_the_construction() -> None:
    """The pin rides the ONE prepare funnel, and a group takes it whole: two models of one
    construction cannot serve two different execution paths and still be one measurement."""
    single = executor_prepare_command(
        _binding(), devices="0", authorized_device_limit_bytes=7, attention_pin="sageattention"
    )
    assert single["attention_pin"] == "sageattention"
    group = executor_prepare_command(
        _binding(models=2),
        devices="0",
        authorized_device_limit_bytes=7,
        attention_pin="sageattention",
    )
    # It sits on the COMMAND, not inside a model row: `_prepare_many` hands the same value
    # to every child prepare, the way it hands them one device authorization.
    assert group["attention_pin"] == "sageattention"
    assert all("attention_pin" not in row["binding"] for row in group["models"])


@pytest.mark.parametrize(
    "term",
    [
        "kernel.gemm=rowwise",  # an axis this runtime does not have
        "kernel.attention",  # no value
        "kernel.attention=",  # an empty pin is not a pin
        "kernel.attention:=sageattention",  # the typed-JSON form of the payload grammar
    ],
)
def test_an_unusable_kernel_term_refuses_instead_of_becoming_a_payload_field(term: str) -> None:
    """The reserved namespace is claimed BEFORE the payload grammar sees a term, so a
    mistyped override can never quietly land on a package's own field."""
    with pytest.raises(CliError) as refused:
        cli_run.split_flags(["marco", term])
    assert refused.value.name == "usage"
    assert "kernel.attention=<name>" in str(refused.value.remedy)


def test_the_claimed_pin_leaves_the_payload_terms_untouched() -> None:
    terms, flags = cli_run.split_flags(
        ["fl2va", "prompt=a cat", "kernel.attention=sageattention", "--seed", "7"]
    )
    assert terms == ["fl2va", "prompt=a cat"]
    assert flags["kernel.attention"] == ["sageattention"]
    assert flags["--seed"] == ["7"]


@needs_executor
def test_a_pin_that_cannot_be_honoured_refuses_through_the_whole_real_path() -> None:
    """THE THREAD, end to end and unmocked: `cozy-runtime run`'s ephemeral worker, the real
    prepare command, a real executor child process.

    `examples/marco-polo` declares no model, so it holds no attention site — and a pin with
    nowhere to go is REFUSED rather than ignored. That is the same rule the A/B rests on: a
    run that reported success while serving something other than what was named would make
    the measurement worthless, so the runtime never quietly drops a pin.
    """
    base = {
        k: v
        for k, v in os.environ.items()
        if not child_env.erased(k) and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
    }
    with tempfile.TemporaryDirectory(prefix="cz-pin.", dir="/tmp") as raw:
        root = Path(raw)
        config = RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            child_base_env=tuple(sorted(base.items())),
        )
        workspace = root / "run"
        workspace.mkdir(parents=True, exist_ok=True)
        body = package_interface.build(discover(PACKAGE))
        with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
            outcome = run_slice(
                config,
                LocalRequest(
                    entrypoint="marco",
                    payload={"message": "marco"},
                    outputs=(),
                    request_id="pin-weightless",
                    kind="entrypoint",
                ),
                workspace=workspace,
                package_code_digest=development_source_digest(PACKAGE),
                package_interface_document=body,
                project=PACKAGE,
                development_tensorfs_root=config.cozy_home / "tensorfs",
                options=WorkerOptions(
                    root=workspace / "worker",
                    devices="0",
                    attention_pin="sageattention",
                ),
            )
    reasons = [fault["reason"] for fault in outcome.faults]
    assert "attention_kernel_unsupported" in reasons, outcome.faults
    assert outcome.result is None, outcome.result
    assert any("sageattention" in str(fault["detail"]) for fault in outcome.faults)


# ---------------------------------------------------------- the served per-request pin (cr-125)


class _Parameter:
    def __init__(self, dtype: str) -> None:
        self.dtype = f"torch.{dtype}"

    def __str__(self) -> str:
        return self.dtype


class _Processor:
    def __init__(self) -> None:
        self._attention_backend: Any = "AUTO"


class _Site:
    """One attention module as `_sites` reads it: a processor, a dtype and a head dimension."""

    def __init__(self, dtype: str = "bfloat16", head_dim: int = 128) -> None:
        self.processor = _Processor()
        self.head_dim = head_dim
        self._dtype = dtype

    def parameters(self) -> Any:
        yield _Parameter(self._dtype)


class _Tree:
    """A torch module tree as `_trees` duck-types one: `modules()` and `parameters()`."""

    def __init__(self, *modules: Any) -> None:
        self._modules = modules

    def modules(self) -> Any:
        return [*self._modules, self]

    def parameters(self) -> Any:
        return iter(())


class _Compiled(_Tree):
    """`torch.compile(module)` — the wrapping spelling, which renames every state_dict key."""

    def __init__(self) -> None:
        super().__init__()
        self._orig_mod = _Tree()


class _CompiledInPlace(_Tree):
    """`nn.Module.compile()` — the in-place spelling, which leaves no wrapper to see."""

    def __init__(self) -> None:
        super().__init__()
        self._compiled_call_impl = lambda *a, **k: None


@pytest.fixture
def diffusers_attention_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_sites` looks up diffusers' attention classes by identity; CI installs no diffusers.

    Only the CLASS LOOKUP is supplied. The walk, the dtype and head-dimension reads, the
    admission rule and the swap itself are this module's own code, unpatched.
    """
    import types

    processor_module = types.ModuleType("diffusers.models.attention_processor")
    processor_module.Attention = _Site  # type: ignore[attr-defined]
    processor_module.MochiAttention = _Site  # type: ignore[attr-defined]
    attention_module = types.ModuleType("diffusers.models.attention")
    attention_module.AttentionModuleMixin = _Site  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    monkeypatch.setitem(sys.modules, "diffusers.models", types.ModuleType("diffusers.models"))
    monkeypatch.setitem(sys.modules, "diffusers.models.attention", attention_module)
    monkeypatch.setitem(sys.modules, "diffusers.models.attention_processor", processor_module)


def _ready(name: str) -> attention.Ready:
    return attention.Ready(attention.BY_NAME[name], member=f"<{name}>")


def test_a_per_request_swap_moves_every_site_and_records_the_kernel_it_moved_to() -> None:
    """THE ARM THE AMENDMENT OWES: two attempts on ONE prepare, two kernels, two records.

    The swap is mechanically a `setattr` across the processors because the fused forward reads
    `_attention_backend` live on every call rather than capturing it at install
    (`fusion_install.py:226`), so one prepared pod serves both arms of an A/B.
    """
    sites = [_Site(), _Site(), _Site()]
    plan = tuple(("dit", site.processor) for site in sites)

    first = attention.Swap(_ready("flash-attn3"), plan).apply()
    assert [site.processor._attention_backend for site in sites] == ["<flash-attn3>"] * 3
    assert first.totals() == {"flash-attn3": 3}
    assert first.line() == "flash-attn3 x3 (PINNED flash-attn3) — dit: flash-attn3"

    second = attention.Swap(_ready("cudnn"), plan).apply()
    assert [site.processor._attention_backend for site in sites] == ["<cudnn>"] * 3
    assert second.totals() == {"cudnn": 3}
    assert second.line() == "cudnn x3 (PINNED cudnn) — dit: cudnn"
    # Two attempts, two DIFFERENT records off one construction. A pin that rendered the same
    # line twice would report an A/B that did not happen.
    assert first.line() != second.line()


@pytest.mark.parametrize(
    ("root", "names"),
    [
        (_Compiled(), ("OptimizedModule",)),
        (_CompiledInPlace(), ("compiled in place",)),
    ],
)
def test_a_pin_on_a_compiled_module_refuses_instead_of_setting_a_field_nobody_reads(
    root: Any, names: tuple[str, ...]
) -> None:
    """THE GUARD IS A REFUSAL, NOT AN ABSENCE (cr-125 amendment).

    A compiled graph was traced with the kernel it had, so a later `setattr` moves nothing —
    and an A/B whose pin did not take is WORSE than no A/B, because it reports a difference
    that is not there. Both compile spellings are seen: the wrapper leaves `_orig_mod`, the
    in-place form leaves only `_compiled_call_impl`.
    """
    with pytest.raises(attention.AttentionRefusal) as refused:
        attention.plan(_device(90), {"dit": root}, "flash-attn3")
    assert refused.value.code == "attention_kernel_frozen"
    for name in names:
        assert name in str(refused.value)
    # And it names what would have to be true, plus where the pin DOES take.
    assert "kernel.attention=flash-attn3" in str(refused.value)


def test_an_installed_group_refuses_the_swap_before_the_kernel_is_even_looked_up() -> None:
    """`parallel/cp.py` orders backend, then parallelism, then compile. Once collectives are
    installed the ordering is spent, so the swap refuses rather than arriving too late."""
    with pytest.raises(attention.AttentionRefusal) as refused:
        attention.plan(_device(90), {"dit": _Tree()}, "sageattention", "a group is installed")
    assert refused.value.code == "attention_kernel_frozen"
    assert "a group is installed" in str(refused.value)


def test_a_mistyped_kernel_refuses_per_request_exactly_as_it_does_per_prepare() -> None:
    """The served path reuses `pinned`, so the override's vocabulary has ONE definition."""
    with pytest.raises(attention.AttentionRefusal) as refused:
        attention.plan(_device(90), {"dit": _Tree()}, "flash-attn3-fp9")
    assert refused.value.code == "attention_kernel_unknown"
    assert "flash-attn3-fp8" in str(refused.value)


def test_a_site_the_kernel_cannot_serve_refuses_with_nothing_set(
    diffusers_attention_classes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A HALF-SWAPPED CONSTRUCTION WOULD SERVE TWO KERNELS AND RECORD ONE.

    `plan` proves every site admits the kernel before `Swap.apply` sets any of them. Only the
    baked-image resolution is substituted — this runner has no GPU and no diffusers — and the
    walk, the admission rule and the ordering under test are real.
    """
    monkeypatch.setattr(attention, "pinned", lambda name, device: [_ready(name)])
    good, bad = _Site(head_dim=128), _Site(head_dim=256)
    with pytest.raises(attention.AttentionRefusal) as refused:
        attention.plan(_device(90), {"dit": _Tree(good, bad)}, "sageattention")
    assert refused.value.code == "attention_kernel_unsupported"
    assert "head dimension 256" in str(refused.value)
    assert good.processor._attention_backend == "AUTO"
    assert bad.processor._attention_backend == "AUTO"

    swapped = attention.plan(_device(90), {"dit": _Tree(good)}, "sageattention").apply()
    assert good.processor._attention_backend == "<sageattention>"
    assert swapped.pin == "sageattention"


def test_a_construction_with_no_attention_site_refuses_the_pin_rather_than_ignoring_it(
    diffusers_attention_classes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention, "pinned", lambda name, device: [_ready(name)])
    with pytest.raises(attention.AttentionRefusal) as refused:
        attention.plan(_device(90), {"vae": _Tree()}, "cudnn")
    assert refused.value.code == "attention_kernel_unsupported"
    assert "never ignored" in str(refused.value)


def test_an_unpinned_attempt_sends_no_kernel_key_to_the_executor_at_all() -> None:
    """ABSENT BY DEFAULT, and absent means the key is not there — the executor branches on
    presence, so a serving worker nobody pinned cannot reach the swap path."""
    assert attempts._attention_pin({}) == {}
    assert attempts._attention_pin({"attention_kernel": ""}) == {}
    assert attempts._attention_pin({"attention_kernel": "cudnn"}) == {"attention_kernel": "cudnn"}


def test_the_wire_carries_the_pin_inside_the_invocation_digest() -> None:
    """Two attempts differing ONLY by their pin are two different documents. If the pin rode
    beside the digest instead of inside it, one attempt could be replayed as the other."""
    bare = pb.InvocationSpec(payload_digest="sha256:" + "aa" * 32)
    pinned_spec = pb.InvocationSpec(payload_digest="sha256:" + "aa" * 32)
    pinned_spec.attention_kernel = "flash-attn3-fp8"
    assert documents.digest_of(documents.canonical_bytes(bare)) != documents.digest_of(
        documents.canonical_bytes(pinned_spec)
    )
    assert (
        documents.read(documents.canonical_bytes(pinned_spec), pb.InvocationSpec)[
            "attention_kernel"
        ]
        == "flash-attn3-fp8"
    )


def test_the_pre_admission_half_of_a_pin_touches_no_device(
    diffusers_attention_classes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`prepare_request` runs while the OTHER arm of the A/B may still be generating, so its
    check may not fire a resolution probe — `_computes` is a real kernel call, and one landing
    there would sit inside somebody else's step timings. `check` therefore never resolves."""

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("the pre-admission check resolved a kernel")

    monkeypatch.setattr(attention, "_resolve", refuse)
    monkeypatch.setattr(attention, "pinned", refuse)
    assert attention.check(_device(90), {"dit": _Tree(_Site())}, "sdpa").name == "sdpa"
    # And it still raises every refusal it can afford to: the vocabulary, the card, the sites.
    for name, kernel, roots in (
        ("attention_kernel_unknown", "nope", {"dit": _Tree(_Site())}),
        ("attention_kernel_unsupported", "sageattention", {"dit": _Tree(_Site())}),
        ("attention_kernel_frozen", "sdpa", {"dit": _Compiled()}),
    ):
        device = _device(80) if name == "attention_kernel_unsupported" else _device(90)
        with pytest.raises(attention.AttentionRefusal) as refused:
            attention.check(device, roots, kernel)
        assert refused.value.code == name


def test_a_typo_on_a_compiled_module_is_answered_as_a_typo() -> None:
    """Order matters in a refusal ladder. Telling a developer their construction is compiled,
    when what they did was misspell `sdpa`, sends them at the wrong thing entirely."""
    with pytest.raises(attention.AttentionRefusal) as refused:
        attention.check(_device(90), {"dit": _Compiled()}, "sdap")
    assert refused.value.code == "attention_kernel_unknown"
