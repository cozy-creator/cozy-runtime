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

from cozy_runtime.author._observations import (
    NAME_CAP,
    TEXT_CAP,
    EventRing,
    ObservationRefusal,
)
from cozy_runtime.cli import host
from cozy_runtime.cli import run as cli_run
from cozy_runtime.cli.io import CliError
from cozy_runtime.internal import attention, child_env, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.local import LocalRequest, development_source_digest, run_slice
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.plan import DeclaredBinding, ModelBinding
from cozy_runtime.internal.worker.session import WorkerOptions, executor_prepare_command
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
    kernel's floor never sees it, and every device sees the floor last."""
    for device in (_device(0, kind="cpu"), _device(70), _device(89)):
        assert [c.name for c in attention.for_device(device)] == ["sdpa"]
    assert [c.name for c in attention.for_device(_device(90))] == ["flash-attn3", "sdpa"]
    assert [c.name for c in attention.for_device(_device(100))] == ["flash-attn3", "sdpa"]
    # Resolution then drops what the image does not ship, in order and without raising.
    for device in (_device(0, kind="cpu"), _device(89), _device(90)):
        offered = [entry.name for entry in attention.available(device)]
        ranked = [candidate.name for candidate in attention.for_device(device)]
        assert offered == [name for name in ranked if name in offered]


def test_a_candidate_that_fails_to_resolve_for_any_reason_drops_out_of_auto() -> None:
    """PLANTED RED ARM. Auto-selection has no refusal arm — the floor takes everything — so a
    kernel that fails to resolve for ANY reason, not only a typed one, must drop out of the
    ranking rather than fail a prepare that was only trying to go faster. The plant is the
    shape that would really happen: a baked snapshot built for another ABI, whose loader
    raises whatever it raises. A PIN is the opposite and keeps its typed refusal."""
    broken = attention.BY_NAME["flash-attn3"]
    real = attention._resolve

    def planted(candidate: attention.Candidate) -> attention.Ready:
        if candidate is broken:
            raise RuntimeError("ImportError: undefined symbol: _ZN3c105ErrorC1E")
        return real(candidate)

    attention._resolve = planted
    try:
        assert "flash-attn3" not in [entry.name for entry in attention.available(_device(90))]
        assert "flash-attn3" not in attention.image_kernels()
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

    # The refusals double as discoverability: each names what would have to be true.
    assert out["unknown"]["code"] == "attention_kernel_unknown"
    assert out["too_old"]["code"] == "attention_kernel_unsupported"
    assert out["absent"]["code"] == "attention_kernel_absent"
    assert "sdpa" in out["image_kernels"]

    # And the pin computes: one real forward through the pinned site.
    assert out["forward"] == {"shape": [1, 64, 1024], "dtype": "torch.bfloat16"}


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
