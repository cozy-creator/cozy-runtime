"""Select and install attention backends for constructed diffusers modules.

Selection checks device architecture, installed artifacts, a numerical smoke test,
site dtype/head dimension, and context-parallel support. Explicit pins refuse
unsupported sites. Automatic selection preserves attention activation precision;
quantizing those activations requires an explicit backend choice.

The optional FP8 backend has an H100 kernel benchmark, including quantization.
That result does not establish full-video speed or output quality; matched H3
outputs exposed detail loss despite passing its numerical smoke test.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from importlib import import_module, metadata
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from cozy_runtime.author._attention import AttentionContext
from cozy_runtime.internal import attention_fp8, attention_sol, attention_upstream
from cozy_runtime.internal.encoding import DeviceFacts

try:
    from kernels import get_local_kernel
except ImportError as _absent_kernels:  # pragma: no cover - exercised by the absent arm
    KERNELS_ABSENT = f"does not import ({_absent_kernels})"
else:
    KERNELS_ABSENT = ""

#: Where the worker image bakes `kernels` Hub snapshots (tensorhub's cu130 Dockerfile):
#: `<distribution>/build/<variant>/` holds one built variant and its metadata. Nothing here is
#: fetched — a pod's executor has no network for it.
IMAGE_KERNELS_DIR = Path("/opt/cozy/kernels")


class AttentionRefusal(RuntimeError):
    """An unsupported explicit selection or unavailable attention implementation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class Candidate:
    """One attention kernel the runtime knows how to reach, and what it needs to run."""

    name: str
    backend: str
    min_sm: int  # CUDA capability, e.g. 90 for sm90; zero also admits CPU.
    dtypes: frozenset[str]  # Empty admits all dtypes.
    max_head_dim: int  # Zero is unbounded.
    max_sm: int = 0  # Baked binaries use their artifact's architecture metadata instead.
    baked: str = ""
    distribution: str = ""
    jit: bool = False  # Source kernels use the declared device range, not binary metadata.
    rel_l2_max: float = 0.0  # Numerical smoke-test threshold, not an output-quality guarantee.
    why: str = ""
    sms: frozenset[int] = frozenset()
    head_dims: frozenset[int] = frozenset()
    import_name: str = ""


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="flash-attn3",
        backend="_flash_3_hub",
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=256,
        baked="flash-attn3",
        rel_l2_max=2e-2,
        why=(
            "Quality-preserving default. H100 H3 N109104: FA3 BF16 about 506 ms, "
            "FA4 about 501 ms, cuDNN about 616 ms; no consistent FA4 gain. "
            "Retained: tracker/h3-acceleration/ATTENTION-UPSTREAM-2026-09-14.md; "
            "outputs/h3-attention-oracle-20260914/ATTENTION-RESULTS.md. "
            "This is a curated H100 ranking, not a universal hardware winner."
        ),
    ),
    Candidate(
        name="sdpa",
        backend="native",
        min_sm=0,
        dtypes=frozenset(),
        max_head_dim=0,
        max_sm=0,
        rel_l2_max=2e-2,
        why="PyTorch scaled_dot_product_attention; supports all site dtypes and head dimensions.",
    ),
)

PINNABLE: tuple[Candidate, ...] = (
    Candidate(
        name="flash-attn3-fp8",
        backend=attention_fp8.BACKEND,
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=128,
        baked="flash-attn3",
        rel_l2_max=6e-2,
        why=(
            "Explicit experiment: matched H3 clips change. The corrected experimental "
            "FA3 FP8 build took 781.3 s versus BF16 907.1 s for 15 s video; qualification "
            "is specific to that build, not every FA3 FP8 binary. "
            "Evidence: outputs/h3-kernel-quality-20260914/comparison15/REPORT.md."
        ),
    ),
    Candidate(
        name="flash-attn4",
        backend="flash_4_hub",
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=256,
        max_sm=110,
        baked="flash-attn4",
        jit=True,
        rel_l2_max=2e-2,
        why=(
            "Unranked: matched H100 BF16 kernels showed no consistent FA4 gain over FA3. "
            "Evidence: outputs/h3-attention-oracle-20260914/"
            "torch214-cudnn924/qualification/REPORT.md."
        ),
    ),
    Candidate(
        name="cudnn",
        backend="_native_cudnn",
        min_sm=80,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=128,
        max_sm=0,
        rel_l2_max=2e-2,
        why="H100 SXM reference: 34.99 s per DiT forward versus native dispatch 34.70 s; h3a-012.",
    ),
    Candidate(
        name="sageattention",
        backend="_sage_qk_int8_pv_fp8_cuda_sm90",
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=128,
        max_sm=90,
        distribution="sageattention",
        rel_l2_max=2e-1,
        why=(
            "Explicit experiment: H100 15 s/30-step H3 generation took 801.2 s versus "
            "BF16 907.1 s; matched clips differ. No automatic quality approval. "
            "Evidence: outputs/h3-kernel-quality-20260914/comparison15/REPORT.md."
        ),
    ),
    Candidate(
        name="kitchen-int8",
        backend=attention_upstream.KITCHEN_BACKEND,
        min_sm=90,
        max_sm=103,
        sms=frozenset({90, 100, 103}),
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=256,
        distribution="comfy-kitchen",
        rel_l2_max=2e-1,
        why="Explicit dense INT8 experiment; H3 quality/timing qualification pending (h3a-060).",
    ),
    Candidate(
        name="flashinfer-bf16-fp8",
        backend=attention_upstream.FLASHINFER_BACKEND,
        min_sm=100,
        max_sm=103,
        sms=frozenset({100, 103}),
        dtypes=frozenset({"bfloat16"}),
        max_head_dim=128,
        head_dims=frozenset({128}),
        distribution="flashinfer-python",
        import_name="flashinfer",
        rel_l2_max=2e-1,
        why="Explicit BF16 Q/K + FP8 P/V experiment; H3 qualification pending (h3a-060).",
    ),
    Candidate(
        name="sol-attn",
        backend="_cozy_sol_bf16",
        min_sm=90,
        max_sm=90,
        sms=frozenset({90}),
        dtypes=frozenset({"bfloat16"}),
        max_head_dim=128,
        head_dims=frozenset({128}),
        distribution="sol-attn",
        rel_l2_max=2e-2,  # Native all-keys arithmetic control, not sparse approximation error.
        why="Explicit sparse experiment; eager model layout and optional Ulysses degrees 2/4.",
    ),
)

BY_NAME: dict[str, Candidate] = {c.name: c for c in (*CANDIDATES, *PINNABLE)}


@dataclass(frozen=True, slots=True)
class Ready:
    """One candidate PROVEN present in this environment, with diffusers' member for it."""

    candidate: Candidate
    member: Any
    location: str = ""
    """The selected local snapshot variant directory; empty for ordinary modules."""
    admission: dict[str, Any] | None = None
    """Retained arithmetic/contract checks; never a model-quality qualification."""
    artifact: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.candidate.name


def _registry() -> Any:
    """Diffusers' attention backend registry, or a typed absence.

    Any Diffusers exposing the per-site dispatch registry qualifies; each backend is
    still proven by registration and a numerical smoke test before it serves.
    """
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    except ImportError as exc:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"this environment's diffusers exposes no attention backend registry: {exc}",
        ) from exc
    return _AttentionBackendRegistry


def _member(backend: str) -> Any:
    if backend == attention_fp8.BACKEND:
        # OURS, not diffusers': registered into its registry on first use (cr-136).
        return attention_fp8.member()
    upstream = {
        "_cozy_sol_bf16": attention_sol.sol_attention,
        attention_upstream.KITCHEN_BACKEND: attention_upstream.kitchen_int8_attention,
        attention_upstream.FLASHINFER_BACKEND: attention_upstream.flashinfer_mixed_attention,
    }
    if backend in upstream:
        from cozy_runtime.internal.attention_registry import register

        return register(
            backend, upstream[backend], supports_context_parallel=backend == "_cozy_sol_bf16"
        )
    from diffusers.models.attention_dispatch import AttentionBackendName

    return AttentionBackendName(backend)


def _capability(declared: str) -> tuple[int, int, str] | None:
    """One architecture as an artifact spells it — `"9.0a"`, `"8.0"`, `"sm_120"` — or None.

    None is "this runtime cannot read what the artifact declared", and every caller treats it
    as NOT covered, because the alternative is calling a kernel we could not check.
    """
    entry = declared.strip().lower().removeprefix("sm_").removeprefix("compute_")
    suffix = entry[-1] if entry[-1:] in ("a", "f") else ""
    digits = entry.removesuffix(suffix) if suffix else entry
    major, _, minor = digits.partition(".")
    if not minor:
        major, minor = digits[:-1], digits[-1:]
    if not (major.isdigit() and minor.isdigit()):
        return None
    return int(major), int(minor), suffix


def _covers(declared: str, sm: int) -> bool:
    """Does device code built for `declared` have an image for a device of capability `sm`?"""
    parsed = _capability(declared)
    if parsed is None:
        return False
    major, minor, suffix = parsed
    if (major, minor) == (sm // 10, sm % 10):
        return True
    return suffix != "a" and major == sm // 10 and minor <= sm % 10


def _declared_archs(variant: Path) -> list[str]:
    """The architectures a `kernels` build variant says it carries, from its own metadata.

    `build/<variant>/metadata.json` is written by kernel-builder beside the `.so` it describes,
    so this is the artifact's statement about itself and not a second copy of it. An empty
    answer is a refusal at the call site, never an admission.
    """
    try:
        document = json.loads((variant / "metadata.json").read_text())
        archs = document["backend"]["archs"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [str(arch) for arch in archs] if isinstance(archs, list) else []


def _snapshot_location(candidate: Candidate) -> Path:
    """Prefer an ordinary dependency wheel; never hide a broken installed candidate."""
    distribution = "cozy-kernel-" + candidate.baked
    try:
        installed = metadata.distribution(distribution)
    except metadata.PackageNotFoundError:
        return IMAGE_KERNELS_DIR / candidate.baked
    package = distribution.replace("-", "_")
    files = installed.files or ()
    if not any(str(file).startswith(package + "/build/") for file in files):
        raise AttentionRefusal(
            "attention_kernel_absent", f"{distribution} carries no recorded kernel snapshot"
        )
    return Path(str(installed.locate_file(package)))


def _artifact(distribution: str) -> dict[str, Any]:
    """Identify installed bytes using the wheel inventory already retained by the closure."""
    installed = metadata.distribution(distribution)
    record = installed.read_text("RECORD")
    if not record:
        raise AttentionRefusal(
            "attention_kernel_absent", f"{distribution} has no installed wheel RECORD"
        )
    result: dict[str, Any] = {
        "distribution": installed.metadata["Name"],
        "version": installed.version,
        "record_digest": "sha256:" + hashlib.sha256(record.encode()).hexdigest(),
    }
    direct = installed.read_text("direct_url.json")
    if direct:
        try:
            origin = json.loads(direct)
            archive = origin.get("archive_info", {})
            digest = archive.get("hashes", {}).get("sha256")
            if not digest and str(archive.get("hash", "")).startswith("sha256="):
                digest = archive["hash"].removeprefix("sha256=")
            if not digest:
                # uv can retain a hash-qualified wheel URL with empty archive_info.
                digest = parse_qs(urlsplit(origin.get("url", "")).fragment).get("sha256", [""])[0]
            if (
                isinstance(digest, str)
                and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest.lower())
            ):
                result["wheel_digest"] = "sha256:" + digest.lower()
        except (ValueError, TypeError, AttributeError):
            pass
    urls = installed.metadata.get_all("Project-URL") or []
    for url in urls:
        label, _, value = url.partition(",")
        if label.strip().lower() == "source":
            result["source"] = value.strip()
    return result


def _baked_archs(candidate: Candidate) -> dict[str, list[str]]:
    """Every installed or image variant, and the architectures each one declares."""
    builds = _snapshot_location(candidate) / "build" if candidate.baked else None
    if builds is None or not builds.is_dir():
        return {}
    return {
        variant.name: _declared_archs(variant)
        for variant in sorted(builds.iterdir())
        if (variant / "metadata.json").is_file()
    }


def _admit_arch(candidate: Candidate, archs: dict[str, list[str]], device: DeviceFacts) -> None:
    """Refuse a card that none of these baked variants has device code for."""
    if candidate.jit:
        # NOTHING TO READ, so nothing is checked here and the static `max_sm` on the row is
        # the whole ceiling. A source snapshot declares no architectures because it carries
        # no device code, and treating that silence as the same refusal an empty
        # `metadata.json` earns from a COMPILED variant would refuse the kernel on every
        # card it serves.
        return
    if device.kind != "cuda" or not archs:
        return
    if any(_covers(arch, device.sm) for declared in archs.values() for arch in declared):
        return
    raise AttentionRefusal(
        "attention_kernel_unsupported",
        f"{candidate.name} declares {archs} and none of those builds carries device code "
        f"for {device.name} (sm{device.sm}). This is a REFUSAL rather than an attempt because "
        "the attempt is not survivable: FlashAttention-3's launcher prints `no kernel image is "
        "available for execution on the device` and exits the process from C++, so neither the "
        "numeric self-check nor auto-selection's own fallback would ever run (h3a-022 measured "
        "it on an sm_120 RTX PRO 6000 with this exact snapshot). A worker image or "
        "installed dependency wheel carrying this architecture changes the answer",
    )


def _load_baked(candidate: Candidate, device: DeviceFacts) -> tuple[Any, str]:
    """Load the selected local snapshot into Diffusers' Hub slot without fetching."""
    if KERNELS_ABSENT:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} loads through the `kernels` distribution, which {KERNELS_ABSENT}",
        )
    location = _snapshot_location(candidate)
    builds = location / "build"
    variants = (
        sorted(p for p in builds.iterdir() if (p / "metadata.json").is_file())
        if (builds.is_dir())
        else []
    )
    if not variants:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} has no snapshot under {builds}; install the ordinary "
            f"cozy-kernel-{candidate.baked} dependency before preparation. "
            "This executor has no network to fetch it",
        )
    try:
        module = get_local_kernel(location)
    except Exception as exc:
        # A snapshot this torch cannot load is an ABSENT kernel, not a crash: the image
        # baked a variant built for another ABI, and the floor still serves.
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} does not load from {location}: {type(exc).__name__}: {exc}",
        ) from exc
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    # kernels supports both build/<variant>/__init__.py and a nested package.
    loaded = next(
        (parent for parent in module_path.parents if parent.parent == builds.resolve()), None
    )
    if loaded is None or not (loaded / "metadata.json").is_file():
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} resolved outside its selected snapshot: {module_path}; "
            f"expected a variant under {builds}",
        )
    _admit_arch(candidate, {loaded.name: _declared_archs(loaded)}, device)
    if candidate.backend == attention_fp8.BACKEND:
        # The fp8 path binds the snapshot's forward to ITS OWN backend and leaves the bf16
        # entry's Hub slot alone, so pinning one never rewires the other (cr-136).
        return attention_fp8.bind(_lookup(module, "flash_attn_func", candidate.name)), str(loaded)
    from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY

    member = _member(candidate.backend)
    if member not in _HUB_KERNELS_REGISTRY:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"diffusers does not resolve {candidate.backend!r} through its Hub-kernel "
            "registry, so a baked snapshot cannot be placed in its slot",
        )
    config = _HUB_KERNELS_REGISTRY[member]
    config.kernel_fn = _lookup(module, config.function_attr, candidate.name)
    for attr, slot in (
        ("wrapped_forward_attr", "wrapped_forward_fn"),
        ("wrapped_backward_attr", "wrapped_backward_fn"),
    ):
        dotted = getattr(config, attr)
        if dotted:
            setattr(config, slot, _lookup(module, dotted, candidate.name))
    return member, str(loaded)


def _lookup(module: Any, dotted: str, name: str) -> Any:
    entry: Any = module
    for part in dotted.split("."):
        entry = getattr(entry, part, None)
        if entry is None:
            raise AttentionRefusal("attention_kernel_absent", f"{name} does not expose {dotted}")
    return entry


#: The self-check's shape: (batch, sequence, heads, head dimension). This crosses two
#: 128-row blocks. Sol adds a ragged row and padding; optional kernels may JIT on first use.
PROBE: tuple[int, int, int, int] = (1, 256, 4, 128)


def _computes(entry: Ready, device: DeviceFacts) -> dict[str, Any] | None:
    """PROVE the kernel computes attention on THIS device, or refuse it by name."""
    if device.kind != "cuda":
        # There is no device to prove anything on, and the only entry a CPU executor admits is
        # torch's own eager dispatch. Silence here is the honest answer, not a skipped gate.
        return None
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    batch, sequence, heads, dim = PROBE
    if entry.name == "sol-attn":
        sequence += 1  # Exercise the sparse kernel's partial final block as well.
    dtype = (
        torch.bfloat16
        if not entry.candidate.dtypes or "bfloat16" in entry.candidate.dtypes
        else torch.float16
    )
    target = torch.device("cuda", device.index)
    seed = torch.Generator(device=target).manual_seed(0x5A6E)
    shape = (batch, sequence, heads, dim)
    query, key, value = (
        (torch.randn(shape, generator=seed, device=target, dtype=torch.float32) * 0.5).to(dtype)
        for _ in range(3)
    )
    with torch.no_grad(), torch.cuda.device(target):
        if entry.name == "sol-attn":
            return _probe_sol(entry, device, query, key, value)
        got = dispatch_attention_fn(query, key, value, backend=entry.member)
    if isinstance(got, tuple):
        got = got[0]

    q_ref, k_ref, v_ref = (tensor.transpose(1, 2).float() for tensor in (query, key, value))
    exact = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref).transpose(1, 2)
    _validate_probe(entry, device, got, query, exact)
    return None


def _probe_sol(
    entry: Ready, device: DeviceFacts, query: Any, key: Any, value: Any
) -> dict[str, Any]:
    """Separate native arithmetic from the wrapper's approximate sparse policy."""
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope

    inputs = (query, key, value)
    originals = tuple(t.clone() for t in inputs)
    sequence, prefix, padding = query.shape[1], 33, 15
    q_ref, k_ref, v_ref = (t.transpose(1, 2).float() for t in inputs)
    exact = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref).transpose(1, 2)
    # Public sink semantics keep all KV blocks exact inside the actual native
    # kernel. No Runtime dense-prefix overwrite can make this control pass.
    native = attention_sol._native(query, key, value, None, sequence)
    _validate_probe(entry, device, native, query, exact)
    if any(
        not torch.equal(t.view(torch.uint8), old.view(torch.uint8))
        for t, old in zip(inputs, originals, strict=True)
    ):
        raise AttentionRefusal("attention_kernel_wrong", "Sol all-keys control modified its inputs")

    padded = tuple(
        torch.cat((t, torch.full_like(t[:, :padding], float("nan"))), dim=1) for t in inputs
    )
    before = tuple(t.view(torch.uint8).clone() for t in padded)
    probe: Any = torch.nn.Module()
    probe.forward = lambda q, k, v: dispatch_attention_fn(q, k, v, backend=entry.member)
    attention_sol.install_site(probe, "probe", "attention")
    try:
        with (
            attention_scope(AttentionLayout(sequence, prefix, 0)),
            attention_sol.observing() as counts,
        ):
            sparse = probe(*padded)
            repeated = probe(*padded)
    finally:
        attention_sol.remove_site(probe, "probe", "attention")
    _validate_probe_contract(entry, device, sparse, padded[0], padded[0])
    expected_prefix = attention_sol._dense(query[:, :prefix], key, value, None)
    if (
        not torch.equal(sparse[:, :prefix], expected_prefix)
        or torch.count_nonzero(sparse[:, sequence:]).item() != 0
        or not torch.equal(sparse, repeated)
        or any(
            not torch.equal(t.view(torch.uint8), old) for t, old in zip(padded, before, strict=True)
        )
        or counts != dict(sparse=2, dense_step=0, dense_path=0, dense_prefix=2)
    ):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            "Sol sparse policy violated protected-prefix, padding, repeat "
            "or input-ownership contracts",
        )
    return {
        "arithmetic_control": "native_all_keys_selected",
        "arithmetic_rel_l2_to_fp32": float((native.float() - exact).norm() / exact.norm()),
        "arithmetic_rel_l2_max": entry.candidate.rel_l2_max,
        "sparse_policy_rel_l2_to_fp32": float(
            (sparse[:, :sequence].float() - exact).norm() / exact.norm()
        ),
        "sparse_error_meaning": "approximation_only_not_model_quality",
        "shape": list(query.shape),
        "protected_prefix": prefix,
        "padding": padding,
        "repeat_equal": True,
        "inputs_unchanged": True,
        "calls": counts,
    }


def _validate_probe(entry: Ready, device: DeviceFacts, got: Any, query: Any, exact: Any) -> None:
    """Check a real result's contract and small numerical smoke, not model quality."""
    got = _validate_probe_contract(entry, device, got, query, exact)
    disagreement = float((got - exact).norm() / exact.norm())
    if disagreement > entry.candidate.rel_l2_max:
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} disagrees with exact fp32 attention at rel L2 {disagreement:.4g} on "
            f"{device.name} (sm{device.sm}), past its declared budget of "
            f"{entry.candidate.rel_l2_max:.4g}. A kernel that returns its uninitialized output "
            "buffer raises nothing and has the right shape (h3a-013), so this is the only "
            "thing that can tell the difference",
        )


def _validate_probe_contract(
    entry: Ready, device: DeviceFacts, got: Any, query: Any, exact: Any
) -> Any:
    """Geometry and finite output are required even when approximation changes the answer."""
    import torch

    if not isinstance(got, torch.Tensor) or got.dtype != query.dtype or got.device != query.device:
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} must return {query.dtype} on {query.device}, matching its input",
        )
    if tuple(got.shape) != tuple(exact.shape):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} returned {tuple(got.shape)}; attention over "
            f"{tuple(query.shape)} requires "
            f"{tuple(exact.shape)}",
        )
    got = got.float()
    if not bool(torch.isfinite(got).all()):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} returned non-finite values on {device.name} (sm{device.sm}); it is "
            "not serving attention and a NaN would pass every ratio test",
        )
    return got


def _resolve(candidate: Candidate, device: DeviceFacts) -> Ready:
    """One candidate, proven present here AND proven to compute, or a typed refusal."""
    _admit_arch(candidate, _baked_archs(candidate), device)
    registry = _registry()
    member = _member(candidate.backend)
    if member not in registry._backends:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"diffusers registers no backend {candidate.backend!r}",
        )
    if candidate.baked:
        loaded_member, location = _load_baked(candidate, device)
        try:
            artifact = _artifact("cozy-kernel-" + candidate.baked)
        except metadata.PackageNotFoundError:
            artifact = None
        ready = Ready(candidate, loaded_member, location, artifact=artifact)
        return replace(ready, admission=_computes(ready, device))
    if candidate.distribution:
        try:
            installed = metadata.version(candidate.distribution)
        except metadata.PackageNotFoundError as exc:
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"{candidate.name} needs the {candidate.distribution} distribution, which "
                "this environment does not install",
            ) from exc
        try:
            import_module(candidate.import_name or candidate.distribution.replace("-", "_"))
        except Exception as exc:
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"{candidate.distribution}=={installed} does not import: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
    if candidate.name == "sol-attn" and not attention_sol.dense_identity():
        dense = next(
            (entry for entry in available(device) if _admits(entry.candidate, "bfloat16", 128)),
            None,
        )
        if dense is None:
            raise AttentionRefusal(
                "attention_kernel_absent", "Sol has no qualified dense reference"
            )
        attention_sol.bind_dense(
            registry._backends[dense.member], dense.name + ":" + dense.location
        )
    ready = Ready(
        candidate,
        member,
        artifact=_artifact(candidate.distribution) if candidate.distribution else None,
    )
    return replace(ready, admission=_computes(ready, device))


def _device_admits(candidate: Candidate, device: DeviceFacts) -> str:
    """Empty when this DEVICE is inside the candidate's DECLARED range, else why it is not."""
    if candidate.sms and device.sm not in candidate.sms:
        return f"{candidate.name} serves SM {sorted(candidate.sms)}, not sm{device.sm}"
    if candidate.min_sm and (device.kind != "cuda" or device.sm < candidate.min_sm):
        return (
            f"{candidate.name} needs sm{candidate.min_sm} or better; this executor holds "
            f"{device.kind} sm{device.sm}"
        )
    if candidate.max_sm and device.kind == "cuda" and device.sm > candidate.max_sm:
        serves = (
            f"sm{candidate.max_sm} only"
            if candidate.min_sm == candidate.max_sm
            else f"sm{candidate.min_sm} through sm{candidate.max_sm}"
        )
        return (
            f"{candidate.name} serves {serves}; this executor holds sm{device.sm}, and the "
            "binary behind it carries no device code for that architecture. Calling it anyway "
            "returns `cudaErrorNoKernelImageForDevice` or kills the process from C++ rather "
            "than raising something this runtime could recover from (h3a-022)"
        )
    return ""


def for_device(device: DeviceFacts) -> list[Candidate]:
    """The ranked table filtered by DEVICE alone — no import, no image. The pure half.

    Pure means it can only apply the bounds the TABLE declares. A baked kernel's real ceiling
    is in its artifact, so it is applied one step later, at resolution — which is why this
    list can still name a kernel that `available` then drops.
    """
    return [candidate for candidate in CANDIDATES if not _device_admits(candidate, device)]


def available(device: DeviceFacts) -> list[Ready]:
    """The ranked kernels this device and environment can actually run, fastest first.

    Never raises: the floor takes everything, so auto-selection has no refusal arm. A ranked
    kernel this image does not ship is simply not in the answer, and an environment with no
    diffusers at all offers nothing — which is right, because it holds no attention site.
    """
    ready: list[Ready] = []
    for candidate in for_device(device):
        try:
            ready.append(_resolve(candidate, device))
        except Exception:
            # BROAD ON PURPOSE. Auto-selection has no refusal arm — the floor takes
            # everything — so a kernel that fails to resolve for ANY reason must drop out of
            # the ranking rather than fail the prepare that was only trying to go faster. A
            # PIN is the opposite and keeps its typed refusals (`pinned`). A kernel that
            # RESOLVES BUT DOES NOT COMPUTE now leaves by this same door, which is the point:
            # auto-selection silently taking a dead kernel is what would have shipped
            # corrupted video (h3a-013).
            continue
    return ready


def pinned(name: str, device: DeviceFacts) -> list[Ready]:
    """The single kernel the execution-path override names, or a typed refusal (cr-125).

    A pin does not fall back. That is the whole point: a run that silently served another
    kernel would make the A/B it was asked for meaningless.
    """
    candidate = BY_NAME.get(name)
    if candidate is None:
        raise AttentionRefusal(
            "attention_kernel_unknown",
            f"{name!r} is not an attention kernel this runtime knows; it knows {sorted(BY_NAME)}",
        )
    unsupported = _device_admits(candidate, device)
    if unsupported:
        # THE REFUSAL IS THE DISCOVERABILITY. A developer pinning a kernel this card cannot
        # run is told, in the same sentence, what this image and this card actually offer —
        # which is what `image_kernels` is for, and it lists what COMPUTES rather than what
        # imports (cr-127) so the answer never points at another trap.
        raise AttentionRefusal(
            "attention_kernel_unsupported",
            f"{unsupported}. This environment runs {image_kernels(device)} on {device.name}",
        )
    return [_resolve(candidate, device)]


# ------------------------------------------------------------------------------ the pin


def _trees(roots: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield each constructed module tree, preserving component labels."""
    seen: set[int] = set()
    queue: deque[tuple[str, Any, int]] = deque(
        (name, root, 0) for name, root in sorted(roots.items())
    )
    while queue:
        label, value, depth = queue.popleft()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if callable(getattr(value, "modules", None)) and callable(
            getattr(value, "parameters", None)
        ):
            yield label, value
            continue
        if depth >= 3:
            continue
        if isinstance(value, Mapping):
            for key, item in value.items():
                queue.append((str(key), item, depth + 1))
        elif hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                if not name.startswith("_"):
                    queue.append((name, item, depth + 1))


def _attention_records(roots: Mapping[str, Any]) -> Iterator[tuple[str, str, Any, Any]]:
    """All Diffusers attention processors, including those with no backend dispatch."""
    try:
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_processor import Attention, MochiAttention
    except ImportError:
        return
    classes = (Attention, MochiAttention, AttentionModuleMixin)
    seen: set[int] = set()
    for component, tree in _trees(roots):
        for path, module in tree.named_modules():
            processor = getattr(module, "processor", None)
            if id(module) not in seen and isinstance(module, classes) and processor is not None:
                seen.add(id(module))
                yield component, path, module, processor


def _site_records(roots: Mapping[str, Any]) -> Iterator[tuple[str, str, Any, Any]]:
    """Only processors implementing Diffusers' per-site backend selection contract."""
    for record in _attention_records(roots):
        if hasattr(record[3], "_attention_backend"):
            yield record


def _unmanaged_reason(component: str, path: str, processor: Any) -> str:
    return (
        f"{component}/{path} uses {type(processor).__name__}, which has no per-site "
        "attention backend selection"
    )


def _sites(roots: Mapping[str, Any]) -> Iterator[tuple[str, Any, Any]]:
    for component, _path, module, processor in _site_records(roots):
        yield component, module, processor


def _head_dim(module: Any) -> int:
    """This site's per-head dimension, or 0 when the module does not say.

    A site whose head dimension cannot be read is treated as unknown, which admits only the
    unbounded floor — the runtime does not guess a shape into a kernel that would fault on it.
    """
    declared = getattr(module, "head_dim", None)
    if isinstance(declared, int) and declared > 0:
        return declared
    heads = getattr(module, "heads", None)
    inner = getattr(module, "inner_dim", None)
    if inner is None:
        to_q = getattr(module, "to_q", None)
        inner = getattr(to_q, "out_features", None)
    if isinstance(heads, int) and heads > 0 and isinstance(inner, int) and inner > 0:
        return inner // heads
    return 0


def _projection_dtype(projection: Any) -> str:
    """Encoded output contract, or an ordinary floating Linear's result dtype."""
    import torch

    declared = getattr(projection, "out_dtype", None)
    if declared is None and type(projection) is torch.nn.Linear:
        declared = projection.weight.dtype
    name = str(declared).removeprefix("torch.")
    return name if name in {"float16", "bfloat16", "float32", "float64"} else ""


def _dtype(module: Any) -> str:
    """Infer attention activation precision from the active Q/K/V projection contracts.

    Never inspect unrelated norm parameters or encoded weight buffers. Unknown or
    disagreeing projection contracts admit only an unbounded dtype backend (SDPA).
    Floating Linear inference assumes no external autocast changes its output dtype.
    """
    q, k, v = (getattr(module, name, None) for name in ("to_q", "to_k", "to_v"))
    packed = getattr(module, "to_qkv", None)
    fused = getattr(module, "fused_projections", None)
    projections: tuple[Any, ...]
    if packed is not None and (fused is True or (fused is None and q is None)):
        projections = (packed,)
    elif q is not None and getattr(module, "to_kv", None) is not None and fused is True:
        projections = (q, module.to_kv)
    elif q is not None and k is not None and v is not None:
        projections = (q, k, v)
    else:
        return ""
    dtypes = {_projection_dtype(projection) for projection in projections}
    return next(iter(dtypes)) if len(dtypes) == 1 and "" not in dtypes else ""


def _admits(candidate: Candidate, dtype: str, head_dim: int) -> bool:
    if candidate.head_dims and head_dim not in candidate.head_dims:
        return False
    if candidate.dtypes and dtype not in candidate.dtypes:
        return False
    return not (candidate.max_head_dim and (head_dim == 0 or head_dim > candidate.max_head_dim))


def _context_parallel(member: Any) -> bool:
    """Does diffusers' registry mark this backend as one it can shard? Read at the pin and
    never restated on a row: the backend's own registration is the authority, and a static
    copy of it would be the drift `_admit_arch` exists to refuse."""
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    return bool(_AttentionBackendRegistry._is_context_parallel_available(member))


@dataclass(frozen=True, slots=True)
class Applied:
    """What the runtime pinned, per component and per kernel. The BOOT record, and all of it."""

    #: component -> kernel name -> attention sites pinned to it.
    hosts: dict[str, dict[str, int]]
    #: kernel name -> the baked variant directory it loaded from, when it loaded from one.
    locations: dict[str, str]
    #: the ranked kernels this environment and device offered, fastest first.
    offered: tuple[str, ...]
    #: set when the caller PINNED one through the execution-path override (cr-125).
    pin: str = ""
    admission: dict[str, dict[str, Any]] | None = None
    artifacts: dict[str, dict[str, Any]] | None = None

    def totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kernels in self.hosts.values():
            for name, sites in kernels.items():
                out[name] = out.get(name, 0) + sites
        return out

    def document(self) -> dict[str, Any]:
        return {
            "kernels": self.totals(),
            "hosts": {name: dict(kernels) for name, kernels in self.hosts.items()},
            "offered": list(self.offered),
            "locations": dict(self.locations),
            "pin": self.pin,
            **({"admission": self.admission} if self.admission else {}),
            **({"artifacts": self.artifacts} if self.artifacts else {}),
            # RENDERED HERE, not by the worker: one renderer for the one line, in the module
            # that knows what it means.
            "line": self.line(),
        }

    def line(self) -> str:
        """The ONE line a pod prints at boot. It is the whole per-run attention record."""
        return line(self.totals(), self.hosts, list(self.offered), self.pin)


def line(
    totals: Mapping[str, int],
    hosts: Mapping[str, Mapping[str, int]],
    offered: Sequence[str],
    pin: str,
) -> str:
    """The boot line, from the facts alone — so a GROUP prepare renders the same sentence from
    its merged facts as a single one does from its own `Applied` (cr-124)."""
    if not totals:
        return "NONE — this construction holds no backend-selectable diffusers attention site"
    chosen = ", ".join(f"{name} x{sites}" for name, sites in sorted(totals.items()))
    where = "; ".join(
        f"{component}: {'+'.join(sorted(kernels))}" for component, kernels in sorted(hosts.items())
    )
    how = f"PINNED {pin}" if pin else f"auto over {list(offered)}"
    return f"{chosen} ({how}) — {where}"


def merge(documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """A GROUP prepare's one attention fact, from its models' own `Applied.document()`s."""
    hosts: dict[str, dict[str, int]] = {}
    kernels: dict[str, int] = {}
    offered: list[str] = []
    locations: dict[str, str] = {}
    pins: set[str] = set()
    admission: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for parameter, row in documents.items():
        for component, chosen in (row.get("hosts") or {}).items():
            hosts[f"{parameter}/{component}"] = dict(chosen)
        for name, sites in (row.get("kernels") or {}).items():
            kernels[name] = kernels.get(name, 0) + int(sites)
        locations.update(row.get("locations") or {})
        for name in row.get("offered") or ():
            if name not in offered:
                offered.append(name)
        if row.get("pin"):
            pins.add(str(row["pin"]))
        if row.get("admission"):
            admission[parameter] = row["admission"]
        if row.get("artifacts"):
            artifacts[parameter] = row["artifacts"]
    if not kernels:
        return None
    pin = "+".join(sorted(pins))
    return {
        "kernels": kernels,
        "hosts": hosts,
        "offered": offered,
        "locations": locations,
        "pin": pin,
        **({"admission": admission} if admission else {}),
        **({"artifacts": artifacts} if artifacts else {}),
        "line": line(kernels, hosts, offered, pin),
    }


def _pin_parts(pin: str) -> tuple[str, str]:
    if "=" not in pin:
        return "", pin
    component, name = pin.split("=", 1)
    if not component or not name or "=" in name or any(ch.isspace() for ch in pin):
        raise AttentionRefusal(
            "attention_kernel_unknown", f"invalid attention override {pin!r}; use component=backend"
        )
    return component, name


def for_model(pin: str, parameters: Sequence[str]) -> str:
    """Translate an aggregate model/component pin to one construction's local scope."""
    scope, backend = _pin_parts(pin)
    if "/" not in scope:
        return pin
    model, component = scope.split("/", 1)
    return f"{component}={backend}" if model in parameters else ""


def _matches(component: str, scope: str) -> bool:
    if not scope or component == scope:
        return True
    model, qualified, name = scope.partition("/")
    if not qualified:
        return component.rsplit("/", 1)[-1] == scope
    owners, owned, component_name = component.partition("/")
    return bool(owned and model in owners.split("+") and name == component_name)


def _eligible(
    candidate: Candidate, device: DeviceFacts, dtype: str, head_dim: int, degree: int
) -> bool:
    if _device_admits(candidate, device) or not _admits(candidate, dtype, head_dim):
        return False
    try:
        _admit_arch(candidate, _baked_archs(candidate), device)
        if candidate.distribution:
            metadata.version(candidate.distribution)
        member = _member(candidate.backend)
        return member in _registry()._backends and (degree == 1 or _context_parallel(member))
    except Exception:
        return False


def install(
    ready: list[Ready],
    roots: Mapping[str, Any],
    pin: str = "",
    degree: int = 1,
    *,
    device: DeviceFacts | None = None,
    choose: Callable[[AttentionContext], str | None] | None = None,
) -> Applied:
    """Validate a complete per-site selection before changing any processor."""
    scope, name = _pin_parts(pin)
    hosts: dict[str, dict[str, int]] = {}
    locations: dict[str, str] = {}
    assignments: list[tuple[str, str, Any, Any, Ready]] = []
    resolved = {entry.name: entry for entry in ready}
    matched = False
    eligible_sites: dict[tuple[str, int], tuple[str, ...]] = {}
    for component, path, module, processor in _attention_records(roots):
        dtype, head_dim = _dtype(module), _head_dim(module)
        forced = bool(name and _matches(component, scope))
        matched |= forced
        if not hasattr(processor, "_attention_backend"):
            reason = _unmanaged_reason(component, path, processor)
            if forced:
                raise AttentionRefusal("attention_kernel_unsupported", reason)
            if choose is not None:
                assert device is not None
                selection = choose(
                    AttentionContext(
                        component,
                        path,
                        device.kind,
                        device.name,
                        device.sm,
                        dtype,
                        head_dim,
                        degree,
                        (),
                        "",
                        reason,
                    )
                )
                if selection is not None:
                    raise AttentionRefusal("attention_kernel_unsupported", reason)
            continue

        offered = [
            entry
            for entry in ready
            if (entry.name == name if forced else not pin or entry.candidate in CANDIDATES)
            and _admits(entry.candidate, dtype, head_dim)
            and (degree == 1 or _context_parallel(entry.member))
        ]
        if not offered:
            detail = _site_refusal(
                name if forced else "no available attention kernel", component, dtype, head_dim
            )
            if degree > 1:
                detail = AttentionRefusal(
                    detail.code,
                    f"{detail}; sequence-parallel degree {degree} requires context parallelism",
                )
            raise detail
        entry = offered[0]
        if choose is not None and not forced:
            assert device is not None
            key = (dtype, head_dim)
            if key not in eligible_sites:
                eligible_sites[key] = tuple(
                    c.name
                    for c in BY_NAME.values()
                    if _eligible(c, device, dtype, head_dim, degree)
                )
            eligible = eligible_sites[key]
            context = AttentionContext(
                component,
                path,
                device.kind,
                device.name,
                device.sm,
                dtype,
                head_dim,
                degree,
                eligible,
                entry.name,
                entry.candidate.why,
            )
            selection = choose(context)
            if selection is not None:
                if not isinstance(selection, str) or selection not in eligible:
                    raise AttentionRefusal(
                        "attention_kernel_unsupported",
                        f"model chose {selection!r} for {component}/{path}; "
                        f"eligible choices are {eligible}",
                    )
                if selection not in resolved:
                    resolved[selection] = pinned(selection, device)[0]
                entry = resolved[selection]
        assignments.append((component, path, module, processor, entry))
        hosts.setdefault(component, {})
        hosts[component][entry.name] = hosts[component].get(entry.name, 0) + 1
        if entry.location:
            locations[entry.name] = entry.location
    if name and not matched:
        raise AttentionRefusal(
            "attention_kernel_unsupported", f"{pin!r} matches no attention component"
        )
    _install_sites(
        [
            (entry, component, path, module)
            for component, path, module, _processor, entry in assignments
        ]
    )
    for _component, _path, _module, processor, entry in assignments:
        processor._attention_backend = entry.member
    proofs = {name: entry.admission for name, entry in resolved.items() if entry.admission}
    artifacts = {name: entry.artifact for name, entry in resolved.items() if entry.artifact}
    return Applied(hosts, locations, tuple(resolved), pin, proofs or None, artifacts or None)


def _install_sites(records: Sequence[tuple[Ready, str, str, Any]]) -> None:
    """Install metadata hooks atomically, preserving hooks owned by preparation."""
    added: list[tuple[str, str, Any]] = []
    try:
        for entry, component, path, module in records:
            if entry.name != "sol-attn":
                continue
            local = component.rsplit("/", 1)[-1]
            if getattr(module, "_cozy_sol_site", None) is None:
                added.append((local, path, module))
            attention_sol.install_site(module, local, path)
    except BaseException:
        for component, path, module in reversed(added):
            if getattr(module, "_cozy_sol_site", None) is not None:
                attention_sol.remove_site(module, component, path)
        raise


def observed(roots: Mapping[str, Any], pin: str = "") -> Applied:
    """Read actual processor choices; never substitute the prepared recommendation."""
    names = {candidate.backend: candidate.name for candidate in BY_NAME.values()}
    hosts: dict[str, dict[str, int]] = {}
    for component, _module, processor in _sites(roots):
        backend = getattr(processor, "_attention_backend", None)
        value = getattr(backend, "value", str(backend))
        name = names.get(value, value)
        hosts.setdefault(component, {})
        hosts[component][name] = hosts[component].get(name, 0) + 1
    artifacts: dict[str, dict[str, Any]] = {}
    for name in {name for kernels in hosts.values() for name in kernels}:
        candidate = BY_NAME.get(name)
        if candidate is None:
            continue
        distribution = (
            "cozy-kernel-" + candidate.baked if candidate.baked else candidate.distribution
        )
        if distribution:
            # The image compatibility snapshot has no wheel distribution.
            with suppress(metadata.PackageNotFoundError):
                artifacts[name] = _artifact(distribution)
    return Applied(hosts, {}, (), pin, artifacts=artifacts or None)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Prepared processor choices, restored without resolution or policy re-execution."""

    choices: tuple[tuple[Any, Any], ...]

    def restore(self) -> None:
        for processor, member in self.choices:
            processor._attention_backend = member


def snapshot(roots: Mapping[str, Any]) -> Snapshot:
    return Snapshot(
        tuple((processor, processor._attention_backend) for _, _, processor in _sites(roots))
    )


@contextmanager
def override(
    defaults: Snapshot, roots: Mapping[str, Any], swap: Swap | None, pin: str = ""
) -> Iterator[Applied]:
    """Observe one temporary request selection and restore it even when execution raises."""
    defaults.restore()
    temporary_sites = (
        [
            (component.rsplit("/", 1)[-1], path, module)
            for component, path, module in swap.modules
            if getattr(module, "_cozy_sol_site", None) is None
        ]
        if swap is not None and swap.entry.name == "sol-attn"
        else []
    )
    try:
        applied = None
        if swap is not None:
            applied = swap.apply()
        facts = observed(roots, pin)
        yield (
            replace(
                facts,
                locations=applied.locations,
                admission=applied.admission,
                artifacts={**(facts.artifacts or {}), **(applied.artifacts or {})} or None,
            )
            if applied is not None
            else facts
        )
    finally:
        try:
            for component, path, module in temporary_sites:
                if getattr(module, "_cozy_sol_site", None) is not None:
                    attention_sol.remove_site(module, component, path)
        finally:
            defaults.restore()


def _site_refusal(name: str, component: str, dtype: str, head_dim: int) -> AttentionRefusal:
    return AttentionRefusal(
        "attention_kernel_unsupported",
        f"{name} does not serve {component}'s attention: {dtype or 'unknown'} activations "
        f"at head dimension {head_dim or 'unknown'}. A pinned kernel is never quietly replaced",
    )


def _baked_in(roots: Mapping[str, Any]) -> str:
    for label, tree in _trees(roots):
        for module in tree.modules():
            if hasattr(module, "_orig_mod"):
                return f"{label} is an OptimizedModule wrapping {type(module._orig_mod).__name__}"
            if getattr(module, "_compiled_call_impl", None) is not None:
                return f"{label} holds {type(module).__name__} compiled in place"
    return ""


@dataclass(frozen=True, slots=True)
class Swap:
    """A validated per-request attention backend change, applied atomically."""

    entry: Ready
    sites: tuple[tuple[str, Any], ...]
    pin: str = ""
    modules: tuple[tuple[str, str, Any], ...] = ()

    def apply(self) -> Applied:
        hosts: dict[str, dict[str, int]] = {}
        _install_sites(
            [(self.entry, component, path, module) for component, path, module in self.modules]
        )
        for component, processor in self.sites:
            processor._attention_backend = self.entry.member
            hosts.setdefault(component, {})
            hosts[component][self.entry.name] = hosts[component].get(self.entry.name, 0) + 1
        locations = {self.entry.name: self.entry.location} if self.entry.location else {}
        proofs = {self.entry.name: self.entry.admission} if self.entry.admission else None
        artifacts = {self.entry.name: self.entry.artifact} if self.entry.artifact else None
        return Applied(
            hosts, locations, (self.entry.name,), self.pin or self.entry.name, proofs, artifacts
        )


def _is_selected(defaults: Snapshot, roots: Mapping[str, Any], pin: str) -> bool:
    """A repeated prepared pin is a no-op only at every site in its nonempty scope."""
    scope, name = _pin_parts(pin)
    candidate = BY_NAME.get(name)
    if candidate is None:
        return False
    prepared = {id(processor): member for processor, member in defaults.choices}
    records = [row for row in _attention_records(roots) if _matches(row[0], scope)]
    return bool(records) and all(
        getattr(processor, "_attention_backend", None) == candidate.backend
        and prepared.get(id(processor)) == candidate.backend
        for _, _, _, processor in records
    )


def admit(pin: str, degree: int) -> None:
    """Refuse a pin no construction here could serve before anything is constructed."""
    _, name = _pin_parts(pin)
    if not name:
        return
    candidate = BY_NAME.get(name)
    if candidate is None:
        raise AttentionRefusal(
            "attention_kernel_unknown",
            f"unknown attention kernel {name!r}; this runtime knows {sorted(BY_NAME)}",
        )
    if degree > 1:
        _registry()
        if not _context_parallel(_member(candidate.backend)):
            raise AttentionRefusal(
                "attention_kernel_unsupported",
                f"{name} does not support context parallelism, which this sequence-parallel "
                f"degree {degree} construction requires",
            )


def check(device: DeviceFacts, roots: Mapping[str, Any], name: str, frozen: str = "") -> Candidate:
    scope, backend = _pin_parts(name)
    candidate = BY_NAME.get(backend)
    if candidate is None:
        raise AttentionRefusal(
            "attention_kernel_unknown",
            f"unknown attention kernel {backend!r}; this runtime knows {sorted(BY_NAME)}",
        )
    selected = {component: root for component, root in roots.items() if _matches(component, scope)}
    blocked = frozen or _baked_in(selected)
    if blocked:
        raise AttentionRefusal(
            "attention_kernel_frozen",
            f"{name} cannot be swapped into this prepared construction: {blocked}. "
            "Send the pin with the request so the worker prepares a pinned construction "
            "(or set it in the model attention policy) before compilation or parallel-group "
            "construction",
        )
    unsupported = _device_admits(candidate, device)
    if unsupported:
        raise AttentionRefusal("attention_kernel_unsupported", unsupported)
    sites = 0
    for component, path, module, processor in _attention_records(selected):
        if not hasattr(processor, "_attention_backend"):
            raise AttentionRefusal(
                "attention_kernel_unsupported", _unmanaged_reason(component, path, processor)
            )
        dtype, head_dim = _dtype(module), _head_dim(module)
        if not _admits(candidate, dtype, head_dim):
            raise _site_refusal(candidate.name, component, dtype, head_dim)
        sites += 1
    if not sites:
        raise AttentionRefusal(
            "attention_kernel_unsupported", f"{name} matches no diffusers attention site"
        )
    return candidate


def plan(device: DeviceFacts, roots: Mapping[str, Any], name: str, frozen: str = "") -> Swap:
    candidate = check(device, roots, name, frozen)
    scope, _ = _pin_parts(name)
    entry = pinned(candidate.name, device)[0]
    records = [
        (component, path, module, processor)
        for component, path, module, processor in _site_records(roots)
        if _matches(component, scope)
    ]
    return Swap(
        entry,
        tuple((c, p) for c, _, _, p in records),
        name,
        tuple((c, path, m) for c, path, m, _ in records),
    )


def select(
    device: DeviceFacts,
    roots: Mapping[str, Any],
    pin: str = "",
    degree: int = 1,
    *,
    choose: Callable[[AttentionContext], str | None] | None = None,
) -> Applied:
    """Select once at preparation; optional model policy and request pins are validated."""
    scope, name = _pin_parts(pin)
    ready = pinned(name, device) if name and not scope else available(device)
    if scope:
        ready = [entry for entry in ready if entry.name != name] + pinned(name, device)
    return install(ready, roots, pin, degree, device=device, choose=choose)


def image_kernels(device: DeviceFacts) -> str:
    """What this environment and device can run, for a refusal message (cr-125)."""
    shipped: list[str] = []
    for candidate in (*CANDIDATES, *PINNABLE):
        if _device_admits(candidate, device):
            continue
        try:
            _resolve(candidate, device)
        except Exception:
            continue
        shipped.append(candidate.name)
    return ", ".join(shipped) or "none"


def baked_variants() -> dict[str, Any]:
    """The baked snapshots under `IMAGE_KERNELS_DIR`, for the boot record and for triage."""
    out: dict[str, Any] = {}
    for candidate in (*CANDIDATES, *PINNABLE):
        builds = IMAGE_KERNELS_DIR / candidate.baked / "build" if candidate.baked else None
        if builds is None or not builds.is_dir():
            continue
        for variant in sorted(builds.iterdir()):
            metadata_file = variant / "metadata.json"
            if metadata_file.is_file():
                out[candidate.name] = {
                    "variant": variant.name,
                    **json.loads(metadata_file.read_text()),
                }
    return out
