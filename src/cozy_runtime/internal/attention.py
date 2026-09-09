"""Select and install attention backends for constructed diffusers modules.

Selection checks device architecture, installed artifacts, a numerical smoke test,
site dtype/head dimension, and context-parallel support. Explicit pins refuse
unsupported sites. Automatic selection tries candidates in order.

The FP8 rank is based on an H100 kernel benchmark, including quantization. It is
inexact; that benchmark does not establish full-video speed or output quality.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from cozy_runtime.internal import attention_fp8
from cozy_runtime.internal.anima_optimization import DIFFUSERS_VERSION
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
    """A typed refusal on the attention axis. Only a PIN can raise one; auto never does."""

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


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="flash-attn3-fp8",
        backend=attention_fp8.BACKEND,
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=128,
        baked="flash-attn3",
        rel_l2_max=6e-2,
        why=(
            "H100 SXM kernel benchmark: 1.292x/1.317x versus BF16 FA3 at sequences "
            "37,710/103,966, including quantization; h3a-013 results3/runtime_check.json. "
            "Inexact; no clip qualification."
        ),
    ),
    Candidate(
        name="flash-attn3",
        backend="_flash_3_hub",
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=256,
        baked="flash-attn3",
        rel_l2_max=2e-2,
        why="H100 SXM attention: 22.84 s versus cuDNN 28.45 s per DiT forward; h3a-012.",
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
        name="flash-attn4",
        backend="flash_4_hub",
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=256,
        max_sm=110,
        baked="flash-attn4",
        jit=True,
        rel_l2_max=2e-2,
        why="Unranked: limited B200 kernel screen in h3a-022, no controlled serving comparison.",
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
            "H100 SXM kernel benchmark: 1.12x versus BF16 FA3; h3a-013. "
            "Inexact and slower than the FP8 FA3 benchmark."
        ),
    ),
)

BY_NAME: dict[str, Candidate] = {c.name: c for c in (*CANDIDATES, *PINNABLE)}


@dataclass(frozen=True, slots=True)
class Ready:
    """One candidate PROVEN present in this environment, with diffusers' member for it."""

    candidate: Candidate
    member: Any
    location: str = ""
    """A baked kernel's variant directory; empty otherwise."""

    @property
    def name(self) -> str:
        return self.candidate.name


def _registry() -> Any:
    """diffusers' attention backend registry at the reviewed version, or a typed absence."""
    try:
        installed = metadata.version("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise AttentionRefusal(
            "attention_kernel_absent", "diffusers is not installed; attention backends are its"
        ) from exc
    if installed != DIFFUSERS_VERSION:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"the reviewed attention backends are diffusers=={DIFFUSERS_VERSION}; this "
            f"environment installs {installed}",
        )
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    return _AttentionBackendRegistry


def _member(backend: str) -> Any:
    if backend == attention_fp8.BACKEND:
        # OURS, not diffusers': registered into its registry on first use (cr-136).
        return attention_fp8.member()
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


def _baked_archs(candidate: Candidate) -> dict[str, list[str]]:
    """Every baked variant of this candidate, and the architectures each one declares."""
    builds = IMAGE_KERNELS_DIR / candidate.baked / "build" if candidate.baked else None
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
        f"{candidate.name} is baked as {archs} and none of those builds carries device code "
        f"for {device.name} (sm{device.sm}). This is a REFUSAL rather than an attempt because "
        "the attempt is not survivable: FlashAttention-3's launcher prints `no kernel image is "
        "available for execution on the device` and exits the process from C++, so neither the "
        "numeric self-check nor auto-selection's own fallback would ever run (h3a-022 measured "
        "it on an sm_120 RTX PRO 6000 with this exact snapshot). Only a worker image baking a "
        "build that carries this architecture changes the answer",
    )


def _load_baked(candidate: Candidate, device: DeviceFacts) -> tuple[Any, str]:
    """Load the image's baked snapshot into diffusers' Hub slot so nothing is fetched."""
    if KERNELS_ABSENT:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} loads through the `kernels` distribution, which {KERNELS_ABSENT}",
        )
    location = IMAGE_KERNELS_DIR / candidate.baked
    builds = location / "build"
    variants = (
        sorted(p for p in builds.iterdir() if (p / "metadata.json").is_file())
        if (builds.is_dir())
        else []
    )
    if not variants:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} is not baked under {builds}; the worker image bakes it there "
            "and this executor has no network to fetch it",
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
    loaded = Path(str(getattr(module, "__file__", ""))).resolve().parent
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
            f"diffusers=={DIFFUSERS_VERSION} does not resolve {candidate.backend!r} through "
            "its Hub-kernel registry, so a baked snapshot cannot be placed in its slot",
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


#: The self-check's shape: (batch, sequence, heads, head dimension). Small enough that the
#: whole proof costs under a millisecond of a prepare that already takes minutes, and large
#: enough to cross a real kernel's tiling — 256 rows is two 128-row blocks, and 128 is the head
#: dimension every entry in both tables serves.
PROBE: tuple[int, int, int, int] = (1, 256, 4, 128)


def _computes(entry: Ready, device: DeviceFacts) -> None:
    """PROVE the kernel computes attention on THIS device, or refuse it by name."""
    if device.kind != "cuda":
        # There is no device to prove anything on, and the only entry a CPU executor admits is
        # torch's own eager dispatch. Silence here is the honest answer, not a skipped gate.
        return
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    batch, sequence, heads, dim = PROBE
    dtype = (
        torch.bfloat16
        if not entry.candidate.dtypes or "bfloat16" in entry.candidate.dtypes
        else torch.float16
    )
    seed = torch.Generator(device="cuda").manual_seed(0x5A6E)
    shape = (batch, sequence, heads, dim)
    query, key, value = (
        (torch.randn(shape, generator=seed, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
        for _ in range(3)
    )
    with torch.no_grad():
        got = dispatch_attention_fn(query, key, value, backend=entry.member)
    if isinstance(got, tuple):
        got = got[0]

    exact = torch.nn.functional.scaled_dot_product_attention(
        *(tensor.transpose(1, 2).float() for tensor in (query, key, value))
    ).transpose(1, 2)
    if tuple(got.shape) != tuple(exact.shape):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} returned {tuple(got.shape)} where attention over {shape} is "
            f"{tuple(exact.shape)}",
        )
    got = got.float()
    if not bool(torch.isfinite(got).all()):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} returned non-finite values on {device.name} (sm{device.sm}); it is "
            "not serving attention and a NaN would pass every ratio test",
        )
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


def _resolve(candidate: Candidate, device: DeviceFacts) -> Ready:
    """One candidate, proven present here AND proven to compute, or a typed refusal."""
    _admit_arch(candidate, _baked_archs(candidate), device)
    registry = _registry()
    member = _member(candidate.backend)
    if member not in registry._backends:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"diffusers=={DIFFUSERS_VERSION} registers no backend {candidate.backend!r}",
        )
    if candidate.baked:
        loaded_member, location = _load_baked(candidate, device)
        ready = Ready(candidate, loaded_member, location)
        _computes(ready, device)
        return ready
    if candidate.distribution:
        try:
            installed = metadata.version(candidate.distribution)
        except metadata.PackageNotFoundError as exc:
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"{candidate.name} needs the {candidate.distribution} distribution, which "
                "this image does not install",
            ) from exc
        try:
            import_module(candidate.distribution.replace("-", "_"))
        except Exception as exc:
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"{candidate.distribution}=={installed} does not import: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
    ready = Ready(candidate, member)
    _computes(ready, device)
    return ready


def _device_admits(candidate: Candidate, device: DeviceFacts) -> str:
    """Empty when this DEVICE is inside the candidate's DECLARED range, else why it is not."""
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
    """The RANKED kernels this device and this image can actually run, fastest first.

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
            f"{unsupported}. This image runs {image_kernels(device)} on {device.name}",
        )
    return [_resolve(candidate, device)]


# ------------------------------------------------------------------------------ the pin


def _sites(roots: Mapping[str, Any]) -> Iterator[tuple[str, Any, Any]]:
    """(component, attention module, its processor) for every diffusers attention site.

    The walk follows public attributes and mapping values three levels deep and stops at torch
    module trees, where `modules()` finds the attention classes diffusers' own
    `set_attention_backend` looks for. A component's label is the name the fill plane gave it.
    """
    try:
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_processor import Attention, MochiAttention
    except ImportError:
        # No diffusers, no diffusers attention site. Nothing to pin, and not a refusal.
        return

    classes = (Attention, MochiAttention, AttentionModuleMixin)
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
            for module in value.modules():
                processor = getattr(module, "processor", None)
                if isinstance(module, classes) and processor is not None:
                    yield label, module, processor
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


def _dtype(module: Any) -> str:
    for parameter in module.parameters():
        return str(parameter.dtype).removeprefix("torch.")
    return ""


def _admits(candidate: Candidate, dtype: str, head_dim: int) -> bool:
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
    #: the ranked kernels this image and device offered, fastest first.
    offered: tuple[str, ...]
    #: set when the caller PINNED one through the execution-path override (cr-125).
    pin: str = ""

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
        return "NONE — this construction holds no diffusers attention site"
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
    if not kernels:
        return None
    pin = "+".join(sorted(pins))
    return {
        "kernels": kernels,
        "hosts": hosts,
        "offered": offered,
        "locations": locations,
        "pin": pin,
        "line": line(kernels, hosts, offered, pin),
    }


def install(
    ready: list[Ready], roots: Mapping[str, Any], pin: str = "", degree: int = 1
) -> Applied:
    """Pin the fastest admitted kernel on every attention site of the constructed components."""
    hosts: dict[str, dict[str, int]] = {}
    locations: dict[str, str] = {}
    for component, module, processor in _sites(roots):
        dtype, head_dim = _dtype(module), _head_dim(module)
        for entry in ready:
            if not _admits(entry.candidate, dtype, head_dim):
                continue
            if degree > 1 and not _context_parallel(entry.member):
                continue
            processor._attention_backend = entry.member
            hosts.setdefault(component, {})
            hosts[component][entry.name] = hosts[component].get(entry.name, 0) + 1
            if entry.location:
                locations[entry.name] = entry.location
            break
        else:
            raise AttentionRefusal(
                "attention_kernel_unsupported",
                f"{ready[0].name if ready else 'no attention kernel'} does not serve "
                f"{component}'s attention: "
                f"{dtype or 'unknown'} activations at head dimension "
                f"{head_dim or 'unknown'}"
                + (
                    f", sharded at sequence-parallel degree {degree} (a kernel that does not "
                    "serve context parallelism cannot take a sharded site)"
                    if degree > 1
                    else ""
                )
                + ". A pinned kernel is never quietly replaced",
            )
    return Applied(hosts, locations, tuple(entry.name for entry in ready), pin)


def select(
    device: DeviceFacts, roots: Mapping[str, Any], pin: str = "", degree: int = 1
) -> Applied:
    """The whole decision, in one call: what this device and image offer, pinned on the sites
    of a placement sharded at `degree` (1 = unsharded)."""
    return install(pinned(pin, device) if pin else available(device), roots, pin, degree)


def image_kernels(device: DeviceFacts) -> str:
    """What this image and device can actually RUN, for a refusal message (cr-125)."""
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
