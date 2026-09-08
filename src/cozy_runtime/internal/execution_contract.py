"""A lane's execution contract: 8-bit attention pinned, proven per attempt, or refused (cr-108).

attention-quantization.md §2-3. A CozyTensors header may carry an inline config named
`execution` that names the device, the weights route and the attention kernel plus its scheme.
This module reads it, refuses typed where this environment cannot honour it, pins the kernel on
every diffusers model of the constructed pipeline, and attests on every attempt that the
contracted entry is what ran: its call count against the pinned attention sites times the host
forwards, the scheme kwargs and activation dtype each call carried, and the CUDA symbol one
profiled call launched.

There is no fallback. A lane that promises a kernel refuses where the kernel is absent or the
device is not the contract's; an attempt whose count or symbol is missing fails typed. The
runtime names no kernel: the contract names the entry, and diffusers' own reviewed wrapper for
that entry is what `dispatch_attention_fn` calls.

A kernel is either an installed distribution (`sageattention`, resolved by `importlib`) or one
the worker image BAKES from a `kernels` Hub snapshot (`flash-attn3`, h3a-012: bf16, -16 %/step
on sm90), resolved by path under `IMAGE_KERNELS_DIR` and never over the network a pod's
executor does not have. The contract's `class` says which numerics class the kernel is against
the reference route; the hub fences publication on it.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import textwrap
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, cast

import msgspec
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import canonical
from cozy_runtime.internal.anima_optimization import DIFFUSERS_VERSION
from cozy_runtime.internal.encoding import DeviceFacts, device_line
from cozy_runtime.internal.resolution import ROUTE_KERNEL

try:
    from kernels import get_local_kernel
except ImportError as _absent_kernels:  # pragma: no cover - exercised by the typed refusal arm
    KERNELS_ABSENT = f"does not import ({_absent_kernels})"
else:
    KERNELS_ABSENT = ""

#: The header config's name. One source of truth, inline, beside the construction configs.
CONFIG_NAME = "execution"

#: The contract's activation dtype spellings, and the torch dtype name each one is.
ACTIVATIONS = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

#: The numerics class the contract claims for its kernel against the reference bf16 SDPA
#: route. `same`: bit-different in reduction order only (FA3 bf16, h3a-012: max abs 1.2e-7,
#: audio bit-identical) — the hub publishes it with no gate. `quantized`: 8-bit attention —
#: the hub publishes it only past a PASS assessment. The hub owns the allowlist of which
#: kernel may claim which class; the runtime reads the vocabulary and applies the kernel.
CLASSES = frozenset({"same", "quantized"})


@dataclass(frozen=True, slots=True)
class Baked:
    """A kernel the worker image bakes: diffusers' backend for it, and the local name its
    reviewed wrapper binds the kernel function to (`func = _HUB_KERNELS_REGISTRY[...]`)."""

    backend: str
    call: str


#: Where the worker image bakes `kernels` Hub snapshots (tensorhub's cu130 Dockerfile,
#: th-182): `<distribution>/build/<variant>/` holds one built variant and its metadata, the
#: same shape as the fused-glue artifact under `/opt/cozy/fusion`. A contract for a baked
#: kernel pins the Hub revision and that variant; nothing here is fetched.
IMAGE_KERNELS_DIR = Path("/opt/cozy/kernels")
BAKED = {"flash-attn3": Baked(backend="_flash_3_hub", call="func")}

_DEVICE = re.compile(r"^sm\d{2,3}$")
_ENTRY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")

#: The REVIEWED diffusers wrappers at `DIFFUSERS_VERSION`, by backend name: the source digest of
#: the function `dispatch_attention_fn` hands (query, key, value, is_causal, scale) to the
#: contracted entry through. The runtime never spells a kernel's calling convention; diffusers
#: does, once per backend, and a contract is honoured only through a wrapper whose source was read
#: (the `anima_optimization._SOURCE_DIGESTS` pattern). A backend this table does not name refuses.
_WRAPPER_DIGESTS = {
    "_sage_qk_int8_pv_fp8_cuda_sm90": (
        "5ffc8c98fda849a0f65cf799c4a0f17aa709a92d3ee71dfc5e83ace3845a3964"
    ),
    "_sage_qk_int8_pv_fp8_cuda": (
        "2d1a5c12b6244b2472663f143ed17b50aabd6cdf8f4507b8fc4a5b0d3a51701c"
    ),
    "_sage_qk_int8_pv_fp16_cuda": (
        "20318297bce785026031d4b58622590a36b5497e18e486acec1f4a37aa0cc103"
    ),
    "_sage_qk_int8_pv_fp16_triton": (
        "d2c352e8f12f6cf2d20737e76a7f358507b204db5a61a13f9d8e466f794c9e3c"
    ),
    "sage": "712c6c1c6722a7a16cc22a23a5d32080caf7a1638b09476b576cfeb3b75486a1",
    "_flash_3_hub": "a7a41608ec7418588c8dc9b45c9ce8649d76f2272bfab2982d0d9f97e77dcd14",
}


class ExecutionContractRefusal(RuntimeError):
    """A typed refusal on the contract axis: invalid, unsupported, kernel_absent, unobserved."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _invalid(detail: str) -> ExecutionContractRefusal:
    return ExecutionContractRefusal("execution_contract_invalid", detail)


def _absent(detail: str) -> ExecutionContractRefusal:
    return ExecutionContractRefusal("execution_contract_kernel_absent", detail)


# ------------------------------------------------------------------------- the contract


class Weights(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    route: str


class Attention(msgspec.Struct, frozen=True, forbid_unknown_fields=True, omit_defaults=True):
    distribution: str
    version: str
    entry: str
    kwargs: dict[str, bool | int | float | str | None]
    scheme: str
    #: A baked kernel's Hub revision and build variant; empty for an installed distribution.
    revision: str = ""
    variant: str = ""


class ExecutionContract(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The closed schema of `header["configs"]["execution"]`."""

    device: str
    activations: str
    class_: str = msgspec.field(name="class")
    weights: Weights
    attention: Attention

    def document(self) -> dict[str, Any]:
        return dict(msgspec.to_builtins(self))

    def digest(self) -> str:
        encoded = canonical.write(self.document())
        return (
            "sha256:" + hashlib.sha256(b"cozy.runtime.execution-contract\0" + encoded).hexdigest()
        )


def read(configs: object) -> ExecutionContract | None:
    """`header["configs"]` -> the lane's contract, or None when the header carries none."""
    if not isinstance(configs, Mapping) or CONFIG_NAME not in configs:
        return None
    raw = configs[CONFIG_NAME]
    if not isinstance(raw, bytes):
        raise _invalid(f"the {CONFIG_NAME!r} config is not inline bytes")
    try:
        contract = msgspec.json.decode(raw, type=ExecutionContract)
    except msgspec.ValidationError as exc:
        raise _invalid(f"{CONFIG_NAME}: {exc}") from exc
    except msgspec.DecodeError as exc:
        raise _invalid(f"{CONFIG_NAME}: not JSON: {exc}") from exc
    try:
        if canonical.write(json.loads(raw)) != raw:
            raise _invalid(f"{CONFIG_NAME}: not canonical JSON")
    except canonical.CanonicalError as exc:
        raise _invalid(f"{CONFIG_NAME}: not canonical JSON: {exc}") from exc
    if not _DEVICE.match(contract.device):
        raise _invalid(f"{CONFIG_NAME}.device: {contract.device!r} is not an `sm<NN>` class")
    if contract.activations not in ACTIVATIONS:
        raise _invalid(
            f"{CONFIG_NAME}.activations: {contract.activations!r} is not one of "
            f"{sorted(ACTIVATIONS)}"
        )
    if contract.class_ not in CLASSES:
        raise _invalid(f"{CONFIG_NAME}.class: {contract.class_!r} is not one of {sorted(CLASSES)}")
    if contract.weights.route not in ROUTE_KERNEL:
        raise _invalid(
            f"{CONFIG_NAME}.weights.route: {contract.weights.route!r} is not one of "
            f"{sorted(ROUTE_KERNEL)}"
        )
    attention = contract.attention
    for name, value in (
        ("distribution", attention.distribution),
        ("version", attention.version),
        ("scheme", attention.scheme),
    ):
        if not value:
            raise _invalid(f"{CONFIG_NAME}.attention.{name}: empty")
    if not _ENTRY.match(attention.entry):
        raise _invalid(f"{CONFIG_NAME}.attention.entry: {attention.entry!r} is not a dotted name")
    try:
        Version(attention.version)
    except InvalidVersion as exc:
        raise _invalid(f"{CONFIG_NAME}.attention.version: {exc}") from exc
    if attention.distribution in BAKED:
        if not _REVISION.match(attention.revision):
            raise _invalid(
                f"{CONFIG_NAME}.attention.revision: {attention.revision!r} is not the 40-hex "
                f"Hub revision {attention.distribution} is baked from"
            )
        if not attention.variant:
            raise _invalid(
                f"{CONFIG_NAME}.attention.variant: empty; name the baked build variant of "
                f"{attention.distribution}"
            )
    elif attention.revision or attention.variant:
        raise _invalid(
            f"{CONFIG_NAME}.attention.revision/variant name a baked kernel; "
            f"{attention.distribution} is an installed distribution"
        )
    return contract


# --------------------------------------------------------------------------- the checks


def check_device(contract: ExecutionContract, device: DeviceFacts) -> None:
    """The contract's device class must be THIS card's. Before any allocation."""
    if device.kind != "cuda" or f"sm{device.sm}" != contract.device:
        raise ExecutionContractRefusal(
            "execution_contract_unsupported",
            f"the lane's execution contract names {contract.device}; this executor's device "
            f"is {device_line(device)} ({device.predicate}). The ladder's next rung serves it",
        )


@dataclass(frozen=True, slots=True)
class Kernel:
    """The contracted entry, resolved in this environment, and diffusers' wrapper for it."""

    entry: Callable[..., Any]
    entry_name: str
    installed_version: str
    backend: str
    wrapper: Callable[..., Any]
    wrapper_digest: str
    live: dict[str, str]
    """Contract kwargs the wrapper feeds from its own parameter (`is_causal=is_causal`): the
    value is decided per call by dispatch, so it is verified per call, by that parameter."""
    location: str = ""
    """A baked kernel's variant directory; empty for an installed distribution."""


def _same(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


def _entry_call(wrapper: Callable[..., Any], call_name: str) -> ast.Call:
    tree = ast.parse(textwrap.dedent(inspect.getsource(wrapper)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == call_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == call_name)
        )
    ]
    if len(calls) != 1:
        raise _absent(
            f"diffusers' {wrapper.__name__} calls {call_name} {len(calls)} times; the "
            "attestation counts one kernel call per wrapper call"
        )
    return calls[0]


def _coverage(
    contract: ExecutionContract,
    kernel_entry: Callable[..., Any],
    call_name: str,
    wrapper: Callable[..., Any],
) -> dict[str, str]:
    """Every contract kwarg is what the entry would RUN under diffusers' wrapper.

    A wrapper passes some kwargs as constants (`tensor_layout="NHD"`), some from its own
    parameters (`is_causal=is_causal`), and leaves the rest to the entry's signature defaults
    (`qk_quant_gran`). The first and last are checked here; the middle is returned for the
    per-call check, because its value is dispatch's to decide. `call_name` is what the wrapper
    calls the entry: its own name, or the local a Hub kernel is bound to.
    """
    signature = inspect.signature(kernel_entry)
    wrapper_signature = inspect.signature(wrapper)
    call = _entry_call(wrapper, call_name)
    positional = list(signature.parameters)[: len(call.args)]
    constants: dict[str, object] = {}
    variables: dict[str, str] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        if isinstance(keyword.value, ast.Constant):
            constants[keyword.arg] = keyword.value.value
        else:
            variables[keyword.arg] = keyword.value.id if isinstance(keyword.value, ast.Name) else ""
    entry = f"{contract.attention.distribution}.{contract.attention.entry}"
    live: dict[str, str] = {}
    for name, expected in contract.attention.kwargs.items():
        parameter = signature.parameters.get(name)
        if (
            parameter is None
            or parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            or name in positional
        ):
            raise _absent(f"{entry} takes no keyword {name!r}, which the contract's scheme names")
        if name in constants:
            actual, where = constants[name], f"diffusers' {wrapper.__name__} passes"
        elif name in variables:
            source = variables[name]
            if source not in wrapper_signature.parameters:
                raise _absent(
                    f"diffusers' {wrapper.__name__} computes {name!r} for {entry}; the "
                    "contract's value cannot be verified against an expression"
                )
            live[name] = source
            continue
        elif parameter.default is inspect.Parameter.empty:
            raise _absent(f"{entry} requires {name!r} and diffusers' {wrapper.__name__} omits it")
        else:
            actual, where = parameter.default, f"{entry}'s signature default is"
        if not _same(actual, expected):
            raise _absent(
                f"the contract's {name}={expected!r} is not what would run: {where} {actual!r}"
            )
    return live


def _lookup(module: Any, dotted: str, distribution: str) -> tuple[Any, Any]:
    """`(owner, entry)` for a dotted name on a loaded module, or `kernel_absent`."""
    owner: Any = module
    entry: Any = module
    for part in dotted.split("."):
        owner = entry
        entry = getattr(entry, part, None)
        if entry is None:
            raise _absent(f"{distribution}.{dotted} does not exist")
    if not callable(entry):
        raise _absent(f"{distribution}.{dotted} is not callable")
    return owner, entry


def _diffusers_backends() -> dict[Any, Callable[..., Any]]:
    """diffusers' attention registry at the reviewed version, or `kernel_absent`."""
    try:
        diffusers_version = metadata.version("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise _absent(
            "diffusers is not installed; attention backends are pinned through it"
        ) from exc
    if diffusers_version != DIFFUSERS_VERSION:
        raise _absent(
            f"the reviewed attention wrappers are diffusers=={DIFFUSERS_VERSION}; this "
            f"environment installs {diffusers_version}"
        )
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    return cast(dict[Any, Callable[..., Any]], _AttentionBackendRegistry._backends)


def _reviewed(backend: str, wrapper: Callable[..., Any]) -> str:
    """The wrapper's source digest, which must be the one reviewed for its backend."""
    digest = hashlib.sha256(inspect.getsource(wrapper).encode()).hexdigest()
    expected = _WRAPPER_DIGESTS.get(backend)
    if expected is None:
        raise _absent(f"diffusers backend {backend!r} has no reviewed wrapper in this runtime")
    if digest != expected:
        raise _absent(
            f"diffusers' {wrapper.__name__} has source sha256:{digest}, reviewed "
            f"sha256:{expected}; refusing the contract rather than guessing across patched source"
        )
    return digest


def resolve_kernel(contract: ExecutionContract) -> Kernel:
    """The contracted entry in THIS environment, or `execution_contract_kernel_absent`."""
    baked = BAKED.get(contract.attention.distribution)
    return _resolve_baked(contract, baked) if baked else _resolve_installed(contract)


def _resolve_installed(contract: ExecutionContract) -> Kernel:
    attention = contract.attention
    try:
        installed = metadata.version(attention.distribution)
    except metadata.PackageNotFoundError as exc:
        raise _absent(
            f"{attention.distribution}=={attention.version} is not installed in this "
            "environment; the lane's execution contract needs it"
        ) from exc
    if Version(installed) != Version(attention.version):
        raise _absent(
            f"the contract needs {attention.distribution}=={attention.version}; this "
            f"environment installs {installed}"
        )
    try:
        module = import_module(attention.distribution.replace("-", "_"))
    except Exception as exc:  # an import error names the missing extension, not just a module
        raise _absent(
            f"{attention.distribution}=={installed} does not import: {type(exc).__name__}: {exc}"
        ) from exc
    owner, entry = _lookup(module, attention.entry, attention.distribution)
    entry_name = attention.entry.rsplit(".", 1)[-1]
    found: list[tuple[str, Callable[..., Any]]] = []
    for member, wrapper in _diffusers_backends().items():
        names = wrapper.__code__.co_names
        if entry_name not in names:
            continue
        bound = wrapper.__globals__
        if any(bound.get(n) is entry or bound.get(n) is owner for n in names):
            found.append((str(member.value), wrapper))
    if len(found) != 1:
        raise _absent(
            f"diffusers=={DIFFUSERS_VERSION} binds "
            + (f"{len(found)} attention backends to" if found else "no attention backend to")
            + f" {attention.distribution}.{attention.entry}"
            + (f" ({', '.join(name for name, _ in found)})" if found else "")
        )
    backend, wrapper = found[0]
    digest = _reviewed(backend, wrapper)
    live = _coverage(contract, entry, entry_name, wrapper)
    return Kernel(entry, entry_name, installed, backend, wrapper, digest, live)


def _resolve_baked(contract: ExecutionContract, baked: Baked) -> Kernel:
    """The baked snapshot under `IMAGE_KERNELS_DIR`, loaded by path through `kernels`.

    diffusers resolves a Hub kernel over the network on first use (`get_kernel(repo,
    version=1)` reads tags); the loaded module is placed in that registry slot instead, so
    the pin never reaches for a network the executor does not have.
    """
    attention = contract.attention
    name = f"{attention.distribution}@{attention.revision[:12]}/{attention.variant}"
    if KERNELS_ABSENT:
        raise _absent(f"{name} loads through the `kernels` distribution, which {KERNELS_ABSENT}")
    location = IMAGE_KERNELS_DIR / attention.distribution
    variant_dir = location / "build" / attention.variant
    if not (variant_dir / "metadata.json").is_file():
        raise _absent(
            f"{name} is not baked at {variant_dir}; the worker image bakes it there and this "
            "executor has no network to fetch it"
        )
    built = json.loads((variant_dir / "metadata.json").read_text())
    if str(built.get("version")) != attention.version:
        raise _absent(
            f"{variant_dir} is {attention.distribution} version {built.get('version')!r}; the "
            f"contract needs {attention.version}"
        )
    try:
        module = get_local_kernel(location)
    except Exception as exc:
        raise _absent(f"{name} does not load from {location}: {type(exc).__name__}: {exc}") from exc
    loaded = Path(str(getattr(module, "__file__", ""))).resolve().parent
    if loaded != variant_dir.resolve():
        raise _absent(
            f"this torch resolves {attention.distribution} to build variant {loaded.name!r} "
            f"where the contract names {attention.variant!r}"
        )
    _, entry = _lookup(module, attention.entry, attention.distribution)
    from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY, AttentionBackendName

    member = AttentionBackendName(baked.backend)
    wrapper = _diffusers_backends()[member]
    digest = _reviewed(baked.backend, wrapper)
    live = _coverage(contract, entry, baked.call, wrapper)
    config = _HUB_KERNELS_REGISTRY[member]
    config.kernel_fn = entry
    for attr, slot in (
        ("wrapped_forward_attr", "wrapped_forward_fn"),
        ("wrapped_backward_attr", "wrapped_backward_fn"),
    ):
        dotted = getattr(config, attr)
        if dotted:
            setattr(config, slot, _lookup(module, dotted, attention.distribution)[1])
    return Kernel(
        entry,
        attention.entry.rsplit(".", 1)[-1],
        attention.version,
        baked.backend,
        wrapper,
        digest,
        live,
        str(variant_dir),
    )


# ------------------------------------------------------------------------ application


@dataclass(slots=True)
class AppliedContract:
    """One contract pinned on one constructed model, with its live counters."""

    contract: ExecutionContract
    kernel: Kernel
    hosts: dict[str, int]
    """host label -> attention sites pinned on it (calls one forward makes)."""
    calls: int = 0
    forwards: dict[str, int] = field(default_factory=dict)
    mismatch: str = ""
    #: CUDA symbol -> self ms, summed over the profiled calls: ONE forward of every pinned
    #: site (`profile_budget` calls), so the packed sequence's O(S^2) kernel is what leads and
    #: not whichever small site a forward happens to enter first.
    kernels: dict[str, float] = field(default_factory=dict)
    profiled: int = 0
    profile_budget: int = 0
    profile_error: str = ""
    opened_calls: int = 0
    opened_forwards: dict[str, int] = field(default_factory=dict)

    def profile_document(self) -> dict[str, Any]:
        ranked = sorted(self.kernels.items(), key=lambda row: -row[1])
        return {
            "kernel_symbol": ranked[0][0] if ranked else "",
            "kernel_self_ms": round(ranked[0][1], 3) if ranked else 0.0,
            "kernels": [{"symbol": symbol, "self_ms": round(ms, 3)} for symbol, ms in ranked[:6]],
            "profiled_calls": self.profiled,
            "profile_error": self.profile_error,
        }

    def document(self) -> dict[str, Any]:
        return {
            "digest": self.contract.digest(),
            "contract": self.contract.document(),
            "backend": self.kernel.backend,
            "installed_version": self.kernel.installed_version,
            "location": self.kernel.location,
            "wrapper_digest": "sha256:" + self.kernel.wrapper_digest,
            "hosts": dict(self.hosts),
            "sites": sum(self.hosts.values()),
            **self.profile_document(),
        }


def _hosts(model: object) -> list[tuple[str, Any]]:
    """Every OUTERMOST diffusers model reachable from the constructed model, labelled.

    The walk follows public attributes and mapping values three levels deep (`model.pipe
    .components["fl2va_dit"]`) and stops at torch module trees, where `modules()` finds the
    outermost object that answers `set_attention_backend`; a nested one is covered by its
    parent's pin and is not counted twice.
    """
    found: list[tuple[str, Any]] = []
    seen: set[int] = set()
    queue: deque[tuple[str, Any, int]] = deque([("model", model, 0)])
    while queue:
        label, value, depth = queue.popleft()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if callable(getattr(value, "modules", None)) and callable(
            getattr(value, "parameters", None)
        ):
            covered: set[int] = set()
            for module in value.modules():
                if id(module) in covered:
                    continue
                if callable(getattr(module, "set_attention_backend", None)):
                    found.append(
                        (label if module is value else f"{label}.{type(module).__name__}", module)
                    )
                    covered.update(id(m) for m in module.modules())
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
    return found


def _profiled(
    wrapper: Callable[..., Any], kwargs: dict[str, Any]
) -> tuple[Any, list[tuple[str, float]], str]:
    """ONE call under torch's profiler: (result, CUDA symbol -> self ms, error)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    query = kwargs.get("query")
    if getattr(getattr(query, "device", None), "type", "") != "cuda":
        return wrapper(**kwargs), [], "the attention inputs are not on a CUDA device"
    try:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            result = wrapper(**kwargs)
            torch.cuda.synchronize()
    except Exception as exc:
        return wrapper(**kwargs), [], f"{type(exc).__name__}: {exc}"[:300]
    rows = [
        (str(event.key), float(event.self_device_time_total) / 1000.0)
        for event in prof.key_averages()
        if getattr(event.device_type, "name", "") == "CUDA"
    ]
    return result, rows, "" if rows else "the profiler saw no CUDA kernel under the call"


def apply(contract: ExecutionContract, kernel: Kernel, model: object) -> AppliedContract:
    """Pin the backend on every diffusers model of the constructed pipeline and arm attestation.

    `set_attention_backend` marks every attention processor of a host; the registry entry for
    that backend is then replaced by a counting, verifying proxy around diffusers' own wrapper,
    and a forward pre-hook counts each host's forwards. Calls == sites x forwards is the
    per-attempt fact; the first call is profiled for the CUDA symbol.
    """
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

    member = AttentionBackendName(kernel.backend)
    hosts: dict[str, int] = {}
    pinned: list[tuple[str, Any]] = []
    for label, host in _hosts(model):
        try:
            host.set_attention_backend(kernel.backend)
        except Exception as exc:
            raise _absent(
                f"diffusers refused backend {kernel.backend!r} on {label}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        sites = sum(
            1
            for module in host.modules()
            if getattr(getattr(module, "processor", None), "_attention_backend", None) == member
        )
        if sites:
            hosts[label] = sites
            pinned.append((label, host))
    if not hosts:
        raise ExecutionContractRefusal(
            "execution_contract_unobserved",
            "no attention site reachable from the constructed model accepts backend "
            f"{kernel.backend!r}; the contract cannot be applied, so it cannot be proven",
        )
    applied = AppliedContract(
        contract,
        kernel,
        hosts,
        forwards=dict.fromkeys(hosts, 0),
        profile_budget=sum(hosts.values()),
    )
    activations = ACTIVATIONS[contract.activations]
    entry = f"{contract.attention.distribution}.{contract.attention.entry}"
    wrapper_defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(kernel.wrapper).parameters.items()
    }

    def attested(**kwargs: Any) -> Any:
        applied.calls += 1
        if not applied.mismatch:
            for name, source in kernel.live.items():
                value = kwargs.get(source, wrapper_defaults.get(source))
                expected = contract.attention.kwargs[name]
                if not _same(value, expected):
                    applied.mismatch = (
                        f"call {applied.calls}: {name}={value!r} reached {entry} where the "
                        f"contract says {expected!r}"
                    )
                    break
            dtype = str(getattr(kwargs.get("query"), "dtype", "")).removeprefix("torch.")
            if not applied.mismatch and dtype != activations:
                applied.mismatch = (
                    f"call {applied.calls}: attention ran on {dtype or 'unknown'} activations "
                    f"where the contract says {contract.activations}"
                )
        if applied.profiled < applied.profile_budget:
            result, rows, error = _profiled(kernel.wrapper, kwargs)
            applied.profiled += 1
            for symbol, ms in rows:
                applied.kernels[symbol] = applied.kernels.get(symbol, 0.0) + ms
            applied.profile_error = applied.profile_error or error
            return result
        return kernel.wrapper(**kwargs)

    _AttentionBackendRegistry._backends[member] = attested
    for label, host in pinned:

        def forwarded(_module: Any, _args: Any, label: str = label) -> None:
            applied.forwards[label] += 1

        host.register_forward_pre_hook(forwarded)
    return applied


class Attestation:
    """The per-attempt fact over every applied contract of one executor."""

    def __init__(self, applied: list[AppliedContract]) -> None:
        self.applied = applied

    def observe(self, run: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
        """ONE bracketed step — a warm dry step (cr-110), or any call that should attest."""
        self.open()
        result = run()
        return result, self.close()

    def open(self) -> None:
        for row in self.applied:
            row.opened_calls = row.calls
            row.opened_forwards = dict(row.forwards)
            row.mismatch = ""

    def close(self) -> dict[str, Any]:
        calls = expected = 0
        forwards: dict[str, int] = {}
        sites: dict[str, int] = {}
        refusals: list[str] = []
        profile: dict[str, Any] = {}
        for row in self.applied:
            made = row.calls - row.opened_calls
            ran = {
                label: row.forwards[label] - row.opened_forwards.get(label, 0)
                for label in row.hosts
            }
            predicted = sum(row.hosts[label] * ran[label] for label in row.hosts)
            calls += made
            expected += predicted
            forwards.update(ran)
            sites.update(row.hosts)
            entry = f"{row.contract.attention.distribution}.{row.contract.attention.entry}"
            if made != predicted:
                refusals.append(
                    f"{entry} was called {made} time(s) where the pinned sites predict "
                    f"{predicted} ("
                    + " + ".join(
                        f"{row.hosts[label]} sites x {ran[label]} forwards on {label}"
                        for label in ran
                    )
                    + ")"
                )
            if row.mismatch:
                refusals.append(row.mismatch)
            if made and row.profiled and not row.kernels:
                refusals.append(
                    "no CUDA kernel was observed under the contracted entry: " + row.profile_error
                )
            if not profile and row.profiled:
                profile = row.profile_document()
        first = self.applied[0].contract
        return {
            "digest": self.applied[0].contract.digest(),
            "entry": f"{first.attention.distribution}.{first.attention.entry}",
            "backend": self.applied[0].kernel.backend,
            "scheme": first.attention.scheme,
            "calls": calls,
            "expected_calls": expected,
            "forwards": forwards,
            "sites": sites,
            "kernel_symbol": str(profile.get("kernel_symbol", "")),
            "kernel_self_ms": float(profile.get("kernel_self_ms", 0.0)),
            "kernels": list(profile.get("kernels", [])),
            "refusal": ("execution_contract_unobserved: " + "; ".join(refusals))
            if refusals
            else "",
        }
