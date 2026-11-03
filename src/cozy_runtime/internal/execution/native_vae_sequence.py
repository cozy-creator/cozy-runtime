"""Immutable H3 window geometry, invocation order and output assembly.

Native policy chooses one declared geometry. The coordinator supplies real queried
plans, physical buffer owners, fresh stage admissions and checked completion. No
Python capacity ranking, nested split, retry or eager decoder fallback exists.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

GEOMETRY = (7, 16, 16)
TOKENS, WIDTH, BLOCKS = 1797, 2048, 36
PIXEL_SHAPE = (3, 28, 256, 256)


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def window_sizes(batch: int) -> tuple[int, ...]:
    """Legal balanced geometry only; no room, allocator or byte policy is consulted."""
    _positive(batch, "batch")
    sizes = [batch]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + 1) // 2)
    return tuple(sizes)


@dataclass(frozen=True)
class Variant:
    batch: int
    outer: int
    inner: int

    def __post_init__(self) -> None:
        for value, name in ((self.batch, "batch"), (self.outer, "outer"), (self.inner, "inner")):
            _positive(value, name)
        if self.outer > self.batch or self.inner > self.outer:
            raise ValueError("window geometry exceeds its declared tile domain")

    @property
    def identity(self) -> str:
        return f"outer:{self.outer}/inner:{self.inner}/tail:{self.batch % self.outer}"

    def windows(self) -> tuple[Window, ...]:
        return tuple(
            Window(start, min(self.outer, self.batch - start), self.inner)
            for start in range(0, self.batch, self.outer)
        )


def variants(batch: int) -> tuple[Variant, ...]:
    """Full B first, then legal smaller windows. Native policy owns the ranking result."""
    return tuple(
        Variant(batch, outer, inner)
        for outer in window_sizes(batch)
        for inner in window_sizes(outer)
    )


def resolve(declarations: tuple[Variant, ...], identity: str) -> Variant:
    """Resolve an exact declared ID; this does not select/admit a geometry."""
    matches = tuple(variant for variant in declarations if variant.identity == identity)
    if len(matches) != 1:
        raise ValueError("native selection must name exactly one declared geometry")
    return matches[0]


@dataclass(frozen=True)
class Window:
    start: int
    count: int
    inner: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("window start must be a nonnegative integer")
        _positive(self.count, "window count")
        _positive(self.inner, "inner window")

    @property
    def stop(self) -> int:
        return self.start + self.count

    @property
    def block_chunk(self) -> int:
        # Ragged geometry is fixed in the declaration, not chosen from memory pressure.
        return min(self.inner, self.count)

    @property
    def block_tail(self) -> int:
        return self.count % self.block_chunk


@dataclass(frozen=True)
class Operation:
    window: Window
    kind: str
    block: int | None
    input_owner: str
    output_owner: str
    phase_role: str

    @property
    def identity(self) -> tuple[int, int, str, int | None]:
        return self.window.start, self.window.count, self.kind, self.block


def operations(variant: Variant) -> tuple[Operation, ...]:
    """One body per logical invocation; full-B geometry is strictly blockwise."""
    result = []
    for window in variant.windows():
        result.append(Operation(window, "post_quant", None, "input", "post_result", "edge"))
        result.append(Operation(window, "prefix", None, "post_result", "hidden16", "edge"))
        current = "hidden16"
        for index in range(BLOCKS):
            target = "hidden32_a" if index % 2 == 0 else "hidden32_b"
            result.append(Operation(window, "block", index, current, target, "supplied_block"))
            current = target
        result.append(Operation(window, "tail", None, current, "pixels", "tail_functional"))
    return tuple(result)


@dataclass(frozen=True)
class Lifetime:
    owner: str
    introduced: str
    last_use: str
    retire: str


def lifetimes() -> tuple[Lifetime, ...]:
    """Per-window local lifetimes plus whole-call input/output and query owners.

    Recreating retired locals for the next window needs fresh native admission of
    the same fixed geometry; it is not a new geometry choice or a replay.
    """
    return (
        Lifetime(
            "input", "caller", "post_quant", "caller-owned; retain through its consuming stream"
        ),
        Lifetime(
            "pixels", "native allocation bootstrap", "output handoff", "last downstream owner"
        ),
        Lifetime(
            "post_result", "window allocation preparation", "prefix", "checked prefix completion"
        ),
        Lifetime(
            "hidden16",
            "window allocation preparation",
            "block:0",
            "retained with block borrowers through checked block:35 completion",
        ),
        Lifetime(
            "rotary_cos_sin",
            "window allocation preparation",
            "block:35",
            "checked block:35 completion",
        ),
        Lifetime(
            "hidden32_a",
            "window allocation preparation",
            "block:35",
            "checked block:35 completion before tail admission",
        ),
        Lifetime("hidden32_b", "window allocation preparation", "tail", "checked tail completion"),
        Lifetime(
            "block_scratch",
            "window allocation preparation",
            "block:35",
            "checked block:35 completion before tail admission",
        ),
        Lifetime(
            "tail_norm_y_mean_rstd",
            "functional LayerNorm inside tail",
            "tail",
            "checked tail completion",
        ),
        Lifetime(
            "source_views",
            "one stage's ordinary source hooks/binding",
            "that stage",
            "existing checked source-use lifetime; never pin all 36",
        ),
        Lifetime(
            "query_bias_index_context_plans",
            "real provider preparation",
            "last borrowing invocation",
            "checked completion and final borrower release",
        ),
    )


def tail_functional_allocations(window: Window) -> tuple[tuple[str, int], ...]:
    """Exact real functional outputs, independently admitted at the tail phase."""
    rows = window.count * TOKENS
    return (("norm_output", rows * WIDTH * 4), ("norm_mean", rows * 4), ("norm_rstd", rows * 4))


def pixel_allocation(batch: int) -> int:
    _positive(batch, "batch")
    return batch * math.prod(PIXEL_SHAPE) * 2


class OutputAssembly:
    """Views into the actual fixed returned owner; no allocation or hidden join copy."""

    def __init__(self, output: Tensor, batch: int) -> None:
        _positive(batch, "batch")
        if (
            type(output) is not Tensor
            or output.layout is not torch.strided
            or output.dtype is not torch.float16
            or tuple(output.shape) != (batch, *PIXEL_SHAPE)
            or not output.is_contiguous()
            or output.is_conj()
            or output.is_neg()
        ):
            raise ValueError("sequence output must be its declared contiguous FP16 owner")
        self.output, self.batch = output, batch

    def view(self, window: Window) -> Tensor:
        if window.stop > self.batch:
            raise ValueError("window exceeds the fixed output owner")
        return self.output.narrow(0, window.start, window.count)

    def validate_result(self, window: Window, result: Tensor) -> None:
        target = self.view(window)
        if (
            type(result) is not Tensor
            or result.device != target.device
            or result.dtype != target.dtype
            or tuple(result.shape) != tuple(target.shape)
            or tuple(result.stride()) != tuple(target.stride())
            or result.is_conj()
            or result.is_neg()
            or result.data_ptr() != target.data_ptr()
        ):
            raise ValueError("tail result is not its exact assigned output view")


@dataclass(frozen=True)
class Ticket:
    operation: Operation
    position: int
    owner: object


class Cursor:
    """Host submission order only; never a CUDA completion or admission certificate."""

    def __init__(self, variant: Variant) -> None:
        self.operations = operations(variant)
        self.position = 0
        self.active: Ticket | None = None
        self.failure: BaseException | None = None
        self._owner = object()

    def claim(self) -> Ticket:
        if (
            self.failure is not None
            or self.active is not None
            or self.position == len(self.operations)
        ):
            raise RuntimeError("sequence cannot repeat, overlap or resume a failed invocation")
        self.active = Ticket(self.operations[self.position], self.position, self._owner)
        return self.active

    def _check(self, ticket: Ticket) -> None:
        if ticket.owner is not self._owner or ticket is not self.active:
            raise RuntimeError("sequence ticket is not the current invocation")

    def submitted(self, ticket: Ticket) -> None:
        self._check(ticket)
        self.position += 1
        self.active = None

    def failed(self, ticket: Ticket, error: BaseException) -> None:
        self._check(ticket)
        self.failure = error
        self.active = None
        error.varena_no_retry = True  # type: ignore[attr-defined]

    @property
    def host_done(self) -> bool:
        return (
            self.failure is None and self.active is None and self.position == len(self.operations)
        )


@dataclass(frozen=True)
class EdgeAllocations:
    """Snapshot of a real prepared provider, not a substitute for its live owners."""

    stage: str
    preparation: tuple[tuple[str, int], ...]
    supplied: tuple[tuple[str, int], ...]
    during_execute: tuple[tuple[str, int], ...]
    workspace_bytes: int

    @property
    def all_execution_allocations(self) -> tuple[tuple[str, int], ...]:
        return self.supplied + self.during_execute


def edge_allocations(stage: str, prepared: Any) -> EdgeAllocations:
    """Require the actual queried edge preparation; unknown workspace is never zero."""
    from cozy_runtime.internal.execution import (
        native_vae_output as tail,
    )
    from cozy_runtime.internal.execution import (
        native_vae_post_quant as post,
    )
    from cozy_runtime.internal.execution import (
        native_vae_prefix as prefix,
    )

    types = {"post_quant": post.Prepared, "prefix": prefix.Prepared, "tail": tail.Prepared}
    if stage not in types or type(prepared) is not types[stage]:
        raise TypeError("sequence allocation metadata requires the actual prepared edge")
    if prepared.pid != os.getpid():
        raise RuntimeError("prepared edge metadata cannot be inherited after fork")
    from cozy_runtime._kernels import _dense

    if type(prepared.plan) is not _dense.Plan or type(prepared.context) is not _dense.Context:
        raise TypeError("sequence metadata requires a real native query and Context")
    info = prepared.plan.info()
    if info != prepared.info or info["bias_address"] != prepared.bias.data_ptr():
        raise ValueError("prepared edge query or bias owner changed")
    workspace = info["workspace_bytes"]
    description = prepared.description
    if stage == "tail":
        supplied = description.preallocations(workspace)
        functional = description.functional_allocations
    else:
        supplied, functional = description.allocations(workspace), ()
    preparation = (("bias", prepared.bias.numel() * prepared.bias.element_size()),)
    return EdgeAllocations(stage, preparation, supplied, functional, workspace)


# A failed completion keeps the actual borrowers alive until executor teardown.
_UNPROVEN: list[Any] = []


def _bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * dtype.itemsize


def _block_id(count: int, inner: int) -> str:
    return f"batch:{inner}/tail:{count % inner}"


@dataclass
class _WindowPlans:
    post: Any
    prefix: Any
    tail: Any
    blocks: tuple[Any, ...]

    def specs(self, inner: int) -> tuple[tuple[str, tuple[int, ...], Any], ...]:
        specs = []
        for stage, prepared in (("post", self.post), ("prefix", self.prefix), ("tail", self.tail)):
            specs.extend(
                (f"{stage}.{name}", shape, dtype)
                for name, shape, dtype in prepared.description.buffer_specs(
                    prepared.info["workspace_bytes"]
                )
                if not (stage == "tail" and name == "result")
            )
        count = self.prefix.description.batch
        specs.extend(
            (name, (count, TOKENS, WIDTH), torch.float32) for name in ("hidden32_a", "hidden32_b")
        )
        # One shared physical bank per scratch role, sized from every actual query.
        # The shapes describe actual tensors, not opaque reserve or a policy estimate.
        scratch: dict[str, tuple[int, Any]] = {}
        for prepared in self.blocks:
            for name, shape, dtype in prepared.buffer_specs(_block_id(count, inner)):
                if name == "output":
                    continue
                prior = scratch.get(name)
                if prior is not None and prior[1] != dtype:
                    raise ValueError("VAE scratch role changed dtype between blocks")
                scratch[name] = (max(math.prod(shape), 0 if prior is None else prior[0]), dtype)
        specs.extend((f"block.{name}", (count,), dtype) for name, (count, dtype) in scratch.items())
        return tuple(specs)


def _key(role: str, declaration: object) -> str:
    import hashlib

    from cozy_runtime.internal.execution import phase

    semantic = phase.configuration(declaration)
    if semantic is None:
        raise ValueError("VAE sequence requires a complete immutable phase identity")
    return f"runtime.vae_sequence.{role}/1:" + hashlib.sha256(semantic.encode()).hexdigest()


def _admitted(request: Any, stream: Any, key: str, choices: tuple[Any, ...], run: Any) -> Any:
    import tensorfs.varena.torch as vt

    from cozy_runtime.internal.execution import phase

    device = request.memory.device.index or 0
    request.memory.remember_working_phase(key, (request,))
    selection = vt.split_select(
        device, key, choices, preferred=choices[0][0], stream=stream.cuda_stream
    )
    identity = selection["choice_id"]
    phase.choice(("native-split", key, identity))
    token = phase.begin_split(device, key, identity, selection, stream=stream.cuda_stream)
    record = {
        "site": key,
        "selection": selection,
        "choice_id": identity,
        "additional_allocations": next(a for name, a in choices if name == identity),
    }
    request.working_splits[key] = record
    submitted = False
    try:
        result = run(identity)
        submitted = True
        return result
    finally:
        record["submission"] = vt.split_end(device, token, submitted=submitted)


def _release(custody: list[Any], owner: Any) -> None:
    for index, held in enumerate(custody):
        if held is owner:
            del custody[index]
            return
    raise RuntimeError("VAE custody owner was not retained")


def _fresh(prepared: Any) -> Any:
    # Each scheduled invocation receives its own one-shot claim and lock. Query
    # resources are immutable borrowers; an accepted invocation is never reset.
    return type(prepared)(
        prepared.description,
        prepared.context,
        prepared.stream,
        prepared.policy,
        prepared.bias,
        prepared.plan,
        prepared.info,
        prepared.pid,
    )


def _edge(
    request: Any,
    stream: Any,
    stage: str,
    template: Any,
    value: Tensor,
    buffers: Any,
    custody: list[Any],
) -> Any:
    prepared = _fresh(template)
    bound = prepared.bind(value, buffers)
    custody.append(bound)
    allocations = template.description.functional_allocations if stage == "tail" else ()
    declaration = (
        stage,
        template.description.input_metadata,
        template.policy,
        tuple(sorted(template.info["algorithm"].items())),
        template.info["workspace_bytes"],
        template.info["library_version"],
        tuple(
            (name, shape, dtype)
            for name, shape, dtype in template.description.buffer_specs(
                template.info["workspace_bytes"]
            )
        ),
    )
    key = _key(stage, declaration)

    def execute(identity: str) -> Any:
        result = bound.execute()
        bound.wait()  # checked S completion, including functional tail outputs
        return result

    result = _admitted(
        request, stream, key, (("fixed", tuple(size for _, size in allocations)),), execute
    )
    _release(custody, bound)
    return result


def _prepare_windows(
    vae: Any, value: Tensor, request: Any, stream: Any, declarations: tuple[Variant, ...]
) -> dict[int, _WindowPlans]:
    from cozy_runtime._kernels import _dense
    from cozy_runtime.internal import dense_linear
    from cozy_runtime.internal.execution import native_vae as block
    from cozy_runtime.internal.execution import native_vae_output as tail
    from cozy_runtime.internal.execution import native_vae_post_quant as post
    from cozy_runtime.internal.execution import native_vae_prefix as prefix
    from cozy_runtime.internal.execution.native_vae_metadata import TensorSpec

    context = getattr(request.memory, "_dense_context", None)
    if context is None:
        context = dense_linear.checked(_dense.Context, value.device.index or 0, stream.cuda_stream)
        request.memory._dense_context = context
    counts: dict[int, set[int]] = {}
    for variant in declarations:
        for window in variant.windows():
            counts.setdefault(window.count, set()).add(window.block_chunk)
    result = {}
    for count, chunks in counts.items():
        inputs = TensorSpec.contiguous((count, 24, *GEOMETRY), value.dtype, value.device)
        half = TensorSpec.contiguous(inputs.shape, torch.float16, value.device)
        hidden = TensorSpec.contiguous((count, TOKENS, WIDTH), torch.float32, value.device)
        descriptions = (
            post.describe(vae.post_quant_conv, inputs),
            prefix.describe(vae.decoder, half, geometry=GEOMETRY),
            tail.describe(vae.decoder.norm_out, vae.decoder.proj_out, hidden, geometry=GEOMETRY),
        )
        if any(d is None for d in descriptions):
            raise ValueError("H3 VAE edge metadata is outside the complete native contract")
        preparations = tuple(
            module.prepare(description, context=context, stream=stream)
            for module, description in zip((post, prefix, tail), descriptions, strict=True)
        )
        if any(p is None for p in preparations):
            raise ValueError("H3 VAE edge provider declined its actual queried geometry")
        blocks = tuple(
            block.prepare_block(
                block.describe_block(
                    module,
                    hidden.shape,
                    torch.float16 if index == 0 else torch.float32,
                    value.device,
                ),
                chunks=tuple(sorted(chunks, reverse=True)),
            )
            for index, module in enumerate(vae.decoder.transformer_blocks)
        )
        result[count] = _WindowPlans(*preparations, blocks)
    return result


def _execute_windows(
    vae: Any,
    value: Tensor,
    request: Any,
    stream: Any,
    plans: dict[int, _WindowPlans],
    declarations: tuple[Variant, ...],
    custody: list[Any],
) -> Tensor:
    from cozy_runtime.internal.execution import native_vae as block
    from cozy_runtime.internal.execution import native_vae_output as tail
    from cozy_runtime.internal.execution import native_vae_post_quant as post
    from cozy_runtime.internal.execution import native_vae_prefix as prefix

    all_specs = {v.identity: plans[v.outer].specs(v.inner) for v in declarations}
    preparation = tuple(
        (
            count,
            tuple(
                (stage, (("bias", prepared.bias.numel() * prepared.bias.element_size()),))
                for stage, prepared in (("post", p.post), ("prefix", p.prefix), ("tail", p.tail))
            ),
            tuple(
                (
                    index,
                    (
                        ("bias", b.bias.numel() * b.bias.element_size()),
                        ("index", b.index.numel() * b.index.element_size()),
                    ),
                )
                for index, b in enumerate(p.blocks)
            ),
        )
        for count, p in plans.items()
    )
    choices = tuple(
        (
            v.identity,
            (
                pixel_allocation(v.batch),
                *(_bytes(shape, dtype) for _, shape, dtype in all_specs[v.identity]),
            ),
        )
        for v in declarations
    )
    key = _key(
        "initial_allocation",
        (
            tuple((v.identity, all_specs[v.identity]) for v in declarations),
            tuple(value.shape),
            value.dtype,
        ),
    )

    def allocate(identity: str) -> tuple[Variant, OutputAssembly, dict[str, Tensor]]:
        selected = resolve(declarations, identity)
        output = torch.empty(
            (selected.batch, *PIXEL_SHAPE), dtype=torch.float16, device=value.device
        )
        owners = {
            name: torch.empty(shape, dtype=dtype, device=value.device)
            for name, shape, dtype in all_specs[identity]
        }
        custody.extend((output, owners))
        return selected, OutputAssembly(output, selected.batch), owners

    selected, assembly, owners = _admitted(request, stream, key, choices, allocate)
    cursor = Cursor(selected)
    request.working_splits[key]["sequence"] = {
        "geometry": selected.identity,
        "initial_selection_is_future_reservation": False,
        "later_foreign_pressure_can_refuse": True,
        "preference": "full batch first; native phase-preservation ranking remains authoritative",
        "projection": "declared fused-bias Lt alternatives",
        "completed_windows": 0,
        "already_live_preparation_allocations": preparation,
        "queried_contexts_and_plans": "actual retained prepared dense owners; no guessed workspace",
    }
    record = request.working_splits[key]["sequence"]
    for wi, window in enumerate(selected.windows()):
        p = plans[window.count]
        if wi:
            specs = p.specs(window.block_chunk)

            def allocate_window(identity: str, specs: Any = specs) -> dict[str, Tensor]:
                physical = {
                    name: torch.empty(shape, dtype=dtype, device=value.device)
                    for name, shape, dtype in specs
                }
                custody.append(physical)
                return physical

            owners = _admitted(
                request,
                stream,
                _key("window_allocation", (selected.identity, window, specs)),
                ((selected.identity, tuple(_bytes(shape, dtype) for _, shape, dtype in specs)),),
                allocate_window,
            )

        def step(kind: str, run: Any, window: Window = window) -> Any:
            ticket = cursor.claim()
            if ticket.operation.kind != kind or ticket.operation.window != window:
                raise RuntimeError("VAE sequence submission differs from its immutable schedule")
            try:
                result = run()
                cursor.submitted(ticket)
                return result
            except BaseException as error:
                cursor.failed(ticket, error)
                raise

        def edge_buffers(
            stage: str,
            module: Any,
            output: Tensor | None = None,
            p: _WindowPlans = p,
            owners: dict[str, Tensor] = owners,
        ) -> Any:
            values = {
                name: owners[f"{stage}.{name}"]
                for name, _, _ in getattr(p, stage).description.buffer_specs(
                    getattr(p, stage).info["workspace_bytes"]
                )
                if not (stage == "tail" and name == "result")
            }
            if output is not None:
                values["result"] = output
            if module is prefix:
                return prefix.Buffers(values)
            return module.Buffers(
                values["input_rows"],
                values["dense_result"],
                values["result"],
                values.get("weight_cast"),
                values.get("workspace"),
            )

        x = step(
            "post_quant",
            lambda p=p, window=window: _edge(
                request,
                stream,
                "post_quant",
                p.post,
                value.narrow(0, window.start, window.count),
                edge_buffers("post", post),
                custody,
            ),
        )
        hidden, rotary = step(
            "prefix",
            lambda p=p, x=x: _edge(
                request, stream, "prefix", p.prefix, x, edge_buffers("prefix", prefix), custody
            ),
        )
        del x
        for name in tuple(owners):
            if name.startswith("post.") or (
                name.startswith("prefix.")
                and name not in ("prefix.hidden", "prefix.cosine", "prefix.sine")
            ):
                del owners[name]
        invocations = []
        custody.append(invocations)
        for index, prepared in enumerate(p.blocks):
            identity = _block_id(window.count, window.block_chunk)
            values = {
                name: owners[f"block.{name}"].view(-1)[: math.prod(shape)].view(shape)
                for name, shape, _ in prepared.buffer_specs(identity)
                if name != "output"
            }
            output = owners["hidden32_a" if index % 2 == 0 else "hidden32_b"]
            buffers = block.BlockBuffers(
                output,
                values["packed"],
                values["expanded"],
                values.get("weight"),
                values.get("workspace"),
                tuple(values[f"attention:{i}"] for i in range(7)),
                window.block_chunk,
                TOKENS,
            )
            invocation = prepared.bind(hidden, rotary, identity, buffers)
            invocations.append(invocation)
            hidden = step("block", invocation)
        stream.synchronize()  # all borrowed scratch/rotary/input owners may now retire
        invocations.clear()
        _release(custody, invocations)
        del invocation, buffers, values, output, rotary
        for name in tuple(owners):
            if name.startswith(("block.", "prefix.")) or name == "hidden32_a":
                del owners[name]
        pixels = step(
            "tail",
            lambda p=p, hidden=hidden, window=window: _edge(
                request,
                stream,
                "tail",
                p.tail,
                hidden,
                edge_buffers("tail", tail, assembly.view(window)),
                custody,
            ),
        )
        assembly.validate_result(window, pixels)
        del hidden, pixels
        stream.synchronize()
        owners.clear()
        _release(custody, owners)
        record["completed_windows"] += 1
    if not cursor.host_done:
        raise RuntimeError("VAE sequence did not submit its complete declared body")
    return assembly.output


def decode_vae_tiles(vae: Any, value: Tensor, *, block_layout: Any) -> Tensor:
    """Final restricted H3 path. Unsupported contracts fail before numerical writes.

    Current S produces/allocates/executes every local owner. Caller inputs and source
    views follow Runtime's ordinary S dependency contract. All borrowers survive
    checked S completion; a failure never retries or falls back to the old decoder.
    """
    from contextlib import ExitStack, nullcontext

    from cozy_runtime import fusion
    from cozy_runtime.author.kernels import Plan
    from cozy_runtime.internal import triton_kernels
    from cozy_runtime.internal.execution import catch, phase
    from cozy_runtime.internal.residency import _REQUEST

    request = _REQUEST.get()
    if (
        request is None
        or request.idle
        or type(value) is not Tensor
        or not value.is_cuda
        or value.device != request.memory.device
        or value.requires_grad
        or value.is_conj()
        or value.is_neg()
        or not value.is_contiguous()
        or tuple(value.shape[1:]) != (24, *GEOMETRY)
        or value.dtype not in (torch.float16, torch.float32)
        or value.shape[0] < 1
        or torch.is_grad_enabled()
        or catch.uncommitted_scope()
        or len(vae.decoder.transformer_blocks) != BLOCKS
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.float16
    ):
        raise ValueError(
            "native H3 tiles require the declared independent CUDA FP16-autocast inference call"
        )
    stream = torch.cuda.current_stream(value.device)
    if stream.cuda_stream <= 2 or torch.cuda.is_current_stream_capturing():
        raise ValueError("native H3 tiles require the actual owned eager stream")
    declarations = variants(value.shape[0])
    custody: list[Any] = [value, vae]
    with ExitStack() as stack:
        with request.memory.lock:
            models = tuple(m for m in request.memory.models.values() if m.module is vae)
        for model in models:
            stack.enter_context(model.passing(executing=True))
        # The actual installed declaration remains mandatory; there is no eager substitute.
        if not fusion.install(
            Plan(value.device, lambda kernels: triton_kernels.load(kernels, device=value.device)),
            block_layout,
            vae.decoder,
        ):
            raise ValueError("native H3 VAE requires its actual compiled block declaration")
        for module in vae.decoder.transformer_blocks:
            site = getattr(module, "_cozy_dit", None)
            if site is None or site.layout != block_layout:
                raise ValueError("native H3 VAE block declaration changed")
            if site.switch.launch is None:
                site.switch.launch = triton_kernels.load(block_layout.kernels, device=value.device)
        try:
            plans = _prepare_windows(vae, value, request, stream, declarations)
            custody.append(plans)
            scope = (
                phase.observing(value.device.index or 0, "eager", request.working_phases)
                if phase._ACTIVE.get() is None
                else nullcontext(None)
            )
            with scope as observation:
                output = _execute_windows(vae, value, request, stream, plans, declarations, custody)
                stream.synchronize()
                if observation is not None:
                    observation.complete(
                        (
                            "h3-native-tiles/1",
                            tuple(value.shape),
                            value.dtype,
                            block_layout.kernels.identity,
                        )
                    )
            return output
        except BaseException as error:
            error.varena_no_retry = True
            try:
                stream.synchronize()
            except BaseException:
                _UNPROVEN.append(custody)
                raise
            raise
