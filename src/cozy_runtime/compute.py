"""Prepared resident CUDA work. Python constructs once; the GPU owns loop continuation.

This deliberately accepts one deterministic, fixed-shape tensor step, not arbitrary Python
control flow. Randomness must be explicit tensor inputs. Serving never compiles this module.
"""

from __future__ import annotations

import hashlib
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, nullcontext, suppress
from contextvars import ContextVar
from typing import Any, Literal, NoReturn

from tensorfs import varena

from cozy_runtime import graphs
from cozy_runtime.author._context import CURRENT, Context
from cozy_runtime.author._errors import CapabilityError, RuntimeFailure
from cozy_runtime.author._model import LANDED, PLACED, Building
from cozy_runtime.author._prepared import Progress
from cozy_runtime.graphs import domain
from cozy_runtime.internal import kernel_ready

_CAPTURE_GROUP: ContextVar[Any] = ContextVar("cozy_sequence_capture_group", default=None)


def _zero_dropout_attention(func: Any, args: Any, kwargs: Any) -> bool:
    """Built-in forward SDPA is tagged seeded even when its dropout is a fixed zero."""
    schema = func._schema
    if schema.overload_name or schema.name not in (
        # The tagged selector chooses a backend; zero-dropout dispatch is deterministic.
        "aten::_fused_sdp_choice",
        "aten::scaled_dot_product_attention",
        "aten::_scaled_dot_product_attention_math",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_cudnn_attention",
    ):
        return False
    # Flash, efficient/cuDNN and public/math schemas put dropout at different positions.
    for position, argument in enumerate(schema.arguments):
        if argument.name == "dropout_p":
            value = kwargs.get(
                "dropout_p", args[position] if position < len(args) else argument.default_value
            )
            return type(value) in (int, float) and value == 0
    return False


def _native_error(error: Exception) -> NoReturn:
    import torch

    if getattr(error, "load_unavailable", False):
        raise CapabilityError(str(error), code="prepared_load_unavailable") from error
    if getattr(error, "eligibility_refusal", False):
        raise CapabilityError(str(error), code="prepared_graph_ineligible") from error

    code = getattr(error, "error_code", None)
    if code == 2:
        raise torch.OutOfMemoryError(str(error)) from error
    if isinstance(code, int):
        fault = torch.AcceleratorError(str(error))
        fault.error_code = code
        raise fault from error
    raise error


class PreparedRepeat:
    """A reusable GPU repeat with resident state, explicit timestep/noise inputs and owners.

    `step(index)` executes only during preparation. `index` is a one-element CUDA int64
    tensor; use device tensor indexing, never `.item()` or a Python branch. The step updates
    the declared state in place. All other inputs must remain immutable through completion.
    Capture retains external ATen dependencies, the graph pool, and explicit owners.
    Placed component roots or explicit weight bindings name the complete dependency set.
    """

    supports_component_load = True

    def __init__(
        self,
        native: Any,
        state: tuple[Any, ...],
        device: int,
        steps: int,
        dependencies: tuple[Any, ...],
        bindings: tuple[Any, ...],
        landed: int,
    ) -> None:
        self._native, self.state, self.device, self.steps = native, state, device, steps
        self._active: Execution | None = None
        self._submission = threading.RLock()
        self._dependencies, self._bindings = dependencies, bindings
        self._closed = False
        self._owned_bindings: tuple[Any, ...] = ()
        self._binding_cleanup = ExitStack()
        self._load: Any = None
        self._load_used = False
        self._capture_cache: dict[str, int] = {}
        self._load_ordering_requested = "checked"
        self._sequence_owner: Any = None
        self._load_ordering = "checked"
        self._landed = landed

    @staticmethod
    def _layout(tensor: Any) -> tuple[Any, ...]:
        storage = tensor.untyped_storage()
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
            tensor.storage_offset(),
            tensor.data_ptr(),
            storage.data_ptr(),
            storage.nbytes(),
        )

    def _invalid_reason(self) -> str | None:
        if self._closed:
            return "prepared repeat is closed"
        if self._landed != LANDED[0]:
            return "prepared kernel implementations changed"
        for reference, layout in self._dependencies:
            tensor = reference()
            if tensor is None or self._layout(tensor) != layout:
                return "prepared tensor storage or geometry changed"
        for reference, identity in self._bindings:
            binding = reference()
            if binding is None or not binding.valid() or binding.identity != identity:
                return "prepared weight binding or fill identity changed"
        return None

    def _validate(self) -> None:
        if reason := self._invalid_reason():
            raise RuntimeError(reason)

    def valid(self) -> bool:
        """Whether this definition's owners, layouts and implementations still match.

        Input values may change in place between completed submissions. This query neither
        prepares missing weights nor reserves a future submission; submit checks again.
        Unexpected storage/device errors propagate rather than becoming a cache miss.
        """
        with self._submission:
            return self._invalid_reason() is None

    @classmethod
    def capture_stage(
        cls,
        step: Callable[[Any], object],
        *,
        state: Sequence[Any],
        owner: object,
        paths: tuple[str, ...] = ("",),
        load_ordering: Literal["checked", "events", "auto"] = "events",
    ) -> PreparedRepeat:
        """Record one immutable raw model block/phase at its existing managed addresses.

        `owner` is a placed component root or a retained subset binding. For a root,
        `paths` selects its named block subtrees plus common weights; () selects only common
        weights. An explicit binding already defines its paths and accepts only the default.
        The callback runs
        only during preparation and writes explicit GPU boundary state. Use PreparedSequence
        to advance these stages or repeat a block range without Python callbacks. A shared
        GPU timestep belongs in declared state and advances in the authored sampler stage.

        Source/fill/generation and parameter geometry remain checked on every submission;
        model claims cover the entire native sequence. Capture-private weight views are
        released after checked preparation, and native detachment refuses escaped aliases.
        Direct stored native-role buffers are supported when the existing route declares
        exact source aliases. Decoding, derived/final buffers, packed F4 views and
        undeclared/stateful module behavior remain ineligible.
        """
        import torch

        tensors = tuple(state)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("model stage state must contain CUDA tensors")
        original = tuple(tensor.detach() for tensor in tensors)
        layouts = tuple(cls._layout(tensor) for tensor in tensors)

        def record(index: Any) -> object:
            try:
                return step(index)
            finally:
                changed = [
                    i for i, tensor in enumerate(tensors) if cls._layout(tensor) != layouts[i]
                ]
                for i in changed:
                    # Restore metadata before capture cleanup copies saved values; never
                    # write mutable-state restoration into an aliased immutable weight.
                    tensors[i].data = original[i]
                if changed:
                    raise CapabilityError(
                        "model stage changed boundary tensor metadata",
                        code="prepared_graph_ineligible",
                    )

        from cozy_runtime.internal.residency import Binding

        if not isinstance(paths, tuple) or any(not isinstance(path, str) for path in paths):
            raise ValueError("model stage paths must be a tuple of module paths")
        selected = None
        if isinstance(owner, Binding):
            if paths != ("",):
                raise ValueError("an explicit model binding already selects its stage paths")
        elif paths != ("",):
            placed = PLACED.get(owner) if isinstance(owner, torch.nn.Module) else None
            if placed is None:
                raise CapabilityError(
                    "model stage paths require a placed component root",
                    code="prepared_graph_ineligible",
                )
            selected = placed[0].retain(placed[1], paths)
            owner = selected
        try:
            return cls.capture(
                record,
                state=tensors,
                steps=1,
                owners=(owner,),
                load_component=owner,
                load_ordering=load_ordering,
                _detach_stage=True,
            )
        finally:
            if selected is not None:
                selected.close()

    @classmethod
    def capture(
        cls,
        step: Callable[[Any], object],
        *,
        state: Sequence[Any],
        steps: int,
        owners: Sequence[object] = (),
        resident_only: bool = False,
        load: Any = None,
        load_component: Any = None,
        load_ordering: Literal["checked", "events", "auto"] = "checked",
        prefix: Callable[[], object] | None = None,
        suffix: Callable[[], object] | None = None,
        dtypes: Mapping[object, Any] | None = None,
        _detach_stage: bool = False,
    ) -> PreparedRepeat:
        """Capture a fixed body, optionally fed by one native LoadSlot.

        A load slot is initialized capture storage, not a cold-model compiler. Capture seals
        its unpublished Fill; one later submission reads its actual stored bytes and, for an
        encoded slot, decodes them through TensorFS before the native graph starts. A native
        worker checks composed completion before publishing results. With load_ordering
        "events", native producer events order compute and a GPU input guard skips invalid
        decoded values. Late ownership/device failures discard output and retire; they do
        not imply that no body ran. "auto" selects events only with both native capabilities,
        otherwise checked ordering. The selected ordering is reported in facts. Loaded
        submissions are one-shot and require wait/request join
        before output access. Pure resident repeats retain their existing reuse contract.

        Optional prefix/suffix tensor work runs once before/after the repeat. The suffix
        runs only after every iteration completed without cancellation at its GPU boundary.
        A phase already admitted on the GPU completes before its owners can be released;
        cancellation does not preempt a running prefix or suffix. Resident definitions can
        be submitted again with new input/state values at the same layouts and identities.
        All three phases share the declared mutable state and retained resident owners;
        this does not release phase weights or implement streaming between phases.

        `dtypes` selects an explicit final float dtype for named placed component roots.
        Stored bytes keep their original dtype; conversion fills managed final buffers once.
        A changed representation retires older idle graphs before preparation. The selected
        representation may remain cached after a preparation refusal; rollback covers the
        declared mutable state, not cache contents. Resident-only capture refuses a change.

        `load_component` names one placed component root or explicit binding already listed
        in owners. Immutable
        raw block layouts use its existing regions: ready bytes are retained and only misses
        are loaded by the native submission. Capture uses private initialized aliases, never
        ordinary published weights. This requires a complete resident phase; unsupported
        transforms or capacity refuse before submission. These inputs are read-only during
        authored computation because their regions cache source bytes. State-storage aliases
        and ATen-declared writes are refused before execution; custom/native kernels must
        honor that same contract. Opting in also requires data-independent preparation:
        every execution must read these dynamic inputs, without caching tensors derived from
        zero initialization in Python during warmup. Raw layout eligibility does not prove
        this property for arbitrary modules; unknown/stateful modules need resident preparation.
        Generic streaming is not implied.

        Placed component roots are resolved to owned weight bindings. `resident_only=True`
        refuses missing weights before running the body; it never fills them for capture.
        Missing residency, unsupported hooks or changed model state raises CapabilityError
        with code `prepared_graph_ineligible`. A caller may continue eagerly after a capture
        refusal or a capture OOM, but must never retry after submit has begun.
        """
        import torch

        from cozy_runtime.internal.residency import (
            _CAPTURE_BLOCKS,
            Binding,
            ResidencyError,
            StaleBinding,
        )

        try:
            import cozy_runtime._kernels._execution as _execution
        except ModuleNotFoundError as error:
            if error.name != "cozy_runtime._kernels._execution":
                raise
            raise CapabilityError(
                "this Runtime wheel has no prebuilt GPU execution module",
                code="prepared_graph_ineligible",
            ) from error

        if (prefix is not None or suffix is not None) and not getattr(
            _execution.ResidentRepeat, "supports_phases", False
        ):
            raise CapabilityError(
                "this Runtime execution module has no resident phase composition",
                code="prepared_graph_ineligible",
            )

        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 < steps <= 2**32 - 1:
            raise ValueError("repeat steps must be a positive uint32")
        tensors = tuple(state)
        group = _CAPTURE_GROUP.get()
        if group is not None:
            group._prepare(tensors)
        if not tensors or any(not isinstance(t, torch.Tensor) or not t.is_cuda for t in tensors):
            raise ValueError("prepared repeat state must contain CUDA tensors")
        device = tensors[0].device.index
        if device is None or any(t.device.index != device for t in tensors):
            raise ValueError("prepared repeat uses one CUDA device")
        if any(t.is_inference() for t in tensors):
            raise CapabilityError(
                "prepared mutable state must be allocated outside inference_mode",
                code="prepared_graph_ineligible",
            )
        if _detach_stage and (load_component is None or len(owners) != 1):
            raise ValueError("a detached model stage requires exactly one named raw binding")
        if load_component is not None:
            if load is not None or resident_only or not any(load_component is o for o in owners):
                raise ValueError(
                    "load_component needs one named owner, no load slot and nonresident preparation"
                )
            if not isinstance(load_component, Binding) and (
                not isinstance(load_component, torch.nn.Module)
                or PLACED.get(load_component) is None
            ):
                raise CapabilityError(
                    "native component loading requires a placed root",
                    code="prepared_graph_ineligible",
                )
        if load is not None or load_component is not None:
            slot_type = getattr(varena, "LoadSlot", None)
            if (
                slot_type is None
                or (load is not None and not isinstance(load, slot_type))
                or not callable(getattr(_execution.ResidentRepeat, "continuation", None))
            ):
                raise CapabilityError(
                    "native loading requires a TensorFS LoadSlot",
                    code="prepared_load_unavailable",
                )
            if load_component is not None and not getattr(
                slot_type, "supports_region_input", False
            ):
                raise CapabilityError(
                    "this TensorFS has no native component input preparation",
                    code="prepared_graph_ineligible",
                )
            event_capable = bool(getattr(slot_type, "supports_event_input", False)) and bool(
                getattr(_execution.ResidentRepeat, "supports_event_input", False)
            )
            if load_ordering not in ("checked", "events", "auto"):
                raise ValueError("load_ordering must be checked, events or auto")
            if load_ordering == "events" and not event_capable:
                raise CapabilityError(
                    "event-ordered loading requires matching native input capabilities",
                    code="prepared_load_events_unavailable",
                )
            selected_ordering = (
                "events" if load_ordering != "checked" and event_capable else "checked"
            )
            if load is not None:
                owners = (*owners, load)
        elif load_ordering != "checked":
            raise ValueError("load_ordering requires a native LoadSlot")
        with (
            torch.cuda.device(device),
            torch.inference_mode(False),
            ExitStack() as lent,
            ExitStack() as cleanup,
            ExitStack() as preparing,
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare a repeat outside graph capture")
            changes: list[Callable[[], None]] = []
            for root, dtype in (dtypes or {}).items():
                if not isinstance(root, torch.nn.Module) or not any(root is o for o in owners):
                    raise CapabilityError(
                        "prepared dtype keys must be component roots named in owners",
                        code="prepared_graph_ineligible",
                    )
                placed = PLACED.get(root)
                if placed is None:
                    raise CapabilityError(
                        "prepared dtype overrides require placed component roots",
                        code="prepared_graph_ineligible",
                    )
                try:
                    changes.append(
                        placed[0].represent(placed[1], dtype, resident_only=resident_only)
                    )
                except StaleBinding as error:
                    raise CapabilityError(str(error), code="prepared_graph_ineligible") from error
            for change in changes:
                try:
                    change()
                except StaleBinding as error:
                    raise CapabilityError(str(error), code="prepared_graph_ineligible") from error
            resolved: list[object] = []
            bindings: dict[int, Any] = {}
            models: dict[int, Any] = {}
            idle: set[int] = set()
            for owner in owners:
                placed = PLACED.get(owner) if isinstance(owner, torch.nn.Module) else None
                if isinstance(owner, Binding):
                    placed = PLACED.get(owner.model.module)
                if placed is not None:
                    mode, _ = placed[0].lora_choices((placed[1],))[placed[1]]
                    if mode == "per_step":
                        raise CapabilityError(
                            "prepared repeats require a fixed baked LoRA representation",
                            code="prepared_graph_ineligible",
                        )
                if isinstance(owner, Binding) or placed is not None:
                    binding = bindings.get(id(owner))
                    if binding is None:
                        if isinstance(owner, Binding):
                            if not owner.valid():
                                raise CapabilityError(
                                    "capture requires a valid weight binding",
                                    code="prepared_graph_ineligible",
                                )
                            binding = owner.model.retain(
                                tuple(b.path for b in owner.groups if b.path)
                            )
                        else:
                            assert placed is not None
                            binding = placed[0].retain(placed[1], ("",))
                        bindings[id(owner)] = binding
                        cleanup.callback(binding.close)
                        model = binding.model
                        if not model.memory.cuda or model.memory.device.index not in (None, device):
                            raise CapabilityError(
                                "prepared owners must use the state's CUDA device",
                                code="prepared_graph_ineligible",
                            )
                        models[id(model.module)] = model.module
                        if why := graphs._hooked(model.module):
                            raise CapabilityError(why, code="prepared_graph_ineligible")
                        if owner is load_component:
                            if _detach_stage:
                                # Lent through detach_stage; taken back as capture returns.
                                lent.enter_context(model.lending())
                            try:
                                load = binding.prepare_load(stage=_detach_stage)
                            except ResidencyError as error:
                                # StaleBinding and a concurrent-use/_claim refusal are both
                                # pre-submission: fall back to an eager resident load.
                                raise CapabilityError(
                                    str(error), code="prepared_graph_ineligible"
                                ) from error
                            except RuntimeError as error:
                                # Native region refusals (busy, capture-active, ineligible
                                # descriptor, or not-local inputs) map to a fallback code; OOM,
                                # device faults and poisoning keep their fatal classification.
                                _native_error(error)
                            if _detach_stage and load is None:
                                raise CapabilityError(
                                    "this TensorFS cannot detach an already-resident model stage",
                                    code="prepared_graph_ineligible",
                                )
                            if load is not None:
                                resolved.append(load)
                        if resident_only or (owner is load_component and load is None):
                            memory = model.memory
                            if id(memory) not in idle:
                                preparing.enter_context(
                                    memory.request(graphs.IDLE_REQUEST, idle=True)
                                )
                                idle.add(id(memory))
                            if why := binding.unresident():
                                raise CapabilityError(
                                    f"capture requires the component's weights to be resident"
                                    f" ({why})",
                                    code="prepared_graph_ineligible",
                                )
                        try:
                            preparing.enter_context(binding.prepare())
                        except StaleBinding as error:
                            raise CapabilityError(
                                str(error), code="prepared_graph_ineligible"
                            ) from error
                    resolved.append(binding)
                else:
                    resolved.append(owner)
                    if isinstance(owner, torch.nn.Module):
                        if why := graphs._hooked(owner):
                            raise CapabilityError(why, code="prepared_graph_ineligible")
                        models[id(owner)] = owner
            snapshots: list[Any] = []
            immutable_buffers = (
                frozenset().union(
                    *(binding.immutable_buffer_ids() for binding in bindings.values())
                )
                if _detach_stage
                else frozenset()
            )
            unbound_buffers = (
                immutable_buffers
                - frozenset().union(
                    *(binding.immutable_buffer_ids(selected=True) for binding in bindings.values())
                )
                if _detach_stage
                else frozenset()
            )
            undeclared_inputs = (
                tuple(
                    area
                    for binding in bindings.values()
                    for area in binding.undeclared_input_ranges()
                )
                if _detach_stage
                else ()
            )
            declared_blocks = (
                frozenset(id(block) for binding in bindings.values() for block in binding.groups)
                if _detach_stage
                else None
            )
            capture_blocks = None
            plan = None
            bypass = graphs.BYPASS.set(True)
            try:
                if declared_blocks is not None:
                    capture_blocks = _CAPTURE_BLOCKS.set(declared_blocks)
                with torch.no_grad():
                    try:
                        snapshots = [
                            graphs._State(torch, module, immutable_buffers=immutable_buffers)
                            for module in models.values()
                        ]
                        plan = cls._capture_ready(
                            step,
                            state=tensors,
                            steps=steps,
                            owners=resolved,
                            prefix=prefix,
                            suffix=suffix,
                            _release_loaded_inputs=_detach_stage,
                            _immutable_buffers=immutable_buffers,
                            _unbound_buffers=unbound_buffers,
                            _undeclared_inputs=undeclared_inputs,
                        )
                    finally:
                        changed = [
                            change for snapshot in snapshots if (change := snapshot.restore())
                        ]
                        if snapshots:
                            torch.cuda.current_stream().synchronize()
                    if changed:
                        raise CapabilityError(
                            "recording changed the model's state: " + ", ".join(changed),
                            code="prepared_graph_ineligible",
                        )
            except BaseException:
                if plan is not None:
                    plan.close()
                raise
            finally:
                if capture_blocks is not None:
                    _CAPTURE_BLOCKS.reset(capture_blocks)
                graphs.BYPASS.reset(bypass)
            assert plan is not None
            if load is not None:
                try:
                    load.seal()
                except BaseException:
                    plan.close()
                    raise
                if _detach_stage:
                    try:
                        # Parameter slots and capture streams are restored before dropping
                        # any capture-only weight view. Native detachment checks real Pins.
                        preparing.close()
                        snapshots.clear()
                        load = next(iter(bindings.values())).detach_stage()
                    except BaseException as error:
                        plan.close()
                        if isinstance(error, Exception):
                            _native_error(error)
                        raise
                plan._load = load
                plan._load_ordering_requested = load_ordering
                plan._load_ordering = selected_ordering
            plan._owned_bindings = tuple(bindings.values())
            for binding in plan._owned_bindings:
                model = binding.model
                graphs.own_prepared(model.module, plan)
                if model.retire_graphs is None:
                    root = weakref.ref(model.module)

                    def retire(root: Any = root) -> None:
                        if (module := root()) is not None:
                            graphs.retire(torch, module)

                    model.retire_graphs = retire
            plan._binding_cleanup = cleanup.pop_all()
            return plan

    @classmethod
    def _capture_ready(
        cls,
        step: Callable[[Any], object],
        *,
        state: Sequence[Any],
        steps: int,
        owners: Sequence[object] = (),
        prefix: Callable[[], object] | None = None,
        suffix: Callable[[], object] | None = None,
        _release_loaded_inputs: bool = False,
        _immutable_buffers: frozenset[int] = frozenset(),
        _unbound_buffers: frozenset[int] = frozenset(),
        _undeclared_inputs: tuple[tuple[Any, int, int], ...] = (),
    ) -> PreparedRepeat:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves

        from cozy_runtime._kernels import _execution
        from cozy_runtime.internal.residency import Binding

        tensors = tuple(state)
        device = tensors[0].device.index
        assert device is not None
        group = _CAPTURE_GROUP.get()
        generated: dict[int, tuple[Any, Any, Any, int, int]] = {}
        held: dict[int, Any] = {id(t): t for t in tensors}
        landed = LANDED[0]
        immutable: tuple[Any, ...] = tuple(
            tensor
            for owner in owners
            for tensor in (
                owner.immutable_inputs()
                if isinstance(owner, Binding)
                else getattr(owner, "immutable_inputs", ())
                if isinstance(owner, varena.LoadSlot)
                else ()
            )
        )
        ranges = tuple(
            dict.fromkeys(
                (tensor.device, int(storage.data_ptr()), int(storage.data_ptr()) + storage.nbytes())
                for tensor in immutable
                if (storage := tensor.untyped_storage()).nbytes()
            )
        )

        def overlaps_input(tensor: Any, areas: Any = ranges) -> bool:
            if not isinstance(tensor, torch.Tensor):
                return False
            storage = tensor.untyped_storage()
            first, size = int(storage.data_ptr()), int(storage.nbytes())
            return bool(size) and any(
                tensor.device == device and first < end and begin < first + size
                for device, begin, end in areas
            )

        if any(overlaps_input(tensor) for tensor in tensors):
            raise CapabilityError(
                "mutable state overlaps immutable loaded input storage",
                code="prepared_graph_ineligible",
            )
        held.update((id(tensor), tensor) for tensor in immutable)
        slots = [owner for owner in owners if isinstance(owner, varena.LoadSlot)]
        detached_bindings = [owner for owner in owners if isinstance(owner, Binding)]
        if _release_loaded_inputs and (
            not immutable
            or len(slots) != 1
            or len(owners) != len(slots) + len(detached_bindings)
            or len(detached_bindings) > 1
            or any(binding._load_slot is not slots[0] for binding in detached_bindings)
        ):
            raise CapabilityError(
                "detached stage capture requires one existing-region raw input slot",
                code="prepared_graph_ineligible",
            )

        class CaptureOwners(TorchDispatchMode):  # type: ignore[misc]  # Torch is optional in checks.
            def __init__(self, retain: bool) -> None:
                super().__init__()
                self.retain = retain
                self.produced: dict[int, Any] = {}

            def __torch_dispatch__(
                self, func: Any, types: Any, args: Any = (), kwargs: Any = None
            ) -> Any:
                if _unbound_buffers or _undeclared_inputs:
                    for tensor in tree_leaves((args, kwargs or {})):
                        if id(tensor) in _unbound_buffers or overlaps_input(
                            tensor, _undeclared_inputs
                        ):
                            raise CapabilityError(
                                "model stage reads a checkpoint buffer outside its declared paths",
                                code="prepared_graph_ineligible",
                            )
                if torch.Tag.nondeterministic_seeded in func.tags and not _zero_dropout_attention(
                    func, args, kwargs or {}
                ):
                    raise RuntimeError(
                        "GPU-driven repeats require explicit noise/RNG tensor inputs"
                    )
                if ranges or _immutable_buffers:
                    for position, argument in enumerate(func._schema.arguments):
                        if argument.alias_info is None or not argument.alias_info.is_write:
                            continue
                        value = (
                            args[position]
                            if position < len(args)
                            else (kwargs or {}).get(argument.name)
                        )
                        if any(
                            id(tensor) in _immutable_buffers or overlaps_input(tensor)
                            for tensor in tree_leaves(value)
                        ):
                            raise CapabilityError(
                                "authored computation writes immutable loaded input storage",
                                code="prepared_graph_ineligible",
                            )
                for tensor in tree_leaves((args, kwargs or {})):
                    prior = self.produced.get(id(tensor))
                    if (
                        self.retain
                        and isinstance(tensor, torch.Tensor)
                        and (prior is None or prior() is not tensor)
                    ):
                        if group is not None:
                            group._external(tensor)
                        held[id(tensor)] = tensor
                input_storage = (
                    {
                        (tensor.device, tensor.untyped_storage().data_ptr())
                        for tensor in tree_leaves((args, kwargs or {}))
                        if isinstance(tensor, torch.Tensor)
                    }
                    if group is not None and self.retain
                    else set()
                )
                result = func(*args, **(kwargs or {}))
                for tensor in tree_leaves(result):
                    if isinstance(tensor, torch.Tensor):
                        self.produced[id(tensor)] = weakref.ref(tensor)
                        if group is not None and self.retain:
                            storage = tensor.untyped_storage()
                            pointer, size = int(storage.data_ptr()), int(storage.nbytes())
                            aliases_generated = any(
                                tensor.device == device and pointer < end and start < pointer + size
                                for _, _, device, start, end in generated.values()
                            )
                            if size and (
                                (tensor.device, pointer) not in input_storage or aliases_generated
                            ):
                                generated[id(tensor)] = (
                                    weakref.ref(tensor),
                                    weakref.ref(storage),
                                    tensor.device,
                                    pointer,
                                    pointer + size,
                                )
                return result

        with torch.cuda.device(device), torch.no_grad():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare a repeat outside graph capture")
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device) if group is None else group._domain.stream
            index = torch.zeros(1, dtype=torch.int64, device=device)
            # Native control writes this storage even when the authored body does not.
            held[id(index)] = index
            saved = tuple(t.clone() for t in tensors)
            stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            before = torch.cuda.CUDAGraph(keep_graph=True) if prefix is not None else None
            after = torch.cuda.CUDAGraph(keep_graph=True) if suffix is not None else None
            phases = [(graph, lambda: step(index))]
            if before is not None:
                phases.insert(0, (before, prefix))
            if after is not None:
                phases.append((after, suffix))
            native = None
            try:
                with torch.cuda.stream(stream), CaptureOwners(False):
                    for _, operation in phases:
                        operation()  # instantiate each phase's library workspaces before capture
                    for target, original in zip(tensors, saved, strict=True):
                        target.copy_(original)
                stream.synchronize()
                # Like torch.cuda.graph's preparation, return inactive eager segments
                # before the graph asks for its own pool. Warmup and its state restore
                # are joined; live tensors, weight pins and existing graph pools remain
                # owned. This is a prelaunch boundary, never allocation recovery.
                reserved_before = int(torch.cuda.memory_reserved(device))
                allocated_before = int(torch.cuda.memory_allocated(device))
                torch.cuda.empty_cache()
                reserved_after = int(torch.cuda.memory_reserved(device))
                capture_cache = {
                    "torch_allocated_before_bytes": allocated_before,
                    "torch_reserved_before_bytes": reserved_before,
                    "torch_reserved_after_bytes": reserved_after,
                    "torch_returned_bytes": max(reserved_before - reserved_after, 0),
                }
                context = CURRENT.get()
                pause = context._pause_stall_watch if context is not None else None
                with (
                    pause() if pause is not None else nullcontext(),
                    kernel_ready.recording(),
                    varena.capture(device),
                    torch.cuda.stream(stream),
                    CaptureOwners(True),
                ):
                    for recording, operation in phases:
                        if group is None:
                            recording.capture_begin(capture_error_mode="global")
                            try:
                                operation()
                            finally:
                                recording.capture_end()
                        else:
                            # Same owned library/control streams as the established Domain.
                            # Capture begins on control so Torch cannot reset the live cuBLAS
                            # library workspace when recording the next stage.
                            control = group._domain.control
                            control.wait_stream(stream)
                            with torch.cuda.stream(control):
                                recording.capture_begin(
                                    pool=group._pool.handle, capture_error_mode="global"
                                )
                                try:
                                    group._domain.fork.record(control)
                                    stream.wait_event(group._domain.fork)
                                    with torch.cuda.stream(stream):
                                        try:
                                            operation()
                                        finally:
                                            group._domain.join.record(stream)
                                            control.wait_event(group._domain.join)
                                finally:
                                    recording.capture_end()
                            control.synchronize()
                            group._domain.graphs.add(recording)
                stream.synchronize()
                for target, original in zip(tensors, saved, strict=True):
                    target.copy_(original)
                current.synchronize()
                # CUDAGraph owns capture allocations; its live template is cloned into a child
                # of the native conditional body. Its executable is never borrowed or replayed.
                phase_args = {}
                if before is not None or after is not None:
                    phase_args = {
                        "prefix_graph": before.raw_cuda_graph() if before is not None else 0,
                        "suffix_graph": after.raw_cuda_graph() if after is not None else 0,
                    }
                # Internal raw-stage capture retains addresses through its native source
                # descriptor after this method returns. Remove only immutable input storage
                # owners here; the caller must then drop every private view and ask native
                # detachment to refuse any remaining foreign alias before admitting execution.
                retained_owners = owners
                if _release_loaded_inputs:
                    held = {
                        key: tensor for key, tensor in held.items() if not overlaps_input(tensor)
                    }
                    retained_owners = ()
                if group is not None:
                    areas = tuple(
                        (device, start, end) for _, _, device, start, end in generated.values()
                    )
                    held = {
                        key: tensor
                        for key, tensor in held.items()
                        if not overlaps_input(tensor, areas)
                    }
                    if any(
                        tensor() is not None or storage() is not None
                        for tensor, storage, _, _, _ in generated.values()
                    ):
                        raise CapabilityError(
                            "a captured temporary escaped its sequence stage; "
                            "use an external boundary tensor",
                            code="prepared_graph_ineligible",
                        )
                    group._areas.extend(areas)
                native = _execution.ResidentRepeat(
                    graph.raw_cuda_graph(),
                    index.data_ptr(),
                    device,
                    steps,
                    [
                        *(recording for recording, _ in phases),
                        stream,
                        index,
                        *held.values(),
                        *(tensor.untyped_storage() for tensor in held.values()),
                        *retained_owners,
                        *(() if group is None else (group._domain, group._pool)),
                    ],
                    **phase_args,
                )
            except Building as error:
                raise CapabilityError(
                    f"capture waits for kernel readiness: {error}",
                    code="prepared_graph_ineligible",
                ) from error
            except _execution.NativeExecutionError as error:
                _native_error(error)
            finally:
                # Restore state even when capture/eligibility fails; no prepared object escapes.
                # A failed warm-up may have queued work before throwing on the host.
                current.wait_stream(stream)
                for target, original in zip(tensors, saved, strict=True):
                    target.copy_(original)
                current.synchronize()
                if native is None:
                    for recording, _ in phases:
                        recording.reset()
                    held.clear()
            if landed != LANDED[0]:
                native.close_owner()
                held.clear()
                raise CapabilityError(
                    "kernel implementations changed during recording",
                    code="prepared_graph_ineligible",
                )
        dependencies = tuple((weakref.ref(t), cls._layout(t)) for t in held.values())
        bindings = tuple(
            (weakref.ref(owner), owner.identity) for owner in owners if isinstance(owner, Binding)
        )
        # The local mode class has a __class__ closure cycle. Native ownership has taken its
        # own references; do not leave GPU tensors retained until an unrelated Python GC pass.
        held.clear()
        result = cls(native, tensors, device, steps, dependencies, bindings, landed)
        result._capture_cache = capture_cache
        return result

    def _retire_if_idle(self) -> None:
        sequence = self._sequence_owner() if self._sequence_owner is not None else None
        if sequence is not None and not sequence._closing:
            sequence._retire_if_idle()
            return
        with self._submission:
            if self._closed:
                return
            if (
                self._active is not None
                and self._active._finished is None
                and not self._active.poll().done
            ):
                raise RuntimeError("cannot retire prepared work while its GPU execution is active")
            self.close()

    def _release_owned_bindings(self) -> None:
        bindings, self._owned_bindings = self._owned_bindings, ()
        for binding in bindings:
            graphs.forget_prepared(binding.model.module, self)
        self._binding_cleanup.close()

    def close(self) -> None:
        sequence = self._sequence_owner() if self._sequence_owner is not None else None
        if sequence is not None and not sequence._closing:
            sequence.close()
            return
        with self._submission:
            if self._closed:
                return
            settled_error: Exception | None = None
            if self._active is not None:
                active = self._active
                # A result waiter may hold _observation while waiting for completion.
                # Deliver cancellation first; it must never need that observer's lock.
                if active._finished is None:
                    active.cancel()
                try:
                    active._join()
                except Exception as error:
                    if active._finished is None:
                        raise  # No checked completion: preserve every graph/storage owner.
                    settled_error = error
            try:
                self._native.close_owner()
            except Exception as error:
                _native_error(error)
            self._closed = True
            self._release_owned_bindings()
            self._load = None
            if settled_error is not None:
                _native_error(settled_error)

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()

    def facts(self) -> dict[str, object]:
        """Facts from the actual native graph owner; a PreparedWork protocol is not evidence."""
        return {
            **self._native.facts(),
            "capture_cache": dict(self._capture_cache),
            "load_ordering_requested": self._load_ordering_requested,
            "load_ordering": self._load_ordering,
        }

    def submit(self, context: Context | None = None) -> Execution:
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise CapabilityError(
                "prepared submission during CUDA capture is not supported",
                code="prepared_capture_active",
            )
        if context is not None:
            context.raise_if_cancelled()
            if context.device.type != "cuda" or context.device.index not in (None, self.device):
                raise ValueError("prepared work does not belong to this request's CUDA device")
        with self._submission:
            if self._sequence_owner is not None and self._sequence_owner() is not None:
                raise RuntimeError("this prepared stage belongs to a PreparedSequence")
            self._validate()
            if self._load is not None and self._load_used:
                raise RuntimeError("a prepared native load is one-shot; prepare a new slot")
            if self._active is not None and not self._active.poll().done:
                raise RuntimeError("a prepared repeat is already running")
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise CapabilityError(
                        "prepared submission during CUDA capture is not supported",
                        code="prepared_capture_active",
                    )
                outputs = tuple(torch.empty_like(t) for t in self.state)
                readout = torch.cuda.Event()
            if context is not None:
                context.raise_if_cancelled()
            claimed: list[Any] = []
            try:
                for binding in self._owned_bindings:
                    if binding.model not in claimed:
                        binding.model._claim()
                        claimed.append(binding.model)
                producer = torch.cuda.current_stream(self.device).cuda_stream
                job = None
                if self._load is None:
                    epoch = self._native.launch(producer)
                else:
                    # Arm on this OS thread before a TensorFS worker takes over. Its native
                    # continuation owns the recorded caller-ready event and execution epoch.
                    capsule, epoch = self._native.continuation(producer)
                    if self._load_ordering == "events":
                        job = self._load.submit_native(capsule, event_ordered=True)
                    else:
                        job = self._load.submit_native(capsule)
                    self._load_used = True
                self._active = Execution(self, context, epoch, outputs, readout, claimed, job)
                self._active._offer_queued()
                return self._active
            except BaseException as error:
                if isinstance(error, Exception) and getattr(error, "eligibility_refusal", False):
                    for model in reversed(claimed):
                        model._release()
                    _native_error(error)  # native preflight rejected before any launch/mutation
                # A signal or error after launch must not lose an in-flight native owner.
                try:
                    self._native.close_owner()
                    self._closed = True
                    self._load = None
                    for model in reversed(claimed):
                        model._release()
                    self._release_owned_bindings()
                except Exception as cleanup:
                    if getattr(cleanup, "error_code", None) not in (None, 2):
                        _native_error(cleanup)
                    raise RuntimeFailure(
                        "prepared submission completion is unproven; retire executor",
                        code="device_fault",
                    ) from cleanup
                if isinstance(error, Exception):
                    _native_error(error)
                raise


class Execution:
    """Observer/control handle. No observation or CPU loop is needed for GPU continuation."""

    def __init__(
        self,
        plan: PreparedRepeat | PreparedSequence,
        context: Context | None,
        epoch: int,
        outputs: tuple[Any, ...],
        readout: Any,
        models: list[Any],
        load_job: Any = None,
    ) -> None:
        self.plan, self.context = plan, context
        self._finished: Progress | None = None
        self._epoch, self._outputs, self._readout = epoch, outputs, readout
        self._observation = threading.RLock()
        self._models = models
        self._load_job = load_job
        from cozy_runtime.internal.residency import _REQUEST

        self._load_request = _REQUEST.get()
        self._load_recorded = False
        self._failure: Exception | None = None
        self._queued_eligible = context is not None and load_job is None
        self._queued_offered: str | None = None

    def _offer_queued(self) -> None:
        # Scratch, outputs and resident weights are already retained. This optional
        # observer work never advances GPU execution or admits a streamed future workspace.
        if (
            self._queued_eligible
            and self._queued_offered is None
            and self._load_request is not None
        ):
            self._queued_offered = self._load_request.prefetch_queued()

    def facts(self) -> dict[str, object]:
        """Native execution and, when present, its actual owned input read."""
        facts = self.plan.facts()
        if self._load_job is not None:
            facts["native_load"] = dict(self._load_job.facts())
        return facts

    def completion_proven(self) -> bool:
        """Whether native work and readout/drain completed, independently of its result.

        A failed load sets _finished only after TensorFS certifies checked cleanup
        and this execution releases every model claim.
        poll() and wait() must still report that load's original error.
        """
        with self._observation:
            return self._finished is not None

    def _release_models(self) -> None:
        while self._models:
            self._models[-1]._release()
            self._models.pop()

    def _record_load(self, facts: Mapping[str, Any]) -> None:
        if self._load_recorded or not facts.get("done", False):
            return
        self._load_recorded = True
        if self._load_request is None or not facts.get("read_done", False):
            return
        for binding in self.plan._owned_bindings:
            if binding._load_slot is self.plan._load:
                for block, nbytes in zip(binding.groups, facts["input_read_bytes"], strict=True):
                    if nbytes:
                        # The native worker did this I/O. Record bytes once after completion;
                        # its separate native clocks are not Python read-wait measurements.
                        self._load_request.read(block, 0, int(nbytes))

    def _failed_load(self, error: Exception) -> None:
        """A native input/sequence job may prove cleanup; an exception is not that proof."""
        if self._load_job is None:
            return
        self._failure = error
        facts = self._load_job.facts()
        self._record_load(facts)
        if facts.get("completion_proven", False):
            # No readout of failed/partial input. Public observers retain the original error;
            # close may now retire the graph and its bindings after this native certificate.
            self._release_models()
            self._finished = Progress(0, bool(facts.get("canceled", False)), True)

    def _finish(self, completed: int, canceled: bool) -> Progress:
        import torch

        if self._finished is None:
            if self._load_job is not None:
                self._record_load(self._load_job.facts())
            with torch.cuda.device(self.plan.device), torch.no_grad():
                for output, state in zip(self._outputs, self.plan.state, strict=True):
                    output.copy_(state)
                self._readout.record(torch.cuda.current_stream(self.plan.device))
                self._readout.synchronize()
            self._release_models()
            self._finished = Progress(completed, canceled, True)
        return self._finished

    def poll(self) -> Progress:
        with self._observation:
            if self._failure is not None:
                _native_error(self._failure)
            if self._finished is not None:
                return self._finished
            try:
                completed, canceled, done = (
                    self.plan._native.poll(self._epoch)
                    if self._load_job is None
                    else self._load_job.poll()
                )
                if done:
                    completed, canceled = (
                        self.plan._native.wait(self._epoch)
                        if self._load_job is None
                        else self._load_job.wait()
                    )
            except Exception as error:
                self._failed_load(error)
                _native_error(error)
            # The input certificate covers producer/graph completion, not this readout.
            # Never reinterpret a copy, synchronization or model-release failure as a
            # clean input failure merely because the load worker finished successfully.
            if done:
                return self._finish(completed, canceled)
            # The one queued CPU preflight can finish after submit. Re-offer only while
            # native work remains active; latching an actual assignment bounds admission.
            self._offer_queued()
            return Progress(completed, canceled, False)

    def cancel(self) -> None:
        try:
            if self._load_job is None:
                self.plan._native.cancel(self._epoch)
            else:
                self._load_job.cancel()
        except Exception as error:
            _native_error(error)

    def _join(self) -> None:
        with self._observation:
            if self._finished is None:
                try:
                    completed, canceled = (
                        self.plan._native.wait(self._epoch)
                        if self._load_job is None
                        else self._load_job.wait()
                    )
                except Exception as error:
                    self._failed_load(error)
                    _native_error(error)
                self._finish(completed, canceled)

    def wait(self) -> tuple[Any, ...]:
        if self._failure is not None:
            _native_error(self._failure)
        if self.context is None:
            self._join()
        else:
            while not self.poll().done:
                if self.context.cancelled:
                    self.cancel()
                time.sleep(0.01)
            self.context.raise_if_cancelled()
        return self._outputs

    def close(self) -> None:
        with self.plan._submission:
            if self.plan._active is not self:
                raise RuntimeError("stale execution token")
            # Share the definition owner's join/readout path. In particular, don't race
            # another observer into two readouts or release the same model claim twice.
            self.plan.close()


class PreparedSequence:
    """Own ordered prepared stages; one submission advances them entirely in native code.

    Stages are distinct definitions on one CUDA device. Their boundary state is ordinary
    persistent GPU storage: a later stage may read a previous stage's state at the same
    address. Output values are copied only after the whole native sequence finishes.
    Preparation is still explicit Python work. The native worker may perform host-required
    I/O and checked CUDA waits; sequential execution is not a single GPU graph launch.

    Constructing this object transfers definition ownership: close/idle retirement of any
    stage closes the whole sequence. Use PreparedRepeat's explicit prefix/body/suffix
    capture instead when a package author chooses one shape-compatible composite graph.
    Neither choice is a consumer/request setting or an automatic fusion decision.

    `prefetch=False` is author execution-plan policy: omit speculative next-stage reads
    when other work has priority. Native demand execution and existing cached weights keep
    the same semantics. True permits bounded lookahead only when actual capacity admits it.
    """

    @classmethod
    def record(cls) -> _SequenceCapture:
        """Capture an ordered program in one private scratch pool, then finish ownership.

        Use `with PreparedSequence.record() as recording`, its capture/capture_stage
        methods, and recording.finish(outputs=..., repeat=...). Every GPU boundary must be
        allocated outside capture. Intermediate definitions cannot launch independently.
        """
        return _SequenceCapture()

    def __init__(
        self,
        stages: Sequence[PreparedRepeat],
        *,
        outputs: Sequence[Any] | None = None,
        repeat: tuple[int, int, int] | None = None,
        prefetch: bool = True,
    ) -> None:
        import torch

        try:
            import cozy_runtime._kernels._execution as native
        except ModuleNotFoundError as error:
            if error.name != "cozy_runtime._kernels._execution":
                raise
            raise CapabilityError(
                "native sequence module is unavailable", code="prepared_graph_ineligible"
            ) from error
        if not callable(getattr(native, "NativeSequence", None)):
            raise CapabilityError(
                "native sequence module is unavailable", code="prepared_graph_ineligible"
            )
        if type(prefetch) is not bool:
            raise TypeError("sequence prefetch policy must be a boolean")
        self.prefetch = prefetch
        self.stages = tuple(stages)
        if not self.stages or len(self.stages) > 65536:
            raise ValueError("a prepared sequence needs 1..65536 stages")
        if any(not isinstance(stage, PreparedRepeat) for stage in self.stages):
            raise TypeError("sequence stages must be PreparedRepeat definitions")
        if len({id(stage) for stage in self.stages}) != len(self.stages):
            raise ValueError("sequence definitions must be distinct; repeat inside a stage")
        if repeat is not None:
            if (
                not isinstance(repeat, tuple)
                or len(repeat) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in repeat)
            ):
                raise ValueError("repeat must be (first_stage, exclusive_end, positive_passes)")
            begin, end, passes = repeat
            if not 0 <= begin < end <= len(self.stages) or not 0 < passes <= 2**32 - 1:
                raise ValueError("invalid bounded repeat range")
            if (end - begin) * passes + len(self.stages) - (end - begin) > 2**32 - 1:
                raise ValueError("repeated stage count exceeds uint32")
            if passes > 1 and not getattr(native.ResidentRepeat, "supports_reusable_input", False):
                raise CapabilityError(
                    "native stage rearm is unavailable", code="prepared_graph_ineligible"
                )
        self._repeat = repeat
        self.device = self.stages[0].device
        if any(stage.device != self.device for stage in self.stages):
            raise ValueError("a native sequence belongs to one CUDA device")
        self.state = tuple(self.stages[-1].state if outputs is None else outputs)
        declared = {id(tensor) for stage in self.stages for tensor in stage.state}
        if not self.state or any(
            not isinstance(t, torch.Tensor) or id(t) not in declared for t in self.state
        ):
            raise ValueError("sequence outputs must be declared persistent stage state")
        self.steps = sum(stage.steps for stage in self.stages)
        if repeat is not None:
            begin, end, passes = repeat
            self.steps += (passes - 1) * sum(stage.steps for stage in self.stages[begin:end])
        self._submission = threading.RLock()
        self._active: _SequenceExecution | None = None
        self._native: Any = None
        self._capture_domain: Any = None
        # A rejected constructor owns no definitions. In particular its destructor must
        # not close a stage already transferred to another live sequence.
        self._closed = True
        self._closing = False
        self._load = None
        self._owned_bindings: tuple[Any, ...] = ()
        with self._locked_stages():
            for stage_index, stage in enumerate(self.stages):
                stage._validate()
                if stage._sequence_owner is not None and stage._sequence_owner() is not None:
                    raise RuntimeError("a prepared stage already belongs to another sequence")
                if stage._active is not None and not stage._active.poll().done:
                    raise RuntimeError("a running definition cannot enter a new sequence")
                if stage._load is not None and not callable(
                    getattr(stage._load, "continuation", None)
                ):
                    raise CapabilityError(
                        "this input loader cannot defer its native stage",
                        code="prepared_load_unavailable",
                    )
                if (
                    self._repeated(stage_index)
                    and stage._load is not None
                    and not getattr(stage._load, "supports_reusable_input", False)
                ):
                    raise CapabilityError(
                        "this input loader cannot rearm a native stage",
                        code="prepared_load_unavailable",
                    )
            owner = weakref.ref(self)
            for stage in self.stages:
                stage._sequence_owner = owner
            self._closed = False

    def _repeated(self, index: int) -> bool:
        return (
            self._repeat is not None
            and self._repeat[2] > 1
            and self._repeat[0] <= index < self._repeat[1]
        )

    def _locked_stages(self) -> ExitStack:
        locks = ExitStack()
        try:
            for stage in sorted(self.stages, key=id):
                locks.enter_context(stage._submission)
            return locks
        except BaseException:
            locks.close()
            raise

    def valid(self) -> bool:
        with self._submission, self._locked_stages():
            return (
                not self._closed
                and (self._active is None or self._active._failure is None)
                and all(stage.valid() for stage in self.stages)
            )

    def facts(self) -> dict[str, object]:
        stages = [stage.facts() for stage in self.stages]
        return {
            "sequence": {} if self._native is None else self._native.facts(),
            "stages": stages,
            "host_graph_launches": sum(int(stage["host_graph_launches"]) for stage in stages),
            "native_stage_advancement": True,
        }

    def submit(self, context: Context | None = None) -> Execution:
        import torch

        from cozy_runtime._kernels import _execution

        if not callable(getattr(_execution, "NativeSequence", None)):
            raise CapabilityError(
                "native sequence module is unavailable", code="prepared_graph_ineligible"
            )
        if torch.cuda.is_current_stream_capturing():
            raise CapabilityError(
                "sequence submission during capture is unsupported", code="prepared_capture_active"
            )
        if context is not None:
            context.raise_if_cancelled()
            if context.device.type != "cuda" or context.device.index not in (None, self.device):
                raise ValueError("prepared sequence does not belong to this request's CUDA device")
        with self._submission, self._locked_stages():
            if self._closed:
                raise RuntimeError("prepared sequence is closed")
            for stage in self.stages:
                stage._validate()
                if stage._active is not None and not stage._active.poll().done:
                    raise RuntimeError("a sequence stage is already running")
            with torch.cuda.device(self.device):
                outputs = tuple(torch.empty_like(value) for value in self.state)
                readout = torch.cuda.Event()
                producer = torch.cuda.current_stream(self.device).cuda_stream
            claimed: list[Any] = []
            capsules: list[Any] = []
            reads: list[tuple[int, Any, tuple[Any, ...]]] = []
            native = None
            execution = None
            phase_controllers: dict[Any, Any] = {}
            phase_definitions: dict[tuple[Any, tuple[str, ...]], tuple[str, tuple[Any, ...]]] = {}
            from cozy_runtime.internal.residency import _REQUEST

            request = _REQUEST.get()
            phase_type = getattr(varena, "PhaseController", None)
            try:
                # Every program model is claimed before model-wide phase identities are
                # snapshotted. Unrelated declared siblings are not decorated or borrowed.
                for stage in self.stages:
                    for binding in stage._owned_bindings:
                        if binding.model not in claimed:
                            binding.model._claim()
                            claimed.append(binding.model)
                for stage_index, stage in enumerate(self.stages):
                    stage._validate()  # claims now fence concurrent model identity changes
                    if self._repeated(stage_index):
                        capsule, _ = stage._native.continuation(producer, reusable=True)
                    else:
                        capsule, _ = stage._native.continuation(producer)
                    if stage._load is not None:
                        capsule, observation = stage._load.continuation(
                            capsule, event_ordered=stage._load_ordering == "events"
                        )
                        reads.append((stage_index, observation, stage._owned_bindings))
                    if callable(phase_type) and request is not None and stage._owned_bindings:
                        memories = {binding.model.memory for binding in stage._owned_bindings}
                        if len(memories) == 1:
                            memory = next(iter(memories))
                            with memory.lock:
                                exclusive = (
                                    request.memory is memory
                                    and not request.idle
                                    and not memory._active_beside(request)
                                )
                                names = tuple(
                                    sorted(
                                        set().union(
                                            *(
                                                request.phase(binding.model.name)
                                                for binding in stage._owned_bindings
                                            )
                                        )
                                    )
                                )
                                selected = [
                                    model
                                    for model in claimed
                                    if model.memory is memory and model.name in names
                                ]
                                if exclusive and memory.native is not None:
                                    definition = phase_definitions.get((memory, names))
                                    if definition is None:
                                        regions = tuple(
                                            {
                                                id(region): region
                                                for model in selected
                                                for block in model.groups()
                                                for region in (block.region, block.final_region)
                                                if region is not None
                                            }.values()
                                        )
                                        # Model source/fill/generation identity belongs to this
                                        # submission. Actual readiness is inspected natively
                                        # at phase entry, never guessed from this snapshot.
                                        identity = repr(
                                            tuple(
                                                sorted(
                                                    (
                                                        model.name,
                                                        model.wanted_identity(),
                                                        model.generation,
                                                    )
                                                    for model in selected
                                                )
                                            )
                                        )
                                        # Only the compact digest crosses the capability and
                                        # progress boundaries; complete immutable identities
                                        # can contain large per-weight route tables.
                                        digest = hashlib.sha256(identity.encode()).hexdigest()
                                        label = "+".join(
                                            sorted(model.component for model in selected)
                                        )[:128]
                                        key = f"{label}:{digest}"
                                        definition = key, regions
                                        phase_definitions[(memory, names)] = definition
                            if exclusive and memory.native is not None:
                                controller = phase_controllers.get(memory)
                                if controller is None:
                                    controller = phase_type(memory.native)
                                    phase_controllers[memory] = controller
                                capsule = controller.wrap(capsule, *definition)
                    capsules.append(capsule)
                native = (
                    _execution.NativeSequence(capsules, self.device)
                    if self._repeat is None
                    else _execution.NativeSequence(capsules, self.device, self._repeat)
                )
                execution = _SequenceExecution(
                    self, context, outputs, readout, claimed, native, reads, phase_controllers
                )
                self._native, self._active = native, execution
                for stage in self.stages:
                    stage._active = execution
                # Every observer/capsule is strongly owned before optional Weak lookahead
                # links are installed. Admission allocates no source bytes and launches no work.
                execution._prepare_lookahead()
                # All Python/native owners exist before the sole native worker can start.
                native.start()
                execution._offer_queued()
                return execution
            except BaseException as error:
                if native is not None:
                    try:
                        native.close_owner()
                    except Exception as cleanup:
                        raise RuntimeFailure(
                            "native sequence submission completion is unproven; retire executor",
                            code="device_fault",
                        ) from cleanup
                else:
                    # No sequence worker exists. Each capsule's provider owns abandonment of
                    # its still-armed epoch; closing definitions checks all caller streams.
                    self._closing = True
                    try:
                        for stage in self.stages:
                            stage.close()
                        self._closed = True
                    except Exception as cleanup:
                        raise RuntimeFailure(
                            "native sequence preparation cleanup is unproven; retire executor",
                            code="device_fault",
                        ) from cleanup
                    finally:
                        self._closing = False
                if execution is not None:
                    execution._release_models()
                else:
                    while claimed:
                        claimed[-1]._release()
                        claimed.pop()
                if isinstance(error, Exception):
                    _native_error(error)
                raise

    def _retire_if_idle(self) -> None:
        with self._submission:
            if (
                self._active is not None
                and self._active._finished is None
                and not self._active.poll().done
            ):
                raise RuntimeError("cannot retire a sequence while GPU execution is active")
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        with self._submission, self._locked_stages():
            if self._closed:
                return
            error = None
            if self._active is not None:
                if self._active._finished is None:
                    self._active.cancel()
                try:
                    self._active._join()
                except Exception as failure:
                    if self._active._finished is None:
                        raise
                    error = failure
            if self._native is not None:
                try:
                    self._native.close_owner()
                except Exception as failure:
                    _native_error(failure)
            self._closing = True
            try:
                for stage in self.stages:
                    stage.close()
                if self._capture_domain is not None:
                    self._capture_domain.finish()
                    self._capture_domain = None
                self._closed = True
            finally:
                self._closing = False
            if error is not None:
                _native_error(error)

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


class _SequenceCapture:
    """Preparation-only ownership; uses the existing sequence for every execution."""

    def __init__(self) -> None:
        self.stages: list[PreparedRepeat] = []
        self._domain: Any = None
        self._pool: Any = None
        self._areas: list[tuple[Any, int, int]] = []
        self._submission = threading.RLock()
        self._closing = False
        self._entered = self._finished = self._closed = self._capturing = False

    def __enter__(self) -> _SequenceCapture:
        if self._entered or self._closed:
            raise RuntimeError("sequence recording is one-shot")
        self._entered = True
        return self

    def __exit__(self, *_: Any) -> None:
        if not self._finished:
            self.close()

    def _external(self, tensor: Any) -> None:
        storage = tensor.untyped_storage()
        pointer, size = int(storage.data_ptr()), int(storage.nbytes())
        if size and any(
            tensor.device == device and pointer < end and start < pointer + size
            for device, start, end in self._areas
        ):
            raise CapabilityError(
                "sequence boundaries must not alias captured temporary storage",
                code="prepared_graph_ineligible",
            )

    def _prepare(self, state: tuple[Any, ...]) -> None:
        import torch

        if not self._entered or self._closed or self._finished or not state:
            raise RuntimeError("capture inside an open sequence recording")
        for tensor in state:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                raise ValueError("sequence state must contain CUDA tensors")
            self._external(tensor)
        device = state[0].device.index
        if device is None or any(t.device.index != device for t in state):
            raise ValueError("sequence capture belongs to one device")
        if self._domain is None:
            self._domain = domain.Domain(torch, None)
            if self._domain.device != device:
                raise ValueError("sequence capture must use the current request device")
            try:
                self._domain.initialize()
                self._domain.prime_blas()
            except Building as error:
                raise CapabilityError(
                    f"capture waits for kernel readiness: {error}",
                    code="prepared_graph_ineligible",
                ) from error
            self._pool = domain.Pool(torch.cuda.graph_pool_handle())
        elif self._domain.device != device:
            raise ValueError("sequence captures cannot span devices")
        if why := self._domain.refusal():
            raise CapabilityError(why, code="prepared_graph_ineligible")

    def _capture(self, method: Any, step: Any, **parameters: Any) -> PreparedRepeat:
        with self._submission:
            if _CAPTURE_GROUP.get() is not None:
                raise RuntimeError("nested sequence capture is unsupported")
            if self._capturing:
                raise RuntimeError("nested sequence preparation is unsupported")
            self._capturing = True
            token = _CAPTURE_GROUP.set(self)
            try:
                plan = method(step, **parameters)
                if self._closed or self._domain is None or self._domain.closed:
                    plan.close()
                    raise RuntimeError("sequence recording retired during preparation")
                with plan._submission:
                    plan._sequence_owner = weakref.ref(self)
                    self.stages.append(plan)
                return plan
            finally:
                _CAPTURE_GROUP.reset(token)
                self._capturing = False

    def capture(self, step: Any, **parameters: Any) -> PreparedRepeat:
        return self._capture(PreparedRepeat.capture, step, **parameters)

    def capture_stage(self, step: Any, **parameters: Any) -> PreparedRepeat:
        return self._capture(PreparedRepeat.capture_stage, step, **parameters)

    def finish(
        self, *, outputs: Sequence[Any] | None = None, repeat: tuple[int, int, int] | None = None
    ) -> PreparedSequence:
        with self._submission, ExitStack() as locks:
            if not self._entered or self._closed or self._finished:
                raise RuntimeError("sequence recording is not open")
            for stage in sorted(self.stages, key=id):
                locks.enter_context(stage._submission)
            owner = weakref.ref(self)
            for stage in self.stages:
                if stage._sequence_owner is None or stage._sequence_owner() is not self:
                    raise RuntimeError("sequence recording lost a captured definition")
                stage._sequence_owner = None
            try:
                sequence = PreparedSequence(self.stages, outputs=outputs, repeat=repeat)
            except BaseException:
                for stage in self.stages:
                    stage._sequence_owner = owner
                raise
            sequence._capture_domain = self._domain
            self._domain = None
            self._finished = True
            return sequence

    def _retire_if_idle(self) -> None:
        if self._capturing:
            raise RuntimeError("cannot retire a sequence during stage preparation")
        self.close()

    def close(self) -> None:
        with self._submission:
            if self._closed or self._finished:
                return
            if self._capturing:
                raise RuntimeError("cannot close a sequence during stage preparation")
            self._closing = True
            try:
                for stage in self.stages:
                    stage.close()
                if self._domain is not None:
                    self._domain.finish()
                self._closed = True
            finally:
                self._closing = False

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


class _SequenceExecution(Execution):
    def __init__(self, plan, context, outputs, readout, models, native, reads, phases=None):
        super().__init__(plan, context, 0, outputs, readout, models, native)
        self._stage_reads = reads
        # Only all-resident sequences have already retained every future stage workspace.
        # Source-backed stages keep their existing separate current-stage lookahead policy.
        self._queued_eligible = context is not None and not reads
        self._phase_controllers = phases or {}
        self._prefetch_cleanup = ExitStack()
        self._prefetch_admission: list[dict[str, object]] = []

    def _prepare_lookahead(self) -> None:
        if not self.plan.prefetch:
            self._prefetch_admission.append({"state": "disabled_by_author"})
            return
        request = self._load_request
        candidates: dict[Any, list[tuple[Any, ...]]] = {}
        for current, following in zip(self._stage_reads, self._stage_reads[1:], strict=False):
            index, job, bindings = current
            next_index, next_job, next_bindings = following
            if next_index != index + 1:
                continue  # no resident gaps or repeat wraparound in the first slice
            repeat = self.plan._repeat
            if repeat is not None and repeat[2] > 1 and next_index == repeat[1]:
                continue  # the suffix is next only on the final pass
            if (
                not bindings
                or not next_bindings
                or not getattr(job, "supports_native_lookahead", False)
                or not getattr(next_job, "supports_native_lookahead", False)
                or not callable(getattr(job, "link_next", None))
            ):
                continue
            memories = {binding.model.memory for binding in (*bindings, *next_bindings)}
            if len(memories) != 1:
                continue
            memory = next(iter(memories))
            regions = {
                id(region): region
                for binding in (*bindings, *next_bindings)
                for _, region in binding._stage_regions
            }
            current_bytes = sum(
                int(region.allocated_bytes)
                for binding in bindings
                for _, region in binding._stage_regions
            )
            next_bytes = sum(
                int(region.allocated_bytes)
                for binding in next_bindings
                for _, region in binding._stage_regions
            )
            # Native reports the same actual reservations. Refuse skew or hidden inputs;
            # do not use Python logical tensor sizes as allocation admission.
            if (
                not regions
                or int(job.reservation_bytes) != current_bytes
                or int(next_job.reservation_bytes) != next_bytes
            ):
                continue
            required = sum(int(region.allocated_bytes) for region in regions.values())
            missing = sum(
                int(region.allocated_bytes) for region in regions.values() if not region.resident
            )
            candidates.setdefault(memory, []).append(
                (index, next_index, job, next_job, required, missing, next_bytes)
            )
        for memory, pairs in candidates.items():
            required = max(pair[4] for pair in pairs)
            missing = max(pair[5] for pair in pairs)
            restore = memory._stage_prefetch_window(required, missing, request, self)
            if restore is None:
                self._prefetch_admission.append({"state": "capacity_skip", "pair_bytes": required})
                continue
            self._prefetch_cleanup.callback(restore)
            for index, next_index, job, next_job, pair_bytes, _, next_bytes in pairs:
                linked = bool(job.link_next(next_job, max_prefetch_bytes=next_bytes))
                self._prefetch_admission.append(
                    {
                        "from_stage": index,
                        "to_stage": next_index,
                        "pair_bytes": pair_bytes,
                        "state": "linked" if linked else "native_refusal",
                    }
                )

    def _release_models(self) -> None:
        # Base execution calls this only after checked native completion/readout. If the
        # drain is unproven, owners and this window lease remain retained for retirement.
        self._prefetch_cleanup.close()
        # Native phase entry changed preferences without a Python callback. After every
        # source/prefetch/consumer is checked, invalidate only our stale eager-policy cache;
        # the next ordinary phase recomputes it. Preserve another live request's policy.
        for memory in self._phase_controllers:
            with memory.lock:
                request = self._load_request
                if (
                    request is not None
                    and not memory._active_beside(request)
                    and memory._phase_request == request.id
                ):
                    memory._phase_names = frozenset()
                    memory._phase_request = None
        super()._release_models()

    def facts(self) -> dict[str, object]:
        facts = self.plan.facts()
        facts["sequence"] = dict(self._load_job.facts())
        facts["stage_prefetch_admission"] = list(self._prefetch_admission)
        facts["native_phases"] = [
            dict(controller.facts()) for controller in self._phase_controllers.values()
        ]
        facts["stage_inputs"] = [
            {"stage": index, **dict(job.facts())} for index, job, _ in self._stage_reads
        ]
        return facts

    def _record_load(self, facts: Mapping[str, Any]) -> None:
        if self._load_recorded or not facts.get("done", False):
            return
        for _, job, bindings in self._stage_reads:
            loaded = job.facts()
            if self._load_request is not None and (
                loaded.get("read_done", False) or getattr(job, "supports_native_lookahead", False)
            ):
                # Lookahead jobs expose complete padded ledgers after aggregate done,
                # including prefetched bytes when demand was canceled before starting.
                blocks = [block for binding in bindings for block in binding.groups]
                if blocks:
                    prefetched = loaded.get("prefetch_input_read_bytes", [0] * len(blocks))
                    for block, nbytes, speculative in zip(
                        blocks, loaded["input_read_bytes"], prefetched, strict=True
                    ):
                        nbytes, speculative = int(nbytes), int(speculative)
                        if not 0 <= speculative <= nbytes:
                            raise RuntimeError(
                                "native prefetch byte accounting exceeds total input reads"
                            )
                        if nbytes > speculative:
                            self._load_request.read(block, 0, nbytes - speculative)
                        if speculative:
                            self._load_request.read(block, 0, speculative, speculative=True)
        self._load_recorded = True
