"""Exact-timestep AdaLN tables for the official Diffusers MiniMax-H3 DiT.

The extension deliberately inherits Diffusers' forward unchanged. It replaces only the
time/AdaLN modules whose checkpoint-specific outputs were precomputed at the timestep rows
labelled by the checkpoint before those dynamic modules were pruned. The remaining
transformer graph, including attention, feed-forward, RoPE, input/output projections,
and packed-row indexing, stays upstream code.

A turbo overlay (`turbo.py`) attaches beside the DiT and is served per forward: the forward
that names its bank in `attention_kwargs` reads the overlay's tables, heads and low-rank
factors through the same modules; every other forward reads the base tables it always did.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

import torch
from diffusers import MiniMaxH3Transformer3DModel
from torch import nn
from torch.nn import functional as F

from cozy_runtime.author import ConformanceError

from .turbo import (
    ATTENTION_KWARG,
    LORA_FAMILIES,
    OVERLAY_KWARG,
    TURBO_BANK,
    TurboArming,
    TurboOverlay,
    _LoRAHook,
)


class _Armed:
    """One module's turbo source for the forward in flight; a plain object, so placing it
    on a module registers nothing and the DiT's state dict is what it was."""

    __slots__ = ("source", "step")

    def __init__(self, source: Any, step: int = 0) -> None:
        self.source = source
        self.step = step


class _AdaLNPrunedTimestepLookup(nn.Module):  # type: ignore[misc]
    """Map exact float32 timestep values to the checkpoint's labelled rows."""

    def __init__(self, timesteps: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer(
            "timesteps", torch.tensor(tuple(timesteps), dtype=torch.float32), persistent=False
        )
        self._armed: _Armed | None = None

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.rows(timestep)

    def rows(self, timestep: torch.Tensor) -> torch.Tensor:
        if self._armed is not None:
            return cast(torch.Tensor, self._armed.source.rows(timestep))
        values = timestep.to(dtype=torch.float32).reshape(-1, 1)
        matches = values == self.timesteps.reshape(1, -1)
        counts = matches.sum(dim=1)
        if not bool(torch.all(counts == 1)):
            unknown = timestep.detach().float().cpu().tolist()
            raise ConformanceError(
                f"model modulation tables do not cover the requested timesteps: {unknown}; "
                "select a covered step count or regenerate the tables for this schedule",
                code="artifact_config",
            )
        return matches.to(dtype=torch.int64).argmax(dim=1)


class _AdaLNPrunedTimestepRows(nn.Module):  # type: ignore[misc]
    """Keep inherited ``forward`` intact while replacing the removed timestep MLP."""

    def __init__(self) -> None:
        super().__init__()
        # The inherited Diffusers forward calls `get_parameter_dtype(time_embedder)` before
        # this identity. Its helper raises `UnboundLocalError` for a module with no tensor,
        # and a floating anchor would cast these exact integer row ids. This zero-length
        # int64 buffer is therefore the smallest state that preserves the upstream call.
        self.register_buffer("row_dtype", torch.empty(0, dtype=torch.int64), persistent=False)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return rows


class _AdaLNPrunedBlockTable(nn.Module):  # type: ignore[misc]
    """Sparse exact table with the dense local shape Diffusers blocks already consume."""

    def __init__(
        self,
        *,
        hidden_size: int,
        timestep_count: int,
        keys: Sequence[tuple[int, int]],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.table = nn.Parameter(torch.empty(len(keys), 6, hidden_size))
        index = torch.full((timestep_count, 3), -1, dtype=torch.int64)
        for table_row, (timestep_row, modality_tag) in enumerate(keys):
            index[timestep_row, modality_tag] = table_row
        self.register_buffer("table_index", index, persistent=False)
        self._armed: _Armed | None = None

    def forward(self, timestep_rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        source = self if self._armed is None else self._armed.source
        indices = source.table_index.index_select(0, timestep_rows).reshape(-1)
        # The transformer's pre-hook proves every selected pair is present. Missing cells
        # are never observed by Diffusers' existing `adaln_indices`; row zero is only the
        # placeholder needed to preserve that dense local indexing convention.
        rows = source.table.index_select(0, indices.clamp_min(0))
        return cast(
            "tuple[torch.Tensor, ...]",
            rows.reshape(-1, 6 * self.hidden_size).chunk(6, dim=-1),
        )


class _AdaLNPrunedOutputTable(nn.Module):  # type: ignore[misc]
    """Official final RMSNorm with exact precomputed shift/scale rows."""

    def __init__(self, norm: nn.Module, *, hidden_size: int, timestep_count: int) -> None:
        super().__init__()
        self.norm = norm
        self.table = nn.Parameter(torch.empty(timestep_count, 2, hidden_size))
        self._armed: _Armed | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_rows: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        table = self.table if self._armed is None else self._armed.source.table
        shift, scale = table.index_select(0, timestep_rows).unbind(dim=1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class _SelectableHead(nn.Linear):  # type: ignore[misc]
    """The official output head, or the turbo evaluation's collapsed head when armed."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)
        self._armed: _Armed | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._armed is None:
            return F.linear(hidden_states, self.weight, self.bias)
        return cast(torch.Tensor, self._armed.source(hidden_states, self._armed.step))


def _projection_base(module: object) -> object:
    """Public PEFT wrappers keep the model-owned selectable head as their base layer."""
    get_base = getattr(module, "get_base_layer", None)
    return get_base() if callable(get_base) else module


class AdaLNPrunedMiniMaxH3Transformer(MiniMaxH3Transformer3DModel):  # type: ignore[misc]
    """Official MiniMax-H3 transformer with only its exact modulation source replaced."""

    _keep_in_fp32_modules: ClassVar[list[str]] = [
        *MiniMaxH3Transformer3DModel._keep_in_fp32_modules,
        "time_proj",
    ]

    def __init__(
        self,
        *,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.time_proj = _AdaLNPrunedTimestepLookup(table_timesteps)
        self.time_embedder = _AdaLNPrunedTimestepRows()
        for block in self.transformer_blocks:
            block.adaln_proj = _AdaLNPrunedBlockTable(
                hidden_size=int(self.config.hidden_size),
                timestep_count=len(table_timesteps),
                keys=table_block_keys,
            )
        norm_out = cast(Any, self.norm_out)  # type: ignore[has-type]
        self.norm_out = _AdaLNPrunedOutputTable(
            norm_out.norm,
            hidden_size=int(self.config.hidden_size),
            timestep_count=len(table_timesteps),
        )
        for name in ("proj_out", "audio_proj_out"):
            head = cast(nn.Linear, getattr(self, name))
            setattr(self, name, _SelectableHead(head.in_features, head.out_features))
        self._arming: TurboArming | None = None
        self._lora_hooks_installed = False
        self.register_forward_pre_hook(self._arm, with_kwargs=True)
        self.register_forward_hook(self._disarm, always_call=True)

    def install_lora_consumers(self) -> None:
        """Install once during turbo-capable construction, before Runtime fill/fusion."""
        if self._lora_hooks_installed:
            return
        for path, site in self.named_modules():
            if path.startswith(
                ("token_refiner.refiner_blocks.", "transformer_blocks.")
            ) and path.endswith(LORA_FAMILIES):
                site.register_forward_hook(_LoRAHook(self, path))
        self._lora_hooks_installed = True

    def _arm(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        del module
        self._release()
        names = (
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "timestep",
            "timestep_indices",
            "token_tags",
        )
        values = {**dict(zip(names, args, strict=False)), **kwargs}
        timestep = values.get("timestep")
        timestep_indices = values.get("timestep_indices")
        token_tags = values.get("token_tags")
        selector = values.get("attention_kwargs")
        bank = selector.get(ATTENTION_KWARG) if isinstance(selector, Mapping) else None
        if not all(
            isinstance(value, torch.Tensor) for value in (timestep, timestep_indices, token_tags)
        ):
            if bank is not None:
                raise ConformanceError(
                    "a turbo forward needs typed timestep, timestep_indices and token_tags",
                    code="artifact_config",
                )
            return None  # Diffusers owns its ordinary argument diagnostics.
        assert isinstance(timestep, torch.Tensor)
        assert isinstance(timestep_indices, torch.Tensor)
        assert isinstance(token_tags, torch.Tensor)
        if bool(torch.any((token_tags < 0) | (token_tags > 2))):
            raise ConformanceError(
                "packed rows contain a modality tag outside the canonical 0/1/2 set",
                code="artifact_config",
            )
        tables: Any = self
        try:
            if bank is not None:
                overlay = selector.get(OVERLAY_KWARG) if isinstance(selector, Mapping) else None
                if (
                    bank != TURBO_BANK
                    or not isinstance(overlay, TurboOverlay)
                    or not self._lora_hooks_installed
                ):
                    raise ConformanceError(
                        f"bank {bank!r} requires a turbo-capable base and prepared "
                        "overlay component",
                        code="artifact_config",
                        fields=["attention_kwargs", ATTENTION_KWARG],
                    )
                self._engage(overlay, overlay.schedule.step(timestep))
                tables = overlay
            global_rows = self.time_proj.rows(timestep)
            row_timesteps = global_rows.index_select(0, timestep_indices)
            table_index = tables.transformer_blocks[0].adaln_proj.table_index
            present = table_index[row_timesteps, token_tags] >= 0
            if not bool(torch.all(present)):
                raise ConformanceError(
                    "model modulation tables do not cover a requested reference timestep/modality; "
                    "regenerate the tables with coverage for this reference type",
                    code="artifact_config",
                )
        except Exception:
            self._release()
            raise
        if isinstance(selector, Mapping) and OVERLAY_KWARG in selector:
            # The package consumes these keys; Diffusers attention processors receive
            # only their own options. The caller's dictionary is never mutated.
            cleaned = {
                key: value
                for key, value in selector.items()
                if key not in (ATTENTION_KWARG, OVERLAY_KWARG)
            }
            return args, {**kwargs, "attention_kwargs": cleaned}
        return None

    def _engage(self, overlay: TurboOverlay, step: int) -> None:
        self._arming = TurboArming(overlay, step)
        self.time_proj._armed = _Armed(overlay)
        for block, source in zip(self.transformer_blocks, overlay.transformer_blocks, strict=True):
            block.adaln_proj._armed = _Armed(source.adaln_proj)
        self.norm_out._armed = _Armed(overlay.norm_out)
        for name in ("proj_out", "audio_proj_out"):
            head = _projection_base(getattr(self, name))
            if not isinstance(head, _SelectableHead):
                raise ConformanceError(
                    f"{name} is {type(head).__name__}, which cannot serve a turbo head",
                    code="artifact_config",
                )
            head._armed = _Armed(getattr(overlay, name), step)

    def _release(self) -> None:
        """Every forward starts disarmed, whatever the previous one left behind."""
        self._arming = None
        self.time_proj._armed = None
        for block in self.transformer_blocks:
            block.adaln_proj._armed = None
        self.norm_out._armed = None
        for name in ("proj_out", "audio_proj_out"):
            head = _projection_base(getattr(self, name))
            if isinstance(head, _SelectableHead):
                head._armed = None

    def _disarm(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        del module, args, output
        self._release()

    @classmethod
    def from_official_config(
        cls,
        config: Mapping[str, Any],
        *,
        table_timesteps: Sequence[float],
        table_block_keys: Sequence[tuple[int, int]],
    ) -> AdaLNPrunedMiniMaxH3Transformer:
        """Construct from the same upstream config accepted by the official class."""
        parameters = inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters
        kwargs = {name: value for name, value in config.items() if name in parameters}
        return cls(
            table_timesteps=table_timesteps,
            table_block_keys=table_block_keys,
            **kwargs,
        )
