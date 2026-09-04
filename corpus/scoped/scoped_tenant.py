"""cr-100's fixture: one model, ONE COMPONENT-USE SCOPE PER COMPONENT.

`corpus/tenant` is the same synthetic checkpoint under a SINGLE whole-model scope, which is
what a lane arbitrating between two tenants needs (cr-022) — and it is exactly why it cannot
observe cr-100: a generation with one scope over everything ends fully resident under either
placement rung, so `all_resident` and `component_staged` are the same picture. Three scopes
in a row is the shape of a real pipeline (encode, denoise, decode) and the shape a demotion
is VISIBLE in: the staged rung frees what each scope does not declare and stages it back for
the next one.

It is a separate package rather than a second entrypoint on the tenant because two
entrypoints binding one model are two CONSTRUCTIONS — the model slot path is inside the
construction key — and one executor owns exactly one prepared generation.

The workspace each scope takes HALVES until it fits. That models the mechanism decisions row
698 names: cuBLAS/cuDNN workspace selection expands into the room the allocator can give, so
the same model at the same shape peaked at 3.91 GiB alone on a 15.6 GiB card and 1.92 GiB
with a co-tenant beside it. A fixture whose demand is a constant cannot express the thing
the planner has to get right. The package is not allowed to ask how much room there is
(residency is the runtime's and `mem_get_info` is not reachable from package code), which is
also how a real kernel finds out: it asks for what it wants and takes less when the answer
is no.
"""

from __future__ import annotations

from typing import Any

import msgspec

from cozy_runtime.author import App, Config, Context, Loader, Model, Telemetry, uses_components

app = App()


class SweepInput(msgspec.Struct, forbid_unknown_fields=True):
    seed: int = 7
    workspace_mib: int = 2048
    """How much scratch the kernels would LIKE. Not an axis, so every request for this
    entrypoint normalizes to the same shape cell however the appetite is set."""


class SweepOutput(msgspec.Struct):
    checksum: float
    workspace_bytes: int


class TenantPipeline:
    """The three component roots, shaped by the artifact's config."""

    def __init__(self, config: Config) -> None:
        import torch
        from torch import nn

        class Weight(nn.Module):
            def __init__(self, size: int, dtype: Any) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(size, dtype=dtype), requires_grad=False)

        class Attention(nn.Module):
            def __init__(self, hidden: int, norm_q: int) -> None:
                super().__init__()
                self.to_q = nn.Linear(hidden, hidden, bias=False)
                self.to_out = nn.ModuleList([nn.Linear(hidden, hidden, bias=False)])
                self.norm_q = Weight(norm_q, torch.bfloat16)

        class Projection(nn.Module):
            def __init__(self, hidden: int, ff: int) -> None:
                super().__init__()
                self.proj = nn.Linear(ff, hidden, bias=False)

        class FeedForward(nn.Module):
            def __init__(self, hidden: int, ff: int) -> None:
                super().__init__()
                self.net = nn.ModuleList([Projection(hidden, ff)])

        class Block(nn.Module):
            def __init__(self, hidden: int, ff: int, norm_q: int) -> None:
                super().__init__()
                self.attn1 = Attention(hidden, norm_q)
                self.ff = FeedForward(hidden, ff)
                self.norm1 = Weight(hidden, torch.bfloat16)

        class Unet(nn.Module):
            def __init__(self, blocks: int, hidden: int, ff: int, norm_q: int) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([Block(hidden, ff, norm_q) for _ in range(blocks)])

        class TextEncoder(nn.Module):
            def __init__(self, embed: int, norm: int) -> None:
                super().__init__()
                self.embeddings = nn.Embedding(embed, embed)
                self.final_norm = Weight(norm, torch.float32)

        class Decoder(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.conv_in = nn.Linear(width, width, bias=False)
                self.conv_out = nn.Linear(width, width, bias=False)

        class Vae(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.decoder = Decoder(width)

        unet = Unet(
            config.as_int("blocks"),
            config.as_int("hidden"),
            config.as_int("ff"),
            config.as_int("norm_q"),
        ).to(torch.bfloat16)
        text_encoder = TextEncoder(config.as_int("text_embed"), config.as_int("text_norm"))
        text_encoder.embeddings.to(torch.bfloat16)
        self.components: dict[str, Any] = {
            "text_encoder": text_encoder,
            "unet": unet,
            "vae": Vae(config.as_int("vae")),
        }


def build_pipeline(config: Config) -> TenantPipeline:
    return TenantPipeline(config)


class ScopedModel(Model[TenantPipeline]):
    pipe: TenantPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(TenantPipeline, factory=build_pipeline)

    def _read(self, name: str, seed: int) -> Any:
        """One element of every parameter of one component, through the guarded root."""
        import torch

        with torch.inference_mode():
            total = torch.zeros((), dtype=torch.float32, device="cuda")
            for parameter in self.pipe.components[name].parameters():
                value = parameter.flatten()[seed % parameter.numel()].float()
                # The synthetic weights are NOISE: any bit pattern, NaN and inf among them.
                total = total + torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return total

    def _workspace(self, want: int) -> tuple[Any, int]:
        """Take the biggest workspace that fits, HALVING until one does."""
        import torch

        size = max(int(want), 16 << 20)
        while size > (16 << 20):
            try:
                return torch.empty(size, dtype=torch.uint8, device="cuda"), size
            except torch.cuda.OutOfMemoryError:
                size //= 2
        return torch.empty(16 << 20, dtype=torch.uint8, device="cuda"), 16 << 20

    @uses_components("text_encoder")
    def encode(self, seed: int, want: int) -> tuple[Any, int]:
        scratch, size = self._workspace(want)
        total = self._read("text_encoder", seed)
        del scratch
        return total, size

    @uses_components("unet")
    def denoise(self, seed: int, want: int) -> Any:
        scratch, _size = self._workspace(want)
        total = self._read("unet", seed)
        del scratch
        return total

    @uses_components("vae")
    def decode(self, seed: int, want: int) -> Any:
        scratch, _size = self._workspace(want)
        total = self._read("vae", seed)
        del scratch
        return total


@app.entrypoint
def sweep(ctx: Context, payload: SweepInput, model: ScopedModel, tel: Telemetry) -> SweepOutput:
    """Three scopes in a row, each declaring ONE component."""
    import torch

    view = model.for_request(ctx, seed=payload.seed)
    want = max(int(payload.workspace_mib), 0) << 20
    encoded, size = model.encode(view._seed, want)
    denoised = model.denoise(view._seed, want)
    decoded = model.decode(view._seed, want)
    with torch.inference_mode():
        checksum = float(torch.tanh(encoded + denoised + decoded))
    tel.metric("checksum", checksum)
    tel.metric("workspace_bytes", size)
    return SweepOutput(checksum=checksum, workspace_bytes=size)
