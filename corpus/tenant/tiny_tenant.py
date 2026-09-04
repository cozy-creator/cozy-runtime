"""A model-bearing TENANT package for the shared-lane proofs (cr-022).

The model is the synthetic checkpoint `tfs checkpoint write --plain` produces (tfs-015):
`unet` blocks of attention/feed-forward linears in bf16, a `text_encoder` embedding, a
`vae` pair of f32 linears. It computes nothing worth keeping; what it does is occupy real
device memory through the real fill plane, so a lane holding two of it has a real decision
to make. Sizes come from the artifact's own config, which the fixture writer derives from
the header — the package names no shape of its own.

Two entrypoints, because a placement rung is only observable when the scopes DIFFER.
`touch` opens one component-use scope over EVERY component: the scope's demand is the whole
model, which is exactly the demand a lane must arbitrate between two tenants that do not
both fit (cr-022). A single scope is also why `touch` cannot see cr-100 at all — it ends
fully resident under either rung, so the staged rung and the resident one are the same
picture. `sweep` opens one scope PER COMPONENT, which is the shape of a real pipeline
(encode, denoise, decode) and the shape a demotion is visible in: under `component_staged`
each scope frees what it does not declare, so a generation that entered fully resident
leaves holding one component.
"""

from __future__ import annotations

from typing import Any

import msgspec

from cozy_runtime.author import App, Config, Context, Loader, Model, Telemetry, uses_components

app = App()


class TouchInput(msgspec.Struct, forbid_unknown_fields=True):
    seed: int = 7


class SweepInput(msgspec.Struct, forbid_unknown_fields=True):
    seed: int = 7
    workspace_mib: int = 2048
    """How much scratch the kernels would LIKE. Not an axis, so every request for this
    entrypoint normalizes to the same shape cell however the appetite is set."""


class TouchOutput(msgspec.Struct):
    checksum: float
    components: list[str]


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


def build_tenant(config: Config) -> TenantPipeline:
    return TenantPipeline(config)


class TenantModel(Model[TenantPipeline]):
    pipe: TenantPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(TenantPipeline, factory=build_tenant)

    @uses_components("text_encoder", "unet", "vae")
    def touch(self, seed: int) -> float:
        """Read one element of every component through the guarded roots: the scope
        admits the WHOLE model, so a tenant that is not resident is staged here."""
        import torch

        with torch.inference_mode():
            total = torch.zeros((), dtype=torch.float32, device="cuda")
            for member in self.pipe.components.values():
                for parameter in member.parameters():
                    value = parameter.flatten()[seed % parameter.numel()].float()
                    # The synthetic weights are NOISE: any bit pattern, NaN and inf among
                    # them. The checksum is bounded and finite so the metric can be spelled.
                    total = total + torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return float(torch.tanh(total))

    def _read(self, name: str, seed: int) -> Any:
        """One element of every parameter of one component, through the guarded root."""
        import torch

        with torch.inference_mode():
            total = torch.zeros((), dtype=torch.float32, device="cuda")
            for parameter in self.pipe.components[name].parameters():
                value = parameter.flatten()[seed % parameter.numel()].float()
                total = total + torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return total

    def _workspace(self, want: int) -> tuple[Any, int]:
        """Take the biggest workspace that fits, HALVING until one does.

        This is the fixture's whole reason to exist. Real kernels do this: cuBLAS and cuDNN
        select a workspace from what the allocator can give them, so the same model at the
        same shape peaked at 3.91 GiB alone on a 15.6 GiB card and 1.92 GiB with a co-tenant
        beside it — reproducibly, same bits out (decisions 698, cr-097). A peak measured on
        a roomier card is therefore an upper bound the kernels CHOSE, and a fixture whose
        demand is a constant cannot express the thing the planner has to get right. The
        package is not allowed to ask how much room there is (the runtime owns residency
        and `mem_get_info` is not reachable from here), which is also how a real kernel
        finds out: it asks for what it wants and takes less when the answer is no.
        """
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
def touch(ctx: Context, payload: TouchInput, model: TenantModel, tel: Telemetry) -> TouchOutput:
    view = model.for_request(ctx, seed=payload.seed)
    checksum = model.touch(view._seed)
    tel.metric("checksum", checksum)
    return TouchOutput(checksum=checksum, components=sorted(model.pipe.components))


@app.entrypoint
def sweep(ctx: Context, payload: SweepInput, model: TenantModel, tel: Telemetry) -> SweepOutput:
    """Three scopes in a row, the shape of a pipeline: encode, denoise, decode.

    Each scope declares ONE component, so `all_resident` and `component_staged` are
    genuinely different plans over the same weights — the resident rung runs all three
    scopes against the set the fill left, the staged rung frees what each scope does not
    declare and stages it back for the next one. That difference is the whole subject of
    cr-100 and is invisible to a package with a single whole-model scope.
    """
    view = model.for_request(ctx, seed=payload.seed)
    want = max(int(payload.workspace_mib), 0) << 20
    encoded, size = model.encode(view._seed, want)
    denoised = model.denoise(view._seed, want)
    decoded = model.decode(view._seed, want)
    import torch

    with torch.inference_mode():
        checksum = float(torch.tanh(encoded + denoised + decoded))
    tel.metric("checksum", checksum)
    tel.metric("workspace_bytes", size)
    return SweepOutput(checksum=checksum, workspace_bytes=size)
