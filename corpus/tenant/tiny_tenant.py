"""A model-bearing TENANT package for the shared-lane proofs (cr-022).

The model is the synthetic checkpoint `tfs checkpoint write --plain` produces (tfs-015):
`unet` blocks of attention/feed-forward linears in bf16, a `text_encoder` embedding, a
`vae` pair of f32 linears. It computes nothing worth keeping; what it does is occupy real
device memory through the real fill plane, so a lane holding two of it has a real decision
to make. Sizes come from the artifact's own config, which the fixture writer derives from
the header — the package names no shape of its own.

One entrypoint, `touch`, opens one component-use scope over EVERY component: the scope's
demand is the whole model, which is exactly the demand a lane must arbitrate between two
tenants that do not both fit. A second, `retouch`, binds the SAME model under the same
parameter name: two entrypoints of one construction, which one executor serves (h3a-018).
"""

from __future__ import annotations

from typing import Any

import msgspec

from cozy_runtime.author import App, Config, Context, Loader, Model, Telemetry, uses_components

app = App()


class TouchInput(msgspec.Struct, forbid_unknown_fields=True):
    seed: int = 7


class TouchOutput(msgspec.Struct):
    checksum: float
    components: list[str]


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


@app.entrypoint
def touch(ctx: Context, payload: TouchInput, model: TenantModel, tel: Telemetry) -> TouchOutput:
    view = model.for_request(ctx, seed=payload.seed)
    checksum = model.touch(view._seed)
    tel.metric("checksum", checksum)
    return TouchOutput(checksum=checksum, components=sorted(model.pipe.components))


@app.entrypoint
def retouch(ctx: Context, payload: TouchInput, model: TenantModel, tel: Telemetry) -> TouchOutput:
    """The same model through a second entrypoint: one construction, two bindings."""
    view = model.for_request(ctx, seed=payload.seed)
    checksum = model.touch(view._seed)
    tel.metric("checksum", checksum)
    return TouchOutput(checksum=checksum, components=sorted(model.pipe.components))
