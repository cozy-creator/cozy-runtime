"""cr-075 PR-2: the two probe surfaces over the one core, proven byte-for-byte equal.

One real installed-package-shaped layout (package.toml + module, discovered and imported
for real), real lane trees (safetensors written by the derive writer, read back through the
verb's Backend into a real `Loader.construct` fill), one real free-running loop shared by
both surfaces — the entrypoint handler the verb drives through the real invocation kernel
IS the loop the in-job arm drives directly. Nothing is mocked. torch is the `derive` extra
(absent from CI's lanes), so every arm follows the `needs_torch` discipline of
test_activation_probe.py and runs wherever torch is installed (`uv sync --extra derive`).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import Model, probe_activations
from cozy_runtime.cli import probe as probe_verb
from cozy_runtime.cli.io import CliError, Options
from cozy_runtime.internal.discovery import discover
from cozy_runtime.probe import REPORT_FORMAT, ProbeRefusal


def _torch_here() -> str:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"torch must be importable in the test interpreter for the probe arms: {exc}"
    return ""


NO_TORCH = _torch_here()
pytestmark = pytest.mark.skipif(bool(NO_TORCH), reason=NO_TORCH or "")

SEED = 7
STEPS = 4

ORDERED_TAPS = (
    "denoiser.embed",
    "denoiser.proj_in",
    "denoiser.blocks.0",
    "denoiser.blocks.1",
    "denoiser.blocks.2",
    "denoiser.blocks.3",
    "denoiser.proj_out",
)

# ------------------------------------------------------- the real serving package fixture

PACKAGE_TOML = '[application]\nobject = "tiny_probe_package:app"\n'

PACKAGE_MODULE = '''\
"""tiny-probe: the surfaces' fixture serving package — the probe core's tiny stack behind
one constructing slot and one entrypoint whose handler IS the free-running loop."""

import msgspec
from cozy_runtime.author import App, Context, Loader, Model, uses_components

app = App()

FEATURES = 8
WIDTH = 16
TOKENS = 6


def _denoiser_class():
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
            self.norm = nn.LayerNorm(WIDTH)
            self.mlp = nn.Sequential(
                nn.Linear(WIDTH, WIDTH * 2), nn.GELU(), nn.Linear(WIDTH * 2, WIDTH)
            )

        def forward(self, hidden, emb):
            attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
            hidden = hidden + attended
            return hidden + self.mlp(self.norm(hidden) + emb)

    class TinyDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, WIDTH)
            self.proj_in = nn.Linear(FEATURES, WIDTH)
            self.blocks = nn.ModuleList(Block() for _ in range(4))
            self.proj_out = nn.Linear(WIDTH, FEATURES)

        def forward(self, latent, timestep, conditioning):
            wide = self.proj_in.weight.dtype
            emb = self.embed(timestep.to(wide)) + conditioning.to(wide)
            hidden = self.proj_in(latent.to(wide))
            for block in self.blocks:
                hidden = block(hidden, emb)
            return self.proj_out(hidden)

    return TinyDenoiser


class TinyPipeline:
    def __init__(self, config):
        import torch

        torch.manual_seed(0)
        self.components = {"denoiser": _denoiser_class()()}


def build_pipeline(config):
    return TinyPipeline(config)


def drive_loop(call, seed, steps):
    """The ONE free-running loop both surfaces record: euler-ish over the model's output."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(1, TOKENS, FEATURES, generator=generator)
    conditioning = torch.randn(1, 1, WIDTH, generator=generator)
    for index in range(steps):
        timestep = torch.full((1, 1, 1), 1.0 - index / steps)
        eps = call(latent, timestep, conditioning)
        latent = latent - 0.3 * eps.float()


class TinyModel(Model[TinyPipeline]):
    pipe: TinyPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(TinyPipeline, factory=build_pipeline)

    @uses_components("denoiser")
    def denoise(self, latent, timestep, conditioning):
        return self.pipe.components["denoiser"](latent, timestep, conditioning)


class ProbeRequest(msgspec.Struct):
    seed: int
    steps: int


class ProbeReply(msgspec.Struct):
    steps: int


@app.entrypoint
def generate(ctx: Context, payload: ProbeRequest, model: TinyModel) -> ProbeReply:
    drive_loop(model.denoise, payload.seed, payload.steps)
    return ProbeReply(steps=payload.steps)


class PingRequest(msgspec.Struct):
    message: str


class PingReply(msgspec.Struct):
    message: str


@app.entrypoint
def ping(payload: PingRequest) -> PingReply:
    return PingReply(message=payload.message)
'''


class Stack:
    """One discovered project plus its three real lane trees."""

    def __init__(self, root: Path) -> None:
        self.project = root / "package"
        self.project.mkdir()
        (self.project / "package.toml").write_text(PACKAGE_TOML)
        (self.project / "tiny_probe_package.py").write_text(PACKAGE_MODULE)
        self.module: ModuleType = discover(self.project).module
        self.reference = root / "lane-fp32-ref"
        self.scaled = root / "lane-scaled"
        self.subscaled = root / "lane-subscaled"
        _write_lane(self.reference, self._denoiser())
        _write_lane(self.scaled, self._denoiser(scale="blocks.2"))
        _write_lane(self.subscaled, self._denoiser(scale="blocks.2.mlp.0"))

    def _denoiser(self, scale: str | None = None) -> Any:
        import torch

        denoiser = self.module.build_pipeline(None).components["denoiser"]
        if scale is not None:
            with torch.no_grad():
                for parameter in denoiser.get_submodule(scale).parameters():
                    parameter.mul_(1.05)
        return denoiser

    def lane_pipeline(self, tree: Path) -> Any:
        """The lane exactly as the verb fills it, loaded the in-job way: the package's own
        factory, then the lane tree's stored bytes into its state dict."""
        import torch

        from cozy_runtime.derive import safetensors_io

        pipeline = self.module.build_pipeline(None)
        state: dict[str, Any] = {}
        for path in sorted(tree.glob("*.safetensors")):
            header, base = safetensors_io.read_header(path)
            for key, row in header.items():
                raw = safetensors_io.read_raw(path, header, base, key)
                values = safetensors_io.to_f32(raw, str(row["dtype"]), list(row["shape"]))
                state[key.removeprefix("denoiser.")] = torch.from_numpy(values.copy())
        pipeline.components["denoiser"].load_state_dict(state)
        return pipeline


def _write_lane(tree: Path, denoiser: Any) -> None:
    from cozy_runtime.derive import safetensors_io

    writer = safetensors_io.Writer()
    for key, tensor in denoiser.state_dict().items():
        writer.add(
            f"denoiser.{key}",
            "F32",
            list(tensor.shape),
            safetensors_io.from_f32(tensor.detach().numpy(), "F32"),
        )
    tree.mkdir(parents=True, exist_ok=True)
    writer.write(tree / "weights.safetensors")
    (tree / "config.json").write_text(json.dumps({}))


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Stack:
    return Stack(tmp_path_factory.mktemp("probe-surfaces"))


def _verb(stack: Stack, *args: str) -> Any:
    return probe_verb.run(list(args), Options(json=True, dir=str(stack.project)))


def _payload() -> tuple[str, str]:
    return (f"seed={SEED}", f"steps={STEPS}")


def _over_floor(report: dict[str, Any], k: float, epsilon: float = 1e-9) -> list[str]:
    floor = {entry["tap"]: entry for entry in report["floor_aggregates"]}
    over: list[str] = []
    for entry in report["aggregates"]:
        worst = entry["worst"]["rel_l2"]
        floor_worst = floor[entry["tap"]]["worst"]["rel_l2"]
        if worst is not None and worst > k * max(floor_worst or 0.0, epsilon):
            over.append(entry["tap"])
    return over


# ------------------------------------------------------------------------- proof arms


def test_both_surfaces_emit_byte_canonical_report_1(stack: Stack, tmp_path: Path) -> None:
    """The verb over the installed-package-shaped layout and the in-job author API over
    the same lanes and the same loop produce the IDENTICAL canonical report/1 bytes — one
    mechanism, two callers. The corrupted block is named through both."""
    out = tmp_path / "report.json"
    result = _verb(
        stack,
        "generate",
        "--component",
        "denoiser",
        "--reference",
        f"fp32-ref={stack.reference}",
        "--candidate",
        f"scaled={stack.scaled}",
        "--out",
        str(out),
        *_payload(),
    )
    assert result.canonical_json is not None
    assert out.read_bytes() == result.canonical_json

    reference = stack.lane_pipeline(stack.reference)
    candidate = stack.lane_pipeline(stack.scaled)

    def call(*args: Any, **kwargs: Any) -> Any:
        return reference.components["denoiser"](*args, **kwargs)

    report = probe_activations(
        reference,
        candidate,
        lambda: stack.module.drive_loop(call, SEED, STEPS),
        component="denoiser",
        reference_label="fp32-ref",
        candidate_label="scaled",
    )
    assert canonical_json.encode(report) == result.canonical_json

    assert report["format"] == REPORT_FORMAT
    assert [tap["path"] for tap in report["taps"]] == list(ORDERED_TAPS)
    assert _over_floor(report, k=10.0)[0] == "denoiser.blocks.2"
    upstream = set(ORDERED_TAPS[:4])
    for row in report["rows"]:
        assert row["nan_inf"] == {"ref": 0, "cand": 0}
        if row["tap"] in upstream:
            assert row["rel_l2"] == 0.0 and row["cosine"] == 1.0
    assert canonical_json.decode(result.canonical_json) == report


def test_verb_zoom_flag_names_the_sublayer(stack: Stack) -> None:
    """`--zoom` arrives with the surfaces: the verb reruns one block subdivided and the
    corrupted sublayer's exact module path is the first tap over K x floor."""
    result = _verb(
        stack,
        "generate",
        "--component",
        "denoiser",
        "--zoom",
        "denoiser.blocks.2",
        "--reference",
        f"fp32-ref={stack.reference}",
        "--candidate",
        f"subscaled={stack.subscaled}",
        *_payload(),
    )
    report = result.document
    assert isinstance(report, dict)
    assert report["zoom"] == "denoiser.blocks.2"
    assert {tap["path"] for tap in report["taps"]} == {
        "denoiser.blocks.2",
        "denoiser.blocks.2.attn",
        "denoiser.blocks.2.norm",
        "denoiser.blocks.2.mlp.0",
        "denoiser.blocks.2.mlp.2",
    }
    assert _over_floor(report, k=10.0)[0] == "denoiser.blocks.2.mlp.0"


def test_in_job_swap_is_scoped_to_the_drive_and_always_restored(stack: Stack) -> None:
    """The recording stand-in exists exactly between swap and restore: the caller's loop
    sees it, the returned pipeline never does — a failing drive included (#573's lesson:
    no proxy survives into a live generation)."""
    reference = stack.lane_pipeline(stack.reference)
    candidate = stack.lane_pipeline(stack.scaled)
    original = reference.components["denoiser"]
    seen: list[object] = []

    def drive() -> None:
        seen.append(reference.components["denoiser"])
        stack.module.drive_loop(
            lambda *a, **k: reference.components["denoiser"](*a, **k), SEED, STEPS
        )

    probe_activations(reference, candidate, drive, component="denoiser")
    assert seen and seen[0] is not original
    assert getattr(seen[0], "embed", None) is original.embed  # attributes reach the real one
    assert reference.components["denoiser"] is original

    def failing() -> None:
        raise RuntimeError("the loop died mid-generation")

    with pytest.raises(RuntimeError):
        probe_activations(reference, candidate, failing, component="denoiser")
    assert reference.components["denoiser"] is original


def test_component_resolution_is_typed_never_guessed(stack: Stack) -> None:
    reference = stack.lane_pipeline(stack.reference)

    with pytest.raises(ProbeRefusal) as unresolved:
        probe_activations(object(), reference, lambda: None, component="denoiser")
    assert unresolved.value.code == "component_unresolved"

    class TwoPipes(Model[object]):
        first: object
        second: object

    holder = TwoPipes.for_test(first=reference, second=stack.lane_pipeline(stack.scaled))
    with pytest.raises(ProbeRefusal) as ambiguous:
        probe_activations(holder, reference, lambda: None, component="denoiser")
    assert ambiguous.value.code == "component_ambiguous"


def test_verb_refusals_are_typed(stack: Stack, tmp_path: Path) -> None:
    lanes = (
        "--reference",
        f"fp32-ref={stack.reference}",
        "--candidate",
        f"scaled={stack.scaled}",
    )

    with pytest.raises(CliError) as slotless:
        _verb(stack, "ping", "--component", "denoiser", *lanes, "message=hi")
    assert slotless.value.name == "structural"

    with pytest.raises(CliError) as unknown_fn:
        _verb(stack, "nowhere", "--component", "denoiser", *lanes)
    assert unknown_fn.value.name == "not_found"

    with pytest.raises(CliError) as no_lane:
        _verb(
            stack,
            "generate",
            "--component",
            "denoiser",
            "--reference",
            f"fp32-ref={tmp_path / 'absent'}",
            "--candidate",
            f"scaled={stack.scaled}",
        )
    assert no_lane.value.name == "not_found"

    with pytest.raises(CliError) as no_component:
        _verb(stack, "generate", "--component", "nowhere", *lanes, *_payload())
    assert no_component.value.name == "component_unresolved"
