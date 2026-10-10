"""H3 audio decode scopes one real Torch setting and keeps the official block path."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.modular_pipelines.minimax_h3.decoders import (  # noqa: E402
    MiniMaxH3AudioDecodeStep,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402

from cozy_runtime.models.minimax_h3.official import OfficialH3Pipeline  # noqa: E402


def _settings() -> tuple[object, ...]:
    cudnn = torch.backends.cudnn
    return (
        cudnn.enabled,
        cudnn.benchmark,
        cudnn.benchmark_limit,
        cudnn.allow_tf32,
        cudnn.fp32_precision,
        cudnn.depthwise_kernel,
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
    )


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_only_determinism_changes_and_restores_after_decode(previous: bool, fails: bool) -> None:
    cudnn = torch.backends.cudnn
    with cudnn.flags(
        enabled=False,
        benchmark=True,
        benchmark_limit=7,
        deterministic=previous,
        allow_tf32=False,
        fp32_precision=cudnn.fp32_precision,
        depthwise_kernel=cudnn.depthwise_kernel,
    ):
        original = _settings()
        state = PipelineState({"audio": torch.ones(1, 2, 3)})
        calls = []

        def run(task: str, name: str, supplied: object, *, component: str) -> None:
            calls.append((task, name, supplied, component))
            assert torch.backends.cudnn.deterministic is True
            assert _settings() == original
            if fails:
                raise ValueError("decoder failed")

        pipe = SimpleNamespace(_run=run)
        if fails:
            with pytest.raises(ValueError, match="decoder failed"):
                OfficialH3Pipeline.decode_audio(pipe, "fl2va_turbo", state)
        else:
            result = OfficialH3Pipeline.decode_audio(pipe, "fl2va_turbo", state)
            assert result is state.audio
        assert calls == [("fl2va_turbo", "decode.audio", state, "audio_vae")]
        assert torch.backends.cudnn.deterministic is previous
        assert _settings() == original


def test_existing_official_audio_block_keeps_normalization_and_layout() -> None:
    class AudioVAE(torch.nn.Module):
        device = torch.device("cpu")
        config = SimpleNamespace(latents_mean=[1.0] * 32, latents_std=[2.0] * 32)

        def decode(self, latents: object, *, return_dict: bool) -> tuple[object]:
            assert torch.backends.cudnn.deterministic is True
            assert return_dict is False
            assert torch.equal(latents, normalized * 2 + 1)
            return (latents[:, :1, :],)

    normalized = torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)
    vae = AudioVAE()
    pipe = object.__new__(OfficialH3Pipeline)
    pipe.components = {"audio_vae": vae}
    pipe._pipes = {"t2va_turbo": SimpleNamespace(audio_vae=vae, audio_sampling_rate=32000)}
    pipe._blocks = {
        "t2va_turbo": SimpleNamespace(sub_blocks={"decode.audio": MiniMaxH3AudioDecodeStep()})
    }
    state = PipelineState({"audio_latents": normalized, "image": None, "last_image": None})
    previous, other = torch.backends.cudnn.deterministic, _settings()
    result = pipe.decode_audio("fl2va_turbo", state)
    assert torch.equal(result, (normalized * 2 + 1)[:, :1, :].permute(1, 0, 2))
    assert state.sampling_rate == 32000
    assert state.audio_latents is normalized
    assert torch.backends.cudnn.deterministic is previous
    assert _settings() == other
