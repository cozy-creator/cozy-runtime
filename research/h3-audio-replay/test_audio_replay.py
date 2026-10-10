"""CPU checks for input custody, public component contract and scoped flag restoration."""

import hashlib
import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import audio_replay as replay
import torch
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AudioDecodeStep
from diffusers.modular_pipelines.modular_pipeline import PipelineState

from cozy_runtime.author import InvalidRequest, describe


class Blob:
    def __init__(self, raw):
        self.raw = raw

    def read_bytes(self):
        return self.raw


def request():
    tensor = torch.arange(2 * 32 * 3).reshape(2, 32, 3).float()
    raw = tensor.numpy().tobytes()
    return replay.ReplayInput(Blob(raw), hashlib.sha256(raw).hexdigest(), (2, 32, 3))


class Decoder:
    pipe = SimpleNamespace(resident=None)

    def __init__(self):
        self.calls = 0
        self.input_hashes = []
        self.policies = []

    def decode_audio(self, task, state, *, checks):
        assert task == "fl2va_turbo"
        self.calls += 1
        self.input_hashes.append(replay.snapshot(state.audio_latents)[1]["sha256"])
        self.policies.append(torch.backends.cudnn.deterministic)
        value = 1 if torch.backends.cudnn.deterministic else self.calls
        return torch.full((1, 2, 12), float(value)), 32000


class TestReplay(unittest.TestCase):
    def test_artifact_restores_exact_and_refuses_corruption(self):
        payload = request()
        tensor = replay.restore(payload, torch.device("cpu"))
        self.assertEqual(replay.snapshot(tensor)[1]["sha256"], payload.sha256)
        payload.sha256 = "0" * 64
        with self.assertRaises(InvalidRequest):
            replay.restore(payload, torch.device("cpu"))

    def test_public_interface_only_selects_audio_vae(self):
        (surface,) = describe(replay.app)
        self.assertEqual(surface.model_components, {"model": ("audio_vae",)})
        (binding,) = surface.model_bindings
        self.assertEqual(binding.class_key, "H3TurboBase")
        self.assertEqual(binding.components["decode_audio"], ("audio_vae",))
        tensor = replay.restore(request(), torch.device("cpu"))
        state = PipelineState({"audio_latents": tensor, "image": None, "last_image": None})
        block_state = MiniMaxH3AudioDecodeStep().get_block_state(state)
        self.assertIs(block_state.audio_latents, tensor)

    def test_scoped_flags_restore_after_exception(self):
        before = replay.numerical_context()
        try:
            with replay.deterministic_decode_scope():
                inside = replay.numerical_context()
                self.assertTrue(inside["cudnn_deterministic"])
                self.assertFalse(inside["cudnn_benchmark"])
                for name in before:
                    if name not in {"cudnn_deterministic", "cudnn_benchmark"}:
                        self.assertEqual(inside[name], before[name], name)
                raise RuntimeError("test restoration")
        except RuntimeError as exc:
            self.assertEqual(str(exc), "test restoration")
        self.assertEqual(replay.numerical_context(), before)

    def test_six_calls_keep_exact_same_latent_and_only_scoped_policy_changes(self):
        initial = replay.numerical_context()
        decoder = Decoder()
        outputs = SimpleNamespace(save_bytes=lambda raw, **_: raw)
        # Baseline environment is local to this CPU test and restored on exit.
        cudnn = torch.backends.cudnn
        with cudnn.flags(
            enabled=cudnn.enabled,
            benchmark=False,
            benchmark_limit=cudnn.benchmark_limit,
            deterministic=False,
            allow_tf32=cudnn.allow_tf32,
            fp32_precision=cudnn.fp32_precision,
            depthwise_kernel=cudnn.depthwise_kernel,
        ):
            result = replay.audio_replay(
                SimpleNamespace(device=torch.device("cpu"), raise_if_cancelled=lambda: None),
                request(),
                decoder,
                outputs,
                SimpleNamespace(stage=lambda _: nullcontext()),
            )
        report = json.loads(result.diagnostics)
        self.assertEqual(decoder.calls, 6)
        self.assertEqual(decoder.policies, [False] * 3 + [True] * 3)
        self.assertEqual(len(set(decoder.input_hashes)), 1)
        self.assertEqual(
            report["within_case_bytes_equal"], {"baseline": False, "deterministic": True}
        )
        self.assertTrue(all(row["unchanged"] for row in report["latent_custody"]))
        self.assertTrue(report["context_restored_exactly"])
        self.assertEqual(replay.numerical_context(), initial)


if __name__ == "__main__":
    unittest.main()
