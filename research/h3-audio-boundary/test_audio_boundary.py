"""CPU controls for byte custody and original-versus-repeat labeling."""

import hashlib
import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from audio_boundary import audio_finish, snapshot


class Outputs:
    def save_bytes(self, payload, *, media_type):
        return payload


class Decoder:
    pipe = SimpleNamespace(sample_rate=48000)

    def __init__(self, mutate=False):
        self.calls = 0
        self.waveform = torch.zeros(1, 2, 12)
        self.mutate = mutate

    def decode_audio(self, task, state, *, checks):
        assert task == "fl2va_turbo"
        self.calls += 1
        self.waveform.fill_(float(self.calls))
        if self.mutate:
            state.audio_latents.add_(1)
        return self.waveform, 48000


class TestAudioBoundary(unittest.TestCase):
    def finish(self, decoder):
        state = SimpleNamespace(audio_latents=torch.arange(32).reshape(2, 4, 4).float())
        schedule = SimpleNamespace(
            timestep_plan_digest="plan",
            video_sigma_digest="vs",
            audio_sigma_digest="as",
            video_timestep_digest="vt",
            audio_timestep_digest="at",
            sigma_grid_points=9,
            transformer_evaluations=8,
        )
        return audio_finish(
            decoder,
            state,
            schedule,
            ctx=SimpleNamespace(raise_if_cancelled=lambda: None),
            out=Outputs(),
            tel=SimpleNamespace(stage=lambda _: nullcontext()),
            checks=None,
            seed=581204,
        )

    def test_snapshot_preserves_strided_bf16_bytes(self):
        source = torch.arange(30).reshape(5, 6).to(torch.bfloat16).transpose(0, 1)
        expected = source.contiguous().view(torch.uint8).numpy().tobytes()
        stride = source.stride()
        payload, facts = snapshot(source)
        self.assertEqual(payload, expected)
        self.assertEqual(facts["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(facts["dtype"], "torch.bfloat16")
        self.assertEqual(facts["shape"], [6, 5])
        self.assertEqual(source.stride(), stride)
        self.assertEqual(facts["nonfinite_elements"], 0)

    def test_original_bytes_owned_before_decoder_reuses_buffer(self):
        decoder = Decoder()
        output = self.finish(decoder)
        report = json.loads(output.diagnostics)
        self.assertEqual(decoder.calls, 2)
        self.assertEqual(output.original_audio, torch.ones(2, 12).numpy().tobytes())
        self.assertEqual(output.repeat_audio, (2 * torch.ones(2, 12)).numpy().tobytes())
        self.assertFalse(report["decode_repeat_bytes_equal"])
        self.assertEqual([r["label"] for r in report["decoded_audio"]], ["original", "repeat"])
        self.assertTrue(all(row["unchanged"] for row in report["latent_custody"]))
        self.assertEqual(
            hashlib.sha256(output.final_audio_latent).hexdigest(),
            report["final_audio_latent"]["sha256"],
        )

    def test_decoder_latent_mutation_is_reported(self):
        output = self.finish(Decoder(mutate=True))
        report = json.loads(output.diagnostics)
        self.assertEqual([row["unchanged"] for row in report["latent_custody"]], [False, False])
        self.assertNotEqual(
            report["latent_custody"][0]["sha256"], report["latent_custody"][1]["sha256"]
        )


if __name__ == "__main__":
    unittest.main()
