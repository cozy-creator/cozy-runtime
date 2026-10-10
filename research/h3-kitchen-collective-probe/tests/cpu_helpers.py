import unittest

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from kitchen_collective_probe.probe import CHUNK, PREFIX, TOKENS, chunks, token_bounds

from cozy_runtime.internal.attention_producer import RequestState, rope_presentation


class Helpers(unittest.TestCase):
    def test_chunk_global_rows_and_order(self):
        t, h, d = CHUNK + 1, 2, 128
        packed = torch.randn(
            1, t, 3 * h, d, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(581204)
        )
        values = packed.chunk(3, dim=2)
        before = packed.clone()
        parts = list(chunks(torch, values))
        self.assertEqual([x.shape[0] for x in parts], [4096, 1])
        self.assertTrue(torch.equal(torch.cat(parts), packed.reshape(t, 3 * h * d)))
        self.assertTrue(torch.equal(packed, before))
        self.assertTrue(torch.equal(torch.cat(list(chunks(torch, values))), torch.cat(parts)))

    def test_global_rotary_presentation_and_head_independence(self):
        n = 129
        pos = torch.arange(n).repeat(3, 1).t()
        r = MiniMaxH3RotaryPosEmbed()(pos)
        state = RequestState()
        freq = rope_presentation(r, n, state)
        self.assertEqual(tuple(freq.shape), (n, 1, 48, 2, 2))
        self.assertTrue(torch.equal(freq.reshape(n, 48, 2, 2)[:, :, 0, 0], r[0][:, :48]))
        self.assertTrue(torch.equal(freq.reshape(n, 48, 2, 2)[:, :, 0, 1], -r[1][:, :48]))
        self.assertIs(rope_presentation(r, n, state), freq)

    def test_uneven_token_and_global_prefix_partition(self):
        self.assertEqual([token_bounds(r) for r in range(2)], [(0, 54613), (54613, 109225)])
        self.assertEqual(sum(b - a for a, b in map(token_bounds, range(2))), TOKENS)
        self.assertEqual(
            [max(0, min(b, PREFIX) - a) for a, b in map(token_bounds, range(2))], [257, 0]
        )


if __name__ == "__main__":
    unittest.main()
