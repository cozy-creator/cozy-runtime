"""Check against upstream H3 geometry rather than duplicating the fixture constants."""

import inspect
import unittest

from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from projection_probe.probe import DIM, HEADS, INPUT_WIDTH, OUTPUT_WIDTH, TOKENS, token_bounds


class GeometryTest(unittest.TestCase):
    def test_uses_actual_h3_projection_geometry(self):
        fields = inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters
        self.assertEqual(INPUT_WIDTH, fields["hidden_size"].default)
        self.assertEqual(HEADS, fields["num_attention_heads"].default)
        self.assertEqual(DIM, fields["attention_head_dim"].default)
        self.assertEqual(OUTPUT_WIDTH, HEADS * DIM)

    def test_uneven_split_covers_each_row_once(self):
        first, second = token_bounds(0), token_bounds(1)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[1], second[0])
        self.assertEqual(second[1], TOKENS)
        self.assertEqual(first[1] - first[0], second[1] - second[0] + 1)


if __name__ == "__main__":
    unittest.main()
