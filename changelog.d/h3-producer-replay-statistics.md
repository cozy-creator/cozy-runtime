H3's shared-QKV producer retains the original sparse statistics across a same-step block replay, so recovery after a later allocation failure uses the same quantization inputs as the first attempt.
