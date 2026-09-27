A device out-of-memory now blocks only repeats of the same invocation. Before, one
oversized request (for example a ten-reference Qwen edit) made the placement refuse
every later request with `device_shortfall` until it left the lane, including
requests that had already succeeded on the same executor.
