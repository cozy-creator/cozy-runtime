Executor startup measures CUDA context overhead from its own per-device driver row,
so another process allocating during startup cannot inflate future process budgets.
Missing PID attribution stays unknown. Weight-plane facts add `context_measurement`
to identify measurements suitable for training machine admission estimates.
