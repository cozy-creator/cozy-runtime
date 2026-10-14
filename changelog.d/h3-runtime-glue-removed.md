Runtime 0.23.37 keeps no model code: the H3 fused glue, shared-QKV producers and CP rotary hook live in minimax-h3 1.36.0 (#1372, −3,934 lines).

- Deleted `internal/fusion.py`, `fusion_install.py` and `attention_producer.py`; the `fusion=` Model class keyword and its interface field are gone (a package's plan is its `execution_plan`). A class that declares `fusion=` refuses at import (`class_keyword`).
- Context parallelism shards attention where the model's `_cp_plan` splits inputs, instead of naming H3's class.
- `internal.accel.readable` and the `leaves.PreQuantized` alias are gone; `cozy_runtime.author.kernels` owns them.
- minimax-h3 ≤1.35.2 (declare `fusion=`) and anima 0.5.0 (imports `internal.accel.readable`) fail on this Runtime; use minimax-h3 1.36.0 and anima 0.5.1.
