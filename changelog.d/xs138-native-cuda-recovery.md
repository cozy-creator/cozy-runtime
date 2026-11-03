CUDA allocations use native Varena recovery exclusively; Python cache selection and second allocation attempts are removed, while non-CUDA recovery and explicit lifecycle offload remain separate.
