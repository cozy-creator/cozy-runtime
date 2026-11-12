`cozy_runtime.bench`'s GEMM speed and tile runs count their shapes and tile sets under the stage "measure", so a gate never reads them as sampler steps (xs-187).
