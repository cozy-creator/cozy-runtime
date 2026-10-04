Adaptive image VAE decoding now selects its mode before execution and invokes the
decoder once. An OOM/indivisible refusal poisons the retained executor with a typed
recovery-contract refusal instead of automatically replaying possibly changed
input, RNG, buffer, hook or Python state at another tile size. Existing measured
pre-call tiling and reviewed safe operator recovery remain available.
