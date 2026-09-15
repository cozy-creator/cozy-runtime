# Optional built-in model library

`cozy_runtime.models` contains reusable model inference implementations. A built-in uses the same public `Model`, `Loader`, component scopes and sequence-parallel declarations as a model defined in an ordinary package. Custom package models remain supported.

MiniMax H3's standard model and separate Turbo base/adapter are available as:

```python
from cozy_runtime.models.minimax_h3 import H3Model, H3TurboBase, H3TurboLoRA
```

Install the `minimax-h3` optional dependency extra to execute these models. Importing `cozy_runtime`, `cozy_runtime.author`, `cozy_runtime.models` or the MiniMax H3 namespace itself does not import Torch, Diffusers or Transformers. Accessing a model class loads its optional implementation. Static description reads the real builtin source declarations without importing ML libraries or executing model code.

The built-in owns the complete reusable closure: model classes, Diffusers pipeline integration, conditioner, AdaLN/Turbo consumers, VAE tiling, processor/tokenizer and immutable schedule resources. Diffusers still implements the model architecture and sampling operations. The workflow package owns requests, media/reference policy, output encoding and long-form assembly. Its static schema plan copies must match the built-in plan bytes.

H3 supplies per-step `AttentionLayout` facts using its actual packed layout. Text, all reference media and generated audio form the protected prefix; only generated video rows use sparse queries. The Sol policy marks the first ten standard evaluations, or first four Turbo evaluations, dense. The token refiner and first two transformer blocks stay dense throughout. Explicit Turbo `sol_dense_steps` overrides remain available. These facts do not select or authorize an approximate backend. Context is request-local and restored after exceptions. Its closed layout is transported with mirrored component calls for supported Ulysses groups; hardware and full-model qualification remain separate evidence.

The historical SDXL development `corpus` extra uses Transformers 4 and is explicitly incompatible with the H3 extra's Transformers 5 cohort. This is not a claim that every model family can share one interpreter or that the SDXL fixture has been qualified with Transformers 5.

## Development delivery

This API was added after the published Runtime 0.18.2 artifact. A version string alone does not prove the module exists. Development qualification must use the matching source wheel and preserve its source revision/hash; publishing or activating a dependent H3 workflow against the older published artifact remains blocked. No PyPI publication is needed for local/remote development-wheel qualification. A normal release dependency can advance when a real release includes the API.
