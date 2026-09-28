Uploads and `convert_cozytensors` carry everything a pipeline needs.
- A source with no pipeline index (a Civitai single file) pins its converter's reference repository beside the carrier. For SDXL that is stabilityai/stable-diffusion-xl-base-1.0@462165984030. The pin covers the index, the component configs, the scheduler and every tokenizer/processor file. An unreachable reference refuses with `model_reference_unavailable`.
- Diffusers sources pin their own tokenizer/processor folders.
- The index and configs embed inline, without private save-time carrier keys such as `_name_or_path`. Tokenizers embed as derived companion assets.
- Memo keys name the versioned converters.
- Requires tensorfs>=0.3.71. The `tensorfs_capability_missing` probe is removed.
