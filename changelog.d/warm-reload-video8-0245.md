Runtime 0.24.5 keeps models hot, bakes LoRAs into int8 lanes and bundles TensorD 0.5.35.

- A model stays hot between requests: an attempt's end maps back what its activations took, so a warm qwen-image-2 int8 request loads no weights (4090, 1 MP 9.2–9.45 → 8.85 s; 2 MP 22.7–23.4 → 21.7 s). A stage cut into pieces keeps its model's whole components on the device (SeedVR2 1080p 88 → 74.5 s, nothing paged).
- `int8-convrot256/1` takes a LoRA bake (eager and fused); encoded leaves honour autocast as `nn.Linear` does (SeedVR2 int8 under bf16 autocast no longer runs in f32).
- The machine measures its GPU links at start and a group start reads them; the decode qualification is release data, one table per card architecture (sm86, sm89, sm90, sm120).
- Bounded failure messages keep their head and tail around " [...] " (Runtime and TensorD).
- A load reply or device frame of any size crosses the seam (no 64 KiB limit; #1430 with TensorD #288).
- TensorD 0.5.35: a removed package's GPU weights leave the GPU; TensorFS 0.9.3 (an upload is the model its repository describes; concurrent ingests share a store).
