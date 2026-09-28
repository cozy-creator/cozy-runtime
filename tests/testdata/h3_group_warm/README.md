# Small production-boundary H3 warm reproduction (h3a-093)

This fixture creates **3,697,560 bytes of synthetic weights** for a real tiny
upstream H3 transformer and an independent eight-byte overlay. It uses the same
mixed precision policy as Runtime H3. Both Models are prepared, preserving the
multi-model boundary without a large model download or an unverified overlay hook.

The base Model runs the unchanged `OfficialH3Pipeline.warm_dit` method with the
fixture's schedule values. That produces the actual warm geometry: eight video,
eight text and eight audio rows, with an eight-row replicated text refiner. It then
runs 1,029- and 1,032-token cases. There are 56 attention heads of dimension 128.
At degree four the actual 24-token warm has six local rows (~84 KiB per Q/K/V);
the larger cases have up to 258 local rows (~3.53 MiB per Q/K/V). Both allocation
regimes matter; the fixture does not assume expandable allocation caused the failure.

Prepare synthetic checkpoints without touching CUDA:

```sh
CUDA_VISIBLE_DEVICES='' python tests/testdata/h3_group_warm_fixture.py \
  --root /absolute/owned/fixture --prepare-only
```
