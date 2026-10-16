Runtime 0.24.7 brings sparse Sol attention to B200, quantizes each activation once, and bundles TensorD 0.5.37.

- Sol on B200 (#1450): the sm100 kernel compiles without a GPU in the trace; a kernel that fails to compile warns its run. H3 turbo 15 s on a B200: 145.7 → 120.6 s per clip.
- Quantize once (#1446): encoded linears that read one activation share its quantized operand, and a producer's norm, modulation and GELU fold into the int8 activation quantize.
- A run served by a refused encoded leaf says so (#1442): one warning per component, lane, leaf and reason.
- An attempt reports what it holds after its restore (#1448), so TensorD stops charging restored weights as external memory.
- TensorD 0.5.37 pins TensorFS 0.9.4: a plain SDXL Diffusers upload serves in fidika/sdxl.
