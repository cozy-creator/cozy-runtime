# Upstream: llama.cpp b12466

`_kernels._ggml`'s kernels are llama.cpp's, vendored byte for byte under `ggml/` (its own paths) from tag
[`b12466`](https://github.com/ggml-org/llama.cpp/releases/tag/b12466), commit
`ede6106780fef959e40ae309afb68da7b6442883`, archive SHA-256
`d81544fe1f31f2539f3610486eb10e7fe606d88da9d1f85a54c310bc7ac75af7`. MIT, "Copyright (c) 2023-2026 The ggml authors":
`LICENSE` here, carried by the native wheels.

Built: `ggml-cuda/mmvq.cu` (MMVQ), `mmq-few.cu` (few-row MMQ), `quantize.cu` (q8_1 activations) and the MMQ instances
of the six types the `ggml.*` lanes use, with ggml-cuda's own flags. The rest are the headers they include.

Ours, not upstream's: `shim.cu`/`shim.h` (what these sources take from ggml-cuda.cu and ggml.c: the device table, a
context on the caller's stream with the caller's allocator as its pool, ggml's tensor arithmetic for these types),
`shim/` (cuBLAS's names, declared only: nothing here calls cuBLAS), `ids.cu`/`ids.h` (MUL_MAT_ID's MMVQ over a table of experts), `matmul.cu`/`matmul.h` (ggml's MUL_MAT choice
among the three, and MMQ's dense host launch adapted from `ggml_cuda_mul_mat_q` in `mmq.cu`) and `../ggml.cpp`.

One upstream file is patched: `ggml-cuda/mmvq.cu` (upstream SHA-256 `11c41a96…0850659e9`). Its two MUL_MAT_ID kernels
(`mul_mat_vec_q` with one token, `mul_mat_vec_q_moe` with more) read `vx` as a table of each expert's weight when the
experts' channel stride is 0 (`// cozy:`, four lines each): our experts are separate paging units, not one stack
(`ids.cu`). A stacked weight's stride is never 0, so llama.cpp's own calls are unchanged.

A new tag replaces these files whole, reapplies that patch, and updates this list.

| file | SHA-256 |
|---|---|
| `ggml/include/ggml-alloc.h` | `d114067f9099c4a0e2bbc0625c676d4894d9cf0ea27168ee072c7de0e71bb51d` |
| `ggml/include/ggml-backend.h` | `34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a` |
| `ggml/include/ggml-cuda.h` | `9cbfcea49f6da4d07f41dc22fa8dbf870825faaa2a68e34e65fe30ea6c41e042` |
| `ggml/include/ggml.h` | `bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b` |
| `ggml/include/gguf.h` | `e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4` |
| `ggml/src/ggml-common.h` | `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c` |
| `ggml/src/ggml-cuda/common.cuh` | `941bce2299c2ae8bf0838531ccfd4643719362a36879df0e49dc403cf2a2e852` |
| `ggml/src/ggml-cuda/cp-async.cuh` | `5c22f648a81f53b23b6f361ebdff65ae4615bce707ad3270790a501c21c1abd6` |
| `ggml/src/ggml-cuda/graph.cuh` | `e7bb53de23581749c9aacf08ad56cf6682e8d462e6101e14775a1cec753688bd` |
| `ggml/src/ggml-cuda/mma.cuh` | `f99e42aa30a62668271d601722467d5d7df8caf0d448a2bfbb77c8d18eb1564d` |
| `ggml/src/ggml-cuda/mmid.cuh` | `5e0900519e6e8b8c8636b8199daff03b0197bf7a9d06d45276d5cf77477397a7` |
| `ggml/src/ggml-cuda/mmq-config-ampere.cuh` | `6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408` |
| `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | `8bec1e65a3a2a58fab1f406225f6ba2f1a4c38a19793dfa255baff69bc044d70` |
| `ggml/src/ggml-cuda/mmq-config-cdna.cuh` | `314eeb8453834723d5224012f692f70673f86ecc42d2b6d6f2005a0fad124dbc` |
| `ggml/src/ggml-cuda/mmq-config-gcn.cuh` | `499b7698417eb95c050ee0daa834ca4215b0c215c4b04664d859787d616f3217` |
| `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh` | `67a93b6d26d3e16e77fd285dd77e7cca7848fb0b0d68897c2ff1e8cc47102263` |
| `ggml/src/ggml-cuda/mmq-config-pascal-older.cuh` | `48bae44f1f0e667fa02fb898859eb18c03961bdbadf53c1e22d98801ed70affe` |
| `ggml/src/ggml-cuda/mmq-config-rdna2.cuh` | `758be2a33b89018841f432c15cd63009ad212d71eae888a04b4535cb64941aa7` |
| `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh` | `0b6298eb94c563d3f81f345331fe8bfc2ec79c11ea2162d88dfcfaf403c9d53b` |
| `ggml/src/ggml-cuda/mmq-config-rdna3.cuh` | `a1ac4f4fa799f2fb200ead23f3403bb0f729d41109b8967c6d76216768aae0cf` |
| `ggml/src/ggml-cuda/mmq-config-rdna4.cuh` | `e40c9c5fe85607db07ae351dd6ca13c92caaa238e2a62d5134877d9fbcb5c645` |
| `ggml/src/ggml-cuda/mmq.cuh` | `f77df00ef77c22749e5130bf8b5ad695ef1d2ce86829d051c808882f1ac3cd49` |
| `ggml/src/ggml-cuda/mmq-few.cu` | `b2e0e50a2fdcfc90a3426aa1037485ec557deeee64c781deae49130530507b32` |
| `ggml/src/ggml-cuda/mmq-few.cuh` | `8c3147331e494dbe80070541920cacd12ce2b1f1ae59755bc27ff67101f5c8e8` |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | `2583bb2f978afafc8bbcd9cba40e4cd7c1abbecfaba1bfc93bb2b420ee31f283` |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `c038cf45ca87552e8c7c81f71f8c908c7d07460a55a571522abd562640c692b3` |
| `ggml/src/ggml-cuda/mmvq.cu` | `8e31d072abf0794a430d8d0a4e40f7aece13fdf14e8fe648e5de9cf6ee3c7be0` |
| `ggml/src/ggml-cuda/mmvq.cuh` | `b37f8359f519ba90ddecd9cb3ca5f8c2ae05685bfae6813d435d75763c2bb006` |
| `ggml/src/ggml-cuda/quantize.cu` | `c8beca1390a33dc3ed8666662baf5837f1fa435d8a53b712594c130233f408fe` |
| `ggml/src/ggml-cuda/quantize.cuh` | `66a47a5cd9a6624b5d2c2ef89f6b7287408a822332e113795e3f6182d9eb2ac0` |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu` | `2ab47361d40a9ece25406c20693424a466c93bcb5cdb5c06cee8f9d03a806beb` |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q5_0.cu` | `34a443ed0e68b1ad614eb251758d0f43f003ce8d10e7833f9f58a46677d336db` |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q5_1.cu` | `a1312dd1316cd75b4a18e2c4a4c34d54ddfcd92b7fc1e45d4daee101e8d69f00` |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q5_k.cu` | `c98bbe1dd91e03362f4136357c6f4e4e2c5328902fdc6fd857d0f608386ffb2e` |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu` | `989943d6501a74927d079eacca793e94a1233863196c94dd20e459017ed76fc2` |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu` | `b38f4de9a3563debaf4796d6772e1ed72ce6944c2b8b03f9c26bf19191addee7` |
| `ggml/src/ggml-cuda/unary.cuh` | `a8c7e041000ba4e90b37842a3d21003a6d4f83808a118bcb96f86c183ba84e52` |
| `ggml/src/ggml-cuda/vecdotq.cuh` | `5e6882fe30b86f2f1ba294542ec88866c962e8f90500d4e035882eff3942f975` |
| `ggml/src/ggml-cuda/vendors/cuda.h` | `a61fedd72c3cf0d027a1a2836dc5b936e02e954dc8ced7d2cfad47c9e611ec52` |
| `ggml/src/ggml-cuda/vendors/hip.h` | `304bade92dae7fcbb306a651e37da7b612ecb0187032ec16a40641afd74af315` |
| `ggml/src/ggml-cuda/vendors/musa.h` | `59908fc4deb35185d5f8a397f3ad452e267e2b45ccaf41c62416891ec9deb9ec` |
| `ggml/src/ggml-impl.h` | `75465b01eba2b1ecae8634898508e71294fbff0f7f96ee9ad2caf5bae5b6c5fd` |
| `LICENSE` | `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d` |
