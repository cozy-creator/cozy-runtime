# Native CPU VAE degraded path

Tracker #320; owner gpu_qualification. Fresh origin/master b2a93731, branch
test/codex-native-cpu-vae-20261004. No product change, dispatch wrapper or custom
storage is implemented. Root authorized source assessment and one full saved-latent
CPU proof; no GPU work is authorized for this step. Ilulu was explicitly ended,
its holder released, keeper exited, and live all-hub inventory shows zero rentals.

Prefer a CPU target selected before the VAE's first forward when a fidelity-safe
native GPU decode cannot be admitted. Preserve the original BF16 upcast, scale,
full 128x128 latent, full 1024x1024 output and one native decode. The source still
supplies the same 248 VAE tensors; no GPU module/weight copy or whole-forward
recovery is needed. A CPU proof is not a matched benchmark or rented pipeline gate.

Existing Runtime mechanisms already support this target: Weights.host_only opens
the CPU plane; CPU registration maps and binds the complete pinned tier in place;
WeightResidency installs neither GPU block hooks nor its adaptive VAE tiling
wrapper for a CPU plane. PlaneBackend.for_script uses the normal census/fit/fill/
commit path and source read leases. The proposed proof exercises those paths on
the exact locally cached manifest 288440e7… and saved latent 87cad3ed….

The proof constructs one original AutoencoderKL on meta, binds its source weights
through the existing CPU plane, then converts dtype in that same module to BF16
as the original GPU diagnostic did. No deepcopy, GPU-to-CPU module movement,
alternate checkpoint or smaller shape is used. CUDA_VISIBLE_DEVICES is empty;
threads are 2 and the shared CPU lock/nice 19 apply. Before construction and before
the single forward, admit a conservative 16 GiB host allowance against actual
available memory. This snapshot is not an enforced machine host reservation.

The dependency fixture overlays exact H0c SDK on an owned test venv, borrowing
existing Torch 2.14/Diffusers 0.40 and public Store-enabled TensorFS 0.3.93. That
Store path proves native CPU source binding; it does not prove the f089
STORE=False machine-source handoff. Runtime placement in an already-GPU model
slot, shared-source lifetime, CPU host-budget reservation and ordinary CLI/rented
integration remain separate gates.

Compare the saved lossless RGB image against the same-latent untiled 5080 image
and preserved original control using the unchanged 35 dB rule. CPU AVX2 and
GPU CUDA kernels can differ in reduction order; BF16 inputs/weights are preserved
but pixel equality is not presumed. Source and CPU evidence must remain distinct
from GPU/runtime qualification.

Whole CPU decode is smaller in scope than operator streaming or custom managed
accounting: the original complete native forward keeps global normalization and
attention. It needs a CPU target and host admission before entry, plus a bounded
latent transfer for a GPU-produced input. Those integration gates still need
review; no failed GPU decoder may be retried on CPU as whole-forward recovery.
