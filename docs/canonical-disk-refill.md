# Canonical DISK reconstruction and dtype replay

A package may retain execution-dtype weights, such as an SDXL VAE converted from stored
F16 to BF16. Parking those weights keeps their current metadata dtype. A refill must
allocate the canonical stored representation first, validate/fill it strictly, then
replay every requested conversion in order. Replaying only the final dtype would lose
rounding from chains such as F16→BF16→F16.

The fill backend now supplies metadata-only RefillSchema: canonical slot dtypes from
its original destination specs and retained nonpersistent-buffer metadata, plus the
largest single derived-buffer restoration copy. No new payload cache or live storage
snapshot is introduced. Canonicalization occurs only inside transactional refill;
drop and quotes do not change visible parked dtype. Initial-construction dtype/shape
validation stays strict. Unknown slots/shapes/routes refuse before allocation.

LoadDemand separates initial canonical source allocation, settled replay result,
derived restoration overlap and conversion replay peak. Sequential loads account for
prior settled units; whole fills first allocate all canonical sources and restore
derived values, then replay conversions. Restoration is not charged again after an
earlier unit widens. Plain no-conversion derived fills are priced too. The old quote
missed the overlap while _restore_derived copies a value before replacing its placeholder.

Decoded-float routes retain their ordinary canonical float topology and existing decoder
scratch quote. Opaque encoded-GEMM routes retain the prior settled+provider-transient
accounting; their metadata adds only derived restoration excess beyond that existing
provider envelope. Pending conversion of that opaque topology refuses locally instead
of guessing source dtypes or double-charging float reconstruction. No kernel route,
precision policy, storage verification or device authority changes.

On fill or conversion failure, the existing tier transaction restores exact original
parked meta slots, pending/applied chains and truthful measured bytes, including after
earlier units already filled and converted. Cleanup failure remains a device fault.

CPU tests use real TensorFS derived artifacts/verified reads, production strict fill and
both production refill entry points. Only CUDA transport is replaced by native CPU reads;
these are not CUDA DMA or multi-GPU qualification. Tests include BF16 repeats, HOST-tier
cycles, F32 width growth, rounding chains, integer/derived buffers, decoded FP8 parts via
the existing CPU decoder, whole/sequence peak ordering, original dtype-refusal reproduction,
and earlier-unit/postfill failure rollback. Encoded-GEMM accounting has an explicit
arithmetic stand-in; no native encoded GPU execution is claimed.

The feature is isolated above baseline80c39efc. Only its focused commit should be applied
to the reviewed c17 integration/candidate; the separate fresh-master merge is not an
isolated GPU-performance change. No public release or active environment installation
is part of this CPU qualification.
