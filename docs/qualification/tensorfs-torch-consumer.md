# TensorFS Torch API consumer

This branch is stacked on the refreshed native PEFT integration at `4f2d8c3d`,
including Runtime master `cab0c8eb`. It requires the new TensorFS 0.3.44 API;
the lock now resolves the published release. No worker, CLI, database, checkpoint changes
as part of this source patch.

Stored weights now go through `tensorfs.from_parts`, with the checkpoint's exact
logical dtype, shape, encoding ID and physical role specifications. Flat and
keepdim scale encodings stay distinct. Unsupported contracts refuse before the
Linear weight changes. TensorFS adopts the actual buffers; Runtime assigns the
Parameter and retains ownership of base eviction, request factors and admission.

Runtime still owns ephemeral activation quantization and its rank-dependent
wrapper, `KernelPreference.TORCH`, fast accumulation, and separate bias addition.
The change does not select another numerical recipe, enable a provider, merge a
LoRA or convert weights during a forward pass.

Storage accounting, fusion identity checks and H3 finite checks now ask TensorFS
for `physical_tensors`. The built-in H3 fence allows only that inspection function,
not Store access. No full-weight dequantization is needed to count or scan parts.

The native CPU regression writes and reads real rowwise checkpoints in both
layouts. It checks zero-copy binding, the provider probe, native reread version
invalidation, PEFT factor ownership across base eviction/restaging, and refusal
before mutation for mismatched declarations. Existing PEFT, H3 finite-check,
activation, model-reader and adapter-custody tests are also rerun.

Initial paired qualification uses TensorFS source
`b262ff85755f617bea13fe667cadb1cf938b6063`, its exact local Linux wheel SHA256
`a6b890b9234f1d3b540c3a4b0d9ef9f183af0b00a4774f48ca651e2457fed451`,
Torch 2.13 CPU, PEFT 0.20.0 and TorchAO 0.18.0. This is a source integration proof;
full Runtime streamed CUDA fill and full H3 serving still require their own
coherent worker qualification. Earlier isolated H100 measurements belong to the
frozen pre-consumer artifacts, not this new source.

TensorFS 0.3.44 is published from source `87f972a9`. The portable wheel SHA256 is
`5879f3e065753415ddd24955ec57009d68520954ab4910f5336ccec09e315238`.
The regenerated lock uses the ordinary registry artifact. No qualification wheel
is vendored in Git and no machine-local path override is committed. Final source
CI and Runtime streamed CUDA qualification remain separate gates; the user ended
the paid qualification window before the latter probe ran.
