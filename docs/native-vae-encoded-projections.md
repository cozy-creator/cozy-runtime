# H3 VAE stored INT8 projections

Actual model receipt26930 has six `NativeLinear` children per block: Q/K/V/out
`[2048,2048]`, up `[16384,2048]`, down `[2048,8192]`. Each carries stored INT8
codes, one FP32 scale per output row, and FP32 bias, with logical FP16 output.
These are distinct operations from the existing plain float biased-Lt path.
Neither path reinterprets the other's stored weights.

The descriptor requires the exact NativeLinear class/provider and ordinary
Rows contract; arbitrary instance forwards, linear overrides and hooks refuse.
Baked Runtime LoRA hooks retain the established no-update Rows semantics. Raw
codes/scales/bias use the same source identity, raw-view layout, alignment and
current Region address checks as float slots, now against their actual buffer
tables. Ordinary block source hooks still run once before any source read.
Installed RMS wrappers are accepted by their actual saved forward object; an
arbitrary transparent marker, forward or hook is insufficient.

## Exact storage and lifetime

For each native-selected tile window W and block batch A, let M=A*1797. Each
full/ragged projection queries the existing `int8_plan(device, True, M, N, K, N)`.
The returned kind/config/workspace enters the semantic identity. Unsupported
queries decline before body writes. No alternate plan is tried after acceptance.

The block owns its existing FP32 output, packed FP16 row banks, expanded FP16
bank and Kitchen attention buffers. The encoded path adds:

- INT8 activation storage: `max(rows*K)` over actual kind0 full/ragged plans.
- FP32 activation scales: `max(rows)` over those same plans, four bytes each.
- Workspace: maximum actual queried workspace over every float and encoded plan.
- Bias: the existing preparation bank of16384 FP16 elements (32768 bytes); each
  stored bias is copied into its prefix immediately before its projection.

The bias copy has the same FP16 rounding as ordinary `Prebuilt._weight`'s
`bias.to(out_dtype).contiguous()`. Inputs to each projection are already contiguous
FP16 from the existing norm/attention/SwiGLU stages, so no extra activation cast or
layout allocation is hidden. Rows slicing creates views only. The native GEMM
gets only the declared scratch callback and supplied output, activation codes and
scales. There is no `Prebuilt._weight` call that could silently allocate a cast.

Activation codes/scales, bias and workspace are reused sequentially on prepared
stream S. Q/K/V output banks remain live until attention consumes them; the same
existing packed banks then serve out/up/down stages. The sequence's role inventory
takes each role's maximum over all36 actual block queries and physically owns it
through checked S completion. Source uses retain their ordinary consume-on-S
fences. Joined producer streams and supplied-storage alias/span checks are
unchanged. Native policy alone selects W/A; these tensor shapes are declarations,
not a Python capacity decision. Fixed-choice re-admission remains fallible under
foreign pressure, terminal after acceptance, with actual async owners retained.

## Qualification boundary

CPU2.15/d0f2 checks construct actual NativeLinear modules at all three stored
geometries, validate real buffer slots/replacement refusal, verify the FP16 bias
copy exactly, and execute actual installed RMS wrappers against their prior
results. They do not prove CUDA query support, kernel equality, allocation census
or whole-decoder output. Those require the existing real source-backed H3 VAE
probe, one-tile execution and full15s/362-frame/1344x768/30-step/audio gate on the
composed candidate. The original26917/26930 failures remain retained.
