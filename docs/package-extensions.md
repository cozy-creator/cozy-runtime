# Package extensions: kernels, execution plans, attention producers, splittable ops

Model code lives in the model's package; the Runtime stays generic (owner ruling,
2026-10-14). Three public extension points let a package bring what used to need Runtime
code. All are typed and documented at their definitions.

Memory is the Runtime's: a package never sizes a tile, a chunk or a batch for a card.

## Kernels — `cozy_runtime.author.kernels`

Hand-written kernels only (never `torch.compile`), compiled on the GPU that runs them.

```python
GLUE = TritonKernels("my-model-glue", [TritonEntry(_norm_kernel, signature, {"BLOCK": 8192})])
```

Each executor that imports the package submits the set's build for its GPUs when it starts,
beside the weight download (the machine's compile manager, niced, once per key). `Plan.load`
waits for that build and returns a launcher, `launch(kernel, grid, *args)`; a request never
compiles. Launch operands that are unmapped plane weights are mapped for the launch.
`KernelsUnavailable.code` is `kernels_absent`, `kernels_failed` or `kernels_corrupt`.

Also here: `PreQuantized` (the operand an `fp8-rowwise` leaf, provider `ROWWISE_FP8`, takes
already quantized), `readable`, and the Runtime's compiled `rms_rope_split_half` (`native()`
says why a wheel lacks it), plane-safe.

## Execution plans — `Model.execution_plan(plan) -> str`

Called once per construction on every GPU: after the fill, before attention selection and
sequence parallelism, outside the construction guards. Install fused forwards and hooks
here; return the plan's identity (`""` for none), which enters the generation's identity
and the prepare's `execution_plan` fact. A raise refuses the construction with its `code`.

## Attention producers — `AttentionProducer`, `register_attention`

A package's own attention call, selectable by name (`choose_attention`, `--attention-kernel`)
and served through a Runtime sparse base (`base`, e.g. `kitchen-sol-fp4-fixed-means`): the
base's admission, build and dense kernel, with the model's `AttentionLayout`. At selected sites
the module's call reaches `produce(call, *args)`; `ProducerCall` carries the layout, site, the
base's kernel module, the request's `state()`, the Ulysses group and `exchange`, the leaf
quantizer, `original` and `served`. `AttentionProducer.request_state()` reaches the same state
from a package hook. A package name takes precedence over a Runtime one it repeats.

MiniMax-H3 (`minimax_h3/glue.py`, `minimax_h3/producer.py`) and Anima (`anima/optimization.py`)
are the reference users.

## Splittable ops — `split(fn, x, *cuts)`

A package says how an op splits; the Runtime says how finely. `Cut(dim, scale, overlap, align)`
names a dimension `fn` may run on windows of: the output's extent is `scale` times the input's,
starts fall on multiples of `align`, and neighbours share `overlap` input elements that the
output cross-fades (none: the pieces join exactly, as a batch does).

```python
decoded = split(self.decode_tiles, z, Cut(-2, scale=8, overlap=8), Cut(-1, scale=8, overlap=8))
```

The op runs whole when the device can hold it. Its peak bytes per input element are measured
in the executor; later calls start at the fewest pieces that fit what is free plus the idle
weights the plane may unmap (which unmap first). A piece that still runs out of memory runs
again at half the size. A run that split records `<op>.pieces`.
