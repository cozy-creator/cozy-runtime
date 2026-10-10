# H100 Sage2 fixed K-mean research

The completed fixed-input H100 probe5232 found the first recorded head-partition difference in BF16 K smoothing means. At seven/eight partitions it propagated to live quantized K codes and dense output; no collectives were involved.

This isolated package will compare the actual public per-thread Sage2 SM90 adapter, an exact replay through its exposed quantizers/native attention call, and that same replay with a fixed token-reduction tree for K mean. It changes no Runtime default, sparse path, V/P conversion, or attention accumulation.

CPU checks and a later explicitly authorized ordinary-CLI GPU diagnostic must establish public-replay equality, input custody, intermediate and output repeat/partition identity, and complete-attention speed. No GPU is authorized by this document. Existing quality runs continue unchanged.

A partition-stable kernel may differ from the original floating-point reduction order. Synthetic identity does not establish full-video identity or visual quality. A candidate must not impose a speed penalty before any serving integration is considered.
