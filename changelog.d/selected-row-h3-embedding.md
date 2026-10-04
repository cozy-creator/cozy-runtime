When fine-grain planning cannot fit H3's 1.449 GiB plain bf16 token embedding,
read only the requested rows from verified machine object descriptors into a bounded
buffer. Preserve index order, dtype, device and output values without copying the
complete table. Normal resident execution remains available when the budget permits.

CPU descriptor tests pass. CUDA operator, real machine selection and ordinary-CLI
conditioning qualification remain required; this does not establish that all of H3's
downstream activations fit in 1 GiB.
