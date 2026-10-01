Model weights run on the TensorFS weight plane (`tensorfs.plane`, TensorFS 0.3.84). An executor
keeps a pinned host copy of each weight set and maps into VRAM what a stage's budget holds; the
rest streams block by block in the learned execution order. The Worker budgets VRAM per GPU and
tenant, cuts idle tenants least recently used first by unmapping only, and refuses only below the
weights' floor, with numbers. A device out-of-memory at a block boundary lowers the budget and
retries the block instead of ending the executor. An image VAE decode runs untiled when its
measured peak fits, else tiled, halving a tile that does not fit down to 128 px; this replaces the
decode stopgap. `residency.py`, the synchronous paging and staging paths, and the NVML sampler are
gone. SDXL at 1024² on an 8 GB card: 0.63 s/step in 3 GiB, and it completes in 1 GiB. A model
switched away from returns from the pinned copy without reading the disk.
