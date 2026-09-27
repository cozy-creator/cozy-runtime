MiniMax H3's resident-weight numerical check scans FP8 weights in 16M-element reductions
instead of 1M, and uses one fused max-abs reduction per chunk. On the staged placement rung
the DiT is re-staged and re-scanned every request; the old bound split it into ~34k chunks
and put ~2.6 s of launch overhead into denoise step 1.
