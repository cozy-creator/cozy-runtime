# Packaged VAE edge checks

Runtime source d2e5501b combines the three additive prepared VAE consumers and byte-span validation. Wheel0.37.5+dev.gd2e5501b SHA338304db98697f0098f52d983d9ee105950ebea9c16dcc1b6cce74b26ee7b5df preserves835RECORD entries and581 unchanged native members. The outer decoder chooser and shared block path have not been cut over.

The actual packaged prefix/tail clients pass with Torch2.14.0+cpu, source08187d9e, from cached site /home/fidika/.cache/uv/archive-v0/ze1ma6eT_qRYunlo, using the existing torch-cpu Python3.12 executable. All run with CUDA hidden, one CPU thread and nice19. Prefix fused-bias projection is a declared numerical alternative; packing/RoPE are exact. Tail output comparisons are exact. Actual CPU distinct-Storage/shared-address span checks pass on CPU2.14.1.

The first span run used the CUDA-enabled pinned SDK. Its final whole-process libcuda-absence assertion failed; this receipt is preserved and is not a product span failure. Correct CPU environments were then used; no assertions or product code were relaxed. No CUDA provider execution, GPU allocation, raw Source retirement, full H3 output or performance is qualified here.
