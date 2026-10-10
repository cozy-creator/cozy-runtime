# Hopper attention partition diagnostics

Research-only ordinary Cozy CLI application. This refresh retains the fixed-Kmean kernels from PR1291 and calls the supported public Runtime SageAttention2 SM90 adapter. It adds direct BF16 CuTe sparse and Kitchen INT8 dense controls with full N=109225, H=56, D=128 synthetic BF16 operands. Local head groupings 2/4/7/8 run on one H100; they are **not actual multi-GPU tests**.

No quality videos or serving/default changes. Each check will have an independent bounded request and report actual backend identity, repeats, input custody, precision differences and warm timing. Unsupported hardware or an unserved sparse kernel refuses the check instead of silently changing the recipe. Parent owns all paid actions; this source is not GPU-qualified.

The public package pin is 0.23.8/TensorFS0.6.1; exact consumed source and binary identities are recorded as evidence, not arbitrary runtime hash admission gates. The original PR1291 source/capture stays frozen.
