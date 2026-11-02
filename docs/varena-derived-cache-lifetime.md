# Derived cache scope and generation lifetime

A DerivedCache can acquire its Region only after nested component scopes have already
opened. Previously those early scope entries recorded no pin. The first inner get created
one pin; inner exit released it while outer code was still active. Native owner requests
could then discard and rebuild the cache inside that live scope.

Keep an explicit marker for each pre-Region scope and convert every marker to a real Region
pin when the Region is created. Inner exit leaves the outer pin. Ordinary idle discard and
escaped tensor ownership are unchanged. Generation unload now closes the Region and its
native owner explicitly, rather than leaving an empty requestable owner attached to a model
that diagnostics or other Python references may retain.

The actual CPU proof uses a TensorFS file, real resolver/Backend/ModelRegistry/Placement,
and two nested decorated component methods of a tiny real linear model. Baseline8384 loses
the outer pin and builds twice; the candidate keeps one pin, declines discard and builds
once. Both produce exactly1.25 and preserve an escaped4MiB derived tensor. Only the candidate
closes/detaches the Region on real registry unload. CUDA stays uninitialized in this proof.

The ordinary Cozy GPU fixture is staged separately to test native active-owner refusal,
idle discard, escaped storage, rebuild and explicit unload. No GPU or LM-specific KV
qualification follows from the CPU result. No native allocator policy or byte budget changes.
