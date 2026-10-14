An op the GPU has no room for, once weights and idle tenants have given theirs, moves the executor's
own tensors to page-locked host memory in place (largest first, its operands last, at most half the
host's free memory): qwen-image-2 at 1 MP on an 8 GB pool no longer ends `operation_exceeds_device` in
its VAE decode. A closed stage's idle cache goes back to the driver; a block that ran out of memory is
run again with the lists and dicts it was handed as they were, then recovers op by op; pointwise in-place
ops are not copied before they run. Refusals carry a structured shortfall and `spilled` facts.
