A stage that scopes several components (Anima's `render`) no longer keeps the idle ones' weights
leased while the running one runs out of device memory: they give their regions back, a read of
one maps it for that op, and its next block maps its common again.
