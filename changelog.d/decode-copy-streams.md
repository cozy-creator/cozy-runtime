An executor makes its host-map decode copy streams at device init. Made at the first decode on a
full card, the driver refused them (run 4297: `cuStreamCreateWithPriority`, out of memory) and the
run failed. A host-map decode that runs out of device memory now makes room and runs again.
