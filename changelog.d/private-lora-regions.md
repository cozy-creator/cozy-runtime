A region a request's LoRA touches stays this executor's private copy in per-step mode too, so
a refused read-only offer no longer drops and refills it every stage. Baked or partly modified
bytes never claim the stored layout. Plain regions keep Degree 2 attach and export.
