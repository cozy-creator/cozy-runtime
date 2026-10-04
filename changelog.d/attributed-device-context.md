An executor's CUDA context is read from its own per-device driver row, never the device-wide
change during its start, so another process allocating then no longer inflates it. Where the
driver hides the row, the context stays unknown.
