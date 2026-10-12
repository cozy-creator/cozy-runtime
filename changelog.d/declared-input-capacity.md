When a worker omits its input-byte ceiling, entrypoints and CPU jobs now honor each
installed input field's declared `AssetBound(max_bytes=...)`. Undeclared fields keep
the 64 MiB default, and an explicit deployment ceiling still narrows declared bounds.
Large retained artifacts can pass between child calls without a worker protocol change.
