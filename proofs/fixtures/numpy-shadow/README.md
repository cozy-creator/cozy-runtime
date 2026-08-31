# Differently named NumPy-root fixture

Repository-only red fixture. Its distribution name is new, but its wheel exports the base-owned
`numpy` import root; Runtime must refuse it before uv installation.
