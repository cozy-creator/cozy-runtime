"""`Handler`: a package function of any signature."""

from collections.abc import Callable

#: A package's function or hook of any signature: `describe` reads it with `inspect` and the
#: executor calls it with a decoded request's fields, so no static signature covers it.
type Handler = Callable[..., object]  # type: ignore[explicit-any]  # any package signature (above)
