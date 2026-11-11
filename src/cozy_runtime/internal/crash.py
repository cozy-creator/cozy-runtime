"""A fatal signal's record (cr-275). The Linux wheels' `_crash` (`kernels/csrc/crash.c`) writes
every thread's native stack beside faulthandler's Python ones, what the executor was answering
(`note`) and `executor crashed: <signal> in <frame>` to stderr, which the machine keeps with the
run; the process still dies by the signal. `install` goes right after `faulthandler.enable`, which
it chains to; `arm` gives a thread an alternate signal stack. Other wheels carry none: faulthandler
alone."""

from __future__ import annotations

try:
    from cozy_runtime._kernels._crash import arm, install, note
except ImportError:  # the pure wheel and Windows

    def install() -> None:
        pass

    def arm() -> None:
        pass

    def note(text: str, /) -> None:
        pass
