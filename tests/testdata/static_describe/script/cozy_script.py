"""An ordinary client script: Creator snapshots it and generates the entry beside it."""

from pathlib import Path

from cozy_runtime.author import ScriptContext


def main(ctx: ScriptContext) -> None:
    ctx.log(f"running in {Path.cwd()}")
