"""Read a reviewed source recipe without downloading or importing inference."""

from dataclasses import asdict

from cozy_runtime.author import UnsupportedInput
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal.exits import Exit
from cozy_runtime.models.ingestion import huggingface_recipe


def run(args: list[str], opts: Options) -> Result:
    if len(args) != 2:
        raise CliError("usage", "Expected repository and immutable revision", "", Exit.usage)
    try:
        recipe = huggingface_recipe(*args)
    except UnsupportedInput as exc:
        raise CliError(
            exc.code, str(exc), "Use the reviewed source revision", Exit.validation
        ) from exc
    return Result(document={"recipe": asdict(recipe) if recipe else None})
