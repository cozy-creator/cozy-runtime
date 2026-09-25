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
    document = asdict(recipe) if recipe else None
    if document is not None and recipe is not None:
        document["metadata"] = {
            name: {"length": length, "sha256": digest}
            for name, (length, digest) in recipe.metadata.items()
        }
    return Result(document={"recipe": document})
