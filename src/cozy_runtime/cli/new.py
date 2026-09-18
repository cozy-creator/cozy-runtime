"""`cozy-runtime new <name>` — the minimal valid package skeleton (§1.0/§1.7)."""

from __future__ import annotations

import re
from pathlib import Path

from cozy_runtime import __version__
from cozy_runtime.cli import describe as describe_verb
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import package_interface, python_interpreters
from cozy_runtime.internal.exits import Exit

NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")

# Pre-launch: cozy-runtime is not on PyPI, so a scaffold made from a source checkout pins
# the checkout as a uv source. From an installed distribution the pin is omitted.
_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _local_pin() -> str:
    if (_SOURCE_ROOT / "pyproject.toml").is_file():
        return f'cozy-runtime = {{ path = "{_SOURCE_ROOT}" }}  # pre-launch\n'
    return ""


def _pyproject(name: str, pkg: str) -> str:
    return f"""[project]
name = "{name}"
version = "0.1.0"
requires-python = ">={python_interpreters.supported_minors()[0]}"
dependencies = ["cozy-runtime>={__version__},<1.0.0"]

[project.entry-points."cozy.application"]
default = "{pkg}:app"

[project.optional-dependencies]
# CUDA-extras torch convention: hosts install the one admitted CUDA generation.
cu130 = ["torch>=2.13,<3"]

[tool.uv.sources]
{_local_pin()}torch = [{{ index = "pytorch-cu130", extra = "cu130" }}]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{pkg}"]

[tool.mypy]
python_version = "{python_interpreters.supported_minors()[0]}"
files = ["src"]
strict = true
warn_unreachable = true
"""


def _package_toml(pkg: str) -> str:
    return f"""[application]
object = "{pkg}:app"

# Default bindings only — package.toml is a default, never a source of truth.
# [bindings."MyModel"]
# model = "org/model@release"
"""


_MODULE = '''"""__PKG__ — a cozy-runtime package.

Code states capability; package.toml states selection. The SIGNATURE is the contract:
naming `out: Outputs` IS declaring the save capability, and a `model: MyModel` parameter
is what derives `gpu`. Nothing else declares either.
"""

import msgspec

from cozy_runtime.author import App, Context, ImageAsset, ImageFrame, Outputs

app = App()


class GenerateRequest(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    # Axis-named fields normalize into RequestFeatures with nothing authored.
    width: int = 64
    height: int = 64
    seed: int | None = None


class GenerateResult(msgspec.Struct):
    image: ImageAsset
    prompt: str


@app.entrypoint
def generate(ctx: Context, payload: GenerateRequest, out: Outputs) -> GenerateResult:
    # Weightless: no model parameter yet, so this package derives gpu=False. Adding
    # `model: MyModel` binds real weights and gives you `model.for_request(ctx, seed=...)`.
    ctx.raise_if_cancelled()
    pixels = bytearray()
    for y in range(payload.height):
        for x in range(payload.width):
            pixels += bytes((x % 256, y % 256, len(payload.prompt) % 256))
    frame = ImageFrame(payload.width, payload.height, bytes(pixels))
    return GenerateResult(
        image=out.save_image(frame, format="png"), prompt=payload.prompt
    )
'''


def _module(pkg: str) -> str:
    return _MODULE.replace("__PKG__", pkg)


_EXAMPLE = '''"""Runnable, weightless, no GPU: `uv run python -m __PKG__`.

The fakes are capability adapters over the PRODUCTION invocation kernel — the decode,
injection, validation and encode below are the ones a served request runs.
"""

from cozy_runtime.author import describe
from cozy_runtime.author.fakes import invoke_with_fakes

from __PKG__ import app


def main() -> None:
    for surface in describe(app):
        caps = ", ".join(sorted(surface.capabilities)) or "(none)"
        print(f"registered: {surface.name}  gpu={surface.gpu}  capabilities: {caps}")
    envelope = invoke_with_fakes(
        app.get("generate"), {"prompt": "a cozy cabin in the woods", "seed": 7}
    )
    print("result:  ", envelope.result)
    print("outputs: ", envelope.outputs)
    print("features:", envelope.features.values)


if __name__ == "__main__":
    main()
'''


def _example(pkg: str) -> str:
    return _EXAMPLE.replace("__PKG__", pkg)


def _readme(name: str, pkg: str) -> str:
    return f"""# {name}

    uv sync
    uv run python -m {pkg}
    uv run cozy-runtime --json describe

`cozy package publish` invokes Runtime's read-only describe command and sends the canonical
PackageInterface bytes with the source release. Tensorhub independently derives and compares the
same interface from the installed project wheel. Nothing generated is committed here.
"""


def run(name: str, target_dir: str) -> Result:
    if not NAME_RE.match(name):
        raise CliError(
            "usage",
            f"invalid package name {name!r}",
            "use a lowercase name starting with a letter: [a-z][a-z0-9_-]*",
            Exit.usage,
        )
    pkg = name.replace("-", "_")
    target = Path(target_dir).expanduser() / name
    if target.exists() and any(target.iterdir()):
        raise CliError(
            "conflict",
            f"{target} exists and is not empty",
            "choose another name, or remove the directory, or scaffold elsewhere with --dir",
            Exit.conflict,
        )

    files = {
        "pyproject.toml": _pyproject(name, pkg),
        "package.toml": _package_toml(pkg),
        "README.md": _readme(name, pkg),
        ".gitignore": ".venv/\n__pycache__/\ndist/\n",
        f"src/{pkg}/__init__.py": _module(pkg),
        f"src/{pkg}/__main__.py": _example(pkg),
    }
    for rel, body in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    # Run the same derivation publication runs. It validates the scaffold without creating
    # another generated repository file.
    described = describe_verb.run(None, Options(dir=str(target)))
    assert isinstance(described.document, dict)

    return Result(
        rows=(
            ("created", str(target)),
            ("package", pkg),
            ("entry", f"{pkg}:app"),
            ("files", str(len(files))),
            (
                package_interface.DIGEST_KEY,
                package_interface.package_interface_digest(described.document),
            ),
        ),
        commands=(
            f"cd {target} && uv sync",
            f"uv run python -m {pkg}",
            "uv run cozy-runtime --json describe",
        ),
        next=(f"cd {target} && uv sync",),
    )
