"""`cozy-runtime new <name>` — the minimal valid endpoint skeleton (§1.0/§1.7)."""

from __future__ import annotations

import re
from pathlib import Path

from cozy_runtime.cli import describe as describe_verb
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import descriptor
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
requires-python = ">=3.11"
dependencies = ["cozy-runtime"]

[project.optional-dependencies]
# CUDA-extras torch convention: hosts install the extra matching their accelerator.
cu128 = ["torch>=2.6"]

[tool.uv.sources]
{_local_pin()}torch = [{{ index = "pytorch-cu128", extra = "cu128" }}]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{pkg}"]

[tool.mypy]
python_version = "3.11"
files = ["src"]
strict = true
warn_unreachable = true
"""


def _endpoint_toml(pkg: str) -> str:
    return f"""[application]
object = "{pkg}:app"

# Default bindings only — endpoint.toml is a default, never a source of truth.
# [bindings."MyModel"]
# repo = "org/repo@release"
"""


_MODULE = '''"""__PKG__ — a cozy-runtime endpoint.

Code states capability; endpoint.toml states selection. The SIGNATURE is the contract:
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
    # Weightless: no model parameter yet, so this endpoint derives gpu=False. Adding
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
    uv run cozy-runtime describe --check

`endpoint.descriptor.json` was derived by the scaffold and is already `--check`-clean;
re-derive it with `describe --write-descriptor` whenever a signature or schema changes, and
commit the diff — it is how a surface-contract change gets reviewed. `uv.lock` and the
descriptor are the two generated, committed files; the release is
digest(source, lock, manifest), so both re-key it.
"""


def run(name: str, target_dir: str) -> Result:
    if not NAME_RE.match(name):
        raise CliError(
            "usage",
            f"invalid endpoint name {name!r}",
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
        "endpoint.toml": _endpoint_toml(pkg),
        "README.md": _readme(name, pkg),
        ".gitignore": ".venv/\n__pycache__/\ndist/\n",
        f"src/{pkg}/__init__.py": _module(pkg),
        f"src/{pkg}/__main__.py": _example(pkg),
    }
    for rel, body in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    # The skeleton is only MINIMAL VALID once its surface contract is derived and committed
    # (§1.0), so the scaffold runs the same describe the build runs: a fresh endpoint is
    # `describe --check`-clean out of the box, and its first commit already carries the
    # surface a reviewer diffs against.
    written = describe_verb.run(None, Options(dir=str(target)), write=True, check=False)
    assert isinstance(written.document, dict)

    return Result(
        rows=(
            ("created", str(target)),
            ("package", pkg),
            ("entry", f"{pkg}:app"),
            ("files", str(len(files) + 1)),
            (
                descriptor.DIGEST_KEY,
                descriptor.descriptor_digest(descriptor.render(written.document)),
            ),
        ),
        commands=(
            f"cd {target} && uv sync",
            f"uv run python -m {pkg}",
            "uv run cozy-runtime describe --check",
        ),
        next=(
            f"cd {target} && uv sync",
            f"git add {descriptor.FILE} — it is generated, committed, and CI-checked",
        ),
    )
