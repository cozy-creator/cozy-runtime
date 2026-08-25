"""`cozy-runtime new <name>` — the minimal valid endpoint skeleton (§1.0/§1.7)."""

from __future__ import annotations

import re
from pathlib import Path

from cozy_runtime.cli.io import CliError, Result
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


def _module(pkg: str) -> str:
    return f'''"""{pkg} — a cozy-runtime endpoint.

Code states capability; endpoint.toml states selection.
"""

from dataclasses import dataclass

from cozy_runtime.author import App

app = App()


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    prompt: str
    steps: int = 4


@dataclass(frozen=True, slots=True)
class GenerateResult:
    text: str
    steps: int


@app.entrypoint
def generate(request: GenerateRequest) -> GenerateResult:
    # Fake-backed: no weights, no device. Bind a real Model parameter here (cr-002a).
    return GenerateResult(text=f"[{pkg}] {{request.prompt}}", steps=request.steps)
'''


def _example(pkg: str) -> str:
    return f'''"""Runnable fake-backed example: `uv run python -m {pkg}`."""

from {pkg} import GenerateRequest, app, generate


def main() -> None:
    result = generate(GenerateRequest(prompt="a cozy cabin in the woods"))
    print("registered:", ", ".join(r.name for r in app.registrations()))
    print("result:", result.text, f"(steps={{result.steps}})")


if __name__ == "__main__":
    main()
'''


def _readme(name: str, pkg: str) -> str:
    return f"""# {name}

    uv sync
    uv run cozy-runtime describe --write-descriptor
    uv run python -m {pkg}

`endpoint.descriptor.json` and `uv.lock` are generated and committed; the release is
digest(source, lock, manifest).
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

    return Result(
        rows=(
            ("created", str(target)),
            ("package", pkg),
            ("entry", f"{pkg}:app"),
            ("files", str(len(files))),
        ),
        commands=(
            f"cd {target} && uv sync",
            "uv run cozy-runtime describe --write-descriptor",
            f"uv run python -m {pkg}",
        ),
        next=(
            f"cd {target} && uv sync",
            "uv.lock and endpoint.descriptor.json are generated, then committed",
        ),
    )
