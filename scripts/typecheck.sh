#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

# Inspect the actual project environment; do not replace first-party imports
# with Any merely because the checker runs in its own lightweight tool venv.
if [ -n "${COZY_TYPECHECK_PYTHON:-}" ]; then
    typing_python=$COZY_TYPECHECK_PYTHON
elif [ -x .venv/bin/python ]; then
    typing_python="$PWD/.venv/bin/python"
elif [ -x .venv/Scripts/python.exe ]; then
    typing_python="$PWD/.venv/Scripts/python.exe"
else
    typing_python=python3
fi
exec uv run --no-project --python 3.12 --with 'mypy>=1.13' \
    --with 'types-protobuf>=6.30' mypy --config-file pyproject.toml \
    --python-executable "$typing_python" "$@"
