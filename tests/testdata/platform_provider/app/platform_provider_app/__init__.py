"""A normal installed application used by the worker platform replacement proof."""

from importlib.metadata import version

import msgspec
from platform_provider_helper import marker  # type: ignore[import-not-found]

from cozy_runtime.author import App

app = App()


class Request(msgspec.Struct):
    text: str


class Reply(msgspec.Struct):
    runtime: str
    tensorfs: str
    helper: str


@app.entrypoint
def versions(payload: Request) -> Reply:
    return Reply(runtime=version("cozy-runtime"), tensorfs=version("tensorfs"), helper=marker)
