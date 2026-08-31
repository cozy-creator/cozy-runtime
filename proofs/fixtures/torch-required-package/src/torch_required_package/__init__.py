from typing import Literal

from msgspec import Struct

from cozy_runtime import App

app = App()


class Request(Struct):
    message: Literal["marco"]


class Response(Struct):
    message: Literal["polo"]


@app.function
def marco(_payload: Request) -> Response:
    return Response(message="polo")
