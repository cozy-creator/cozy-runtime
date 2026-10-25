"""What a machine asks of the Runtime it installed in a generation, in its own process: package
descriptions (`machine-describe`) and its executors' frame encodings (`frame-codec`). The machine
speaks JSON to these and imports nothing of the Runtime; their requests and replies are read
tolerantly (unknown members ignored).

    cozy-runtime machine-describe   one request on stdin, one reply on stdout
    cozy-runtime frame-codec        one request per stdin line, one answer per stdout line
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path

import msgspec

from cozy_runtime.author._codec import FRAME_MEDIA_TYPES, encode_frame
from cozy_runtime.internal import static_interface
from cozy_runtime.internal.installed import record_digest

# --------------------------------------------------------------------------- machine-describe

#: The package a Runtime's built-in operations run as (the Runtime's rows name them `builtin`).
BUILTIN_PACKAGE = "runtime/operations"


class Describe(msgspec.Struct, tag="describe", tag_field="kind"):
    """A project's interface from its source."""

    project: str
    environment_python: str | None = None


class DescribeInstalled(msgspec.Struct, tag="describe_installed", tag_field="kind"):
    """An installed distribution's interface, without importing it."""

    distribution: str
    environment_python: str


class DescribeEnvironment(msgspec.Struct, tag="describe_environment", tag_field="kind"):
    """This environment's root digest and every other installed App, described statically."""

    root: str
    #: normalized distribution -> the Hub package the machine read from the release's lock
    packages: dict[str, str] = {}
    #: also the root's own installed interface, in the same process
    interface: bool = False


class Callee(msgspec.Struct, frozen=True):
    """Another package's App this environment holds: a job of the root calls its invocables as
    child runs here."""

    distribution: str
    version: str
    application: str
    interface: msgspec.Raw
    source_digest: str = ""
    package: str = ""


class Described(msgspec.Struct, tag="described", tag_field="kind"):
    interface: msgspec.Raw


class DescribedEnvironment(msgspec.Struct, tag="described_environment", tag_field="kind"):
    source_digest: str
    callees: list[Callee]
    #: the root's interface document as JSON text, when asked for
    interface: str = ""


class DescribeFailed(msgspec.Struct, tag="describe_failed", tag_field="kind"):
    code: str
    detail: str


class _Undescribed(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _package_name(found: importlib.metadata.Distribution, packages: dict[str, str]) -> str:
    """The Hub package the machine read from the release's lock, else local code."""
    name = _normalized(found.metadata["Name"])
    package = packages.get(name)
    if package is None:
        return f"local/{name}"
    org, separator, distribution = package.partition("/")
    if not separator or not org or "/" in distribution or _normalized(distribution) != name:
        raise ValueError(f"callee {name!r} names a different package: {package!r}")
    return package


def _environment(request: DescribeEnvironment) -> DescribedEnvironment:
    """This environment's root digest and every other installed App, described statically. An
    App the reader refuses fails the description: left out, its callers would call its raw
    function instead of the package."""
    found = {_normalized(d.metadata["Name"]): d for d in importlib.metadata.distributions()}
    callees = []
    for key, distribution in sorted(found.items()):
        applications = [e for e in distribution.entry_points if e.group == "cozy.application"]
        if key == _normalized(request.root) or not applications:
            continue
        name = distribution.metadata["Name"]
        try:
            document = static_interface.build_installed(
                name, environment_python=Path(sys.executable)
            )
        except Exception as exc:
            raise _Undescribed(
                getattr(exc, "code", type(exc).__name__),
                f"installed package {name} {distribution.version} cannot be described: {exc}",
            ) from exc
        callees.append(
            Callee(
                name,
                distribution.version,
                applications[0].value,
                msgspec.Raw(msgspec.json.encode(document)),
                record_digest(distribution),
                _package_name(distribution, request.packages),
            )
        )
    runtime = found.get("cozy-runtime")
    if runtime is not None:
        # The Runtime's own operations (quantize, prepare_model) are an App every environment
        # holds: a job calls them as child runs, keyed by this Runtime's exact install.
        document = static_interface.build_builtin("operations")
        callees.append(
            Callee(
                "cozy-runtime-operations",
                runtime.version,
                str(document["application"]),
                msgspec.Raw(msgspec.json.encode(document)),
                record_digest(runtime),
                BUILTIN_PACKAGE,
            )
        )
    own = ""
    if request.interface:
        built = static_interface.build_installed(
            request.root, environment_python=Path(sys.executable)
        )
        own = msgspec.json.encode(built).decode()
    return DescribedEnvironment(record_digest(found.get(_normalized(request.root))), callees, own)


def describe() -> int:
    """`machine-describe`: one request on stdin, its reply on stdout; a failure is a reply."""
    reply: Described | DescribedEnvironment | DescribeFailed
    try:
        request = msgspec.json.decode(
            sys.stdin.buffer.read(), type=Describe | DescribeInstalled | DescribeEnvironment
        )
        if isinstance(request, DescribeEnvironment):
            reply = _environment(request)
        elif isinstance(request, DescribeInstalled):
            document = static_interface.build_installed(
                request.distribution, environment_python=Path(request.environment_python)
            )
            reply = Described(msgspec.Raw(msgspec.json.encode(document)))
        else:
            python = Path(request.environment_python) if request.environment_python else None
            document = static_interface.build(Path(request.project), environment_python=python)
            reply = Described(msgspec.Raw(msgspec.json.encode(document)))
    except Exception as exc:
        reply = DescribeFailed(getattr(exc, "code", type(exc).__name__), str(exc))
    sys.stdout.buffer.write(msgspec.json.encode(reply))
    return 0


# --------------------------------------------------------------------------- frame-codec


def _regular(directory: int, name: str) -> int:
    """The spool file `name`, opened as a path: a regular file, never a link or elsewhere."""
    if not name or name != os.path.basename(name) or name in (".", ".."):
        raise ValueError("SDK output has no single owned spool name")
    handle = os.open(name, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            raise ValueError("SDK output is not a regular file")
        return handle
    except BaseException:
        os.close(handle)
        raise


def _read_regular(directory: int, name: str) -> bytes:
    handle = _regular(directory, name)
    try:
        with open(f"/proc/self/fd/{handle}", "rb") as stream:
            return stream.read()
    finally:
        os.close(handle)


def _encode(request: dict[str, object], directory: int) -> dict[str, object]:
    """Each frame encoded into the spool name the machine gave it: its media type, by handle."""
    media: dict[str, str] = {}
    frames = request["frames"]
    assert isinstance(frames, list)
    for frame in frames:
        raw = _read_regular(directory, frame["raw"])
        if len(raw) != frame["raw_bytes"]:
            raise ValueError("registered raw frame length changed")
        name = frame["name"]
        if not name or name != os.path.basename(name) or name in (".", ".."):
            raise ValueError("SDK frame has no owned output binding")
        encoded = encode_frame(frame["codec"], frame.get("facts") or {}, raw)
        temporary = ".codec-" + uuid.uuid4().hex
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600, dir_fd=directory
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
        os.rename(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        media[frame["handle"]] = FRAME_MEDIA_TYPES[frame["codec"]]
    return {"media": media}


def frame_codec() -> int:
    """`frame-codec`: `{spool, frames}` per stdin line, `{media}` or `{error}` per stdout line,
    until stdin ends. One refused output never ends the encoder."""
    for line in sys.stdin:
        answer: dict[str, object]
        try:
            request = json.loads(line)
            directory = os.open(request["spool"], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                answer = _encode(request, directory)
            finally:
                os.close(directory)
        except Exception as exc:
            answer = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
        sys.stdout.write(json.dumps(answer, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0
