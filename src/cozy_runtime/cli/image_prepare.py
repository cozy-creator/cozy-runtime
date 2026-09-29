"""A package-free, bounded seat for the shared image preparation core.

Only the local source path and declared policy cross stdin. A required derivative is
written once under the system temporary directory; the source is never copied unchanged.
"""

from __future__ import annotations

import hashlib
import os
import signal
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import FrameType
from typing import Literal

import msgspec

try:
    import resource
except ImportError:  # Windows has no resource module; optional local preparation defers.
    resource = None  # type: ignore[assignment]

from cozy_runtime.author import _codec
from cozy_runtime.author._decode import (
    DEFAULT_DECODE_LIMITS,
    IMAGE_SOURCE_MAX_BYTES,
    IMAGE_WORKING_MAX_BYTES,
    _av,
    _Budget,
    decode_image_path,
    image_source_geometry,
    image_target_size,
)
from cozy_runtime.author._errors import AuthorError, CapabilityError
from cozy_runtime.author._image_profile import image_preparation_profile
from cozy_runtime.author._markers import IMAGE_PREPARATION_PROFILE, ImagePreparation
from cozy_runtime.author._media import KIND_MEDIA, SNIFF_BYTES, sniff
from cozy_runtime.cli.io import CliError, Result
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.host_paths import input_media_root

_INPUT_LIMIT = 16 << 10


class _Policy(msgspec.Struct):
    profile: str
    max_edge: int | None = None
    max_pixels: int | None = None


class _Request(msgspec.Struct):
    source_path: str
    prepare: _Policy
    media_types: list[str]
    max_bytes: int
    max_decoded_bytes: int


def profile(args: list[str]) -> Result:
    if args:
        raise CliError(
            "usage", "image-preparation-profile takes no arguments", "use --json", Exit.usage
        )
    try:
        codec = _av()
    except CapabilityError:
        codec = None
    return Result(document=image_preparation_profile(codec))


def _request() -> _Request:
    raw = sys.stdin.buffer.read(_INPUT_LIMIT + 1)
    if len(raw) > _INPUT_LIMIT:
        raise ValueError("image preparation input exceeds 16 KiB")
    request = msgspec.json.decode(raw, type=_Request)
    if (
        not Path(request.source_path).is_absolute()
        or not request.prepare.profile
        or len(request.prepare.profile) > 64
        or request.max_bytes <= 0
        or not 0 <= request.max_decoded_bytes <= DEFAULT_DECODE_LIMITS.max_decoded_bytes
        or not request.media_types
        or len(request.media_types) != len(set(request.media_types))
        or not set(request.media_types) <= set(KIND_MEDIA["image"])
    ):
        raise ValueError("image preparation requires absolute paths and valid admitted bounds/MIME")
    return request


def _source(request: _Request) -> tuple[Path, int, str]:
    source = Path(request.source_path)
    facts = source.stat()
    if not stat.S_ISREG(facts.st_mode):
        raise ValueError("image preparation source must be a regular file")
    if facts.st_size > IMAGE_SOURCE_MAX_BYTES:
        raise ValueError("image preparation source exceeds its byte ceiling")
    with source.open("rb") as stream:
        media_type = sniff(stream.read(SNIFF_BYTES))
    if media_type not in request.media_types:
        raise ValueError("original image MIME is not admitted by this input")
    return source, facts.st_size, media_type


def _document(
    status: Literal["prepared", "raw"], path: Path, media_type: str, length: int
) -> dict[str, object]:
    return {
        "status": status,
        "path": str(path),
        "media_type": media_type,
        "length": length,
        "profile": IMAGE_PREPARATION_PROFILE,
    }


def _raw(source: Path, length: int, media_type: str, max_bytes: int) -> Result:
    if length > max_bytes:
        raise ValueError(
            "raw image exceeds admitted bytes and no valid local derivative is available"
        )
    return Result(document=_document("raw", source, media_type, length))


def _prepare(request: _Request) -> Result:
    # Original MIME admission comes before conversion; it is never widened by an encoder.
    source, source_length, source_mime = _source(request)
    policy = ImagePreparation(request.prepare.max_edge, request.prepare.max_pixels)
    if request.prepare.profile != IMAGE_PREPARATION_PROFILE:
        return _raw(source, source_length, source_mime, request.max_bytes)
    geometry = image_source_geometry(source, DEFAULT_DECODE_LIMITS, policy)
    if image_target_size(*geometry, policy) == geometry:
        # Raw fallback keeps normal Runtime validation; do not decode small images twice.
        return _raw(source, source_length, source_mime, request.max_bytes)
    try:
        _av()
    except CapabilityError:
        return _raw(source, source_length, source_mime, request.max_bytes)
    carrier = next(
        (mime for mime in ("image/webp", "image/png") if mime in request.media_types), ""
    )
    if not carrier:
        return _raw(source, source_length, source_mime, request.max_bytes)
    image = decode_image_path(
        source,
        DEFAULT_DECODE_LIMITS,
        _Budget(request.max_decoded_bytes or DEFAULT_DECODE_LIMITS.max_decoded_bytes),
        policy,
    )
    # Choose and encode once. A second format is never tried to improve this result.
    encoded = (
        _codec.encode_webp(image.width, image.height, image.rgb)
        if carrier == "image/webp"
        else _codec.encode_png(image.width, image.height, image.rgb)
    )
    if source_length <= request.max_bytes and source_length <= len(encoded):
        return _raw(source, source_length, source_mime, request.max_bytes)
    if len(encoded) > request.max_bytes:
        return _raw(source, source_length, source_mime, request.max_bytes)
    directory = input_media_root()
    directory.mkdir(mode=0o755, exist_ok=True)
    facts = directory.lstat()
    if not stat.S_ISDIR(facts.st_mode) or facts.st_mode & 0o022:
        raise ValueError(
            "temporary media directory must not be a symlink or writable by other users"
        )
    digest = hashlib.sha256(encoded).hexdigest()
    target = directory / (digest + (".webp" if carrier == "image/webp" else ".png"))
    if not target.exists():
        descriptor, filename = tempfile.mkstemp(prefix=".image-", dir=directory)
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as output:
                if output.write(encoded) != len(encoded):
                    raise OSError("temporary image write was incomplete")
                output.flush()
                os.fchmod(output.fileno(), 0o444)
            with suppress(FileExistsError):
                os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as existing:
        facts = os.fstat(existing.fileno())
        if (
            not stat.S_ISREG(facts.st_mode)
            or facts.st_size != len(encoded)
            or hashlib.file_digest(existing, "sha256").hexdigest() != digest
        ):
            raise ValueError("temporary image file differs from its content hash")
    result = _document("prepared", target, carrier, len(encoded))
    result.update(width=image.width, height=image.height)
    return Result(document=result)


@contextmanager
def _helper_memory_limit() -> Iterator[bool]:
    """Cap this isolated CLI process; never used by the shared Runtime executor."""
    if resource is None or not hasattr(resource, "RLIMIT_AS"):
        yield False
        return
    previous = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = min(
        [IMAGE_WORKING_MAX_BYTES, *(value for value in previous if value != resource.RLIM_INFINITY)]
    )
    try:
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, previous[1]))
    except (OSError, ValueError):
        yield False
        return
    try:
        yield True
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)


def _cancel(_signal: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def run(args: list[str]) -> Result:
    if args:
        raise CliError(
            "usage", "image-prepare reads one request from stdin", "use --json", Exit.usage
        )
    previous = signal.signal(signal.SIGTERM, _cancel)
    try:
        request = _request()
        with _helper_memory_limit() as bounded:
            if not bounded:
                source, length, mime = _source(request)
                return _raw(source, length, mime, request.max_bytes)
            return _prepare(request)
    except MemoryError as error:
        raise CliError(
            "image_preparation_capacity",
            "image preparation exceeded helper memory",
            "use an image within the helper process bound",
            Exit.capacity,
        ) from error
    except KeyboardInterrupt as error:
        raise CliError(
            "canceled", "image preparation canceled", "retry if needed", Exit.canceled
        ) from error
    except AuthorError as error:
        raise CliError(
            error.code, str(error), "supply an admitted bounded image", Exit.validation
        ) from error
    except (OSError, ValueError, msgspec.ValidationError) as error:
        raise CliError(
            "image_preparation_refused",
            str(error),
            "supply one valid declared image preparation request",
            Exit.validation,
        ) from error
    finally:
        signal.signal(signal.SIGTERM, previous)
