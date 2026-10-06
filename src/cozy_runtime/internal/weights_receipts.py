"""Digest spellings and native receipt exchange limits."""

from __future__ import annotations


from cozy_runtime.internal import canonical

MAX_RECEIPT_BYTES = 1 << 20


class ReceiptError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def spell(digest: bytes) -> str:
    if len(digest) != 32:
        raise ReceiptError("malformed_digest", f"{len(digest)} B is not a 32-byte SHA-256")
    return "sha256:" + digest.hex()


def raw(digest: str) -> bytes:
    if canonical.DIGEST_RE.fullmatch(digest) is None:
        raise ReceiptError("malformed_digest", f"{digest!r} is not a lowercase sha256: digest")
    return bytes.fromhex(digest[7:])
