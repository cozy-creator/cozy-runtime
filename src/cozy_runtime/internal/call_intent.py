"""Canonical managed-call identity shared by dispatch and native admission."""

from cozy_runtime import canonical_json
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def canonical_intent(request: pb.ChildCallRequest) -> bytes:
    body = {
        "module": request.module,
        "export": request.export,
        "request": canonical_json.decode(request.request_canonical_bytes),
    }
    if request.HasField("capture"):
        body["capture"] = documents.body(request.capture)
    return canonical_json.encode(body)
