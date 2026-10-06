# Machine and Runtime roles

The Rust machine owns accepted runs, preparation, scheduling, process lifecycle,
software updates and its durable journal. Local and remote clients reach its
`cozy.machine.v1` API. Runtime contains the Python author API and device executor;
it does not expose a Python worker server or worker protobuf facade.

The machine launches `cozy_runtime.internal.executor` through the private bounded
JSON/descriptor channel. The executor imports the selected package, constructs
and invokes it, and consumes machine-granted native TensorFS source/output and
weight-plane handles. TensorFS owns the native storage semantics.

The Runtime CLI reads/scaffolds package interfaces and reports installed
distribution/build provenance. It does not advertise the retired worker wire
contract. The machine API and journal are documented in tensord; the former
Python worker database views and capability-report command are retired.
