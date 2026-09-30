package workerprotov1

// WireMinor is the current release of cozy.worker.v1.
const WireMinor uint32 = 70

// MinCompatibleWireMinor is the oldest peer minor whose preparation and execution this binding
// speaks. It gates those operations only; it never refuses a connection or Claim.
const MinCompatibleWireMinor uint32 = 64

// CapabilityUnavailableCode fails one operation whose peer is outside its range or lacks a
// capability it needs; the detail names the component to update. Other work continues.
const CapabilityUnavailableCode = "capability_unavailable"

const RentalIdleTimeoutSeconds int64 = 900
const MaxRentalKeepaliveRequestIDBytes = 128
