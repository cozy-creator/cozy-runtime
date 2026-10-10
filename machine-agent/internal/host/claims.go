package host

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"sync"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/canonical"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// Every call carries a Claim: an authorized key's Ed25519 signature over ClaimProof/1, which
// binds it to this machine lifetime and leaf. The daemon verifies it, then speaks to its
// Runtime as the Runtime's only record owner, so any number of clients share one machine.

// runtimeOwnerID and the epoch are the ones every Creator has named, so an adopted journal
// and the answers older clients check keep matching.
const (
	runtimeOwnerID    = "cozy-local-client"
	runtimeOwnerEpoch = 1
)

type claims struct {
	authority        *rentalAuthority
	workerID, bootID string
	leafDigest       []byte
	mu               sync.Mutex
	authorized       []ed25519.PublicKey
	changed          chan struct{} // closed when the authorized set changes
	// own is the daemon's key for its Runtime; the Runtime is launched trusting only it.
	own   ed25519.PrivateKey
	claim *pb.Claim
}

func newClaims(workerID, bootID string, leafDigest []byte, authorized []ed25519.PublicKey) (*claims, error) {
	_, own, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	c := &claims{workerID: workerID, bootID: bootID, leafDigest: leafDigest, authorized: authorized,
		own: own, changed: make(chan struct{})}
	proof, err := c.transcript(runtimeOwnerEpoch)
	if err != nil {
		return nil, err
	}
	c.claim = &pb.Claim{RecordOwnerEpoch: runtimeOwnerEpoch, RecordOwnerId: runtimeOwnerID, WorkerId: workerID,
		WorkerBootId: bootID, WireMinor: pb.WireMinor, Proof: ed25519.Sign(own, proof)}
	return c, nil
}

func (c *claims) useOwn(own ed25519.PrivateKey) error {
	c.own = own
	proof, err := c.transcript(runtimeOwnerEpoch)
	if err != nil {
		return err
	}
	c.claim = &pb.Claim{RecordOwnerEpoch: runtimeOwnerEpoch, RecordOwnerId: runtimeOwnerID, WorkerId: c.workerID, WorkerBootId: c.bootID, WireMinor: pb.WireMinor, Proof: ed25519.Sign(own, proof)}
	return nil
}

// ownKey is the public half the Runtime verifies the daemon's Claim against.
func (c *claims) ownKey() string {
	return base64.RawURLEncoding.EncodeToString(c.own.Public().(ed25519.PublicKey))
}

func (c *claims) transcript(epoch uint64) ([]byte, error) {
	return canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: epoch, WorkerId: c.workerID,
		WorkerBootId: c.bootID, WorkerTlsCertificateDigest: c.leafDigest})
}

// verify admits one client Claim. A Claim is never refused over its wire minor: only new
// work checks that (admitWork).
func (c *claims) verify(claim *pb.Claim) (ed25519.PublicKey, error) {
	if claim == nil {
		return nil, status.Error(codes.Unauthenticated, "the call carries no Claim")
	}
	if claim.WorkerId != c.workerID || claim.WorkerBootId != c.bootID {
		return nil, status.Error(codes.Unauthenticated, "the Claim does not name this machine and boot")
	}
	transcript, err := c.transcript(claim.RecordOwnerEpoch)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "rebuild ClaimProof/1: %v", err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, key := range c.authorized {
		if len(claim.Proof) != ed25519.SignatureSize || !ed25519.Verify(key, transcript, claim.Proof) {
			continue
		}
		return key, nil
	}
	return nil, status.Error(codes.Unauthenticated, "the Claim is not signed by a key this machine authorizes")
}

// admitWork checks the one thing new work needs beyond identity: a client that speaks at
// least the minimum minor. Observation, collection and control of started work never ask.
func admitWork(claim *pb.Claim, operation string) error {
	if claim.GetWireMinor() >= pb.MinCompatibleWireMinor {
		return nil
	}
	return status.Errorf(codes.FailedPrecondition, "%s: %s needs worker protocol %d; the client speaks %d; update the cozy CLI",
		pb.CapabilityUnavailableCode, operation, pb.MinCompatibleWireMinor, claim.GetWireMinor())
}

// forRuntime rewrites a verified client message for the Runtime: every Claim becomes the
// daemon's, and every owner epoch the client named becomes the daemon's. The client's
// values come back through answer.
func (c *claims) forRuntime(message proto.Message, wireMinor uint32) (client *pb.Claim) {
	walk(message.ProtoReflect(), func(m protoMessage) {
		if claim, ok := m.Interface().(*pb.Claim); ok {
			if client == nil {
				client = proto.Clone(claim).(*pb.Claim)
			}
			proto.Reset(claim)
			proto.Merge(claim, c.claim)
			claim.WireMinor = min(c.claim.WireMinor, max(wireMinor, pb.MinCompatibleWireMinor))
			return
		}
		setOwner(m, runtimeOwnerEpoch, runtimeOwnerID)
	})
	return client
}

// answer restores the client's owner values in a Runtime answer.
func (c *claims) answer(message proto.Message, client *pb.Claim) {
	if client == nil || message == nil {
		return
	}
	walk(message.ProtoReflect(), func(m protoMessage) {
		if epoch, ok := field(m, "record_owner_epoch"); ok && m.Get(epoch).Uint() == runtimeOwnerEpoch {
			m.Set(epoch, protoValueOfUint64(client.RecordOwnerEpoch))
		}
		if id, ok := field(m, "record_owner_id"); ok && m.Get(id).String() == runtimeOwnerID && client.RecordOwnerId != "" {
			m.Set(id, protoValueOfString(client.RecordOwnerId))
		}
	})
}

func (c *claims) authorizedKeys() []ed25519.PublicKey {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]ed25519.PublicKey(nil), c.authorized...)
}
