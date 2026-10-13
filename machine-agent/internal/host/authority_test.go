package host

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

func TestRentalAuthorityRevokesOnlyClientTransports(t *testing.T) {
	first, _, _ := ed25519.GenerateKey(rand.Reader)
	second, _, _ := ed25519.GenerateKey(rand.Reader)
	c, err := newClaims("worker", "boot", make([]byte, 32), []ed25519.PublicKey{first, second})
	if err != nil {
		t.Fatal(err)
	}
	own := c.claim
	keys := []ed25519.PublicKey{first, second}
	var failure error
	c.authority = &rentalAuthority{fetch: func(context.Context) ([]ed25519.PublicKey, time.Duration, error) { return keys, time.Hour, failure }}
	if _, err := c.refreshAuthority(t.Context()); err != nil {
		t.Fatal(err)
	}
	lost, stopLost := c.watchAuthority(t.Context(), capability.KeyID(first))
	defer stopLost()
	kept, stopKept := c.watchAuthority(t.Context(), capability.KeyID(second))
	defer stopKept()
	keys = []ed25519.PublicKey{second}
	if _, err := c.refreshAuthority(t.Context()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-lost.Done():
	case <-time.After(time.Second):
		t.Fatal("revoked channel remained open")
	}
	if kept.Err() != nil {
		t.Fatal("another device lost its channel")
	}
	if c.claim != own {
		t.Fatal("revocation changed Runtime owner authority")
	}
	failure = errors.New("Hub unavailable")
	if _, err := c.refreshAuthority(t.Context()); err == nil {
		t.Fatal("offline control remained authorized")
	}
	select {
	case <-kept.Done():
	case <-time.After(time.Second):
		t.Fatal("offline authority remained open")
	}
	if len(c.authorizedKeys()) != 0 {
		t.Fatal("stale authority retained")
	}
}

func TestRentalAuthorityLeaseEndsWithoutFurtherRequests(t *testing.T) {
	key, _, _ := ed25519.GenerateKey(rand.Reader)
	c, _ := newClaims("worker", "boot", make([]byte, 32), nil)
	c.authority = &rentalAuthority{fetch: func(context.Context) ([]ed25519.PublicKey, time.Duration, error) {
		return []ed25519.PublicKey{key}, 30 * time.Millisecond, nil
	}}
	if _, err := c.refreshAuthority(t.Context()); err != nil {
		t.Fatal(err)
	}
	call, cancel := c.watchAuthority(t.Context(), capability.KeyID(key))
	defer cancel()
	select {
	case <-call.Done():
	case <-time.After(time.Second):
		t.Fatal("expired lease left channel authorized")
	}
	if len(c.authorizedKeys()) != 0 {
		t.Fatal("expired keys retained")
	}
}

func TestRevocationClosesRealControlStreamAndRefusesNewRPC(t *testing.T) {
	key, private, _ := ed25519.GenerateKey(rand.Reader)
	c, _ := newClaims("worker", "boot", make([]byte, 32), []ed25519.PublicKey{key})
	var revoked atomic.Bool
	c.authority = &rentalAuthority{fetch: func(context.Context) ([]ed25519.PublicKey, time.Duration, error) {
		if revoked.Load() {
			return nil, time.Hour, nil
		}
		return []ed25519.PublicKey{key}, time.Hour, nil
	}}
	m := &Machine{grant: &Grant{WorkerID: "worker", Lifetime: "rental"}, id: &Identity{BootID: "boot"}, claims: c, log: io.Discard}
	listener := bufconn.Listen(1 << 20)
	defer listener.Close()
	server := grpc.NewServer(grpc.UnknownServiceHandler(m.serveRPC))
	defer server.Stop()
	go server.Serve(listener)
	conn, err := grpc.NewClient("passthrough:///authority", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	client := pb.NewWorkerControlClient(conn)
	stream, err := client.Control(ctx)
	if err != nil {
		t.Fatal(err)
	}
	transcript, _ := c.transcript(1)
	claim := &pb.Claim{WorkerId: "worker", WorkerBootId: "boot", RecordOwnerEpoch: 1, Proof: ed25519.Sign(private, transcript)}
	if err := stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}); err != nil {
		t.Fatal(err)
	}
	ack, err := stream.Recv()
	if err != nil || !ack.GetClaimAck().GetAccepted() {
		t.Fatal(ack, err)
	}
	revoked.Store(true)
	if _, err := c.refreshAuthority(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Recv(); err == nil {
		t.Fatal("revoked control stream stayed open")
	}
	if _, err := client.ListMachineExecutions(ctx, &pb.MachineExecutionListQuery{Claim: claim}); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("revoked new RPC: %v", err)
	}
}

func TestRentalMaintenanceHTTPChecksCurrentAuthority(t *testing.T) {
	public, private, _ := ed25519.GenerateKey(rand.Reader)
	c, _ := newClaims("worker", "boot", make([]byte, 32), []ed25519.PublicKey{public})
	live := true
	offline := false
	c.authority = &rentalAuthority{fetch: func(context.Context) ([]ed25519.PublicKey, time.Duration, error) {
		if offline {
			return nil, 0, errors.New("offline")
		}
		if !live {
			return nil, time.Hour, nil
		}
		return []ed25519.PublicKey{public}, time.Hour, nil
	}}
	m := &Machine{grant: &Grant{WorkerID: "worker", Lifetime: "rental"}, claims: c}
	token, err := capability.Mint(private, capability.Grant{Machine: "worker", Action: capability.Maintenance, Expires: time.Now().Add(time.Hour).Unix()})
	if err != nil {
		t.Fatal(err)
	}
	writes := 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { writes++; w.WriteHeader(http.StatusAccepted) })
	request := func() int {
		r := httptest.NewRequest("POST", "/v1/machine/runtime/update", nil)
		r.Header.Set("Authorization", "Cozy-Cap "+token)
		w := httptest.NewRecorder()
		m.serveAuthorizedHTTP(handler, w, r)
		return w.Code
	}
	if code := request(); code != http.StatusAccepted {
		t.Fatal(code)
	}
	live = false
	if code := request(); code != http.StatusForbidden {
		t.Fatal("revoked update accepted", code)
	}
	offline = true
	if code := request(); code != http.StatusServiceUnavailable {
		t.Fatal("offline update accepted", code)
	}
	if writes != 1 {
		t.Fatal("refused maintenance mutated machine", writes)
	}
}

func TestLeaseExpiryDoesNotWaitForStalledRefresh(t *testing.T) {
	key, _, _ := ed25519.GenerateKey(rand.Reader)
	c, _ := newClaims("worker", "boot", make([]byte, 32), nil)
	var calls atomic.Int32
	entered := make(chan struct{})
	c.authority = &rentalAuthority{fetch: func(ctx context.Context) ([]ed25519.PublicKey, time.Duration, error) {
		if calls.Add(1) == 1 {
			return []ed25519.PublicKey{key}, 40 * time.Millisecond, nil
		}
		close(entered)
		<-ctx.Done()
		return nil, 0, ctx.Err()
	}}
	if _, err := c.refreshAuthority(t.Context()); err != nil {
		t.Fatal(err)
	}
	watched, stop := c.watchAuthority(t.Context(), capability.KeyID(key))
	defer stop()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan struct{})
	go func() { defer close(done); _, _ = c.refreshAuthority(ctx) }()
	<-entered
	select {
	case <-watched.Done():
	case <-time.After(time.Second):
		t.Fatal("stalled refresh prevented lease expiry")
	}
	cancel()
	<-done
}
