package host

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
)

// servedMachine serves PodHost through serveRPC in front of a ready Runtime connection.
func servedMachine(t *testing.T, runtime *grpc.ClientConn) pb.PodHostClient {
	t.Helper()
	seed := make([]byte, 32)
	for index := range seed {
		seed[index] = byte(index + 32)
	}
	idle, err := openIdle(filepath.Join(t.TempDir(), "idle.json"), true, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	ready := make(chan struct{})
	close(ready)
	m := &Machine{claims: &claims{workerID: "test-worker", bootID: "test-boot", leafDigest: bytes.Repeat([]byte{0xaa}, 32),
		authorized: []ed25519.PublicKey{ed25519.NewKeyFromSeed(seed).Public().(ed25519.PublicKey)}, claim: progressClaim(t)},
		conn: runtime, idle: idle, owned: true, restarts: &restarts{}, ready: ready, runtimeAck: &pb.ClaimAck{},
		proc: &runtimeProcess{done: make(chan struct{})}, runtimeCaps: []string{"runtime-model-preparation/1"}, log: io.Discard}
	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer(grpc.UnknownServiceHandler(m.serveRPC))
	go server.Serve(listener)
	t.Cleanup(server.Stop)
	conn, err := grpc.NewClient("passthrough:///machine", grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return pb.NewPodHostClient(conn)
}

func TestHostDetachesAndCancelsIdentifiedPreparation(t *testing.T) {
	python := os.Getenv("COZY_RUNTIME_TEST_PYTHON")
	if python == "" {
		t.Skip("set COZY_RUNTIME_TEST_PYTHON to this Runtime's interpreter")
	}
	fixture, root := nativeProgressFixture(t, python)
	runtime, err := grpc.NewClient(fixture.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	raw, err := base64.StdEncoding.DecodeString(fixture.Request)
	if err != nil {
		t.Fatal(err)
	}
	request := &pb.PreparePackageSetRequest{}
	if err := proto.Unmarshal(raw, request); err != nil {
		t.Fatal(err)
	}
	host := servedMachine(t, runtime)
	call := &pb.PreparePackageSetCall{Claim: progressClaim(t), Hub: request.Hub, OperationId: "op-host",
		PackageSet: &pb.DesiredPackageSet{DownloadDelegation: request.DownloadDelegation}}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()

	detached, detach := context.WithCancel(ctx)
	first, err := host.PreparePackageSet(detached, call)
	if err != nil {
		t.Fatal(err)
	}
	reader := &Machine{claims: &claims{claim: progressClaim(t)}}
	models := []preparationModel{{repository: "proof/adapter", manifest: fixture.Manifest}}
	for {
		if rows := reader.preparationProgress(ctx, runtime, models); len(rows) == 1 && rows[0].TransferredBytes > 0 {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("identified preparation did not land its first objects")
		case <-time.After(10 * time.Millisecond):
		}
	}
	detach()
	for err == nil {
		_, err = first.Recv()
	}
	if status.Code(err) != codes.Canceled {
		t.Fatalf("detached stream: %v", err)
	}

	again, err := host.PreparePackageSet(ctx, call)
	if err != nil {
		t.Fatal(err)
	}
	answer, err := host.CancelPreparation(ctx, &pb.CancelPreparationCall{Claim: progressClaim(t), OperationId: "op-host"})
	if err != nil || !answer.Found || len(answer.ModelProgress) != 1 || answer.ModelProgress[0].TransferredBytes == 0 {
		t.Fatalf("cancel answer: %v, %v", answer, err)
	}
	var last *pb.PrepareEvent
	for {
		event, err := again.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("attached stream: %v", err)
		}
		last = event
	}
	if last.GetStage() != pb.PrepareStage_PREPARE_STAGE_REFUSED || last.SafeCode != "preparation_cancelled" {
		t.Fatalf("attached requester did not see its cancellation: %v", last)
	}
	for {
		if _, err := os.Stat(filepath.Join(root, "aborted")); err == nil {
			break // the native fetch closed: cancellation reached TensorFS through the Host
		}
		select {
		case <-ctx.Done():
			t.Fatal("cancelled native fetch kept running")
		case <-time.After(10 * time.Millisecond):
		}
	}
	if answer, err := host.CancelPreparation(ctx, &pb.CancelPreparationCall{Claim: progressClaim(t), OperationId: "op-host"}); err != nil || answer.Found {
		t.Fatalf("repeated cancel is not an idempotent no-op: %v, %v", answer, err)
	}
	forged := progressClaim(t)
	forged.Proof[0] ^= 1
	if _, err := host.CancelPreparation(ctx, &pb.CancelPreparationCall{Claim: forged, OperationId: "op-host"}); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("unclaimed cancel: %v", err)
	}
}

type olderRuntime struct {
	pb.UnimplementedRuntimePreparationServer
}

func TestHostAnswersCapabilityUnavailableForAnOlderRuntime(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	pb.RegisterRuntimePreparationServer(server, olderRuntime{})
	go server.Serve(listener)
	defer server.Stop()
	runtime, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	_, err = servedMachine(t, runtime).CancelPreparation(t.Context(), &pb.CancelPreparationCall{Claim: progressClaim(t), OperationId: "op"})
	if status.Code(err) != codes.FailedPrecondition || !strings.HasPrefix(status.Convert(err).Message(), pb.CapabilityUnavailableCode+":") {
		t.Fatalf("older Runtime: %v", err)
	}
}
