package host

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/canonical"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// The TensorFS transport log is read through the machine's real TLS/gRPC API while its
// Runtime is down: rotated file first, a tail from a line start, chunks under the bound.
func TestTensorFSTransportLogIsReadWhileTheRuntimeIsDown(t *testing.T) {
	root := t.TempDir()
	l := NewLayout(&Grant{Root: root})
	m := &Machine{layout: l, launcher: &directLauncher{root: root}, restarts: &restarts{}, log: io.Discard}
	public, key, _ := ed25519.GenerateKey(rand.Reader)
	port, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := port.Addr().String()
	m.grant = &Grant{Root: root, Lifetime: "persistent", WorkerID: "local-machine", ListenHost: "127.0.0.1",
		WorkerPort: port.Addr().(*net.TCPAddr).Port, Authorized: []ed25519.PublicKey{public}}
	port.Close()
	if m.id, err = prepare(m.grant, l); err != nil {
		t.Fatal(err)
	}
	if m.claims, err = newClaims(m.grant.WorkerID, m.id.BootID, m.id.Digest, m.grant.Authorized); err != nil {
		t.Fatal(err)
	}
	m.restarts.fail(&launchRefusal{"runtime_launch_refused", "fixture Runtime is stopped"})
	listeners, err := m.listen()
	if err != nil {
		t.Fatal(err)
	}
	defer listeners.close()
	roots := x509.NewCertPool()
	roots.AddCert(mustLeaf(m.id.Leaf))
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{RootCAs: roots, ServerName: ServerName})))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	claim := &pb.Claim{RecordOwnerId: "cli", RecordOwnerEpoch: 1, WorkerId: m.grant.WorkerID, WorkerBootId: m.id.BootID, WireMinor: pb.WireMinor}
	proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: 1, WorkerId: claim.WorkerId, WorkerBootId: claim.WorkerBootId, WorkerTlsCertificateDigest: m.id.Digest})
	if err != nil {
		t.Fatal(err)
	}
	claim.Proof = ed25519.Sign(key, proof)
	read := func(query *pb.MachineLogQuery) ([]byte, int, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		stream, err := pb.NewPodHostClient(conn).ReadMachineLog(ctx, query)
		if err != nil {
			return nil, 0, err
		}
		var out []byte
		chunks := 0
		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				return out, chunks, nil
			}
			if err != nil {
				return out, chunks, err
			}
			if len(chunk.GetData()) == 0 || len(chunk.GetData()) > pb.MaxMachineLogChunkBytes {
				t.Fatalf("chunk of %d bytes", len(chunk.GetData()))
			}
			out, chunks = append(out, chunk.GetData()...), chunks+1
		}
	}
	transport := &pb.MachineLogQuery{Claim: claim, Log: pb.MachineLog_MACHINE_LOG_TENSORFS_TRANSPORT}

	if _, chunks, err := read(transport); err != nil || chunks != 0 {
		t.Fatalf("a log TensorFS has not written: %d chunks, %v", chunks, err)
	}
	dir := filepath.Join(l.Store, "logs")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	var older, current bytes.Buffer
	for i := range 2000 {
		fmt.Fprintf(&older, "1790879965.%03d hedge %016x 67108864 ranged lanes=2 rate_bps=5252 left=67043328 measured=true busy=128 level=128\n", i%1000, i)
	}
	fmt.Fprintf(&current, "1790881094.217 won ea78bbefeffbd4ea 67108864 ranged racers=2 after_s=3.0\n")
	fmt.Fprintf(&current, "1790881096.936 walk 4477 101123198626 fetched=4477 cached=0 moved=101123198626 seconds=231.4 level=128 hedges=29 hedges_won=26 outcome=ok\n")
	if err := os.WriteFile(filepath.Join(dir, "transport.log.1"), older.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "transport.log"), current.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	whole := append(append([]byte{}, older.Bytes()...), current.Bytes()...)
	data, chunks, err := read(transport)
	if err != nil || !bytes.Equal(data, whole) || chunks < 2 {
		t.Fatalf("whole log: %d of %d bytes in %d chunks, %v", len(data), len(whole), chunks, err)
	}
	tail := &pb.MachineLogQuery{Claim: claim, Log: transport.Log, TailBytes: uint64(current.Len() + 10)}
	if data, _, err = read(tail); err != nil || !bytes.Equal(data, current.Bytes()) {
		t.Fatalf("tail from a line start: %q, %v", data, err)
	}
	if _, _, err = read(&pb.MachineLogQuery{Claim: claim, Log: pb.MachineLog(99)}); status.Code(err) != codes.NotFound {
		t.Fatalf("a log a newer client names: %v", err)
	}
	stranger := &pb.Claim{RecordOwnerId: "cli", RecordOwnerEpoch: 1, WorkerId: claim.WorkerId, WorkerBootId: claim.WorkerBootId, WireMinor: pb.WireMinor, Proof: make([]byte, ed25519.SignatureSize)}
	if _, _, err = read(&pb.MachineLogQuery{Claim: stranger, Log: transport.Log}); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("an unsigned Claim read the log: %v", err)
	}
}
