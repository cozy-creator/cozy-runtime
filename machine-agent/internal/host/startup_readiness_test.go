package host

import (
	"context"
	"errors"
	"io"
	"net"
	"path/filepath"
	"testing"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

type startupClosurePeer struct {
	pb.UnimplementedWorkerControlServer
	refuse bool
}

func (s *startupClosurePeer) GetMachineExecutionWorkspace(context.Context, *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{ExecutionWorkspaceId: "workspace", SubmissionClose: true}, nil
}
func (s *startupClosurePeer) CloseMachineSubmission(_ context.Context, q *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	if s.refuse {
		return nil, status.Error(codes.Unimplemented, "missing closure")
	}
	return &pb.MachineSubmissionClosure{RequestId: q.RequestId, SubmissionId: q.SubmissionId, ExecutionWorkspaceId: q.ExpectedExecutionWorkspaceId}, nil
}

type startupCommitLauncher struct {
	receipt   *receipt
	committed bool
	refuse    bool
}

func (*startupCommitLauncher) launch(string) (*runtimeProcess, error) {
	return nil, errors.New("not used")
}
func (*startupCommitLauncher) close() {}
func (l *startupCommitLauncher) maintain(op string, _ []string) (string, error) {
	if op != "startup-commit" || l.receipt.Envelope() != nil {
		return "", errors.New("receipt was advertised before private health commit")
	}
	if l.refuse {
		return "", errors.New("commit refused")
	}
	l.committed = true
	return "", nil
}

func TestStartupRentalReceiptIsNotSealedBeforeHealthCommit(t *testing.T) {
	for _, arm := range []string{"closure refused", "commit refused", "healthy"} {
		t.Run(arm, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			server := grpc.NewServer()
			pb.RegisterWorkerControlServer(server, &startupClosurePeer{refuse: arm == "closure refused"})
			go server.Serve(listener)
			defer server.Stop()
			conn, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			r := &receipt{path: filepath.Join(t.TempDir(), "receipt.json"), key: make([]byte, 32)}
			launcher := &startupCommitLauncher{receipt: r, refuse: arm == "commit refused"}
			m := &Machine{grant: &Grant{WorkerPort: 9443, startupPending: true}, claims: &claims{claim: &pb.Claim{}}, receipt: r, conn: conn, launcher: launcher, log: io.Discard, inUpdate: true}
			err = m.commitStartup(&runtimeProcess{incarnation: "fresh", done: make(chan struct{})}, []byte(`{"pod_boot_id":"boot","tls_certificate_der_base64":"leaf"}`))
			if arm == "healthy" {
				if err != nil || !launcher.committed || r.Envelope() == nil || m.startupPending() {
					t.Fatalf("healthy commit: %v", err)
				}
			} else if err == nil || r.Envelope() != nil || !m.startupPending() || len(r.key) != 32 {
				t.Fatalf("failed candidate advertised or spent receipt authority: %v", err)
			}
		})
	}
}
