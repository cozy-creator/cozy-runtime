package host

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/canonical"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// This exercises the installed Python decoder, real read-only SQLite connection,
// and Go media fold. It deliberately supplies only the public read views: any
// accidental dependency on private Runtime tables fails the test.
func TestPublicJournalServesStatusEventsAndMediaWithoutRuntime(t *testing.T) {
	python := os.Getenv("COZY_RUNTIME_TEST_PYTHON")
	if python == "" {
		t.Skip("set COZY_RUNTIME_TEST_PYTHON to an installed Runtime venv interpreter")
	}
	python, err := filepath.Abs(python)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	l := NewLayout(&Grant{Root: root})
	if err := os.MkdirAll(filepath.Join(root, "opt/cozy"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Dir(filepath.Dir(python)), filepath.Join(root, "opt/cozy/python")); err != nil {
		t.Fatal(err)
	}
	setup := `
import hashlib,sqlite3,sys
from pathlib import Path
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
store=Path(sys.argv[1]); journal=store/'.cozy-workspace/journal.sqlite3';journal.parent.mkdir(parents=True)
data=b'completed media survives the coordinator';digest=hashlib.sha256(data).digest();name=digest.hex()
blob=store/'blobs'/name[:2]/name[2:4]/name;blob.parent.mkdir(parents=True);blob.write_bytes(data)
product=documents.canonical_bytes(pb.RunProduct(output='text',op=pb.RUN_PRODUCT_OP_SET,content=pb.Ref(digest=digest,length=len(data)),media_type='text/plain'))
outcome=documents.canonical_bytes(pb.AttemptOutcomeBody(status=pb.OUTCOME_STATUS_SUCCEEDED))
db=sqlite3.connect(journal)
db.executescript('''
CREATE VIEW machine_workspace_v1 AS SELECT 'workspace' AS workspace_id;
CREATE TABLE runs AS SELECT 'cozy-local-client' AS record_namespace,'workspace' AS workspace_id,1 AS run_number,'request-1' AS request_id,1 AS attempt_ordinal,1 AS generation,'succeeded' AS state,0 AS collected,2 AS sequence,0 AS compacted_through,100 AS accepted_at_ms,200 AS finished_at_ms,'local-machine' AS worker_id,'boot' AS worker_boot_id;
CREATE VIEW machine_runs_v1 AS SELECT * FROM runs;
CREATE TABLE events(request_id,sequence,attempt_ordinal,at_ms,kind,body_json,invocation_spec_digest,outcome_id,outcome_digest,outcome_json);
CREATE VIEW machine_events_v1 AS SELECT 'cozy-local-client' AS record_namespace,* FROM events;
''')
db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',('request-1',1,1,100,'product',product,b'','','',b''))
db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',('request-1',2,1,200,'outcome',b'{}',bytes(32),'outcome-1',hashlib.sha256(outcome).digest(),outcome))
db.commit();db.close()
`
	if out, err := childOutput(exec.Command(python, "-I", "-c", setup, l.Store), true); err != nil {
		t.Fatalf("fixture: %v: %s", err, out)
	}
	m := &Machine{layout: l, launcher: &directLauncher{root: root}, restarts: &restarts{}, grant: &Grant{WorkerID: "local-machine"}, id: &Identity{BootID: "boot"}}
	answer, handled, err := m.publicFallback(context.Background(), &pb.MachineExecutionQuery{RequestId: "request-1", ExpectedExecutionWorkspaceId: "workspace"})
	if err != nil || !handled || answer.(*pb.MachineExecutionState).State != "succeeded" {
		t.Fatalf("state: %v %v", answer, err)
	}
	_, _, err = m.publicFallback(context.Background(), &pb.MachineExecutionQuery{RequestId: "request-1", ExpectedExecutionWorkspaceId: "other"})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("wrong workspace: %v", err)
	}
	for _, after := range []uint64{3, 1 << 63} {
		_, _, err = m.publicFallback(context.Background(), &pb.MachineExecutionEventsQuery{Execution: &pb.MachineExecutionQuery{RequestId: "request-1"}, After: after})
		if status.Code(err) != codes.FailedPrecondition {
			t.Fatalf("invalid cursor %d: %v", after, err)
		}
	}
	listAnswer, _, err := m.publicFallback(context.Background(), &pb.MachineExecutionListQuery{NewestFirst: true, AfterNumber: 100})
	if err != nil || len(listAnswer.(*pb.MachineExecutionList).Executions) != 1 {
		t.Fatalf("newest-first incorrectly used ascending cursor: %v %v", listAnswer, err)
	}
	tail, cancelTail := context.WithTimeout(context.Background(), 5*time.Second)
	_, _, err = m.publicFallback(tail, &pb.MachineExecutionEventsQuery{Execution: &pb.MachineExecutionQuery{RequestId: "request-1"}, After: 2, Wait: true})
	cancelTail()
	if err != nil {
		t.Fatalf("terminal event tail did not settle: %v", err)
	}
	snapshot, err := m.Open(1, "text", -1)
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	data, err := io.ReadAll(io.NewSectionReader(snapshot.Body, 0, snapshot.Length))
	if err != nil || string(data) != "completed media survives the coordinator" || !snapshot.Final || !strings.HasPrefix(snapshot.SHA256, "sha256:") {
		t.Fatalf("media: %q %#v %v", data, snapshot, err)
	}
	entries, err := m.Entries(context.Background(), 1, 1)
	if err != nil || len(entries) != 1 || entries[0].Status != "completed" {
		t.Fatalf("events: %+v %v", entries, err)
	}
	if _, err := os.Stat(filepath.Join(l.State, "runs")); !os.IsNotExist(err) {
		t.Fatal("read made a second run journal")
	}

	// An authorized client reads through the real TLS/gRPC API without opening a
	// Control stream. The response marks the saved state as a journal snapshot.
	public, key, _ := ed25519.GenerateKey(rand.Reader)
	port, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := port.Addr().String()
	m.grant = &Grant{Root: root, Lifetime: "persistent", WorkerID: "local-machine", ListenHost: "127.0.0.1", WorkerPort: port.Addr().(*net.TCPAddr).Port, Authorized: []ed25519.PublicKey{public}}
	port.Close()
	m.id, err = prepare(m.grant, l)
	if err != nil {
		t.Fatal(err)
	}
	m.claims, err = newClaims(m.grant.WorkerID, m.id.BootID, m.id.Digest, m.grant.Authorized)
	if err != nil {
		t.Fatal(err)
	}
	m.log = io.Discard
	m.restarts.fail(&launchRefusal{"runtime_launch_refused", "fixture coordinator is unavailable"})
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
	claim := &pb.Claim{RecordOwnerId: "independent-controller", RecordOwnerEpoch: 10, WorkerId: m.grant.WorkerID, WorkerBootId: m.id.BootID, WireMinor: pb.WireMinor}
	for _, epoch := range []uint64{10, 1} {
		claim.RecordOwnerEpoch = epoch
		proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: epoch, WorkerId: claim.WorkerId, WorkerBootId: claim.WorkerBootId, WorkerTlsCertificateDigest: m.id.Digest})
		if err != nil {
			t.Fatal(err)
		}
		claim.Proof = ed25519.Sign(key, proof)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		var header metadata.MD
		state, err := pb.NewPodHostClient(conn).GetMachineExecution(ctx, &pb.MachineExecutionQuery{Claim: claim, RequestId: "request-1", ExpectedExecutionWorkspaceId: "workspace"}, grpc.Header(&header))
		cancel()
		if err != nil || state.GetState() != "succeeded" {
			t.Fatalf("authenticated read without owner handshake: %v %v", state, err)
		}
		if got := header.Get("cozy-runtime-available"); len(got) != 1 || got[0] != "false" {
			t.Fatalf("journal snapshot presented as live Runtime: %v", header)
		}
	}

	resume := `
import hashlib,sqlite3,sys
from pathlib import Path
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
store=Path(sys.argv[1]);db=sqlite3.connect(store/'.cozy-workspace/journal.sqlite3')
if sys.argv[2]=='running':
 data=b'resumed output';digest=hashlib.sha256(data).digest();name=digest.hex()
 blob=store/'blobs'/name[:2]/name[2:4]/name;blob.parent.mkdir(parents=True,exist_ok=True);blob.write_bytes(data)
 product=documents.canonical_bytes(pb.RunProduct(output='text',op=pb.RUN_PRODUCT_OP_SET,content=pb.Ref(digest=digest,length=len(data)),media_type='text/plain'))
 db.execute('UPDATE runs SET attempt_ordinal=2,generation=2,state="running",sequence=3,finished_at_ms=0')
 db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',('request-1',3,2,300,'product',product,b'','','',b''))
else:
 outcome=documents.canonical_bytes(pb.AttemptOutcomeBody(status=pb.OUTCOME_STATUS_SUCCEEDED))
 db.execute('UPDATE runs SET state="succeeded",sequence=4,finished_at_ms=400')
 db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',('request-1',4,2,400,'outcome',b'{}',bytes(32),'outcome-2',hashlib.sha256(outcome).digest(),outcome))
db.commit();db.close()
`
	for _, phase := range []string{"running", "succeeded"} {
		if out, err := childOutput(exec.Command(python, "-I", "-c", resume, l.Store, phase), true); err != nil {
			t.Fatalf("resume fixture: %v %s", err, out)
		}
		current, err := m.Open(1, "text", -1)
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(io.NewSectionReader(current.Body, 0, current.Length))
		current.Close()
		if err != nil || string(body) != "resumed output" || current.Final != (phase == "succeeded") {
			t.Fatalf("historical terminal hid resumed attempt: %q final=%v phase=%s err=%v", body, current.Final, phase, err)
		}
	}
	entries, err = m.Entries(context.Background(), 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.Status != "" && entry.Seq != 4 {
			t.Fatalf("historical outcome ended resumed run: %+v", entries)
		}
	}
}

func TestPublicReadFallbackCannotAnswerMutationWithSameQueryType(t *testing.T) {
	for _, name := range []string{"CollectMachineExecution", "ControlMachineExecution", "CloseMachineSubmission", "AcknowledgeMachineExecutionCollection"} {
		if publicReadMethod(name) {
			t.Fatalf("%s can mutate through public projection", name)
		}
	}
}
