package host

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
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
CREATE VIEW machine_runs_v1 AS SELECT 'workspace' AS workspace_id,1 AS run_number,'request-1' AS request_id,1 AS attempt_ordinal,1 AS generation,'succeeded' AS state,0 AS collected,2 AS sequence,0 AS compacted_through,100 AS accepted_at_ms,200 AS finished_at_ms,'local-machine' AS worker_id,'boot' AS worker_boot_id;
CREATE TABLE events(request_id,sequence,attempt_ordinal,at_ms,kind,body_json,invocation_spec_digest,outcome_id,outcome_digest,outcome_json);
CREATE VIEW machine_events_v1 AS SELECT * FROM events;
''')
db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',('request-1',1,1,100,'product',product,b'','','',b''))
db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',('request-1',2,1,200,'outcome',b'{}',bytes(32),'outcome-1',hashlib.sha256(outcome).digest(),outcome))
db.commit();db.close()
`
	if out, err := exec.Command(python, "-I", "-c", setup, l.Store).CombinedOutput(); err != nil {
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
}

func TestPublicReadFallbackCannotAnswerMutationWithSameQueryType(t *testing.T) {
	for _, name := range []string{"CollectMachineExecution", "ControlMachineExecution", "CloseMachineSubmission", "AcknowledgeMachineExecutionCollection"} {
		if publicReadMethod(name) {
			t.Fatalf("%s can mutate through public projection", name)
		}
	}
}
