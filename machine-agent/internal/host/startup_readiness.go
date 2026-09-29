package host

import (
	"context"
	"errors"
	"fmt"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
)

func (m *Machine) startupPending() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.grant != nil && m.grant.startupPending
}

// The ready gate remains closed until this incarnation has answered authenticated
// workspace and closure RPCs and the privileged launcher commits its journal.
func (m *Machine) commitStartup(p *runtimeProcess, payload []byte) error {
	// These control reads perform no execution. Their response budget bounds this
	// disposable pre-admission health check, never an accepted workload.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	client := pb.NewWorkerControlClient(m.conn)
	workspace, err := client.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: m.claims.claim})
	if err != nil {
		return err
	}
	if workspace.ExecutionWorkspaceId == "" || !workspace.SubmissionClose {
		return errors.New("candidate lacks durable submission closure")
	}
	id := "startup-probe-" + p.incarnation
	closed, err := client.CloseMachineSubmission(ctx, &pb.MachineSubmissionClose{Claim: m.claims.claim, RequestId: id, SubmissionId: id, ExpectedExecutionWorkspaceId: workspace.ExecutionWorkspaceId})
	if err != nil {
		return err
	}
	if closed.RequestId != id || closed.SubmissionId != id || closed.ExecutionWorkspaceId != workspace.ExecutionWorkspaceId || closed.Receipt != nil {
		return errors.New("candidate did not close the unused startup probe")
	}
	if _, err := m.launcher.maintain("startup-commit", []string{p.incarnation}); err != nil {
		return err
	}
	if err := m.attestRuntime(payload); err != nil {
		return err
	}
	m.mu.Lock()
	m.grant.startupPending, m.inUpdate = false, false
	m.mu.Unlock()
	fmt.Fprintln(m.log, "cozy machine: startup update committed after authenticated Runtime readiness")
	return nil
}

func (m *Machine) rollbackStartup(ctx context.Context) error {
	m.stopRuntime()
	if _, err := m.launcher.maintain("startup-rollback", nil); err != nil {
		return fmt.Errorf("startup candidate failed and offline rollback failed: %w", err)
	}
	m.mu.Lock()
	m.grant.startupPending, m.inUpdate = false, false
	m.mu.Unlock()
	if err := m.restarts.clear(); err != nil {
		return err
	}
	if err := m.tfs.initStore(ctx); err != nil {
		return err
	}
	if err := m.launch(); err != nil {
		return err
	}
	fmt.Fprintln(m.log, "cozy machine: failed startup candidate rolled back; verified prior Runtime relaunched")
	return nil
}
