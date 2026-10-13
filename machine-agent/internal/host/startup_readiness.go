package host

import (
	"context"
	"errors"
	"fmt"

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
	// Candidate process progress owns the guard. Its exit cancels control reads;
	// useful preparation has no arbitrary wall-clock deadline.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		select {
		case <-p.done:
			cancel()
		case <-ctx.Done():
		}
	}()
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
	if _, err := m.launcher.maintain("startup-commit", []string{p.incarnation, workspace.ExecutionWorkspaceId}); err != nil {
		return err
	}
	// A later receipt failure cannot undo the committed installation. Keep API
	// readiness closed and report repair, without sending a rollback for this journal.
	p.startupCommitted = true
	if !m.manualUpdateActive() {
		defer m.setUpdating(false) // receipt and resumed cleanup retain admission
	}
	m.mu.Lock()
	m.grant.startupPending = false
	m.mu.Unlock()
	if err := m.attestRuntime(payload); err != nil {
		failure := fmt.Errorf("Runtime installation committed but readiness receipt failed: %w", err)
		m.finishResumedUpdate("failed", failure.Error())
		return failure
	}
	m.finishResumedUpdate("succeeded", "")
	fmt.Fprintln(m.log, "cozy machine: startup update committed after authenticated Runtime readiness")
	return nil
}

func (m *Machine) rollbackStartup(ctx context.Context, cause error) error {
	// The caller owns update admission (and boot may already hold launchMu).
	// Keep that fence through application handoff or the restored launch/error.
	// An intentional process replacement exits without releasing this process's fence.
	defer func() {
		if !m.manualUpdateActive() {
			m.setUpdating(false)
		}
	}()
	detail := cause.Error()
	fmt.Fprintln(m.log, "cozy machine: rolling back startup candidate:", detail)
	m.stopRuntime()
	if _, err := m.launcher.maintain("startup-rollback", []string{detail}); err != nil {
		failure := &launchRefusal{"runtime_update_recovery_required", fmt.Sprintf("candidate failed: %s; offline rollback failed: %v", detail, err)}
		m.mu.Lock()
		m.grant.startupPending, m.proc = false, nil
		m.mu.Unlock()
		m.restarts.fail(failure)
		m.finishResumedUpdate("failed", failure.Error())
		return failure
	}
	m.finishResumedUpdate("rolled_back", detail)
	m.mu.Lock()
	m.grant.startupPending = false
	m.mu.Unlock()
	if err := m.restarts.clear(); err != nil {
		return err
	}
	if err := m.replaceApplication(true); err != nil {
		m.restarts.fail(err)
		return nil
	}
	if err := m.tfs.initStore(ctx); err != nil {
		m.restarts.fail(err)
		return nil
	}
	if err := m.launch(); err != nil {
		m.restarts.fail(err)
		return nil
	}
	fmt.Fprintln(m.log, "cozy machine: failed startup candidate rolled back; verified prior Runtime relaunched")
	return nil
}
