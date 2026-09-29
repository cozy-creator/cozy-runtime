package host

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/outputs"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
)

//go:embed public_read.py
var publicReadScript string

// The guardian selects only explicit public views. Neither the agent nor this
// adapter writes Runtime's journal, copies its records, or reads private tables.
func readPublic(root, store string, args []string) (string, error) {
	if len(args) != 1 || len(args[0]) > 64<<10 {
		return "", errors.New("invalid public read")
	}
	cmd := exec.Command(filepath.Join(root, "opt/cozy/python/bin/python"), "-I", "-c", publicReadScript,
		filepath.Join(store, ".cozy-workspace/journal.sqlite3"), args[0])
	out, err := childOutput(cmd, false)
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			for _, code := range []string{"execution_not_found", "execution_workspace_changed", "execution_cursor_invalid"} {
				if strings.Contains(string(exit.Stderr), code) {
					return "", &launchRefusal{code, "the Runtime journal does not contain the requested execution"}
				}
			}
		}
		return "", &launchRefusal{"runtime_public_reads_unavailable", "the installed Runtime has not created a readable machine journal"}
	}
	return strings.TrimSpace(string(out)), nil
}

func (m *Machine) publicRead(query map[string]any, answer proto.Message) error {
	raw, err := json.Marshal(query)
	if err != nil {
		return err
	}
	if !m.launcherReady() || m.launcher == nil {
		return outputs.ErrUnavailable
	}
	encoded, err := m.launcher.maintain("public-read", []string{string(raw)})
	if err != nil {
		var refused *launchRefusal
		if errors.As(err, &refused) {
			code := codes.Unavailable
			if refused.code == "execution_not_found" {
				code = codes.NotFound
			}
			if refused.code == "execution_workspace_changed" || refused.code == "execution_cursor_invalid" {
				code = codes.FailedPrecondition
			}
			return refusal(code, refused.code, refused.detail)
		}
		return err
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return fmt.Errorf("invalid public read response: %w", err)
	}
	return proto.Unmarshal(data, answer)
}

// publicFallback serves observations while the coordinator is down. Commands
// still require it; journal availability never grants permission to mutate it.
func (m *Machine) publicFallback(ctx context.Context, request proto.Message) (answer proto.Message, handled bool, err error) {
	defer func() {
		if errors.Is(err, outputs.ErrUnavailable) {
			err = unavailable("runtime_starting", "Runtime observations are not ready; retry when the coordinator finishes starting")
		}
	}()
	query := map[string]any{}
	switch q := request.(type) {
	case *pb.MachineExecutionWorkspaceQuery:
		if q.Describe != nil {
			return nil, false, nil
		}
		query["op"], answer = "workspace", &pb.MachineExecutionWorkspace{}
	case *pb.MachineExecutionQuery:
		query["op"], query["request"], query["workspace"] = "get", q.RequestId, q.ExpectedExecutionWorkspaceId
		answer = &pb.MachineExecutionState{}
	case *pb.MachineExecutionListQuery:
		query = map[string]any{"op": "list", "after": q.AfterNumber, "before": q.BeforeNumber, "newest": q.NewestFirst, "states": q.States, "limit": q.Limit}
		for {
			list := &pb.MachineExecutionList{}
			err := m.publicRead(query, list)
			if err != nil || !q.Wait || q.NewestFirst || len(list.Executions) > 0 {
				return list, true, err
			}
			select {
			case <-ctx.Done():
				return nil, true, ctx.Err()
			case <-time.After(time.Second):
			}
		}
	case *pb.MachineExecutionEventsQuery:
		query = map[string]any{"op": "events", "request": q.GetExecution().GetRequestId(), "workspace": q.GetExecution().GetExpectedExecutionWorkspaceId(), "after": q.After, "limit": q.Limit}
		for {
			page := &pb.MachineExecutionEventPage{}
			err := m.publicRead(query, page)
			if err != nil || !q.Wait || len(page.Events) > 0 {
				return page, true, err
			}
			state := &pb.MachineExecutionState{}
			if err := m.publicRead(map[string]any{"op": "get", "request": query["request"], "workspace": query["workspace"]}, state); err != nil {
				return nil, true, err
			}
			switch state.State {
			case "succeeded", "failed", "paused", "canceled":
				return page, true, nil
			}
			select {
			case <-ctx.Done():
				return nil, true, ctx.Err()
			case <-time.After(time.Second):
			}
		}
	default:
		return nil, false, nil
	}
	err = m.publicRead(query, answer)
	if workspace, ok := answer.(*pb.MachineExecutionWorkspace); ok {
		workspace.WorkerId, workspace.WorkerBootId = m.grant.WorkerID, m.id.BootID
	}
	return answer, true, err
}

func (m *Machine) readPublicLog(ctx context.Context, run, after uint64, wait bool) (*runLog, error) {
	list := &pb.MachineExecutionList{}
	if err := m.publicRead(map[string]any{"op": "list", "after": run - 1, "limit": 1}, list); err != nil {
		return nil, err
	}
	if len(list.Executions) == 0 || list.Executions[0].Number != run {
		return nil, outputs.ErrNotFound
	}
	state := list.Executions[0]
	log := newRunLog(run)
	log.attempt = state.AttemptOrdinal
	query := map[string]any{"op": "events", "request": state.RequestId, "workspace": list.ExecutionWorkspaceId, "limit": 256}
	for {
		query["after"] = log.next
		page := &pb.MachineExecutionEventPage{}
		if err := m.publicRead(query, page); err != nil {
			return nil, err
		}
		for _, event := range page.Events {
			log.add(event)
		}
		current := &pb.MachineExecutionState{}
		if err := m.publicRead(map[string]any{"op": "get", "request": state.RequestId, "workspace": list.ExecutionWorkspaceId}, current); err != nil {
			return nil, err
		}
		log.observeState(current)
		if log.next >= page.HeadSequence || len(page.Events) == 0 {
			if !wait || log.terminal || len(log.after(after)) > 0 {
				return log.final(), nil
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Second):
			}
		}
	}
}
