package host

import (
	"context"
	"crypto/ed25519"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/outputs"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/runoutputs"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var _ outputs.Source = (*Machine)(nil)

// Keys is the authorized key set and a channel closed when it changes.
func (m *Machine) Keys() ([]ed25519.PublicKey, <-chan struct{}) {
	m.claims.mu.Lock()
	defer m.claims.mu.Unlock()
	return append([]ed25519.PublicKey(nil), m.claims.authorized...), m.claims.changed
}

// Runtime owns the only execution journal. Reads use its API while it is healthy,
// and its versioned public SQLite views while it is down. The agent never copies runs.

// liveRuntime is the running Runtime, waiting while it boots. It never launches one.
func (m *Machine) liveRuntime(ctx context.Context) (*grpc.ClientConn, error) {
	m.mu.Lock()
	p, ready := m.proc, m.ready
	m.mu.Unlock()
	if p == nil || p.exited() {
		return nil, outputs.ErrUnavailable
	}
	select {
	case <-ready:
		return m.conn, nil
	case <-p.done:
		return nil, outputs.ErrUnavailable
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// request is the query that names run n's execution, asking the Runtime by number.
func (m *Machine) request(ctx context.Context, conn *grpc.ClientConn, run uint64) (*pb.MachineExecutionQuery, uint64, error) {
	list, err := pb.NewWorkerControlClient(conn).ListMachineExecutions(ctx, &pb.MachineExecutionListQuery{
		Claim: m.claims.claim, AfterNumber: run - 1, Limit: 1})
	switch {
	case status.Code(err) == codes.Unimplemented:
		return nil, 0, outputs.ErrUpdateRequired
	case err != nil:
		return nil, 0, err
	case len(list.Executions) == 0 || list.Executions[0].Number != run:
		return nil, 0, outputs.ErrNotFound
	}
	state := list.Executions[0]
	workspace := cmpOr(state.ExecutionWorkspaceId, list.ExecutionWorkspaceId)
	return &pb.MachineExecutionQuery{Claim: m.claims.claim, RequestId: state.RequestId, ExpectedExecutionWorkspaceId: workspace}, state.AttemptOrdinal, nil
}

// runLog is a run's log read from its start, folded into items.
type runLog struct {
	fold     *runoutputs.Fold
	entries  []outputs.Entry
	terminal bool
	next     uint64
	attempt  uint64
}

func newRunLog(run uint64) *runLog { return &runLog{fold: runoutputs.New(strconv.FormatUint(run, 10))} }

// read folds run's log: a finished run from its record, a live one from the Runtime, where with
// wait and nothing after `after` it holds one read open at the tail.
func (m *Machine) read(ctx context.Context, run, after uint64, wait bool) (*runLog, error) {
	if run == 0 {
		return nil, outputs.ErrNotFound
	}
	if m.phase() != "ready" {
		return m.readPublicLog(ctx, run, after, wait)
	}
	conn, err := m.liveRuntime(ctx)
	if err != nil {
		return m.readPublicLog(ctx, run, after, wait)
	}
	request, attempt, err := m.request(ctx, conn, run)
	if err != nil {
		return nil, err
	}
	client := pb.NewWorkerControlClient(conn)
	l := newRunLog(run)
	l.attempt = attempt
	page := func(wait bool) (*pb.MachineExecutionEventPage, error) {
		answer, err := client.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{
			Execution: request, After: l.next, Wait: wait})
		if status.Code(err) == codes.NotFound {
			return nil, outputs.ErrNotFound
		}
		if err != nil {
			return nil, err
		}
		for _, event := range answer.Events {
			l.add(event)
		}
		state, err := client.GetMachineExecution(ctx, request)
		if err != nil {
			return nil, err
		}
		l.observeState(state)
		return answer, nil
	}
	for {
		answer, err := page(false)
		if err != nil {
			return nil, err
		}
		if len(answer.Events) == 0 || l.next >= answer.HeadSequence {
			break
		}
	}
	for wait && !l.terminal && len(l.after(after)) == 0 {
		if _, err := page(true); err != nil {
			return nil, err
		}
		if !l.terminal && len(l.after(after)) == 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Second):
			}
		}
	}
	return l.final(), nil
}

// final names the digest of every item's current revision once the terminal entry follows.
func (l *runLog) final() *runLog {
	if !l.terminal {
		return l
	}
	for i := range l.entries {
		entry := &l.entries[i]
		if item, ok := l.fold.Item(entry.Output, listIndex(entry.Index)); ok && entry.Status == "" && entry.Rev == uint64(item.Current.Rev) {
			entry.SHA256 = item.Current.Digest
		}
	}
	return l
}

func listIndex(index int) uint32 {
	if index < 0 {
		return 0
	}
	return uint32(index)
}

func (l *runLog) add(event *pb.MachineExecutionEvent) {
	if event.AttemptOrdinal > l.attempt {
		l.attempt, l.terminal = event.AttemptOrdinal, false
		l.entries = slices.DeleteFunc(l.entries, func(entry outputs.Entry) bool { return entry.Status != "" })
	}
	l.next = max(l.next, event.Sequence)
	switch {
	case event.Product != nil:
		item, _, err := l.fold.Add(event.Sequence, event.Product)
		if err != nil {
			return
		}
		rev, index := item.Current, -1
		if item.List {
			index = int(item.Index)
		}
		entry := outputs.Entry{Seq: event.Sequence, Output: item.Output, Index: index, Rev: uint64(rev.Rev), Length: rev.Length,
			DurationUS: rev.DurationUs, MediaType: rev.MediaType, Label: rev.Label}
		if rev.AppendedFrom != nil {
			from := uint64(*rev.AppendedFrom)
			entry.AppendedFrom = &from
		}
		l.entries = append(l.entries, entry)

	}
}

func (l *runLog) after(after uint64) []outputs.Entry {
	at := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Seq > after })
	return l.entries[at:]
}

// A canceled attempt may only mean pause. Only the execution's current state
// can mark output revisions final or end a follow.
func (l *runLog) observeState(state *pb.MachineExecutionState) {
	l.entries = slices.DeleteFunc(l.entries, func(entry outputs.Entry) bool { return entry.Status != "" })
	l.terminal = false
	terminal := ""
	switch state.State {
	case "succeeded":
		terminal = "completed"
	case "failed":
		terminal = "failed"
	case "canceled":
		terminal = "canceled"
	}
	if terminal == "" || state.Sequence > l.next {
		return
	}
	l.terminal = true
	l.entries = append(l.entries, outputs.Entry{Seq: state.Sequence, Index: -1, Status: terminal})
}

// Entries answers run's log after `after`, waiting at the tail; at or past the terminal entry
// it answers the terminal entry again.
func (m *Machine) Entries(ctx context.Context, run, after uint64) ([]outputs.Entry, error) {
	l, err := m.read(ctx, run, after, true)
	if err != nil {
		return nil, err
	}
	if entries := l.after(after); len(entries) > 0 {
		return entries, nil
	}
	if l.terminal {
		return l.entries[len(l.entries)-1:], nil
	}
	return nil, ctx.Err()
}

// Open answers an output's current bytes: its parts, in order, from the TensorFS store.
func (m *Machine) Open(run uint64, output string, index int) (outputs.Snapshot, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	l, err := m.read(ctx, run, 0, false)
	if err != nil {
		return outputs.Snapshot{}, err
	}
	item, ok := l.fold.Item(output, listIndex(index))
	if !ok {
		return outputs.Snapshot{}, outputs.ErrNotFound
	}
	rev := item.Current
	body := &parts{}
	for _, part := range rev.Parts {
		file, err := m.object(part.Digest)
		if err != nil {
			body.Close()
			return outputs.Snapshot{}, err
		}
		body.files, body.lengths = append(body.files, file), append(body.lengths, part.Length)
	}
	snapshot := outputs.Snapshot{Body: body, Length: rev.Length, Rev: uint64(rev.Rev), Final: l.terminal, MediaType: rev.MediaType}
	if snapshot.Final {
		snapshot.SHA256 = rev.Digest
	}
	return snapshot, nil
}

// object opens one object of the machine's TensorFS store, sha256:<hex>.
func (m *Machine) object(digest string) (*os.File, error) {
	hex, ok := strings.CutPrefix(digest, "sha256:")
	if !ok || len(hex) != 64 {
		return nil, outputs.ErrNotFound
	}
	file, err := os.OpenFile(filepath.Join(m.layout.Store, "blobs", hex[:2], hex[2:4], hex), os.O_RDONLY|syscall.O_NOFOLLOW, 0)
	if errors.Is(err, os.ErrNotExist) {
		return nil, outputs.ErrNotFound
	}
	return file, err
}

// parts is the concatenation of an output's part objects, read at any offset.
type parts struct {
	files   []*os.File
	lengths []int64
}

func (p *parts) ReadAt(b []byte, off int64) (int, error) {
	n := 0
	for i, file := range p.files {
		if off >= p.lengths[i] {
			off -= p.lengths[i]
			continue
		}
		for n < len(b) && off < p.lengths[i] {
			want := min(int64(len(b)-n), p.lengths[i]-off)
			got, err := file.ReadAt(b[n:n+int(want)], off)
			n, off = n+got, off+int64(got)
			if err != nil && !(errors.Is(err, io.EOF) && int64(got) == want) {
				return n, err
			}
		}
		if n == len(b) {
			return n, nil
		}
		off = 0
	}
	return n, io.EOF
}

func (p *parts) Close() error {
	for _, file := range p.files {
		file.Close()
	}
	return nil
}

// RefreshKeys checks the cached rental lease before a new media session.
func (m *Machine) RefreshKeys(context.Context) error {
	return m.claims.checkAuthority()
}
