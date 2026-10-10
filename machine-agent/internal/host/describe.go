package host

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/build"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// describeMachine answers what this machine is. The daemon measures itself; the Runtime,
// when it runs and knows the call, adds what it measures.
func (m *Machine) describeMachine(stream grpc.ServerStream) error {
	call := &pb.DescribeMachineQuery{}
	client, err := recvClaimed(stream, m, call)
	if err != nil {
		return err
	}
	hostname, _ := os.Hostname()
	host := &pb.MachineHost{Version: build.Version, WireMinor: pb.WireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor,
		Platform: runtime.GOOS + "/" + runtime.GOARCH, OsRelease: osRelease(), Hostname: hostname, Phase: m.phase(),
		Filesystems:     []*pb.MachineFilesystem{filesystem(m.layout.Store), filesystem(m.layout.Root)},
		StartedAtUnixMs: uint64(m.started.UnixMilli())}
	if m.grant.HubOrigin != "" {
		host.Hubs = append(host.Hubs, &pb.MachineHub{Origin: m.grant.HubOrigin, MachineId: m.grant.WorkerID})
	}

	if !m.owned && !m.idle.releasedNow() {
		host.IdleDeadlineUnixMs = uint64(m.idle.deadline().UnixMilli())
	}
	answer := &pb.MachineDescription{WorkerId: m.grant.WorkerID, WorkerBootId: m.id.BootID, Host: host}
	if !m.running() {
		answer.RuntimeAbsent = m.runtimeGone()
		if answer.RuntimeAbsent == "" {
			answer.RuntimeAbsent = "the Runtime coordinator is starting"
		}
		return stream.SendMsg(answer)
	}
	ctx, cancel := context.WithTimeout(stream.Context(), 30*time.Second)
	defer cancel()
	conn, err := m.runtime(ctx)
	if err != nil {
		answer.RuntimeAbsent = "the Runtime is not ready: " + err.Error()
		return stream.SendMsg(answer)
	}
	m.claims.forRuntime(call, client.GetWireMinor())
	measured, err := pb.NewWorkerControlClient(conn).DescribeMachine(ctx, call)
	switch {
	case status.Code(err) == codes.Unimplemented:
		answer.RuntimeAbsent = "this machine's Runtime predates DescribeMachine; update the machine's Runtime"
	case err != nil:
		answer.RuntimeAbsent = "the Runtime did not describe itself: " + status.Convert(err).Message()
	default:
		answer.Runtime = measured.GetRuntime()
	}
	return stream.SendMsg(answer)
}

func filesystem(path string) *pb.MachineFilesystem {
	var fs syscall.Statfs_t
	if syscall.Statfs(path, &fs) != nil {
		return &pb.MachineFilesystem{Path: path}
	}
	return &pb.MachineFilesystem{Path: path, TotalBytes: fs.Blocks * uint64(fs.Bsize), AvailableBytes: fs.Bavail * uint64(fs.Bsize)}
}

func osRelease() string {
	raw, _ := os.ReadFile("/etc/os-release")
	for _, line := range strings.Split(string(raw), "\n") {
		if value, ok := strings.CutPrefix(line, "PRETTY_NAME="); ok {
			return strings.Trim(value, `"`)
		}
	}
	return runtime.GOOS
}

// listModels answers from the Runtime when it runs and knows the call, else from TensorFS
// directly, so an idle machine still says what it holds.
func (m *Machine) listModels(stream grpc.ServerStream) error {
	call := &pb.ModelListQuery{}
	client, err := recvClaimed(stream, m, call)
	if err != nil {
		return err
	}
	if m.running() {
		ctx, cancel := context.WithTimeout(stream.Context(), 30*time.Second)
		defer cancel()
		if conn, err := m.runtime(ctx); err == nil {
			m.claims.forRuntime(call, client.GetWireMinor())
			if list, err := pb.NewWorkerControlClient(conn).ListModels(ctx, call); err == nil {
				return stream.SendMsg(list)
			} else if status.Code(err) != codes.Unimplemented {
				return err
			}
		}
	}
	list, err := m.modelsFromStore(stream.Context())
	if err != nil {
		return err
	}
	return stream.SendMsg(list)
}

func (m *Machine) modelsFromStore(ctx context.Context) (*pb.ModelList, error) {
	dir, err := os.MkdirTemp(m.layout.Tmp, "models-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	list := &pb.ModelList{Store: &pb.StoreUsage{Filesystem: filesystem(m.layout.Store)}}
	if err := m.tfs.run(ctx, nil, nil, "repo", "list", m.tfs.store, "--rows", filepath.Join(dir, "releases.jsonl")); err != nil {
		return nil, status.Errorf(codes.Unavailable, "TensorFS could not list this machine's models: %v", err)
	}
	for _, line := range lines(filepath.Join(dir, "releases.jsonl")) {
		var row struct {
			Kind, Org, Name, Version, Lane string
			SourceSelection                string `json:"source_selection"`
			Manifest                       string `json:"manifest_sha256"`
			Length                         uint64 `json:"manifest_length"`
		}
		if json.Unmarshal([]byte(line), &row) != nil || row.Org == "" {
			continue
		}
		digest, _ := hexBytes(row.Manifest)
		list.Models = append(list.Models, &pb.MachineModel{Kind: row.Kind, Repository: row.Org + "/" + row.Name,
			Version: row.Version, Lane: row.Lane, SourceSelection: row.SourceSelection,
			Manifest: &pb.Ref{Digest: digest, Length: row.Length}})
	}
	if err := m.tfs.run(ctx, nil, nil, "repo", "usage", m.tfs.store, "--rows", filepath.Join(dir, "usage.jsonl")); err == nil {
		for _, line := range lines(filepath.Join(dir, "usage.jsonl")) {
			var row struct {
				Kind, Org, Name string
				Total           uint64 `json:"bytes_total"`
				Unique          uint64 `json:"bytes_unique"`
				UniqueSum       uint64 `json:"bytes_unique_sum"`
				Unreferenced    uint64 `json:"bytes_unreferenced"`
			}
			if json.Unmarshal([]byte(line), &row) != nil {
				continue
			}
			switch row.Kind {
			case "repo":
				list.Repositories = append(list.Repositories, &pb.RepositoryUsage{Repository: row.Org + "/" + row.Name,
					BytesTotal: row.Total, BytesUnique: row.Unique})
			case "store":
				list.Store.BytesTotal, list.Store.BytesUniqueSum, list.Store.BytesUnreferenced = row.Total, row.UniqueSum, row.Unreferenced
			}
		}
	}
	return list, nil
}

func lines(path string) []string {
	raw, _ := os.ReadFile(path)
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}
