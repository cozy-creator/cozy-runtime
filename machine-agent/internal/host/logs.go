package host

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// readMachineLog streams one log this machine keeps, oldest line first (wire 72). TensorFS
// writes its transport decisions beside the store it pulls into and rotates them to one
// older file; the Host reads both from disk, so the log answers while the Runtime is stopped.
func (m *Machine) readMachineLog(stream grpc.ServerStream) error {
	call := &pb.MachineLogQuery{}
	if _, err := recvClaimed(stream, m, call); err != nil {
		return err
	}
	var files []string
	switch call.GetLog() {
	case pb.MachineLog_MACHINE_LOG_TENSORFS_TRANSPORT:
		dir := filepath.Join(m.layout.Store, "logs")
		files = []string{filepath.Join(dir, "transport.log.1"), filepath.Join(dir, "transport.log")}
	default:
		return status.Errorf(codes.NotFound, "this machine keeps no log %s", call.GetLog())
	}
	var data []byte
	for _, file := range files {
		kept, err := os.ReadFile(file)
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return status.Errorf(codes.Internal, "read %s: %v", filepath.Base(file), err)
		}
		data = append(data, kept...)
	}
	if tail := call.GetTailBytes(); tail > 0 && uint64(len(data)) > tail {
		data = data[uint64(len(data))-tail:]
		if line := bytes.IndexByte(data, '\n'); line >= 0 {
			data = data[line+1:]
		}
	}
	for len(data) > 0 {
		n := min(len(data), pb.MaxMachineLogChunkBytes)
		if err := stream.SendMsg(&pb.MachineLogChunk{Data: data[:n]}); err != nil {
			return err
		}
		data = data[n:]
	}
	return nil
}
