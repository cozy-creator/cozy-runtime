package host

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Local package upload: a client streams an operation's carriers (checksummed wheels and
// one source.tar) into the machine, resumably. Progress is the partial file's length; a
// carrier is verified once it is whole, then renamed to its name.

var (
	operationID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$`)
	wheelName   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]{0,250}\.whl$`)
)

const maxUploadChunk = 1 << 20

func validCarrier(name string, digest []byte, length uint64) bool {
	if name == "source.tar" {
		return len(digest) == 0 && length > 0 && length <= 4<<30
	}
	return wheelName.MatchString(name) && len(digest) == sha256.Size && length > 0 && length <= 512<<20
}

// verifiedCarrier is a whole uploaded carrier's path, checked against its length and digest.
func verifiedCarrier(dir, name string, digest []byte, length uint64) (string, error) {
	path := filepath.Join(dir, name)
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("the carrier %s was not uploaded", name)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || uint64(info.Size()) != length {
		return "", fmt.Errorf("the carrier %s differs from its declared length", name)
	}
	if len(digest) > 0 {
		hash := sha256.New()
		if _, err := io.Copy(hash, file); err != nil || !bytes.Equal(hash.Sum(nil), digest) {
			return "", fmt.Errorf("the carrier %s does not match its digest", name)
		}
	}
	return path, nil
}

func (m *Machine) localPackageUpload(stream grpc.ServerStream) error {
	first := &pb.LocalPackageUploadFrame{}
	client, err := recvClaimed(stream, m, first)
	if err != nil {
		return err
	}
	if err := admitWork(client, "local package upload"); err != nil {
		return err
	}
	header := first.GetHeader()
	file := header.GetFile()
	if header == nil || file == nil || !operationID.MatchString(header.OperationId) || !validCarrier(file.Filename, file.Digest, file.Length) {
		return status.Error(codes.InvalidArgument, "a local package upload begins with one valid header")
	}
	dir := m.layout.Stage(header.OperationId)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	final := filepath.Join(dir, file.Filename)
	partial := final + ".partial"
	report := func(received uint64, verified bool) error {
		state := pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING
		if verified {
			state = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
		}
		return stream.SendMsg(&pb.LocalPackageFileStatus{RecordOwnerEpoch: client.RecordOwnerEpoch, WorkerBootId: m.id.BootID,
			OperationId: header.OperationId, Digest: file.Digest, Filename: file.Filename, Length: file.Length,
			State: state, ReceivedBytes: received})
	}
	if _, err := os.Stat(final); err == nil {
		if _, err := verifiedCarrier(dir, file.Filename, file.Digest, file.Length); err != nil {
			return status.Error(codes.FailedPrecondition, "the upload header differs from the carrier this machine holds")
		}
		return report(file.Length, true)
	}
	out, err := os.OpenFile(partial, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer out.Close()
	info, err := out.Stat()
	if err != nil {
		return err
	}
	received := uint64(info.Size())
	if received > file.Length {
		if err := out.Truncate(0); err != nil {
			return err
		}
		received = 0
	}
	if err := report(received, false); err != nil {
		return err
	}
	for received < file.Length {
		frame := &pb.LocalPackageUploadFrame{}
		if err := stream.RecvMsg(frame); errors.Is(err, io.EOF) {
			return nil // the acknowledged prefix waits for a resumed stream
		} else if err != nil {
			return err
		}
		chunk := frame.GetChunk()
		if chunk == nil || len(chunk.Data) == 0 || len(chunk.Data) > maxUploadChunk || chunk.Offset != received ||
			uint64(len(chunk.Data)) > file.Length-received {
			return status.Error(codes.FailedPrecondition, "the chunk does not continue the carrier at its received offset")
		}
		if _, err := out.WriteAt(chunk.Data, int64(received)); err != nil {
			return err
		}
		if err := out.Sync(); err != nil {
			return err
		}
		received += uint64(len(chunk.Data))
		if received < file.Length {
			if err := report(received, false); err != nil {
				return err
			}
		}
	}
	if err := os.Rename(partial, final); err != nil {
		return err
	}
	if _, err := verifiedCarrier(dir, file.Filename, file.Digest, file.Length); err != nil {
		_ = os.Remove(final)
		return status.Error(codes.FailedPrecondition, err.Error())
	}
	return report(received, true)
}
