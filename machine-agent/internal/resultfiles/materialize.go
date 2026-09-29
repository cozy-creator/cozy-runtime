package resultfiles

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/exit"
)

// Materialize exports already accepted bytes from internal publication custody.
// Keep that custody independent of the user-editable copy, including across mounts.
// The destination name comes only from the accepted digest and media type.
func Materialize(source, directory, digest, mediaType string, length int64) (string, *exit.Error) {
	name, problem := Filename(digest, mediaType)
	if problem != nil {
		return "", problem
	}
	if length < 0 || length == math.MaxInt64 {
		return "", exit.Named(exit.Validation, "output_export_length_invalid", "output length is invalid")
	}
	if problem := Preflight(directory); problem != nil {
		return "", problem
	}
	input, err := os.Open(source)
	if err != nil {
		return "", exit.Internalf("cannot open accepted output: %s", err)
	}
	defer input.Close()
	info, err := input.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() != length {
		return "", exit.Named(exit.Conflict, "output_export_source_changed", "accepted output changed before export")
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer root.Close()
	temporary := ".cozy-export-" + randomSuffix()
	output, err := root.OpenFile(temporary, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer func() { output.Close(); _ = root.Remove(temporary) }()
	hash := sha256.New()
	written, err := io.Copy(io.MultiWriter(output, hash), io.LimitReader(input, length+1))
	if err != nil {
		return "", exit.Internalf("cannot export accepted output: %s", err)
	}
	if written != length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != digest {
		return "", exit.Named(exit.Conflict, "output_export_source_changed", "accepted output changed before export")
	}
	if err := output.Sync(); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	if err := output.Close(); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	if err := root.Rename(temporary, name); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	parent, err := root.Open(".")
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer parent.Close()
	if err := parent.Sync(); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	return filepath.Join(directory, name), nil
}

func exportWriteFailure(directory string, err error) *exit.Error {
	return exit.Named(exit.Failed, "output_export_write_failed",
		"cannot export accepted output under %s: %s", directory, ioCause(err))
}
