package resultfiles

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/canonical"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/exit"
)

const TreeMediaType = "application/vnd.cozy.tree-manifest"

type TreeMember struct {
	Path, Digest string
	Length       int64
}

// ReadTreeManifest accepts only the exact bounded native file closure. Export
// paths come from this hash-verified manifest, never from a result JSON field.
func ReadTreeManifest(source, digest string, length, contentBytes int64) ([]TreeMember, *exit.Error) {
	if length <= 0 || length > 1<<20 || contentBytes < 0 {
		return nil, exit.New(exit.Conflict, "tree manifest exceeds its exact bounds")
	}
	file, err := os.Open(source)
	if err != nil {
		return nil, exit.Internalf("cannot open tree manifest: %s", err)
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, length+1))
	hash := sha256.Sum256(raw)
	if err != nil || int64(len(raw)) != length || "sha256:"+hex.EncodeToString(hash[:]) != digest {
		return nil, exit.New(exit.Conflict, "tree manifest changed its accepted identity")
	}
	return ParseTreeManifest(raw, contentBytes)
}

func ParseTreeManifest(raw []byte, contentBytes int64) ([]TreeMember, *exit.Error) {
	fail := func() ([]TreeMember, *exit.Error) {
		return nil, exit.New(exit.Conflict, "tree manifest has an invalid file closure")
	}
	// Runtime writes the manifest and its digest is already verified; members another
	// version adds are ignored.
	if len(raw) == 0 || len(raw) > 1<<20 || contentBytes < 0 {
		return fail()
	}
	var document map[string]json.RawMessage
	if json.Unmarshal(raw, &document) != nil {
		return fail()
	}
	var entries []map[string]json.RawMessage
	if json.Unmarshal(document["entries"], &entries) != nil || len(entries) == 0 {
		return fail()
	}
	var members []TreeMember
	names := map[string]bool{}
	sizes := map[string]int64{}
	var total int64
	for _, entry := range entries {
		var kind, name string
		var blob map[string]json.RawMessage
		var sha string
		var length int64
		if json.Unmarshal(entry["kind"], &kind) != nil || kind != "file" || json.Unmarshal(entry["path"], &name) != nil || json.Unmarshal(entry["blob"], &blob) != nil || json.Unmarshal(blob["sha256"], &sha) != nil || json.Unmarshal(blob["length"], &length) != nil || length < 0 || length > contentBytes-total {
			return fail()
		}
		if name == "." || path.Clean(name) != name || !filepath.IsLocal(filepath.FromSlash(name)) || strings.ContainsAny(name, "\\\x00") || names[name] {
			return fail()
		}
		digest := "sha256:" + sha
		if _, err := canonical.Raw(digest); err != nil {
			return fail()
		}
		if prior, found := sizes[digest]; found && prior != length {
			return fail()
		}
		sizes[digest], names[name] = length, true
		total += length
		members = append(members, TreeMember{Path: name, Digest: digest, Length: length})
	}
	if total != contentBytes {
		return fail()
	}
	for name := range names {
		for parent := path.Dir(name); parent != "."; parent = path.Dir(parent) {
			if names[parent] {
				return fail()
			}
		}
	}
	return members, nil
}

// MaterializeTree copies a complete verified closure into a separate directory.
// Its staging directory is published only after all files and directories sync.
func MaterializeTree(source, directory, name, digest string, length, contentBytes int64) (string, *exit.Error) {
	members, problem := ReadTreeManifest(source, digest, length, contentBytes)
	if problem != nil {
		return "", problem
	}
	if problem := Preflight(directory); problem != nil {
		return "", problem
	}
	destination := filepath.Join(directory, name)
	replaced := ""
	if _, err := os.Lstat(destination); err == nil {
		if verifyExportedTree(destination, members) == nil {
			return destination, nil
		}
		// Another revision of the tree: the new one takes its name whole.
		replaced = filepath.Join(directory, ".cozy-tree-old-"+randomSuffix())
	} else if !os.IsNotExist(err) {
		return "", exportWriteFailure(directory, err)
	}
	temporary, err := os.MkdirTemp(directory, ".cozy-tree-")
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer os.RemoveAll(temporary)
	root, err := os.OpenRoot(temporary)
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer root.Close()
	dirs := map[string]bool{".": true}
	for _, member := range members {
		parent := filepath.FromSlash(path.Dir(member.Path))
		if err := root.MkdirAll(parent, 0755); err != nil {
			return "", exportWriteFailure(directory, err)
		}
		for folder := parent; folder != "."; folder = filepath.Dir(folder) {
			dirs[folder] = true
		}
		input, err := os.Open(filepath.Join(source+".files", strings.TrimPrefix(member.Digest, "sha256:")))
		if err != nil {
			return "", exit.Internalf("cannot open tree member custody: %s", err)
		}
		output, err := root.OpenFile(filepath.FromSlash(member.Path), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
		if err != nil {
			input.Close()
			return "", exportWriteFailure(directory, err)
		}
		hash := sha256.New()
		written, copyErr := io.Copy(io.MultiWriter(output, hash), io.LimitReader(input, member.Length+1))
		input.Close()
		if copyErr != nil || written != member.Length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != member.Digest {
			output.Close()
			return "", exit.New(exit.Conflict, "tree member custody changed before export")
		}
		syncErr := output.Sync()
		closeErr := output.Close()
		if syncErr != nil {
			return "", exportWriteFailure(directory, syncErr)
		}
		if closeErr != nil {
			return "", exportWriteFailure(directory, closeErr)
		}
	}
	folders := make([]string, 0, len(dirs))
	for folder := range dirs {
		folders = append(folders, folder)
	}
	sort.Sort(sort.Reverse(sort.StringSlice(folders)))
	for _, folder := range folders {
		file, err := root.Open(folder)
		if err != nil {
			return "", exportWriteFailure(directory, err)
		}
		syncErr := file.Sync()
		file.Close()
		if syncErr != nil {
			return "", exportWriteFailure(directory, syncErr)
		}
	}
	if replaced != "" {
		if err := os.Rename(destination, replaced); err != nil {
			return "", exportWriteFailure(directory, err)
		}
		defer os.RemoveAll(replaced)
	}
	if err := os.Rename(temporary, destination); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	parent, err := os.Open(directory)
	if err != nil {
		return "", exportWriteFailure(directory, err)
	}
	defer parent.Close()
	if err := parent.Sync(); err != nil {
		return "", exportWriteFailure(directory, err)
	}
	return destination, nil
}

func verifyExportedTree(directory string, members []TreeMember) *exit.Error {
	info, err := os.Lstat(directory)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return exit.New(exit.Conflict, "tree export destination changed")
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return exportWriteFailure(directory, err)
	}
	defer root.Close()
	wanted := map[string]TreeMember{}
	for _, member := range members {
		wanted[member.Path] = member
	}
	err = fs.WalkDir(root.FS(), ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		member, ok := wanted[name]
		if !ok || !entry.Type().IsRegular() {
			return fs.ErrInvalid
		}
		file, err := root.Open(filepath.FromSlash(name))
		if err != nil {
			return err
		}
		hash := sha256.New()
		written, copyErr := io.Copy(hash, io.LimitReader(file, member.Length+1))
		file.Close()
		if copyErr != nil || written != member.Length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != member.Digest {
			return fs.ErrInvalid
		}
		delete(wanted, name)
		return nil
	})
	if err != nil || len(wanted) != 0 {
		return exit.New(exit.Conflict, "existing tree export differs from the accepted result; choose another --out directory")
	}
	return nil
}
