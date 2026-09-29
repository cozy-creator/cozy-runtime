// Package canonical is cozy-creator's half of the digest plane: `cozy.worker.v1`
// messages <-> their canonical DOCUMENT bytes, under tfs-013's writer rules.
//
// Identity in this protocol is never marshaled protobuf (worker-protocol/01): a digest
// that fences MEANING is the SHA-256 of a document's exact canonical bytes, and those
// bytes travel IN the message so a receiver recomputes rather than re-canonicalizes.
//
// The writer is ADAPTED from worker-protocol's own independent Go canonicalizer
// (`scripts/crosslang/canon.go`) — the second implementation that proved the rules are
// written down rather than accidental. The reader serves documents from independently
// deployed peers: it refuses bytes no canonical writer produces, and keeps the members this
// build consumes, ignoring additions from other versions.
//
// Conformance is `go test ./tests/product -run TestCanonicalDocuments`, which renders
// worker-protocol's FROZEN fixture corpus through this codec and compares bytes and ids.
package canonical

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

const (
	intMax   = int64(1)<<53 - 1
	intMin   = -intMax
	depthMax = 12
	// DocMax is the document size cap; bytes past it are refused, never truncated.
	DocMax = 8 << 20
)

// Value is bool | int64 | string | []Value | map[string]Value. No float, no null:
// the protocol profile is integer-only printable ASCII.
type Value any

// Doc is a parsed canonical document — the JSON value, deliberately not a re-encoded
// message. Re-encoding would create a second representation free to disagree with the
// bytes the digest was taken over.
type Doc map[string]Value

// Error is a canonical-document refusal, named by code so a caller can classify it.
type Error struct {
	Code   string
	Detail string
}

func (e *Error) Error() string { return e.Code + ": " + e.Detail }

func refuse(code, format string, args ...any) *Error {
	return &Error{Code: code, Detail: fmt.Sprintf(format, args...)}
}

// Code returns err's refusal code, or "" if it is not a canonical refusal.
func Code(err error) string {
	if e, ok := err.(*Error); ok {
		return e.Code
	}
	return ""
}

// --------------------------------------------------------------------------- digests

// Spell turns the 32 raw bytes protobuf transports into `sha256:<hex>`.
func Spell(raw []byte) (string, error) {
	if len(raw) != 32 {
		return "", refuse("malformed_digest", "%d B is not a 32-byte SHA-256", len(raw))
	}
	return "sha256:" + hex.EncodeToString(raw), nil
}

// Raw turns `sha256:<hex>` back into the 32 bytes protobuf transports.
func Raw(spelled string) ([]byte, error) {
	if !strings.HasPrefix(spelled, "sha256:") || len(spelled) != 71 {
		return nil, refuse("malformed_digest", "%q is not a lowercase sha256: digest", spelled)
	}
	out, err := hex.DecodeString(spelled[7:])
	if err != nil {
		return nil, refuse("malformed_digest", "%q is not lowercase hex", spelled)
	}
	return out, nil
}

// Digest is the raw 32-byte SHA-256 of exactly these bytes — the only digest a
// receiver ever computes.
func Digest(data []byte) []byte {
	sum := sha256.Sum256(data)
	return sum[:]
}

// --------------------------------------------------------------------------- writer

func isDigestField(name string) bool {
	return name == "digest" || strings.HasSuffix(name, "_digest") || strings.HasSuffix(name, "_digests")
}

func fieldValue(fd protoreflect.FieldDescriptor, v protoreflect.Value) (Value, error) {
	switch fd.Kind() {
	case protoreflect.BoolKind:
		return v.Bool(), nil
	case protoreflect.EnumKind:
		return int64(v.Enum()), nil // NUMBERS are normative (R2), names are not
	case protoreflect.Int32Kind, protoreflect.Int64Kind, protoreflect.Sint32Kind,
		protoreflect.Sint64Kind, protoreflect.Sfixed32Kind, protoreflect.Sfixed64Kind:
		n := v.Int()
		if n < intMin || n > intMax {
			return nil, refuse("number_range", "%s=%d outside the interoperable integer range", fd.FullName(), n)
		}
		return n, nil
	case protoreflect.Uint32Kind, protoreflect.Uint64Kind, protoreflect.Fixed32Kind,
		protoreflect.Fixed64Kind:
		u := v.Uint()
		if u > uint64(intMax) {
			return nil, refuse("number_range", "%s=%d outside the interoperable integer range", fd.FullName(), u)
		}
		return int64(u), nil
	case protoreflect.FloatKind, protoreflect.DoubleKind:
		return nil, refuse("non_integer_number", "%s is a float; these documents are integer-only", fd.FullName())
	case protoreflect.StringKind:
		return v.String(), nil
	case protoreflect.BytesKind:
		b := v.Bytes()
		if isDigestField(string(fd.Name())) {
			return Spell(b)
		}
		return base64.StdEncoding.EncodeToString(b), nil
	case protoreflect.MessageKind:
		return body(v.Message())
	}
	return nil, refuse("wrong_type", "%s has no canonical form", fd.FullName())
}

// body walks only the SET fields: a field at its proto3 default is omitted, because
// absence-default equals pre-introduction behavior (R8).
func body(m protoreflect.Message) (map[string]Value, error) {
	out := map[string]Value{}
	var err error
	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		name := string(fd.Name())
		if fd.IsList() {
			l := v.List()
			arr := make([]Value, 0, l.Len())
			for i := 0; i < l.Len(); i++ {
				e, e2 := fieldValue(fd, l.Get(i))
				if e2 != nil {
					err = e2
					return false
				}
				arr = append(arr, e)
			}
			out[name] = arr
			return true
		}
		e, e2 := fieldValue(fd, v)
		if e2 != nil {
			err = e2
			return false
		}
		out[name] = e
		return true
	})
	for _, name := range explicitRepeated[string(m.Descriptor().FullName())] {
		if _, present := out[name]; !present {
			out[name] = []Value{}
		}
	}
	return out, err
}

var explicitRepeated = map[string][]string{
	"cozy.worker.v1.DownloadDelegation":      {"models", "packages"},
	"cozy.worker.v1.Entrypoint":              {"slots"},
	"cozy.worker.v1.Placement":               {"entrypoints", "models"},
	"cozy.worker.v1.MachineExecutionCapture": {"installed_packages", "bindings"},
	"cozy.worker.v1.Slot":                    {"components", "stamps"},
	"cozy.worker.v1.Stamp":                   {"values"},
}

// Format is the canonical `format` tag for one message's document: its full name plus the
// sole pre-release document version. There are no compatibility aliases or version-specific
// readers: Capture uses /2 after removing revision fingerprints; other documents use /1.
func Format(m proto.Message) string {
	name := string(m.ProtoReflect().Descriptor().FullName())
	if name == "cozy.worker.v1.MachineExecutionCapture" {
		return name + "/2"
	}
	return name + "/1"
}

// Document is a message's set fields plus the `format` tag that domain-separates one
// document kind from another.
func Document(m proto.Message) (Doc, error) {
	b, err := body(m.ProtoReflect())
	if err != nil {
		return nil, err
	}
	if _, taken := b["format"]; taken {
		return nil, refuse("unknown_field", "`format` is reserved for the document tag")
	}
	b["format"] = Format(m)
	return b, nil
}

func emitStr(s string, out *strings.Builder) error {
	for i := 0; i < len(s); i++ {
		if s[i] < 0x20 || s[i] > 0x7e {
			return refuse("non_ascii_field", "%q is not printable ASCII", s)
		}
	}
	out.WriteByte('"')
	out.WriteString(strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s))
	out.WriteByte('"')
	return nil
}

func writeValue(v Value, out *strings.Builder, depth int) error {
	if depth > depthMax {
		return refuse("depth_cap", "nesting deeper than %d", depthMax)
	}
	switch t := v.(type) {
	case bool:
		out.WriteString(strconv.FormatBool(t))
	case int64:
		out.WriteString(strconv.FormatInt(t, 10))
	case string:
		return emitStr(t, out)
	case []Value:
		out.WriteByte('[')
		for i, e := range t {
			if i > 0 {
				out.WriteByte(',')
			}
			if err := writeValue(e, out, depth+1); err != nil {
				return err
			}
		}
		out.WriteByte(']')
	case map[string]Value:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys) // sorted at write; never by hand
		out.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				out.WriteByte(',')
			}
			if err := emitStr(k, out); err != nil {
				return err
			}
			out.WriteByte(':')
			if err := writeValue(t[k], out, depth+1); err != nil {
				return err
			}
		}
		out.WriteByte('}')
	case Doc:
		return writeValue(map[string]Value(t), out, depth)
	default:
		return refuse("wrong_type", "%T is not a canonical JSON value", v)
	}
	return nil
}

// Write is the canonical bytes of one value. These are what gets hashed.
func Write(v Value) ([]byte, error) {
	var sb strings.Builder
	if err := writeValue(v, &sb, 0); err != nil {
		return nil, err
	}
	if sb.Len() > DocMax {
		return nil, refuse("size_cap", "%d B over the %d B cap", sb.Len(), DocMax)
	}
	return []byte(sb.String()), nil
}

// Bytes is the whole contract for authoring: a document-shape message in, its one
// identity's bytes out.
func Bytes(m proto.Message) ([]byte, error) {
	doc, err := Document(m)
	if err != nil {
		return nil, err
	}
	return Write(map[string]Value(doc))
}

// Identity is `(canonical bytes, raw 32-byte digest)` — the fenced pair a message
// carries. The orchestrator mints this once per attempt and never recomputes it from a
// second representation.
func Identity(m proto.Message) ([]byte, []byte, error) {
	data, err := Bytes(m)
	if err != nil {
		return nil, nil, err
	}
	return data, Digest(data), nil
}
