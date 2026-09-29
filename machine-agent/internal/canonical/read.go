package canonical

import (
	"bytes"
	"encoding/base64"
	"strconv"
	"strings"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
)

// Read turns canonical bytes from an independently deployed peer into a typed document.
// It refuses what no canonical writer produces — size cap, JSON grammar (no null, no float,
// printable ASCII, only \" and \\), duplicate key, the RE-EMIT LAW (the bytes must be the
// canonical encoding of their own content) — and a `format` tag naming another document.
// Members the message does not declare are another version's additions: they are ignored,
// and an absent member reads as its default (worker-protocol DOCUMENT VERSIONS).
func Read(data []byte, m proto.Message) (Doc, error) {
	obj, err := ReadObject(data)
	if err != nil {
		return nil, err
	}
	d := m.ProtoReflect().Descriptor()
	want := Format(m)
	if obj["format"] != want {
		return nil, refuse("unknown_format", "%v is not %q", obj["format"], want)
	}
	prune(obj, d)
	obj["format"] = want
	if err := semantics(string(d.FullName()), obj); err != nil {
		return nil, err
	}
	return obj, nil
}

// ReadObject validates the shared canonical JSON profile without assigning a
// document schema. It is for exact control documents whose schemas live outside
// worker-protocol (for example PackageBindingRelease). Callers must still
// enforce their closed key set and semantic joins.
func ReadObject(data []byte) (Doc, error) {
	obj, err := parseObject(data)
	if err != nil {
		return nil, err
	}
	again, err := Write(map[string]Value(obj))
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(again, data) {
		return nil, refuse("noncanonical_encoding", "bytes are not the canonical encoding of their own content")
	}
	return obj, nil
}

// parseObject applies the bounded JSON grammar and duplicate-key refusal but
// does not require compact canonical rendering. Descriptor bytes have their own
// exact stored-byte identity and may be pretty-printed; their schema owner, not
// this codec, decides that presentation.
func parseObject(data []byte) (Doc, error) {
	if len(data) > DocMax {
		return nil, refuse("size_cap", "%d B over the %d B cap", len(data), DocMax)
	}
	p := &parser{src: data}
	v, err := p.value(0)
	if err != nil {
		return nil, err
	}
	p.space()
	if p.i != len(p.src) {
		return nil, refuse("noncanonical_encoding", "%d trailing byte(s) after the document", len(p.src)-p.i)
	}
	obj, ok := v.(map[string]Value)
	if !ok {
		return nil, refuse("wrong_type", "a document is a JSON object")
	}
	return Doc(obj), nil
}

// prune keeps the members a message declares, recursively.
func prune(obj map[string]Value, d protoreflect.MessageDescriptor) {
	for key, value := range obj {
		field := d.Fields().ByName(protoreflect.Name(key))
		if field == nil {
			delete(obj, key)
			continue
		}
		if field.Kind() != protoreflect.MessageKind || field.IsMap() {
			continue
		}
		if nested, ok := value.(map[string]Value); ok {
			prune(nested, field.Message())
		}
		if items, ok := value.([]Value); ok {
			for _, item := range items {
				if nested, ok := item.(map[string]Value); ok {
					prune(nested, field.Message())
				}
			}
		}
	}
}

// semantics carries the few document rules the KEY SET cannot state. A closed key set says
// which keys may appear; it cannot say that one of them has exactly one legal spelling of
// "absent", and #485b makes that a wire law rather than a convention.
func semantics(name string, d Doc) error {
	switch name {
	case "cozy.worker.v1.AttemptOutcomeBody":
		return weightsReceiptList(d)
	case "cozy.worker.v1.WeightsReceipt":
		return weightsReceipt(d)
	}
	return nil
}

func weightsReceiptList(d Doc) error {
	raw, present := d["weights_receipts"]
	if !present {
		return nil
	}
	items, ok := raw.([]Value)
	if !ok {
		return refuse("weights_receipt_shape", "weights_receipts is a list")
	}
	if len(items) > pb.MaxWeightsReceipts {
		return refuse("weights_receipt_count_cap", "%d receipts exceeds the %d-item cap",
			len(items), pb.MaxWeightsReceipts)
	}
	aggregate, slots := 0, map[string]bool{}
	for _, item := range items {
		fields, ok := item.(map[string]Value)
		if !ok {
			return refuse("weights_receipt_shape", "an weights_receipts item is not an object")
		}
		receipt, size, err := readWeightsReceiptRef(Doc(fields))
		if err != nil {
			return err
		}
		aggregate += size
		slot := receipt.Str("output_slot")
		if slots[slot] {
			return refuse("weights_receipt_duplicate", "output slot %q has two receipts", slot)
		}
		slots[slot] = true
	}
	if aggregate > pb.MaxWeightsReceiptAggregateBytes {
		return refuse("weights_receipt_aggregate_cap", "%d receipt bytes exceeds the %d-byte cap",
			aggregate, pb.MaxWeightsReceiptAggregateBytes)
	}
	return nil
}

func weightsReceipt(d Doc) error {
	for _, field := range []string{"owner_authority_scope", "request_id", "invocation_spec_digest",
		"output_slot", "weights_transaction_id", "tensorfs_receipt_digest",
		"tensorfs_receipt_canonical_bytes"} {
		if d.Str(field) == "" {
			return refuse("weights_receipt_incomplete", "%s is empty or absent", field)
		}
	}
	nested, err := decodeCanonicalBytes(d.Str("tensorfs_receipt_canonical_bytes"))
	if err != nil {
		return refuse("weights_receipt_nested_malformed", "%s", err)
	}
	if !sameDigest(nested, d.Str("tensorfs_receipt_digest")) {
		return refuse("weights_receipt_nested_digest_mismatch",
			"tensorfs_receipt_digest does not hash the exact carried bytes")
	}
	if _, err := Raw(d.Str("invocation_spec_digest")); err != nil {
		return refuse("weights_receipt_identity_malformed", "invocation_spec_digest: %s", err)
	}
	return nil
}

func readWeightsReceiptRef(ref Doc) (Doc, int, error) {
	if ref.Str("weights_receipt_digest") == "" || ref.Str("weights_receipt_canonical_bytes") == "" {
		return nil, 0, refuse("weights_receipt_ref_shape", "the reference carries a digest and bytes")
	}
	data, err := decodeCanonicalBytes(ref.Str("weights_receipt_canonical_bytes"))
	if err != nil {
		return nil, 0, refuse("weights_receipt_malformed", "%s", err)
	}
	if len(data) == 0 || len(data) > pb.MaxWeightsReceiptBytes {
		return nil, 0, refuse("weights_receipt_item_cap", "%d B is outside the 1..%d B range",
			len(data), pb.MaxWeightsReceiptBytes)
	}
	if !sameDigest(data, ref.Str("weights_receipt_digest")) {
		return nil, 0, refuse("weights_receipt_digest_mismatch",
			"weights_receipt_digest does not hash the exact carried bytes")
	}
	receipt, err := Read(data, &pb.WeightsReceipt{})
	return receipt, len(data), err
}

func decodeCanonicalBytes(value string) ([]byte, error) {
	return base64.StdEncoding.Strict().DecodeString(value)
}

func sameDigest(data []byte, spelled string) bool {
	raw, err := Raw(spelled)
	return err == nil && bytes.Equal(Digest(data), raw)
}

// Str reads one string field off a parsed document; a missing or wrong-typed field
// answers "" rather than a zero value pretending to be a fact.
func (d Doc) Str(key string) string {
	s, _ := d[key].(string)
	return s
}

// Int reads one integer field off a parsed document.
func (d Doc) Int(key string) int64 {
	n, _ := d[key].(int64)
	return n
}

// Sub reads one nested document off a parsed document.
func (d Doc) Sub(key string) Doc {
	m, _ := d[key].(map[string]Value)
	return Doc(m)
}

// Ints reads one repeated integer field off a parsed document. An entry that is not an
// integer is skipped rather than guessed at.
func (d Doc) Ints(key string) []int64 {
	items, _ := d[key].([]Value)
	out := make([]int64, 0, len(items))
	for _, item := range items {
		if n, ok := item.(int64); ok {
			out = append(out, n)
		}
	}
	return out
}

// Strs reads one repeated string field off a parsed document.
func (d Doc) Strs(key string) []string {
	items, _ := d[key].([]Value)
	out := make([]string, 0, len(items))
	for _, item := range items {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// List reads one repeated field off a parsed document as documents. An entry that is not
// an object is skipped rather than guessed at: the caller wanted rows.
func (d Doc) List(key string) []Doc {
	items, _ := d[key].([]Value)
	out := make([]Doc, 0, len(items))
	for _, item := range items {
		if m, ok := item.(map[string]Value); ok {
			out = append(out, Doc(m))
		}
	}
	return out
}

// --------------------------------------------------------------------------- parser

type parser struct {
	src []byte
	i   int
}

// space skips insignificant whitespace. The parser ACCEPTS it so the re-emit law is
// what refuses it — one rule refusing non-canonical bytes, never two.
func (p *parser) space() {
	for p.i < len(p.src) {
		switch p.src[p.i] {
		case ' ', '\t', '\n', '\r':
			p.i++
		default:
			return
		}
	}
}

func (p *parser) value(depth int) (Value, error) {
	if depth > depthMax {
		return nil, refuse("depth_cap", "nesting deeper than %d", depthMax)
	}
	p.space()
	if p.i >= len(p.src) {
		return nil, refuse("malformed_json", "document ends where a value was expected")
	}
	switch c := p.src[p.i]; {
	case c == '{':
		return p.object(depth)
	case c == '[':
		return p.array(depth)
	case c == '"':
		return p.str()
	case c == 't':
		return true, p.lit("true")
	case c == 'f':
		return false, p.lit("false")
	case c == 'n':
		return nil, refuse("wrong_type", "null has no canonical spelling in this profile")
	default:
		return p.number()
	}
}

func (p *parser) lit(word string) error {
	if !bytes.HasPrefix(p.src[p.i:], []byte(word)) {
		return refuse("malformed_json", "expected %q at offset %d", word, p.i)
	}
	p.i += len(word)
	return nil
}

func (p *parser) object(depth int) (Value, error) {
	p.i++ // '{'
	out := map[string]Value{}
	p.space()
	if p.i < len(p.src) && p.src[p.i] == '}' {
		p.i++
		return out, nil
	}
	for {
		p.space()
		if p.i >= len(p.src) || p.src[p.i] != '"' {
			return nil, refuse("key_grammar", "an object key must be a string, at offset %d", p.i)
		}
		key, err := p.str()
		if err != nil {
			return nil, err
		}
		name := key.(string)
		if _, seen := out[name]; seen {
			return nil, refuse("duplicate_key", "key %q appears twice", name)
		}
		p.space()
		if p.i >= len(p.src) || p.src[p.i] != ':' {
			return nil, refuse("malformed_json", "expected ':' at offset %d", p.i)
		}
		p.i++
		v, err := p.value(depth + 1)
		if err != nil {
			return nil, err
		}
		out[name] = v
		p.space()
		if p.i >= len(p.src) {
			return nil, refuse("malformed_json", "unterminated object")
		}
		switch p.src[p.i] {
		case ',':
			p.i++
		case '}':
			p.i++
			return out, nil
		default:
			return nil, refuse("malformed_json", "expected ',' or '}' at offset %d", p.i)
		}
	}
}

func (p *parser) array(depth int) (Value, error) {
	p.i++ // '['
	out := []Value{}
	p.space()
	if p.i < len(p.src) && p.src[p.i] == ']' {
		p.i++
		return out, nil
	}
	for {
		v, err := p.value(depth + 1)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
		p.space()
		if p.i >= len(p.src) {
			return nil, refuse("malformed_json", "unterminated array")
		}
		switch p.src[p.i] {
		case ',':
			p.i++
		case ']':
			p.i++
			return out, nil
		default:
			return nil, refuse("malformed_json", "expected ',' or ']' at offset %d", p.i)
		}
	}
}

func (p *parser) str() (Value, error) {
	p.i++ // '"'
	var sb strings.Builder
	for p.i < len(p.src) {
		c := p.src[p.i]
		switch {
		case c == '"':
			p.i++
			return sb.String(), nil
		case c == '\\':
			if p.i+1 >= len(p.src) {
				return nil, refuse("malformed_json", "escape at end of document")
			}
			switch p.src[p.i+1] {
			case '"':
				sb.WriteByte('"')
			case '\\':
				sb.WriteByte('\\')
			default:
				return nil, refuse("noncanonical_encoding",
					"escape \\%c is outside the profile's two escapes", p.src[p.i+1])
			}
			p.i += 2
		case c < 0x20 || c > 0x7e:
			return nil, refuse("non_ascii_field", "byte 0x%02x in a string; fields are printable ASCII", c)
		default:
			sb.WriteByte(c)
			p.i++
		}
	}
	return nil, refuse("malformed_json", "unterminated string")
}

func (p *parser) number() (Value, error) {
	start := p.i
	for p.i < len(p.src) {
		c := p.src[p.i]
		if (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' {
			p.i++
			continue
		}
		break
	}
	tok := string(p.src[start:p.i])
	if tok == "" {
		return nil, refuse("malformed_json", "expected a value at offset %d", start)
	}
	if strings.ContainsAny(tok, ".eE") {
		return nil, refuse("non_integer_number", "%s is a float; documents are integer-only", tok)
	}
	n, err := strconv.ParseInt(tok, 10, 64)
	if err != nil || n < intMin || n > intMax {
		return nil, refuse("number_range", "%s is outside the interoperable integer range", tok)
	}
	return n, nil
}
