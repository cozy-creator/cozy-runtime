package canonical

// This file is RFC 8785 rendering over PackageInterface/1's bounded I-JSON profile.
// Every numeric value is limited to +/-((2^53)-1), independent of token spelling.
// Worker-protocol documents continue to use the narrower integer-only printable-ASCII
// profile in canonical.go/read.go.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

const jcsDepthMax = 32

type jcsNode struct {
	kind   byte
	truth  bool
	number string
	text   string
	array  []jcsNode
	object []jcsPair
}

type jcsPair struct {
	key   string
	value jcsNode
}

// NormalizeJCS parses one bounded JSON value and returns the canonical bytes of its
// content. Insignificant whitespace, object-key order and equivalent number spellings
// disappear before identity is computed. Duplicate keys and out-of-profile numbers refuse.
func NormalizeJCS(data []byte) ([]byte, error) {
	if len(data) == 0 || len(data) > DocMax {
		return nil, refuse("size_cap", "%d B is outside the 1..%d B range", len(data), DocMax)
	}
	if !utf8.Valid(data) {
		return nil, refuse("malformed_json", "document is not UTF-8")
	}
	if err := validateSurrogateEscapes(data); err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	value, err := readJCS(decoder, 0)
	if err != nil {
		return nil, err
	}
	if token, err := decoder.Token(); err != io.EOF {
		if err != nil {
			return nil, refuse("malformed_json", "%v", err)
		}
		return nil, refuse("trailing_bytes", "trailing JSON value %v", token)
	}
	var out strings.Builder
	if err := writeJCS(&out, value); err != nil {
		return nil, err
	}
	if out.Len() > DocMax {
		return nil, refuse("size_cap", "%d B exceeds the %d B canonical cap", out.Len(), DocMax)
	}
	return []byte(out.String()), nil
}

func validateSurrogateEscapes(data []byte) error {
	inString := false
	for i := 0; i < len(data); i++ {
		switch data[i] {
		case '"':
			inString = !inString
		case '\\':
			if !inString || i+1 >= len(data) {
				continue
			}
			if data[i+1] != 'u' {
				i++
				continue
			}
			first, ok := hexQuad(data, i+2)
			if !ok {
				return refuse("malformed_json", "invalid Unicode escape at offset %d", i)
			}
			if first >= 0xdc00 && first <= 0xdfff {
				return refuse("unicode_scalar", "unpaired low surrogate at offset %d", i)
			}
			if first >= 0xd800 && first <= 0xdbff {
				if i+12 > len(data) || data[i+6] != '\\' || data[i+7] != 'u' {
					return refuse("unicode_scalar", "unpaired high surrogate at offset %d", i)
				}
				second, ok := hexQuad(data, i+8)
				if !ok || second < 0xdc00 || second > 0xdfff {
					return refuse("unicode_scalar", "unpaired high surrogate at offset %d", i)
				}
				i += 11
				continue
			}
			i += 5
		}
	}
	return nil
}

func hexQuad(data []byte, at int) (uint16, bool) {
	if at+4 > len(data) {
		return 0, false
	}
	var value uint16
	for _, char := range data[at : at+4] {
		value <<= 4
		switch {
		case char >= '0' && char <= '9':
			value += uint16(char - '0')
		case char >= 'a' && char <= 'f':
			value += uint16(char-'a') + 10
		case char >= 'A' && char <= 'F':
			value += uint16(char-'A') + 10
		default:
			return 0, false
		}
	}
	return value, true
}

func readJCS(decoder *json.Decoder, depth int) (jcsNode, error) {
	if depth > jcsDepthMax {
		return jcsNode{}, refuse("depth_cap", "nesting deeper than %d", jcsDepthMax)
	}
	token, err := decoder.Token()
	if err != nil {
		return jcsNode{}, refuse("malformed_json", "%v", err)
	}
	switch value := token.(type) {
	case nil:
		return jcsNode{kind: '0'}, nil
	case bool:
		return jcsNode{kind: 'b', truth: value}, nil
	case string:
		return jcsNode{kind: 's', text: value}, nil
	case json.Number:
		canonical, err := jcsNumber(string(value))
		if err != nil {
			return jcsNode{}, err
		}
		return jcsNode{kind: 'n', number: canonical}, nil
	case json.Delim:
		switch value {
		case '[':
			items := []jcsNode{}
			for decoder.More() {
				item, err := readJCS(decoder, depth+1)
				if err != nil {
					return jcsNode{}, err
				}
				items = append(items, item)
			}
			if close, err := decoder.Token(); err != nil || close != json.Delim(']') {
				return jcsNode{}, refuse("malformed_json", "array is not closed")
			}
			return jcsNode{kind: 'a', array: items}, nil
		case '{':
			pairs := []jcsPair{}
			seen := map[string]bool{}
			for decoder.More() {
				keyToken, err := decoder.Token()
				key, ok := keyToken.(string)
				if err != nil || !ok {
					return jcsNode{}, refuse("key_grammar", "object key is not a string")
				}
				if seen[key] {
					return jcsNode{}, refuse("duplicate_key", "key %q appears twice", key)
				}
				seen[key] = true
				item, err := readJCS(decoder, depth+1)
				if err != nil {
					return jcsNode{}, err
				}
				pairs = append(pairs, jcsPair{key: key, value: item})
			}
			if close, err := decoder.Token(); err != nil || close != json.Delim('}') {
				return jcsNode{}, refuse("malformed_json", "object is not closed")
			}
			sort.Slice(pairs, func(i, j int) bool { return utf16Less(pairs[i].key, pairs[j].key) })
			return jcsNode{kind: 'o', object: pairs}, nil
		}
	}
	return jcsNode{}, refuse("wrong_type", "%T has no canonical JSON form", token)
}

func jcsNumber(raw string) (string, error) {
	if !strings.ContainsAny(raw, ".eE") {
		value, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || value < intMin || value > intMax {
			return "", refuse("number_range", "%s is outside the interoperable integer range", raw)
		}
		return strconv.FormatInt(value, 10), nil
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		return "", refuse("number_range", "%s has no finite IEEE-754 spelling", raw)
	}
	if math.Abs(value) > float64(intMax) {
		return "", refuse("number_range", "%s is outside the interoperable numeric range", raw)
	}
	if value == 0 {
		return "0", nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", refuse("number_range", "%s: %v", raw, err)
	}
	return string(encoded), nil
}

func writeJCS(out *strings.Builder, value jcsNode) error {
	switch value.kind {
	case '0':
		out.WriteString("null")
	case 'b':
		out.WriteString(strconv.FormatBool(value.truth))
	case 'n':
		out.WriteString(value.number)
	case 's':
		writeJCSString(out, value.text)
	case 'a':
		out.WriteByte('[')
		for i, item := range value.array {
			if i > 0 {
				out.WriteByte(',')
			}
			if err := writeJCS(out, item); err != nil {
				return err
			}
		}
		out.WriteByte(']')
	case 'o':
		out.WriteByte('{')
		for i, pair := range value.object {
			if i > 0 {
				out.WriteByte(',')
			}
			writeJCSString(out, pair.key)
			out.WriteByte(':')
			if err := writeJCS(out, pair.value); err != nil {
				return err
			}
		}
		out.WriteByte('}')
	default:
		return fmt.Errorf("unknown canonical JSON node %q", value.kind)
	}
	return nil
}

func writeJCSString(out *strings.Builder, value string) {
	out.WriteByte('"')
	for _, char := range value {
		switch char {
		case '"':
			out.WriteString(`\"`)
		case '\\':
			out.WriteString(`\\`)
		case '\b':
			out.WriteString(`\b`)
		case '\t':
			out.WriteString(`\t`)
		case '\n':
			out.WriteString(`\n`)
		case '\f':
			out.WriteString(`\f`)
		case '\r':
			out.WriteString(`\r`)
		default:
			if char < 0x20 {
				fmt.Fprintf(out, `\u%04x`, char)
			} else {
				out.WriteRune(char)
			}
		}
	}
	out.WriteByte('"')
}

func utf16Less(left, right string) bool {
	a, b := utf16.Encode([]rune(left)), utf16.Encode([]rune(right))
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return len(a) < len(b)
}
