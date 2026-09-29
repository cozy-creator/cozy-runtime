package canonical

import (
	"encoding/base64"
	"encoding/json"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// Unmarshal reads one canonical document into its generated message. The same
// digest-field rule as the writer translates canonical SHA-256 spelling to
// protobuf bytes; consumers must not maintain another nested-field decoder.
func Unmarshal(data []byte, message proto.Message) error {
	doc, err := Read(data, message)
	if err != nil {
		return err
	}
	delete(doc, "format")
	converted, err := protobufObject(doc, message.ProtoReflect().Descriptor())
	if err != nil {
		return err
	}
	encoded, err := json.Marshal(converted)
	if err != nil {
		return err
	}
	return (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(encoded, message)
}

func protobufObject(doc Doc, descriptor protoreflect.MessageDescriptor) (map[string]any, error) {
	out := make(map[string]any, len(doc))
	for key, value := range doc {
		field := descriptor.Fields().ByName(protoreflect.Name(key))
		if field == nil {
			return nil, refuse("unknown_field", "%s has no field %s", descriptor.FullName(), key)
		}
		convert := func(item Value) (any, error) {
			if field.Kind() == protoreflect.BytesKind && isDigestField(key) {
				text, ok := item.(string)
				if !ok {
					return nil, refuse("wrong_type", "%s is not a digest string", key)
				}
				raw, err := Raw(text)
				if err != nil {
					return nil, err
				}
				return base64.StdEncoding.EncodeToString(raw), nil
			}
			if field.Kind() == protoreflect.MessageKind {
				nested, ok := item.(map[string]Value)
				if !ok {
					return nil, refuse("wrong_type", "%s is not an object", key)
				}
				return protobufObject(Doc(nested), field.Message())
			}
			return item, nil
		}
		if field.IsList() {
			list, ok := value.([]Value)
			if !ok {
				return nil, refuse("wrong_type", "%s is not a list", key)
			}
			values := make([]any, len(list))
			for i, item := range list {
				next, err := convert(item)
				if err != nil {
					return nil, err
				}
				values[i] = next
			}
			out[key] = values
		} else {
			next, err := convert(value)
			if err != nil {
				return nil, err
			}
			out[key] = next
		}
	}
	return out, nil
}
