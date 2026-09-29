package host

import (
	"google.golang.org/protobuf/reflect/protoreflect"
)

type protoMessage = protoreflect.Message

// walk visits a message and every message it holds, depth first. Bytes are not visited, so
// a large payload costs nothing.
func walk(m protoMessage, visit func(protoMessage)) {
	visit(m)
	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		switch {
		case fd.IsMap():
			if fd.MapValue().Message() != nil {
				v.Map().Range(func(_ protoreflect.MapKey, item protoreflect.Value) bool {
					walk(item.Message(), visit)
					return true
				})
			}
		case fd.IsList():
			if fd.Message() != nil {
				list := v.List()
				for i := 0; i < list.Len(); i++ {
					walk(list.Get(i).Message(), visit)
				}
			}
		case fd.Message() != nil:
			walk(v.Message(), visit)
		}
		return true
	})
}

func field(m protoMessage, name protoreflect.Name) (protoreflect.FieldDescriptor, bool) {
	fd := m.Descriptor().Fields().ByName(name)
	return fd, fd != nil && !fd.IsList() && !fd.IsMap() && m.Has(fd)
}

// setOwner replaces an owner epoch or id a message names with the daemon's.
func setOwner(m protoMessage, epoch uint64, id string) {
	if fd, ok := field(m, "record_owner_epoch"); ok && fd.Kind() == protoreflect.Uint64Kind {
		m.Set(fd, protoreflect.ValueOfUint64(epoch))
	}
	if fd, ok := field(m, "record_owner_id"); ok && fd.Kind() == protoreflect.StringKind {
		m.Set(fd, protoreflect.ValueOfString(id))
	}
}

func protoValueOfUint64(v uint64) protoreflect.Value { return protoreflect.ValueOfUint64(v) }
func protoValueOfString(v string) protoreflect.Value { return protoreflect.ValueOfString(v) }
