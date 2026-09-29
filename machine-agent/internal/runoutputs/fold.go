// Package runoutputs folds a run's output log into its items: the one place that says what an
// output's current bytes are, which revision they are, and how they changed.
//
// A run's outputs are items with stable ids, `<run>/<output>` for a single output and
// `<run>/<output>/<i>` (1-based) for a list element. Each `product` entry the machine journals
// is a new revision of its item, rewritten in place: it appends when its parts extend the
// previous revision's parts, and otherwise replaces them. `rev` is the item's 1-based entry
// ordinal and the ETag is `"r<rev>"`. The content digest is exposed on the final revision only
// (progressive-outputs.md §1–2).
package runoutputs

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/canonical"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
)

// Part is one stored piece of an item's bytes; the bytes are the parts in order.
type Part struct {
	Digest     string
	Length     int64
	DurationUs uint64
	// Source is the machine hold the part is read through (ReadByteTreeObject).
	Source *pb.NativeByteRetentionRequest
}

// Revision is an item's bytes as one log entry left them.
type Revision struct {
	Rev      uint32
	Sequence uint64
	Length   int64
	// AppendedFrom is set exactly when this revision appends: the offset its new bytes start
	// at, which is the previous revision's length. A replace leaves it nil.
	AppendedFrom *int64
	// DurationUs is the media duration so far; 0 for bytes without a timeline.
	DurationUs uint64
	// Digest is the sha256 of the whole bytes. Show it only on the final revision.
	Digest    string
	MediaType string
	Label     string
	// Parts are the current bytes in order; a single-source product is one part.
	Parts []Part
}

// ETag is the revision's HTTP entity tag.
func (r Revision) ETag() string { return fmt.Sprintf(`"r%d"`, r.Rev) }

// PrefixOf answers whether this revision's bytes begin every later revision up to `now`:
// true when each revision after it appended.
func (r Revision) PrefixOf(now Revision) bool {
	if r.Rev > now.Rev || len(r.Parts) > len(now.Parts) {
		return false
	}
	for i, part := range r.Parts {
		if now.Parts[i].Digest != part.Digest || now.Parts[i].Length != part.Length {
			return false
		}
	}
	return true
}

// Item is one output of a run at its current revision.
type Item struct {
	ID string
	// OutputIndex is the item's 0-based position in the order items were first added.
	OutputIndex int
	Output      string
	// Index is the list element's 1-based position; 0 for a single output.
	Index   uint32
	List    bool
	Type    string
	Current Revision
	// History is the item's revisions' lengths and parts: enough to answer an If-Range for
	// any earlier rev without holding its bytes.
	History []Revision
}

// Name is the item's file stem: `<run>-<output>` or `<run>-<output>-<i>`.
func (i Item) Name(run string) string {
	if i.List {
		return fmt.Sprintf("%s-%s-%d", run, i.Output, i.Index)
	}
	return run + "-" + i.Output
}

// Revision answers the item's revision `rev`, if it had one.
func (i Item) Revision(rev uint32) (Revision, bool) {
	if rev == 0 || int(rev) > len(i.History) {
		return Revision{}, false
	}
	return i.History[rev-1], true
}

type key struct {
	output string
	index  uint32
}

// Fold is a run's items, built from its log's product entries in order.
type Fold struct {
	run   string
	items map[key]*Item
	order []*Item
	last  uint64
}

// New folds the output log of run `run` (its number, or machine/number).
func New(run string) *Fold {
	return &Fold{run: run, items: map[key]*Item{}}
}

// Add folds one product entry. An entry at or before the last one folded is a replay and
// changes nothing. It answers the item as the entry left it and whether the entry added it.
func (f *Fold) Add(sequence uint64, product *pb.RunProduct) (Item, bool, error) {
	if sequence <= f.last {
		return Item{}, false, nil
	}
	revision, problem := revisionOf(sequence, product)
	if problem != nil {
		return Item{}, false, problem
	}
	list := product.Op == pb.RunProductOp_RUN_PRODUCT_OP_APPEND
	k := key{output: product.Output}
	if list {
		k.index = product.Index + 1
	} else if product.Op != pb.RunProductOp_RUN_PRODUCT_OP_SET {
		return Item{}, false, fmt.Errorf("product entry %d has no operation", sequence)
	}
	f.last = sequence
	item, known := f.items[k]
	if !known {
		item = &Item{OutputIndex: len(f.order), Output: product.Output, Index: k.index, List: list,
			Type: typeOf(product.MediaType)}
		item.ID = f.run + "/" + product.Output
		if list {
			item.ID = fmt.Sprintf("%s/%d", item.ID, k.index)
		}
		f.items[k] = item
		f.order = append(f.order, item)
	} else if item.List != list {
		return Item{}, false, fmt.Errorf("output %q is both a single output and a list", product.Output)
	}
	revision.Rev = uint32(len(item.History) + 1)
	if known && item.Current.PrefixOf(revision) && len(revision.Parts) > len(item.Current.Parts) {
		from := item.Current.Length
		revision.AppendedFrom = &from
	}
	item.Current = revision
	item.History = append(item.History, revision)
	return *item, !known, nil
}

// Items are the run's items in the order they were first added.
func (f *Fold) Items() []Item {
	items := make([]Item, len(f.order))
	for i, item := range f.order {
		items[i] = *item
	}
	return items
}

// Item answers an item by its output and 1-based list index (0 for a single output).
func (f *Fold) Item(output string, index uint32) (Item, bool) {
	item, ok := f.items[key{output, index}]
	if !ok {
		return Item{}, false
	}
	return *item, true
}

func revisionOf(sequence uint64, product *pb.RunProduct) (Revision, error) {
	if product == nil || product.Output == "" || product.Content == nil {
		return Revision{}, fmt.Errorf("product entry %d is incomplete", sequence)
	}
	digest, err := canonical.Spell(product.Content.Digest)
	if err != nil {
		return Revision{}, fmt.Errorf("product entry %d names no sha256", sequence)
	}
	revision := Revision{Sequence: sequence, Length: int64(product.Content.Length), Digest: digest,
		MediaType: product.MediaType, Label: product.Label}
	if len(product.Parts) == 0 {
		revision.Parts = []Part{{Digest: digest, Length: revision.Length, Source: product.Source}}
		return revision, nil
	}
	var total int64
	for _, part := range product.Parts {
		spelled, err := canonical.Spell(part.GetContent().GetDigest())
		if err != nil {
			return Revision{}, fmt.Errorf("product entry %d has a part with no sha256", sequence)
		}
		length := int64(part.Content.Length)
		total += length
		revision.DurationUs += part.DurationUs
		revision.Parts = append(revision.Parts, Part{Digest: spelled, Length: length, DurationUs: part.DurationUs, Source: part.Source})
	}
	if total != revision.Length {
		return Revision{}, fmt.Errorf("product entry %d's parts are %d bytes, not %d", sequence, total, revision.Length)
	}
	return revision, nil
}

// typeOf is an item's Responses-style type, from its media type.
func typeOf(mediaType string) string {
	switch {
	case mediaType == "application/vnd.cozy.tree-manifest":
		return "tree"
	case strings.HasPrefix(mediaType, "video/"):
		return "video"
	case strings.HasPrefix(mediaType, "image/"):
		return "image"
	case strings.HasPrefix(mediaType, "audio/"):
		return "audio"
	}
	return "file"
}
