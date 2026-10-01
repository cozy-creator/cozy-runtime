package host

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/canonical"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// Preparation progress is optional observation of Runtime-owned native work.
// Sampling never acquires its preparation lock, owns a transfer, or cancels it.
const preparationSampleInterval = 2 * time.Second

// This bounds an optional observation to its sampling cadence. A timed-out read
// drops a sample; it never imposes a deadline on the preparation RPC or native work.
const preparationObservationBudget = preparationSampleInterval

type preparationModel struct {
	repository, manifest string
	labels               *pb.DownloadModelRef
}

func preparationModels(message proto.Message) []preparationModel {
	var delegation []byte
	var native []*pb.NativeModelBinding
	var choices []*pb.ModelChoice
	switch call := message.(type) {
	case *pb.PreparePackageSetCall:
		delegation = call.GetPackageSet().GetDownloadDelegation()
	case *pb.PreparePrivatePlacementCall:
		selected := call.GetPrivatePlacementSet()
		delegation, native = selected.GetDownloadDelegation(), selected.GetNativeModels()
		choices = selected.GetModelChoices()
	}
	var models []preparationModel
	add := func(model *pb.DownloadModelRef) {
		// An unresolved repository choice is not an exact checkpoint. Optional
		// progress waits for exact resolution rather than borrowing another revision.
		if len(model.GetManifest()) == len("sha256:")+64 && strings.HasPrefix(model.Manifest, "sha256:") {
			labels := proto.Clone(model).(*pb.DownloadModelRef)
			labels.Adapters = nil
			models = append(models, preparationModel{model.Model, model.Manifest, labels})
		}
	}
	adapters := func(rows []*pb.DownloadAdapterRef) {
		for _, row := range rows {
			add(&pb.DownloadModelRef{Model: row.Model, Manifest: row.Manifest, Release: row.Release, Lane: row.Lane})
		}
	}
	if len(delegation) > 0 {
		selected := &pb.DownloadDelegation{}
		if _, err := canonical.Read(delegation, selected); err == nil {
			if err := json.Unmarshal(delegation, selected); err == nil {
				for _, model := range selected.Models {
					add(model)
					adapters(model.GetAdapters())
				}
			}
		}
	}
	for _, model := range native {
		adapters(model.GetAdapters())
	}
	for _, choice := range choices {
		manifest := ""
		if len(choice.GetManifest().GetDigest()) == 32 {
			manifest = "sha256:" + hex.EncodeToString(choice.Manifest.Digest)
		}
		add(&pb.DownloadModelRef{Model: choice.Repository, Manifest: manifest, Release: choice.Release,
			Lane: choice.Lane, Slot: choice.Parameter})
		adapters(choice.GetAdapters())
	}
	return models
}

func (selected preparationModel) matches(model *pb.DownloadModelRef) bool {
	// A bare CID authorizes bytes, not another caller's repository label.
	return selected.repository != "" && selected.manifest != "" && selected.manifest == model.GetManifest() &&
		strings.EqualFold(selected.repository, model.GetModel())
}

func (m *Machine) preparationProgress(ctx context.Context, conn *grpc.ClientConn, models []preparationModel) []*pb.PrepareModelProgress {
	if len(models) == 0 {
		return nil
	}
	observing, stop := context.WithTimeout(ctx, preparationObservationBudget)
	defer stop()
	described, err := pb.NewWorkerControlClient(conn).DescribeMachine(observing, &pb.DescribeMachineQuery{Claim: m.claims.claim})
	if err != nil {
		// Older or temporarily unreadable peers keep preparing. Telemetry is not authority.
		return nil
	}
	var observed []*pb.PrepareModelProgress
	for _, row := range described.GetRuntime().GetPreparationProgress() {
		for _, selected := range models {
			if selected.matches(row.GetModel()) {
				projected := proto.Clone(row).(*pb.PrepareModelProgress)
				projected.Model = &pb.DownloadModelRef{Model: selected.repository, Manifest: selected.manifest}
				if labels := selected.labels; labels != nil {
					projected.Model.Release, projected.Model.Lane = labels.Release, labels.Lane
					projected.Model.Package, projected.Model.Slot = labels.Package, labels.Slot
				}
				observed = append(observed, projected)
			}
		}
	}
	return observed
}

func (m *Machine) observePreparation(ctx context.Context, conn *grpc.ClientConn, models []preparationModel, emit func(*pb.PrepareEvent), last *[]*pb.PrepareModelProgress) {
	if len(models) == 0 {
		return
	}
	ticker := time.NewTicker(preparationSampleInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			rows := m.preparationProgress(ctx, conn, models)
			if len(rows) == 0 || ctx.Err() != nil {
				continue
			}
			stage := pb.PrepareStage_PREPARE_STAGE_PREPARING
			for _, row := range rows {
				if row.TotalBytes == 0 || row.TransferredBytes < row.TotalBytes {
					stage = pb.PrepareStage_PREPARE_STAGE_DOWNLOADING
					break
				}
			}
			if stage != pb.PrepareStage_PREPARE_STAGE_DOWNLOADING && len(*last) == 0 {
				// A recent completed flight can still be in DescribeMachine. Without
				// observing its live transfer, do not attribute that traffic to this call.
				continue
			}
			*last = rows
			// Individual native plans can share objects. Their sum is not unique traffic.
			emit(&pb.PrepareEvent{Stage: stage, ModelProgress: rows})
		}
	}
}
