package host

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/canonical"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// A preparation delegates package, model, and memory decisions to Runtime. A refusal that retrying cannot change ends the stream as REFUSED; any
// other ends it Unavailable, and the client asks again.

type prepared struct {
	result *pb.PreparePackageSetResult
	code   string // a refusal's safe code
	detail string
	retry  bool // the refusal leaves the request valid
}

func refused(code string, err error, retry bool) prepared {
	return prepared{code: code, detail: safe(err.Error()), retry: retry}
}

var safeCode = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,127}$`)

func safe(text string) string {
	var out strings.Builder
	for i := 0; i < len(text) && out.Len() < 1024; i++ {
		if text[i] < 0x20 || text[i] > 0x7e {
			out.WriteByte('?')
		} else {
			out.WriteByte(text[i])
		}
	}
	return out.String()
}

// runtimeRefused classifies a Runtime preparation error, keeping its named code.
func runtimeRefused(err error, trailer metadata.MD) prepared {
	code := "runtime_preparation_failed"
	if named := trailer.Get("cozy-error-code"); len(named) == 1 && safeCode.MatchString(named[0]) {
		code = named[0]
	}
	answer := status.Convert(err)
	retry := answer.Code() == codes.Canceled || answer.Code() == codes.DeadlineExceeded || answer.Code() == codes.Unavailable
	return prepared{code: code, detail: safe(fmt.Sprintf("runtime preparation failed (%s): %s", answer.Code(), answer.Message())), retry: retry}
}

// prepare runs one preparation stream: admission, idle hold, stage events and the answer.
func (m *Machine) prepare(stream grpc.ServerStream, message proto.Message,
	work func(ctx context.Context, conn *grpc.ClientConn, emit func(*pb.PrepareEvent)) prepared,
) error {
	client, err := recvClaimed(stream, m, message)
	if err != nil {
		return err
	}
	if err := admitWork(client, "preparation"); err != nil {
		return err
	}
	ctx := stream.Context()
	if err := m.requireRuntimeRange(ctx); err != nil {
		return err
	}
	if err := m.idle.work(time.Now()); err != nil {
		return status.Error(codes.FailedPrecondition, "this machine's idle release is already committed")
	}
	done := m.beginPreparation()
	if done == nil {
		return status.Error(codes.Unavailable, "Runtime activation is waiting for preparations to drain")
	}
	defer done()
	conn, err := m.runtime(ctx)
	if err != nil {
		return err
	}
	var mu sync.Mutex
	var last *pb.PrepareEvent
	emit := func(event *pb.PrepareEvent) {
		mu.Lock()
		defer mu.Unlock()
		if last != nil && proto.Equal(last, event) {
			return
		}
		last = event
		_ = stream.SendMsg(event)
	}
	observed, stopObserving := context.WithCancel(ctx)
	progressDone := make(chan struct{})
	var modelProgress []*pb.PrepareModelProgress
	models := preparationModels(message)
	go func() {
		defer close(progressDone)
		m.observePreparation(observed, conn, models, emit, &modelProgress)
	}()
	answer := work(ctx, conn, emit)
	stopObserving()
	<-progressDone
	if ctx.Err() != nil {
		return status.FromContextError(ctx.Err()).Err()
	}
	if answer.code != "" {
		if answer.retry {
			return status.Errorf(codes.Unavailable, "%s: %s", answer.code, answer.detail)
		}
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_REFUSED, SafeCode: answer.code, SafeDetail: answer.detail})
		return nil
	}
	if answer.result.GetPlacementSet() == nil && answer.result.GetInstalledPackage() == nil {
		return status.Error(codes.DataLoss, "the Runtime returned neither an installation nor a placement")
	}
	if len(modelProgress) > 0 {
		if rows := m.preparationProgress(ctx, conn, models); len(rows) > 0 {
			modelProgress = rows
		}
	}
	emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, PlacementSet: answer.result.PlacementSet,
		InstalledPackage: answer.result.InstalledPackage, ModelProgress: modelProgress})
	return nil
}

func (m *Machine) preparePackageSet(stream grpc.ServerStream) error {
	call := &pb.PreparePackageSetCall{}
	return m.prepare(stream, call, func(ctx context.Context, conn *grpc.ClientConn, emit func(*pb.PrepareEvent)) prepared {
		desired := call.GetPackageSet().GetDownloadDelegation()
		if len(desired) == 0 {
			return refused("package_set_invalid", errors.New("PreparePackageSet names no download set"), false)
		}
		selected := &pb.DownloadDelegation{}
		if _, err := canonical.Read(desired, selected); err != nil {
			return refused("package_set_invalid", fmt.Errorf("download document: %w", err), false)
		}
		var document struct {
			Models []json.RawMessage `json:"models"`
		}
		_ = json.Unmarshal(desired, &document)
		if len(document.Models) > 0 && !m.supportsRuntime("runtime-model-preparation/1") {
			return prepared{code: pb.CapabilityUnavailableCode, detail: "the installed Runtime does not support model preparation; update its Runtime wheel"}
		}
		request := &pb.PreparePackageSetRequest{DownloadDelegation: desired, InstallRoot: m.layout.Installs,
			Application: call.Application, PythonRequires: call.PythonRequires, PythonVersion: call.PythonVersion,
			ModelSlotPaths: call.ModelSlotPaths, ImageInventory: call.ImageInventory,
			LockedRequirements: call.LockedRequirements, PackageInterface: call.PackageInterface, Hub: call.Hub}
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED})
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARING})
		var trailer metadata.MD
		result, err := pb.NewRuntimePreparationClient(conn).PreparePackageSet(ctx, request, grpc.Trailer(&trailer))
		if err != nil {
			return runtimeRefused(err, trailer)
		}
		return prepared{result: result}
	})
}

func (m *Machine) preparePrivatePlacement(stream grpc.ServerStream) error {
	call := &pb.PreparePrivatePlacementCall{}
	return m.prepare(stream, call, func(ctx context.Context, conn *grpc.ClientConn, emit func(*pb.PrepareEvent)) prepared {
		selected := call.GetPrivatePlacementSet()
		if selected.GetOperationId() == "" || selected.GetInstallationId() == "" {
			return refused("placement_invalid", errors.New("an unpublished placement names its prepared operation and installation"), false)
		}
		if len(selected.NativeModels) > 0 && !m.supportsRuntime("runtime-model-preparation/1") {
			return prepared{code: pb.CapabilityUnavailableCode, detail: "the installed Runtime does not support model preparation; update its Runtime wheel"}
		}
		if len(selected.ModelChoices) > 0 && !m.supportsRuntime("runtime-model-overrides/1") {
			return prepared{code: pb.CapabilityUnavailableCode, detail: "the installed Runtime cannot prepare model overrides; update its Runtime wheel"}
		}
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED})
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARING})
		request := &pb.PreparePrivatePlacementRequest{OperationId: selected.OperationId, InstallationId: selected.InstallationId,
			DownloadDelegation: selected.DownloadDelegation, DownloadDelegationSignature: selected.DownloadDelegationSignature,
			NativeModels: selected.NativeModels, ModelChoices: selected.ModelChoices,
			SourceCredentials: selected.SourceCredentials, Hub: selected.Hub, Owner: selected.Owner}
		if len(selected.NativeModels) > 0 || len(selected.ModelChoices) > 0 {
			request.Claim = m.claims.claim
		}
		var trailer metadata.MD
		result, err := pb.NewRuntimePreparationClient(conn).PreparePrivatePlacement(ctx, request, grpc.Trailer(&trailer))
		if err != nil {
			return runtimeRefused(err, trailer)
		}
		return prepared{result: result}
	})
}

// prepareLocalPackage installs uploaded local code. With no upload it asks the Runtime to
// reuse an installation it already has; the client uploads on `local_package_reuse_unavailable`.
func (m *Machine) prepareLocalPackage(stream grpc.ServerStream) error {
	call := &pb.PrepareLocalPackageCall{}
	return m.prepare(stream, call, func(ctx context.Context, conn *grpc.ClientConn, emit func(*pb.PrepareEvent)) prepared {
		selected := call.GetLocalPackageSet()
		if call.Hub != "" {
			peer, err := pb.NewRuntimePreparationClient(conn).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
			if err != nil {
				return runtimeRefused(err, nil)
			}
			if peer.GetWireMinor() < 69 {
				return refused(pb.CapabilityUnavailableCode, errors.New("this Runtime cannot select a scoped Hub for local package preparation; update the Runtime"), false)
			}
		}
		if !operationID.MatchString(selected.GetOperationId()) || selected.GetPackage() == nil || len(selected.Files) == 0 ||
			len(selected.Files) > pb.MaxLocalPackageFiles {
			return refused("local_package_invalid", errors.New("the local package selection is invalid"), false)
		}
		request := &pb.PrepareLocalPackageRequest{OperationId: selected.OperationId, Package: selected.Package, Hub: call.Hub,
			SourceArchive: selected.SourceArchive, InstallRoot: m.layout.Installs, PythonRequires: selected.PythonRequires,
			PythonVersion: selected.PythonVersion, DependencyRequirements: selected.DependencyRequirements}
		dir := m.layout.Stage(selected.OperationId)
		_, uploaded := os.Stat(dir)
		var total int64
		for _, file := range selected.Files {
			if !validCarrier(file.GetFilename(), file.GetDigest(), file.GetLength()) {
				return refused("local_package_invalid", errors.New("the local package file set is invalid"), false)
			}
			carrier := &pb.LocalPackageFile{Digest: file.Digest, Filename: file.Filename, Length: file.Length}
			if uploaded == nil {
				path, err := verifiedCarrier(dir, file.Filename, file.Digest, file.Length)
				if err != nil {
					return refused("local_package_transfer_incomplete", err, false)
				}
				carrier.Path = path
			}
			request.Files = append(request.Files, carrier)
			total += int64(file.Length)
		}
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED, TotalBytes: uint64(total), TransferredBytes: uint64(total)})
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARING})
		var trailer metadata.MD
		result, err := pb.NewRuntimePreparationClient(conn).PrepareLocalPackage(ctx, request, grpc.Trailer(&trailer))
		if err != nil {
			if status.Code(err) == codes.FailedPrecondition && strings.HasPrefix(status.Convert(err).Message(), "local_package_reuse_unavailable:") {
				return refused("local_package_reuse_unavailable", err, false)
			}
			return runtimeRefused(err, trailer)
		}
		if uploaded == nil {
			_ = os.RemoveAll(strings.TrimSuffix(dir, "/wheels"))
		}
		return prepared{result: result}
	})
}

func hexBytes(text string) ([]byte, error) { return hex.DecodeString(text) }
