package host

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

// A preparation lands a package set's models through TensorFS and has the Runtime install
// and place it, reporting each stage as a PrepareEvent. The Runtime installs while the
// models still land. A refusal that retrying cannot change ends the stream as REFUSED; any
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
	answer := work(ctx, conn, emit)
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
	emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, PlacementSet: answer.result.PlacementSet,
		InstalledPackage: answer.result.InstalledPackage})
	return nil
}

func downloading(emit func(*pb.PrepareEvent)) func(landed, total int64) {
	return func(landed, total int64) {
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_DOWNLOADING, TotalBytes: uint64(total), TransferredBytes: uint64(landed)})
	}
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
		source, registered := m.modelSource(call.Hub)
		if !registered {
			return refused("hub_unregistered", fmt.Errorf("this machine is not registered at %s", call.Hub), false)
		}
		modelsOnly := call.Application == "" && len(call.LockedRequirements) == 0 && len(selected.Packages) == 0
		request := &pb.PreparePackageSetRequest{DownloadDelegation: desired, InstallRoot: m.layout.Installs,
			Application: call.Application, PythonRequires: call.PythonRequires, PythonVersion: call.PythonVersion,
			ModelSlotPaths: call.ModelSlotPaths, ImageInventory: call.ImageInventory,
			LockedRequirements: call.LockedRequirements, PackageInterface: call.PackageInterface, Hub: call.Hub}
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED})
		warm, stopWarm := context.WithCancel(ctx)
		defer stopWarm()
		warmed := make(chan prepared, 1)
		if !modelsOnly && len(selected.Models) > 0 {
			go func() { // install the release while its models land
				landing := proto.Clone(request).(*pb.PreparePackageSetRequest)
				landing.ModelsLanding = true
				var trailer metadata.MD
				_, err := pb.NewRuntimePreparationClient(conn).PreparePackageSet(warm, landing, grpc.Trailer(&trailer))
				if err != nil && warm.Err() == nil {
					if answer := runtimeRefused(err, trailer); !answer.retry {
						warmed <- answer
						stopWarm()
					}
				}
				close(warmed)
			}()
		} else {
			close(warmed)
		}
		if err := m.tfs.fetch(warm, source, desired, downloading(emit)); err != nil {
			stopWarm()
			if answer, ok := <-warmed; ok {
				return answer
			}
			var fetch *fetchRefusal
			if errors.As(err, &fetch) {
				return prepared{code: fetch.Code, detail: safe(fetch.Detail), retry: fetch.Resumable}
			}
			return refused("model_fetch_failed", err, true)
		}
		if answer, ok := <-warmed; ok {
			return answer
		}
		if modelsOnly {
			raw, err := canonical.Bytes(&pb.PlacementSet{})
			if err != nil {
				return refused("model_download_receipt_invalid", err, false)
			}
			sum := sha256.Sum256(raw)
			return prepared{result: &pb.PreparePackageSetResult{PlacementSet: &pb.DesiredPlacementSet{
				PlacementSetCanonicalBytes: raw, PlacementSetDigest: sum[:]}}}
		}
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
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED})
		desired := selected.DownloadDelegation
		if adapters := adapterDownloads(selected); adapters != nil {
			desired = adapters
		}
		if len(desired) > 0 {
			source, _ := m.modelSource("")
			if err := m.tfs.fetch(ctx, source, desired, downloading(emit)); err != nil {
				var fetch *fetchRefusal
				if errors.As(err, &fetch) {
					return prepared{code: fetch.Code, detail: safe(fetch.Detail), retry: fetch.Resumable}
				}
				return refused("model_fetch_failed", err, true)
			}
		}
		emit(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARING})
		request := &pb.PreparePrivatePlacementRequest{OperationId: selected.OperationId, InstallationId: selected.InstallationId,
			DownloadDelegation: selected.DownloadDelegation, DownloadDelegationSignature: selected.DownloadDelegationSignature,
			NativeModels: selected.NativeModels}
		if len(selected.NativeModels) > 0 {
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

// adapterDownloads adds the adapters native models bind to the download set.
func adapterDownloads(selected *pb.DesiredPrivatePlacementSet) []byte {
	set := &pb.DownloadDelegation{}
	if len(selected.DownloadDelegation) > 0 {
		if _, err := canonical.Read(selected.DownloadDelegation, set); err != nil {
			return nil
		}
	}
	added := false
	for _, model := range selected.NativeModels {
		for _, a := range model.GetAdapters() {
			set.Models = append(set.Models, &pb.DownloadModelRef{Slot: model.Slot, Model: a.Model, Release: a.Release, Lane: a.Lane, Manifest: a.Manifest})
			added = true
		}
	}
	if !added {
		return nil
	}
	raw, err := canonical.Bytes(set)
	if err != nil {
		return nil
	}
	return raw
}

// prepareLocalPackage installs uploaded local code. With no upload it asks the Runtime to
// reuse an installation it already has; the client uploads on `local_package_reuse_unavailable`.
func (m *Machine) prepareLocalPackage(stream grpc.ServerStream) error {
	call := &pb.PrepareLocalPackageCall{}
	return m.prepare(stream, call, func(ctx context.Context, conn *grpc.ClientConn, emit func(*pb.PrepareEvent)) prepared {
		selected := call.GetLocalPackageSet()
		if !operationID.MatchString(selected.GetOperationId()) || selected.GetPackage() == nil || len(selected.Files) == 0 ||
			len(selected.Files) > pb.MaxLocalPackageFiles {
			return refused("local_package_invalid", errors.New("the local package selection is invalid"), false)
		}
		request := &pb.PrepareLocalPackageRequest{OperationId: selected.OperationId, Package: selected.Package,
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
