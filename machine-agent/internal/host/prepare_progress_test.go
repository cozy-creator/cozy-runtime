package host

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/canonical"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

type progressFixture struct {
	Address               string `json:"address"`
	Request               string `json:"request"`
	Manifest              string `json:"manifest"`
	ManifestLength        uint64 `json:"manifest_length"`
	AlternateManifest     string `json:"alternate_manifest"`
	FirstAdapterManifest  string `json:"first_adapter_manifest"`
	SecondAdapterManifest string `json:"second_adapter_manifest"`
	Total                 uint64 `json:"total"`
}

func nativeProgressFixture(t *testing.T, python string) (progressFixture, string) {
	t.Helper()
	root := t.TempDir()
	script, err := filepath.Abs("../../../tests/model_preparation_progress_server.py")
	if err != nil {
		t.Fatal(err)
	}
	log, err := os.Create(filepath.Join(root, "runtime.log"))
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(python, "-B", script, root)
	cmd.Stdout, cmd.Stderr = log, log
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	exited := make(chan struct{})
	var exitErr error
	go func() {
		exitErr = cmd.Wait()
		close(exited)
	}()
	t.Cleanup(func() {
		_ = os.WriteFile(filepath.Join(root, "finish"), nil, 0600)
		select {
		case <-exited:
			if exitErr != nil {
				body, _ := os.ReadFile(log.Name())
				t.Errorf("Runtime fixture: %v\n%s", exitErr, body)
			}
		case <-time.After(30 * time.Second):
			t.Error("test Runtime did not settle its explicit finish")
		}
		_ = log.Close()
	})
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		body, err := os.ReadFile(filepath.Join(root, "ready.json"))
		if err == nil {
			var ready progressFixture
			if err := json.Unmarshal(body, &ready); err == nil {
				return ready, root
			}
		}
		select {
		case <-exited:
			body, _ := os.ReadFile(log.Name())
			t.Fatalf("Runtime fixture exited before ready: %v\n%s", exitErr, body)
		default:
		}
		time.Sleep(10 * time.Millisecond)
	}
	body, _ := os.ReadFile(log.Name())
	t.Fatalf("Runtime fixture did not become ready\n%s", body)
	return progressFixture{}, root
}

func progressClaim(t *testing.T) *pb.Claim {
	t.Helper()
	seed := make([]byte, 32)
	for index := range seed {
		seed[index] = byte(index + 32)
	}
	claim := &pb.Claim{RecordOwnerId: "owner", RecordOwnerEpoch: 1,
		WorkerId: "test-worker", WorkerBootId: "test-boot", WireMinor: pb.WireMinor}
	digest := make([]byte, 32)
	for index := range digest {
		digest[index] = 0xaa
	}
	proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: 1, WorkerId: claim.WorkerId,
		WorkerBootId: claim.WorkerBootId, WorkerTlsCertificateDigest: digest})
	if err != nil {
		t.Fatal(err)
	}
	claim.Proof = ed25519.Sign(ed25519.NewKeyFromSeed(seed), proof)
	return claim
}

func TestPreparationSamplerReadsSharedNativeTransferAfterObserverReconnect(t *testing.T) {
	python := os.Getenv("COZY_RUNTIME_TEST_PYTHON")
	if python == "" {
		t.Skip("set COZY_RUNTIME_TEST_PYTHON to this Runtime's interpreter")
	}
	fixture, root := nativeProgressFixture(t, python)
	conn, err := grpc.NewClient(fixture.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	raw, err := base64.StdEncoding.DecodeString(fixture.Request)
	if err != nil {
		t.Fatal(err)
	}
	request := &pb.PreparePackageSetRequest{}
	if err := proto.Unmarshal(raw, request); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	prepared := make(chan error, 1)
	go func() {
		_, err := pb.NewRuntimePreparationClient(conn).PreparePackageSet(ctx, request)
		prepared <- err
	}()
	m := &Machine{claims: &claims{claim: progressClaim(t)}}
	models := []preparationModel{{repository: "proof/adapter", manifest: fixture.Manifest}}
	events := make(chan *pb.PrepareEvent, 4)
	observe, detach := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		var rows []*pb.PrepareModelProgress
		m.observePreparation(observe, conn, models, func(event *pb.PrepareEvent) { events <- event }, &rows)
	}()
	select {
	case event := <-events:
		if event.Stage != pb.PrepareStage_PREPARE_STAGE_DOWNLOADING || len(event.ModelProgress) != 1 {
			t.Fatalf("partial native transfer not forwarded: %v", event)
		}
		row := event.ModelProgress[0]
		if row.TotalBytes != fixture.Total || row.TransferredBytes == 0 || row.TransferredBytes >= row.TotalBytes {
			t.Fatalf("native partial bytes changed: %v", row)
		}
		if event.TransferredBytes != 0 || event.TotalBytes != 0 {
			t.Fatal("per-model shared object bytes were summed as unique traffic")
		}
	case <-ctx.Done():
		t.Fatal("no native preparation progress before test deadline")
	}
	detach()
	<-done
	select {
	case err := <-prepared:
		t.Fatalf("detaching observation ended a blocked native preparation: %v", err)
	default:
	}
	// A fresh observer reads the same blocked native flight rather than starting a transfer.
	rows := m.preparationProgress(ctx, conn, models)
	if len(rows) != 1 || rows[0].TransferredBytes == 0 || rows[0].TransferredBytes >= rows[0].TotalBytes {
		t.Fatalf("reconnected observation lost native work: %v", rows)
	}
	if err := os.WriteFile(filepath.Join(root, "release"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	if err := <-prepared; err != nil {
		t.Fatal(err)
	}
	rows = m.preparationProgress(ctx, conn, models)
	if len(rows) != 1 || rows[0].TransferredBytes != fixture.Total {
		t.Fatalf("completed native row not retained: %v", rows)
	}
}

func TestPreparationOnOlderRuntimeHasOnlyAnObservationGap(t *testing.T) {
	python := os.Getenv("COZY_RUNTIME_OLD_TEST_PYTHON")
	if python == "" {
		t.Skip("set COZY_RUNTIME_OLD_TEST_PYTHON to the installed public wire70 Runtime")
	}
	fixture, root := nativeProgressFixture(t, python)
	conn, err := grpc.NewClient(fixture.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	raw, err := base64.StdEncoding.DecodeString(fixture.Request)
	if err != nil {
		t.Fatal(err)
	}
	request := &pb.PreparePackageSetRequest{}
	if err := proto.Unmarshal(raw, request); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	prepared := make(chan error, 1)
	go func() {
		_, err := pb.NewRuntimePreparationClient(conn).PreparePackageSet(ctx, request)
		prepared <- err
	}()
	for {
		if _, err := os.Stat(filepath.Join(root, "blocked")); err == nil {
			break
		}
		select {
		case err := <-prepared:
			t.Fatalf("old Runtime did not enter the real native transfer: %v", err)
		case <-ctx.Done():
			t.Fatal("old Runtime did not reach its test barrier")
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
	m := &Machine{claims: &claims{claim: progressClaim(t)}}
	models := []preparationModel{{repository: "proof/adapter", manifest: fixture.Manifest}}
	described, err := pb.NewWorkerControlClient(conn).DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: m.claims.claim})
	if err != nil || described.GetRuntime().GetWireMinor() != 70 {
		t.Fatalf("test requires the actual older wire70 Runtime: %v, %v", described, err)
	}
	if rows := m.preparationProgress(ctx, conn, models); len(rows) != 0 {
		t.Fatalf("older Runtime invented progress rows: %v", rows)
	}
	observe, detach := context.WithCancel(ctx)
	events := make(chan *pb.PrepareEvent, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		var rows []*pb.PrepareModelProgress
		m.observePreparation(observe, conn, models, func(event *pb.PrepareEvent) { events <- event }, &rows)
	}()
	select {
	case event := <-events:
		t.Fatalf("missing optional observation became a synthetic stage: %v", event)
	case err := <-prepared:
		t.Fatalf("optional observation ended a blocked valid preparation: %v", err)
	case <-time.After(preparationSampleInterval + 500*time.Millisecond):
	}
	detach()
	<-done
	if err := os.WriteFile(filepath.Join(root, "release"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	if err := <-prepared; err != nil {
		t.Fatalf("new Host refused old Runtime's successful preparation: %v", err)
	}
}

func TestBlockedPreparationObservationDoesNotDelayNativeSuccess(t *testing.T) {
	python := os.Getenv("COZY_RUNTIME_TEST_PYTHON")
	if python == "" {
		t.Skip("set COZY_RUNTIME_TEST_PYTHON to this Runtime's interpreter")
	}
	fixture, root := nativeProgressFixture(t, python)
	conn, err := grpc.NewClient(fixture.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	raw, err := base64.StdEncoding.DecodeString(fixture.Request)
	if err != nil {
		t.Fatal(err)
	}
	request := &pb.PreparePackageSetRequest{}
	if err := proto.Unmarshal(raw, request); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	prepared := make(chan error, 1)
	go func() {
		_, err := pb.NewRuntimePreparationClient(conn).PreparePackageSet(ctx, request)
		prepared <- err
	}()
	if err := os.WriteFile(filepath.Join(root, "hold-description"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(root, "description-held")); err == nil {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("test did not hold its authenticated read")
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
	m := &Machine{claims: &claims{claim: progressClaim(t)}}
	models := []preparationModel{{repository: "proof/adapter", manifest: fixture.Manifest}}
	observed := make(chan []*pb.PrepareModelProgress, 1)
	go func() { observed <- m.preparationProgress(ctx, conn, models) }()
	select {
	case rows := <-observed:
		if len(rows) != 0 {
			t.Fatalf("blocked observation invented availability: %v", rows)
		}
	case <-time.After(preparationObservationBudget + 2*time.Second):
		t.Fatal("optional observation did not honor its independent read budget")
	}
	if err := os.WriteFile(filepath.Join(root, "release"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-prepared:
		if err != nil {
			t.Fatalf("observation deadline canceled native preparation: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("native preparation did not succeed while the read remained blocked")
	}
	// The same bounded read is used after success; an unresponsive observer cannot
	// hold the terminal preparation response forever.
	go func() { observed <- m.preparationProgress(ctx, conn, models) }()
	select {
	case rows := <-observed:
		if len(rows) != 0 {
			t.Fatal("blocked post-success observation returned a synthetic row")
		}
	case <-time.After(preparationObservationBudget + 2*time.Second):
		t.Fatal("post-success observation exceeded its read budget")
	}
	_ = os.WriteFile(filepath.Join(root, "release-description"), nil, 0600)
}

func TestSharedCheckpointProgressKeepsCallerLabelsAndRevisionIsolated(t *testing.T) {
	python := os.Getenv("COZY_RUNTIME_TEST_PYTHON")
	if python == "" {
		t.Skip("set COZY_RUNTIME_TEST_PYTHON to this Runtime's interpreter")
	}
	fixture, root := nativeProgressFixture(t, python)
	conn, err := grpc.NewClient(fixture.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	raw, _ := base64.StdEncoding.DecodeString(fixture.Request)
	request := &pb.PreparePackageSetRequest{}
	if err := proto.Unmarshal(raw, request); err != nil {
		t.Fatal(err)
	}
	delegation := &pb.DownloadDelegation{}
	if _, err := canonical.Read(request.DownloadDelegation, delegation); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(request.DownloadDelegation, delegation); err != nil {
		t.Fatal(err)
	}
	makeCall := func(pkg, slot, adapter, manifest string) (*pb.PreparePackageSetRequest, *pb.PreparePackageSetCall) {
		selected := proto.Clone(delegation).(*pb.DownloadDelegation)
		selected.Models[0].Package, selected.Models[0].Slot = pkg, slot
		selected.Models[0].Manifest = manifest
		if adapter != "" {
			adapterManifest := fixture.FirstAdapterManifest
			if adapter == "proof/second-adapter" {
				adapterManifest = fixture.SecondAdapterManifest
			}
			selected.Models[0].Adapters = []*pb.DownloadAdapterRef{{Model: adapter, Manifest: adapterManifest}}
		}
		body, err := canonical.Bytes(selected)
		if err != nil {
			t.Fatal(err)
		}
		prepared := proto.Clone(request).(*pb.PreparePackageSetRequest)
		prepared.DownloadDelegation = body
		return prepared, &pb.PreparePackageSetCall{PackageSet: &pb.DesiredPackageSet{DownloadDelegation: body}}
	}
	first, firstCall := makeCall("private/first", "first.models.character", "proof/first-adapter", fixture.Manifest)
	second, secondCall := makeCall("private/second", "second.models.scene", "proof/second-adapter", fixture.Manifest)
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	ended := make(chan error, 2)
	go func() { _, err := pb.NewRuntimePreparationClient(conn).PreparePackageSet(ctx, first); ended <- err }()
	for {
		if _, err := os.Stat(filepath.Join(root, "blocked")); err == nil {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("shared base did not reach the native barrier")
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
	go func() { _, err := pb.NewRuntimePreparationClient(conn).PreparePackageSet(ctx, second); ended <- err }()
	m := &Machine{claims: &claims{claim: progressClaim(t)}}
	for _, call := range []*pb.PreparePackageSetCall{firstCall, secondCall} {
		rows := m.preparationProgress(ctx, conn, preparationModels(call))
		if len(rows) == 0 {
			t.Fatal("actual authenticated read lost shared base progress")
		}
		for _, row := range rows {
			if len(row.GetModel().GetAdapters()) != 0 {
				t.Fatal("another caller's adapter stack escaped a shared source row")
			}
			if row.GetModel().GetModel() == "proof/adapter" {
				selected := &pb.DownloadDelegation{}
				_ = json.Unmarshal(call.PackageSet.DownloadDelegation, selected)
				if row.Model.Package != selected.Models[0].Package || row.Model.Slot != selected.Models[0].Slot {
					t.Fatalf("shared row retained another caller's labels: %v", row)
				}
			}
		}
	}
	shared, err := pb.NewWorkerControlClient(conn).DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: m.claims.claim})
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range shared.Runtime.PreparationProgress {
		ref := row.GetModel()
		if ref.GetPackage() != "" || ref.GetSlot() != "" || ref.GetRelease() != "" || ref.GetLane() != "" || len(ref.GetAdapters()) != 0 {
			t.Fatalf("shared telemetry contains request-owned labels: %v", ref)
		}
	}
	// A repository-only choice and an exact different revision cannot borrow this flight.
	if rows := m.preparationProgress(ctx, conn, []preparationModel{{repository: "proof/adapter"}}); len(rows) != 0 {
		t.Fatalf("unresolved choice borrowed another checkpoint: %v", rows)
	}
	_, differentCall := makeCall("private/next", "next.models.video", "", fixture.AlternateManifest)
	if rows := m.preparationProgress(ctx, conn, preparationModels(differentCall)); len(rows) != 0 {
		t.Fatalf("different real checkpoint borrowed the active revision: %v", rows)
	}
	// Runtime accepts exact immutable manifests without a repository. Its bytes
	// do not authorize copying a repository label from another caller's flight.
	digest, err := hex.DecodeString(strings.TrimPrefix(fixture.Manifest, "sha256:"))
	if err != nil || fixture.ManifestLength == 0 {
		t.Fatalf("fixture needs its real immutable manifest identity: %v", err)
	}
	bareCall := &pb.PreparePrivatePlacementCall{PrivatePlacementSet: &pb.DesiredPrivatePlacementSet{
		ModelChoices: []*pb.ModelChoice{{Parameter: "bare.models.base",
			Manifest: &pb.Ref{Digest: digest, Length: fixture.ManifestLength}}},
	}}
	if _, err := canonical.Bytes(bareCall); err != nil {
		t.Fatalf("valid bare-CID choice failed typed admission: %v", err)
	}
	if rows := m.preparationProgress(ctx, conn, preparationModels(bareCall)); len(rows) != 0 {
		t.Fatalf("bare CID disclosed another caller's repository label: %v", rows)
	}
	_ = os.WriteFile(filepath.Join(root, "release"), nil, 0600)
	for range 2 {
		if err := <-ended; err != nil {
			t.Fatal(err)
		}
	}
	alternate, _ := makeCall("private/next", "next.models.video", "", fixture.AlternateManifest)
	if _, err := pb.NewRuntimePreparationClient(conn).PreparePackageSet(ctx, alternate); err != nil {
		t.Fatal(err)
	}
	rows := m.preparationProgress(ctx, conn, preparationModels(differentCall))
	if len(rows) != 1 || rows[0].Model.Manifest != fixture.AlternateManifest || rows[0].Model.Slot != "next.models.video" {
		t.Fatalf("resolved revision did not preserve its caller's own labels: %v", rows)
	}
}
