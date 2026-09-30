package host

import (
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"testing"
)

func TestOlderRuntimeRefusesModelOverridesWithoutRejectingBaseSelections(t *testing.T) {
	machine := &Machine{}
	base := &pb.MachineExecutionSubmit{ReleaseRoot: &pb.ReleaseRoot{Models: []*pb.ModelChoice{{Parameter: "model", Repository: "proof/base"}}}}
	if err := machine.requireModelOverrides(base); err != nil {
		t.Fatalf("ordinary older base selection refused: %v", err)
	}
	cases := []*pb.MachineExecutionSubmit{
		{ReleaseRoot: &pb.ReleaseRoot{Models: []*pb.ModelChoice{{Parameter: "model", Adapters: []*pb.DownloadAdapterRef{{Model: "proof/adapter"}}}}}},
		{ReleaseRoot: &pb.ReleaseRoot{Models: []*pb.ModelChoice{{Parameter: "child.models.model", Repository: "proof/base"}}}},
		{CaptureCanonicalBytes: []byte(`{"model_choices":[{"parameter":"child.models.model","adapters":[{"model":"proof/adapter"}]}]}`)},
	}
	for _, request := range cases {
		if err := machine.requireModelOverrides(request); status.Code(err) != codes.FailedPrecondition {
			t.Fatalf("unsupported override did not refuse before forwarding: %v", err)
		}
		machine.runtimeCaps = []string{"runtime-model-overrides/1", "future/7"}
		if err := machine.requireModelOverrides(request); err != nil {
			t.Fatalf("supported override refused: %v", err)
		}
		machine.runtimeCaps = nil
	}
}
