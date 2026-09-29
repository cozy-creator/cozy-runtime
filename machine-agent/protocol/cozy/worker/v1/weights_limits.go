package workerprotov1

// Weights, model-source, and local-package host-exchange bounds are generated beside the schema.
const MaxWeightsReceipts = 16
const MaxWeightsReceiptBytes = 1 << 20
const MaxWeightsReceiptAggregateBytes = 4 << 20
const MaxWeightsDeclarationBytes = 1 << 20
const MaxWeightsInventoryBytes = 4 << 20
const MaxWeightsObjects = 65536
const MaxWeightsGrantURLBytes = 16 << 10

// MaxInlineControlBytes is the ceiling every content-bearing bound here must respect: the
// control stream carries control, not content. It has no carve-out: the tensorhub fence in
// scripts/fence.py convicts any content-bearing bound declared above it.
const MaxInlineControlBytes = 4 << 20
const MaxChildArtifactGrants = 32
const MaxRetainedModelResults = 32
const MaxRetainedModelArtifactBytes = 4096
const MaxModelResultPointerBytes = 1024
const MaxActiveChildCalls = 32
const MaxNativeByteReadChunkBytes = 32 << 10
const MaxInputTreeManifestBytes = 1 << 20
const MaxInputTreeChunkBytes = 1 << 20
const MaxModelSourceHeaderBytes = MaxInlineControlBytes
const MaxCheckpointObjects = 128
const MaxModelSourceFiles = 4096
const MaxModelSourceProfiles = 16
const MaxModelSourceMemberBytes = 1024
const MaxModelSourceProfileBytes = 1024
const MaxModelSourceURIBytes = 4096
const MaxModelSourceLicenseBytes = 4096
const MaxModelSourceURLBytes = 16 << 10
const MaxLocalPackageFiles = 129
const MaxLocalPackageFilenameBytes = 255
const MaxModelSlotPaths = 256
const MaxImageInventoryDistributions = 4096
const MaxLockedRequirementsBytes = 1 << 20
