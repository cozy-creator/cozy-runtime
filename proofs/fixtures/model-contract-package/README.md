# Model-contract control-plane fixture

This package gives the Runtime proof seats one tiny, stamp-free model, one metadata-only
function, and one hidden function. The published proof checks exact model-contract bytes; the
development proof binds `proof/tiny-core@1.0.0` lane `cpu` from a local TensorFS Repository and
checks the same attempt kernel without a Hub PackageRelease, Qualification, or Environment.
