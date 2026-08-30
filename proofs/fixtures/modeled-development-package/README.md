# Modeled development fixture

One visible, stamp-free model entrypoint for the source-checkout development proof. Its authored
default selects `proof/tiny-core@1.0.0` lane `cpu`; the proof builder commits that exact release
to a local TensorFS Repository with one `core.weight` tensor and `{"width":4}` config.

This fixture is deliberately separate from `model-contract-package`: that published proof needs
a mixed metadata/model surface, while the current executor admits one construction per worker.
