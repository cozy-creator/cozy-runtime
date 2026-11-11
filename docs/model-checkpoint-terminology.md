# Model and checkpoint names

A named model is a base model or a LoRA. Each immutable saved state is a
checkpoint. Runtime source uses `model_id` and `derived_model_id` for identities,
and keeps existing checkpoint and derivation hashes unchanged.

Published TensorFS 0.24 uses `repo_id` for the Python Writer keyword, checkpoint
identity getter and executor record field. Those spellings remain only at that
SDK/protocol boundary. They do not create a second Runtime identity. The adapter
uses the supported published interface while Runtime variables and operations use
model/checkpoint terminology; no unpublished SDK version or local wheel path is
required to install or synchronize this RC.

When the supported published SDK contract provides the canonical spelling, the
boundary can migrate with its dependency floor and installed-consumer tests.
Actual Git and upstream provider repositories retain their own terminology.
